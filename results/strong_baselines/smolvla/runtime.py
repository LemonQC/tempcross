"""Pinned native SmolVLA core and processors; no surrogate learning module."""
import os
from pathlib import Path
from baselines.strong.common import ROOT,read,write,digest,resource,now
RUN=ROOT/'smolvla'
os.environ['HF_HOME']=str(RUN/'huggingface')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys
sys.path.insert(0,'/home/users/lemon/baseline-benchmark/vendor/lerobot-smolvla/src')
import hashlib
import time
import numpy as np
import torch
from lerobot.configs.types import PolicyFeature,FeatureType
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from .data import Dataset,DIRECTORY

def config(device='cpu'):
    cfg=PreTrainedConfig.from_pretrained(RUN/'pretrained')
    assert isinstance(cfg,SmolVLAConfig)
    cfg.device=device;cfg.push_to_hub=False;cfg.load_vlm_weights=False
    cfg.input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(9,)),
        'observation.images.camera1':PolicyFeature(type=FeatureType.VISUAL,shape=(3,180,320))}
    cfg.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))}
    assert cfg.max_state_dim==32 and cfg.max_action_dim==32 and cfg.chunk_size==50 and cfg.n_action_steps==50
    assert cfg.num_steps==10 and cfg.rtc_config is None
    return cfg

def processors(cfg):
    stats={k:{a:torch.tensor(b,dtype=torch.float32) for a,b in v.items()}
        for k,v in read(DIRECTORY/'statistics.json').items()}
    return make_smolvla_pre_post_processors(cfg,dataset_stats=stats)

def prepare_batch(pre,batch):
    # Native transition conversion drops this dataset-only boundary flag.
    # Carry it across preprocessing under the exact key consumed by the
    # unchanged official SmolVLAPolicy.forward, never supervise padded futures.
    mask=batch.get('actions_id_pad')
    result=pre(batch)
    if mask is not None:result['actions_id_pad']=mask.to(result['action'].device)
    return result

def load(checkpoint=None,device='cuda'):
    if device=='cuda':assert torch.cuda.is_available(), 'No CPU fallback for a GPU feasibility probe'
    cfg=config('cpu')
    if checkpoint is None:
        # Native strict loader: complete checkpoint, including the frozen VLM.
        policy=SmolVLAPolicy.from_pretrained(RUN/'pretrained',config=cfg,strict=True,local_files_only=True)
    else:
        policy=SmolVLAPolicy(cfg)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['official_source']=='8fff0fde7c79f23a93d845d1a50e985de01f8b8a'
        policy.load_state_dict(saved['model'],strict=True)
    cfg.device=device;policy.to(device)
    return policy,processors(cfg)

def parameter_hashes(policy):
    return {k:dict(sha256=hashlib.sha256(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        parameters=v.numel(),dtype=str(v.dtype),trainable=v.requires_grad) for k,v in policy.named_parameters()}

def probe(size,tag=None):
    directory=RUN/'probe'/(tag or f'batch-{size}');directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists() and read(directory/'result.json')['status']=='PASS':
        print('PASS_ALREADY_SAVED_NO_REPEAT',flush=True);return
    report=dict(status='RUNNING',actual_training_updates=0,batch_size=size,start=resource(),utc=now())
    write(directory/'result.json',report);data=None
    try:
        torch.set_num_threads(4);torch.manual_seed(20261113);torch.cuda.reset_peak_memory_stats()
        policy,(pre,post)=load();before=parameter_hashes(policy);data=Dataset()
        batch,traces=data.batch(0,size);batch=prepare_batch(pre,batch)
        assert batch['action'].shape==(size,50,7) and batch['observation.state'].shape==(size,9)
        images,masks=policy.prepare_images(batch)
        assert len(images)==1 and masks[0].all() and images[0].shape==(size,3,512,512)
        print('NATIVE_MODEL_BATCH_READY',torch.cuda.get_device_name(),flush=True)
        optimizer=policy.config.get_optimizer_preset().build(policy.get_optim_params())
        policy.train();times=[];metrics=[]
        for i in range(3):
            torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            loss,info=policy(batch);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),policy.config.optimizer_grad_clip_norm)
            assert torch.isfinite(norm);optimizer.step();torch.cuda.synchronize()
            times.append(time.monotonic()-start);metrics.append(dict(loss=loss.item(),grad_norm=norm.item()))
            report['actual_training_updates']=i+1;write(directory/'result.json',report)
            print('REAL_NATIVE_UPDATE',i+1,metrics[-1],'seconds',times[-1],flush=True)
        after=parameter_hashes(policy);changed=[k for k in before if before[k]['sha256']!=after[k]['sha256']]
        assert changed and all(before[k]['trainable'] for k in changed)
        assert any('lm_expert' in k for k in changed) and any('state_proj' in k for k in changed)
        counts=sorted(set(int(v['step'].item()) for v in optimizer.state.values() if 'step' in v))
        assert counts==[3]
        checkpoint=directory/'updated.pt'
        torch.save(dict(model=policy.state_dict(),update=3,official_source='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',
            config=policy.config.to_dict() if hasattr(policy.config,'to_dict') else str(policy.config)),checkpoint)
        write(directory/'parameter_audit.json',dict(before=before,after=after,changed=changed))
        report.update(status='PASS',actual_training_updates=3,optimizer_step_counters=counts,
            changed_tensors=len(changed),frozen_tensors_changed=0,
            total_parameters=sum(x['parameters'] for x in before.values()),trainable_parameters=sum(x['parameters'] for x in before.values() if x['trainable']),
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),metrics=metrics,
            update_seconds=times,warm_update_seconds=float(np.mean(times[1:])),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            source='official SmolVLAPolicy.forward/native flow matching/official optimizer preset',
            native_future_padding_mask_preserved=True,traces=traces,end=resource())
    except BaseException as e:
        import traceback
        report.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc(),end=resource());raise
    finally:
        write(directory/'result.json',report)
        if data is not None:data.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--probe-batch',type=int,required=True);p.add_argument('--tag');args=p.parse_args();probe(args.probe_batch,args.tag)
