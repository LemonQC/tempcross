"""Official HRI-EU visual flow policy with declared observation/action adapters."""
import os
os.environ.setdefault('USE_TF','0');os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import time
import hashlib
import numpy as np
import torch
from .native import ConditionalUnet1D,get_resnet,replace_bn_with_gn,ConditionalFlowMatcher,EMAModel,REVISION,unnormalize_data
from .data import RUN,DIRECTORY,Dataset
from baselines.strong.common import out,read,write,digest,resource,now

def build(device='cuda'):
    assert torch.cuda.is_available() or device=='cpu','No silent CPU fallback'
    # These are the author's visual encoder, GN replacement and original
    # default UNet widths/kernel/time embedding. Only interface dimensions vary.
    return torch.nn.ModuleDict({
        'vision_encoder':replace_bn_with_gn(get_resnet('resnet18')),
        'noise_pred_net':ConditionalUnet1D(input_dim=7,global_cond_dim=512+9+768)
    }).to(device)

def condition(nets,batch):
    features=nets['vision_encoder'](batch['image'])
    assert features.shape==(len(batch['image']),512)
    return torch.cat([features,batch['state'],batch['language']],dim=-1)

def flow_loss(nets,batch,matcher=None):
    matcher=matcher or ConditionalFlowMatcher(sigma=0.)
    x1=batch['action'];x0=torch.randn(x1.shape,device=x1.device)
    t,xt,ut=matcher.sample_location_and_conditional_flow(x0,x1)
    vt=nets['noise_pred_net'](xt,t,global_cond=condition(nets,batch))
    return torch.mean((vt-ut)**2)

@torch.no_grad()
def sample(nets,batch):
    # Exact public visual and Robomimic examples: uniform sampling, t=0,
    # one Euler step, predict16 then execute the first8 (obs_horizon=1).
    # Their training samples Gaussian x0. Preserve and disclose this mismatch;
    # do not silently repair the author's stochastic generation mechanism.
    noise=torch.rand(len(batch['image']),16,7,device=batch['image'].device)
    t=torch.zeros(len(noise),device=noise.device)
    velocity=nets['noise_pred_net'](noise,t,global_cond=condition(nets,batch))
    return noise+velocity,noise,velocity

def load(checkpoint):
    nets=build();saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['official_source']==REVISION
    nets.load_state_dict(saved['model'],strict=True);nets.eval()
    return nets,saved

def hashes(nets):
    return {n:dict(shape=list(v.shape),parameters=v.numel(),
        sha256=hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()) for n,v in nets.named_parameters()}

def probe():
    directory=RUN/'probe';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'result.json'
    if path.exists() and read(path)['status']=='PASS':print('PASS_PROBE_ALREADY_DONE_NO_REPEAT');return
    report=dict(status='RUNNING',actual_optimizer_calls=0,batch_size=32,start=resource(),utc=now())
    write(path,report);data=None
    try:
        torch.set_num_threads(4);torch.manual_seed(20261113);torch.cuda.reset_peak_memory_stats()
        nets=build();before=hashes(nets);data=Dataset();batch,traces=data.batch(0,32)
        batch={k:v.cuda() for k,v in batch.items()};optimizer=torch.optim.AdamW(nets.parameters(),lr=1e-4,weight_decay=1e-6)
        ema=EMAModel(nets.parameters(),power=.75)
        torch.cuda.synchronize();started=time.monotonic()
        # Diagnostic discarded update uses official peak LR without scheduler;
        # avoid a zero-LR first warmup call being mistaken for learned weights.
        optimizer.zero_grad(set_to_none=True);loss=flow_loss(nets,batch)
        assert torch.isfinite(loss);loss.backward()
        norm=torch.sqrt(sum((p.grad.float()**2).sum() for p in nets.parameters() if p.grad is not None))
        assert torch.isfinite(norm);optimizer.step();ema.step(nets.parameters())
        report['actual_optimizer_calls']=1;torch.cuda.synchronize();seconds=time.monotonic()-started
        after=hashes(nets);changed=[n for n in before if before[n]['sha256']!=after[n]['sha256']]
        assert any(n.startswith('vision_encoder.') for n in changed) and any(n.startswith('noise_pred_net.') for n in changed)
        counters=sorted(set(int(s['step'].item()) for s in optimizer.state.values() if 'step' in s));assert counters==[1]
        nets.eval();single={k:v[:1] for k,v in batch.items()};times=[]
        for _ in range(3):
            torch.cuda.synchronize();started=time.monotonic();normalized,noise,velocity=sample(nets,single)
            torch.cuda.synchronize();times.append(time.monotonic()-started)
            assert normalized.shape==(1,16,7) and torch.isfinite(normalized).all()
            torch.testing.assert_close(normalized,noise+velocity);assert noise.min()>=0 and noise.max()<1
        checkpoint=directory/'updated.pt'
        torch.save(dict(model=nets.state_dict(),update=1,official_source=REVISION,
            initialization='from scratch, seed20261113; no partial pretrained loading'),checkpoint)
        write(directory/'parameter_audit.json',dict(before=before,after=after,changed=changed))
        report.update(status='PASS',actual_optimizer_calls=1,optimizer_step_counters=counters,changed_tensors=len(changed),
            parameters=sum(p.numel() for p in nets.parameters()),loss=float(loss.detach()),gradient_norm=float(norm),
            real_update_seconds=seconds,one_step_inference_seconds=times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),traces=traces,end=resource(),
            core='native HRI ConditionalUnet1D+GN ResNet18; TorchCFM ConditionalFlowMatcher sigma0; velocity MSE',
            diagnostic_only=True,probe_discarded_before_formal_training=True,probe_scheduler_used=False,
            inference_source='uniform[0,1], single Euler at t0; original Gaussian-training mismatch retained')
        print('PASS_ONE_REAL_NATIVE_OPTIMIZER_UPDATE',report['loss'],seconds,'parameters',report['parameters'],flush=True)
    except BaseException as e:
        import traceback
        report.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        write(path,report);write(out('flow_matching_policy')/'probe.json',report)
        if data is not None:data.close()

if __name__=='__main__':probe()
