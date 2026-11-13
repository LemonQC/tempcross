"""One frozen native SmolVLA expert fine-tuning run; full-state resume."""
import concurrent.futures
import time
import json
import os
import numpy as np
import torch
from .runtime import RUN,load,prepare_batch
from .data import Dataset
from baselines.strong.common import out,read,write,digest,resource,now

def checkpoint(policy,optimizer,scheduler,step,protocol_sha,path,full=True):
    payload=dict(model=policy.state_dict(),step=step,update=step,
        official_source='8fff0fde7c79f23a93d845d1a50e985de01f8b8a',protocol_sha256=protocol_sha)
    if full:payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def validate(policy,pre,data,size):
    # Training uses unchanged native Policy.forward/full32 loss. Selection
    # measures only real7 action dimensions and real (non-padded) frames.
    # This is a declared validation metric, not an alternative training loss.
    policy.eval();real_sum=real_count=native_sum=native_count=0
    with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for offset in range(0,len(data.anchors),size):
            torch.manual_seed(17000+offset)
            raw,_=data.batch(offset,min(size,len(data.anchors)-offset));b=prepare_batch(pre,raw)
            images,masks=policy.prepare_images(b);state=policy.prepare_state(b);actions=policy.prepare_action(b)
            loss=policy.model.forward(images,masks,b['observation.language.tokens'],
                b['observation.language.attention_mask'],state,actions)
            valid=(~b['actions_id_pad']).unsqueeze(-1)
            real_sum+=float((loss[...,:7]*valid).sum());real_count+=int(valid.sum())*7
            native_sum+=float((loss*valid).sum());native_count+=loss.numel()
    policy.train()
    assert real_count>0
    return dict(real7_valid_flow_mse=real_sum/real_count,native_full32_masked_mean=native_sum/native_count)

def main():
    directory=RUN/'training/source_target_state';directory.mkdir(parents=True,exist_ok=True)
    protocol=read(out('smolvla')/'protocol.json');freeze=read(out('smolvla')/'freeze.json')
    protocol_sha=digest(out('smolvla')/'protocol.json');assert freeze['protocol_sha256']==protocol_sha
    assert protocol['full_task_smoke_gate']=='PASS' and protocol['real_update_gate']=='PASS'
    for r in protocol['code_files']+protocol['native_files']+protocol['data_files']:assert digest(r['path'])==r['sha256'],r['path']
    assert read(RUN/'data/pack_index.json')['status']=='PASS'
    state_path=directory/'state.json';resume=directory/'resume.pt'
    previous=read(state_path) if state_path.exists() else {}
    if previous.get('status')=='COMPLETE':print('COMPLETE_NO_RETRAIN',flush=True);return
    torch.set_num_threads(4);torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed'])
    torch.cuda.reset_peak_memory_stats();policy,(pre,post)=load();policy.train()
    optimizer=policy.config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler=policy.config.get_scheduler_preset().build(optimizer,num_training_steps=protocol['training_updates'])
    step=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==protocol_sha and saved['step']==previous['saved_step']
        policy.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);step=int(saved['step'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
        del saved
    report=dict(previous,status='RUNNING',actual_training_updates=step,saved_step=step,protocol_sha256=protocol_sha,
        source='official SmolVLAPolicy.forward/native AdamW/native scheduler',
        initial_pretrained_sha256=digest(RUN/'pretrained/model.safetensors'),
        best_validation=previous.get('best_validation',float('inf')),best_step=previous.get('best_step'),
        elapsed_seconds=previous.get('elapsed_seconds',0),start_resources=previous.get('start_resources',resource()))
    write(state_path,report);data=Dataset();val=Dataset('validation');size=protocol['batch_size']
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending=pool.submit(data.batch,step*size,size);started=time.monotonic()
    with (directory/'training.jsonl').open('a') as stream:
        try:
            while step<protocol['training_updates']:
                raw,traces=pending.result()
                if step+1<protocol['training_updates']:pending=pool.submit(data.batch,(step+1)*size,size)
                batch=prepare_batch(pre,raw);optimizer.zero_grad(set_to_none=True)
                loss,info=policy(batch);assert torch.isfinite(loss),'Nonfinite native training loss'
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),policy.config.optimizer_grad_clip_norm)
                assert torch.isfinite(norm),'Nonfinite gradient norm'
                optimizer.step();scheduler.step();step+=1;report['actual_training_updates']=step
                if step==1 or step%protocol['log_interval']==0:
                    row=dict(update=step,metrics=dict(loss=float(loss),grad_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr']),
                        anchors_consumed=step*size,elapsed_seconds=report['elapsed_seconds']+time.monotonic()-started,utc=now())
                    stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
                if step%protocol['validation_interval']==0 or step==protocol['training_updates']:
                    metrics=validate(policy,pre,val,size);assert all(np.isfinite(v) for v in metrics.values())
                    row=dict(update=step,validation=metrics,utc=now());stream.write(json.dumps(row)+'\n');stream.flush()
                    value=metrics['real7_valid_flow_mse']
                    if value<report['best_validation']:
                        checkpoint(policy,optimizer,scheduler,step,protocol_sha,directory/'best.pt',full=False)
                        report.update(best_validation=value,best_step=step)
                    print('VALIDATION',step,metrics,'BEST',report['best_step'],flush=True)
                if step%protocol['save_interval']==0 or step==protocol['training_updates']:
                    assert __import__('shutil').disk_usage(directory).free>protocol['minimum_free_bytes']
                    checkpoint(policy,optimizer,scheduler,step,protocol_sha,resume)
                    report.update(saved_step=step,elapsed_seconds=report['elapsed_seconds']+time.monotonic()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),resources=resource())
                    started=time.monotonic();write(state_path,report)
                if report['elapsed_seconds']+time.monotonic()-started>protocol['maximum_training_seconds']:
                    raise RuntimeError('Frozen training wall-time cap; no automatic budget extension')
            counts=sorted(set(int(s['step'].item()) for s in optimizer.state.values() if 'step' in s))
            assert counts==[protocol['training_updates']]
            best=directory/'best.pt';report.update(status='COMPLETE',actual_training_updates=step,
                optimizer_step_counters=counts,scheduler_last_epoch=scheduler.last_epoch,anchors_consumed=step*size,
                checkpoint=str(best),checkpoint_sha256=digest(best),
                model_selection='minimum fixed target-validation real7 valid-frame flow MSE; earliest tie')
        except BaseException as e:
            import traceback
            report.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc())
            checkpoint(policy,optimizer,scheduler,step,protocol_sha,resume);report['saved_step']=step
            raise
        finally:
            report['elapsed_seconds']+=time.monotonic()-started;write(state_path,report)
            pool.shutdown(wait=True);data.close();val.close()

if __name__=='__main__':main()
