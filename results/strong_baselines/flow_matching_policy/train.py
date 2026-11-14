"""One finite official visual-CFM run, native EMA, validation-only selection."""
import concurrent.futures
import time
import json
import shutil
import numpy as np
import torch
from .runtime import build,flow_loss,hashes,load
from .native import EMAModel,get_scheduler,ConditionalFlowMatcher,REVISION
from .data import RUN,Dataset
from baselines.strong.common import out,read,write,digest,resource,now

def save(nets,optimizer,scheduler,ema,step,sha,path,full=True):
    state=dict(model=nets.state_dict(),update=step,official_source=REVISION,protocol_sha256=sha,
        checkpoint_model='native EMA' if not full else 'raw training model',ema_optimization_step=ema.optimization_step)
    if full:state.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),ema=ema.state_dict(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)

def validate(nets,data,size=32):
    nets.eval();total=count=0
    with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for offset in range(0,len(data.anchors),size):
            torch.manual_seed(17000+offset)
            batch,_=data.batch(offset,min(size,len(data.anchors)-offset));batch={k:v.cuda() for k,v in batch.items()}
            loss=flow_loss(nets,batch);assert torch.isfinite(loss)
            n=batch['action'].numel();total+=float(loss)*n;count+=n
    nets.train();return total/count

def validation_probe():
    torch.set_num_threads(4);torch.manual_seed(20261113);nets=build();ema=EMAModel(nets.parameters(),power=.75)
    saved=torch.load(RUN/'probe/updated.pt',map_location='cpu',weights_only=False)
    nets.load_state_dict(saved['model'],strict=True);before=hashes(nets)
    data=Dataset('validation');cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone()
    # Deliberately swap between distinct initial EMA and one-update raw weights:
    # prove native store/copy/restore, not a no-op check of equal weights.
    ema.store(nets.parameters());ema.copy_to(nets.parameters())
    first=validate(nets,data);second=validate(nets,data)
    assert first==second and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    ema.restore(nets.parameters());assert hashes(nets)==before
    result=dict(status='PASS_NATIVE_EMA_AND_FIXED_VALIDATION',anchors=256,velocity_mse=first,repeat_velocity_mse=second,
        raw_parameters_exactly_restored=True,CPU_CUDA_RNG_restored=True,optimizer_updates=0,physics_steps=0)
    write(RUN/'validation_probe.json',result);write(out('flow_matching_policy')/'validation_probe.json',result)
    data.close();print(result,flush=True)

def main():
    directory=RUN/'training/source_target_visual';directory.mkdir(parents=True,exist_ok=True)
    protocol=read(out('flow_matching_policy')/'protocol.json');sha=digest(out('flow_matching_policy')/'protocol.json')
    assert read(out('flow_matching_policy')/'freeze.json')['protocol_sha256']==sha
    for r in protocol['code_files']+protocol['native_files']+protocol['data_files']:assert digest(r['path'])==r['sha256'],r['path']
    state_path=directory/'state.json';resume=directory/'resume.pt'
    previous=read(state_path) if state_path.exists() else {}
    if previous.get('status')=='COMPLETE':print('COMPLETE_NO_RETRAIN',flush=True);return
    torch.set_num_threads(4);torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']);torch.cuda.reset_peak_memory_stats()
    nets=build();initial=hashes(nets);expected=read(RUN/'probe/parameter_audit.json')['before']
    assert initial==expected,'Formal initialization must reset exactly to pre-probe random weights'
    write(directory/'initial_parameter_audit.json',dict(status='PASS_RESET_TO_PRE_PROBE',parameters=initial,seed=protocol['seed']))
    optimizer=torch.optim.AdamW(nets.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=protocol['warmup_updates'],num_training_steps=protocol['training_updates'])
    ema=EMAModel(nets.parameters(),power=protocol['ema_power']);step=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['protocol_sha256']==sha and saved['update']==previous['saved_step']
        nets.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        ema.load_state_dict(saved['ema']);ema.to('cuda');step=int(saved['update'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);del saved
    report=dict(previous,status='RUNNING',actual_training_updates=step,saved_step=step,protocol_sha256=sha,
        best_validation=previous.get('best_validation',float('inf')),best_step=previous.get('best_step'),
        source='native HRI visual ResNet18/GN/ConditionalUnet1D, TorchCFM sigma0 velocity loss',
        initial_parameter_audit_sha256=digest(directory/'initial_parameter_audit.json'),
        start_resources=previous.get('start_resources',resource()),elapsed_seconds=previous.get('elapsed_seconds',0))
    write(state_path,report);base_elapsed=report['elapsed_seconds'];started=time.monotonic()
    data=Dataset();val=Dataset('validation');size=protocol['batch_size'];matcher=ConditionalFlowMatcher(sigma=0.)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);pending=pool.submit(data.batch,step*size,size)
    with (directory/'training.jsonl').open('a') as stream:
        try:
            while step<protocol['training_updates']:
                raw,_=pending.result()
                if step+1<protocol['training_updates']:pending=pool.submit(data.batch,(step+1)*size,size)
                batch={k:v.cuda() for k,v in raw.items()};optimizer.zero_grad(set_to_none=True)
                loss=flow_loss(nets,batch,matcher);assert torch.isfinite(loss);loss.backward()
                norm=torch.sqrt(sum((p.grad.float()**2).sum() for p in nets.parameters() if p.grad is not None))
                assert torch.isfinite(norm);optimizer.step();step+=1;scheduler.step();ema.step(nets.parameters())
                report['actual_training_updates']=step
                if step==1 or step%protocol['log_interval']==0:
                    row=dict(update=step,loss=float(loss.detach()),gradient_norm=float(norm),learning_rate=optimizer.param_groups[0]['lr'],
                        anchors_consumed=step*size,elapsed_seconds=base_elapsed+time.monotonic()-started,utc=now())
                    stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
                if step%protocol['validation_interval']==0 or step==protocol['training_updates']:
                    ema.store(nets.parameters());ema.copy_to(nets.parameters())
                    try:
                        metric=validate(nets,val,size);row=dict(update=step,validation=dict(EMA_velocity_mse=metric),utc=now())
                        stream.write(json.dumps(row)+'\n');stream.flush()
                        if metric<report['best_validation']:
                            save(nets,optimizer,scheduler,ema,step,sha,directory/'best.pt',False)
                            report.update(best_validation=metric,best_step=step)
                    finally:ema.restore(nets.parameters())
                    print('VALIDATION',step,metric,'BEST',report['best_step'],flush=True)
                if step%protocol['save_interval']==0 or step==protocol['training_updates']:
                    assert shutil.disk_usage(directory).free>protocol['minimum_free_bytes']
                    save(nets,optimizer,scheduler,ema,step,sha,resume)
                    report.update(saved_step=step,elapsed_seconds=base_elapsed+time.monotonic()-started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),resources=resource())
                    write(state_path,report)
                if base_elapsed+time.monotonic()-started>protocol['maximum_training_seconds']:
                    raise RuntimeError('Frozen wall-time cap; do not extend automatically')
            counts=sorted(set(int(s['step'].item()) for s in optimizer.state.values() if 'step' in s))
            assert counts==[protocol['training_updates']] and scheduler.last_epoch==step and ema.optimization_step==step
            best=directory/'best.pt';report.update(status='COMPLETE',actual_training_updates=step,optimizer_step_counters=counts,
                scheduler_last_epoch=scheduler.last_epoch,ema_optimization_step=ema.optimization_step,anchors_consumed=step*size,
                checkpoint=str(best),checkpoint_sha256=digest(best),selected_checkpoint_training_updates=report['best_step'],
                selected_checkpoint_anchors_consumed=report['best_step']*size,
                selection='minimum fixed256-target-validation EMA velocity MSE; earliest tie; no reference success selection')
        except BaseException as e:
            import traceback
            report.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc())
            save(nets,optimizer,scheduler,ema,step,sha,resume);report['saved_step']=step;raise
        finally:
            report['elapsed_seconds']=base_elapsed+time.monotonic()-started;write(state_path,report)
            pool.shutdown(wait=True);data.close();val.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--validation-probe',action='store_true');a=p.parse_args();validation_probe() if a.validation_probe else main()
