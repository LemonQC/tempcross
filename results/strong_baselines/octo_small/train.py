"""Resumable official Octo full finetuning, guarded by a post-smoke freeze."""
import concurrent.futures
import json
import shutil
import time
from pathlib import Path
import numpy as np
from .runtime import RUN,load_model,optimizer,functions,host_metrics
import jax
import flax
from octo.utils.train_utils import TrainState
from baselines.strong.common import out,write,read,digest,resource,now
from .data import Dataset,DIRECTORY

def atomic_state(state,path):
    temporary=path.with_suffix('.tmp');temporary.write_bytes(flax.serialization.to_bytes(jax.device_get(state)))
    temporary.replace(path)

def validation(state,data,fn,batch_size):
    values=[]
    for i in range(0,len(data.anchors),batch_size):
        batch,_=data.batch(i,min(batch_size,len(data.anchors)-i),state.model)
        loss,_=fn(state.model.params,batch,jax.random.PRNGKey(17000+i))
        values.append((float(loss),len(batch['action'])))
    return sum(a*b for a,b in values)/sum(b for a,b in values)

def main():
    protocol=read(out('octo_small')/'protocol.json');freeze=read(out('octo_small')/'freeze.json')
    assert freeze['protocol_sha256']==digest(out('octo_small')/'protocol.json')
    assert protocol['real_update_gate']=='PASS' and protocol['full_task_smoke_gate']=='PASS'
    pack=read(DIRECTORY/'pack_index.json');assert pack['status']=='PASS' and pack['completed']==640
    for f in read(RUN.parent/'preservation_freeze.json')['files']:assert digest(f['path'])==f['sha256']
    batch_size=protocol['batch_size'];budget=protocol['training_updates']
    directory=RUN/'training/source_target';directory.mkdir(parents=True,exist_ok=True)
    ready=directory/'state.json';resume=directory/'resume.msgpack'
    if ready.exists() and read(ready)['status']=='COMPLETE':
        print('COMPLETE_NO_RETRAIN',flush=True);return
    model=load_model();tx,lr,kwargs=optimizer(model,budget)
    state=TrainState.create(model=model,tx=tx,rng=jax.random.PRNGKey(protocol['seed']))
    previous=read(ready) if ready.exists() else {}
    if resume.exists():
        state=flax.serialization.from_bytes(state,resume.read_bytes())
        assert int(state.step)==previous['saved_step'] and previous['protocol_sha256']==freeze['protocol_sha256']
    update,validate=functions(model,lr);data=Dataset();val=Dataset('validation')
    report=dict(previous,status='RUNNING',protocol_sha256=freeze['protocol_sha256'],
        actual_training_updates=int(state.step),saved_step=int(state.step),optimizer=kwargs,
        initial_pretrained_sha256=digest(RUN/'pretrained/300000/default/checkpoint'),start_resources=resource(),
        best_validation=previous.get('best_validation',float('inf')),best_step=previous.get('best_step'),
        elapsed_seconds=previous.get('elapsed_seconds',0))
    started=time.monotonic();write(ready,report)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    def next_batch(step):return data.batch(step*batch_size,batch_size,model,augment=True)
    pending=pool.submit(next_batch,int(state.step))
    with (directory/'training.jsonl').open('a') as stream:
        try:
            while int(state.step)<budget:
                batch,traces=pending.result();step=int(state.step)
                if step+1<budget:pending=pool.submit(next_batch,step+1)
                state,metrics=update(state,batch);m=host_metrics(metrics)
                assert all(np.isfinite(v) for v in m.values()),'Non-finite native optimization; stop without core change'
                step=int(state.step);report['actual_training_updates']=step
                if step%protocol['log_interval']==0 or step==1:
                    row=dict(update=step,metrics=m,anchors_consumed=step*batch_size,
                        elapsed_seconds=report['elapsed_seconds']+time.monotonic()-started,utc=now())
                    stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
                if step%protocol['validation_interval']==0 or step==budget:
                    value=validation(state,val,validate,batch_size)
                    stream.write(json.dumps(dict(update=step,validation_ddpm_loss=value,utc=now()))+'\n');stream.flush()
                    if value<report['best_validation']:
                        state.model.save_pretrained(step,str(directory/'selected'))
                        report.update(best_validation=value,best_step=step)
                        # Retain one selected native checkpoint; optimizer resume is separate.
                        for old in (directory/'selected').iterdir():
                            if old.is_dir() and old.name.isdigit() and old.name!=str(step):shutil.rmtree(old)
                    print('VALIDATION',step,value,'BEST',report['best_step'],flush=True)
                if step%protocol['save_interval']==0 or step==budget:
                    assert shutil.disk_usage(directory).free>protocol['minimum_free_bytes']
                    atomic_state(state,resume)
                    report.update(saved_step=step,resources=resource(),
                        elapsed_seconds=report['elapsed_seconds']+time.monotonic()-started)
                    started=time.monotonic();write(ready,report)
                if report['elapsed_seconds']+time.monotonic()-started>protocol['maximum_training_seconds']:
                    raise RuntimeError('Frozen training wall-time cap reached; preserve partial result, do not enlarge budget')
            checkpoint=directory/'selected'/str(report['best_step'])/'default/checkpoint'
            report.update(status='COMPLETE',actual_training_updates=int(state.step),
                checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                model_selection='minimum fixed validation DDPM loss; earliest tie',
                anchors_consumed=int(state.step)*batch_size,resources=resource())
        except BaseException as error:
            import traceback
            report.update(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc())
            # If a batch failed, the last successful complete state can be resumed.
            atomic_state(state,resume);report['saved_step']=int(state.step)
            raise
        finally:
            report['elapsed_seconds']+=time.monotonic()-started
            write(ready,report);pool.shutdown(wait=True);data.close();val.close()

if __name__=='__main__':main()
