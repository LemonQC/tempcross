"""One pinned official BC-Transformer configuration and one training seed."""
import csv
import json
import math
import time
from common import ROOT, OUT, digest, read, write
from data import Dataset


def main():
    import torch
    import numpy as np
    from baselines.common.model_factory import make_model
    from baselines.common.train import validation
    import robomimic.utils.torch_utils as TorchUtils
    cfg=read(OUT/'downstream_protocol.json')['training']
    directory=ROOT/'downstream/training'
    if directory.exists():
        if (directory/'status.json').exists() and read(directory/'status.json')['status']=='COMPLETE':return
        raise FileExistsError('Preserve running/incomplete training; do not repeat optimizer updates')
    directory.mkdir(parents=True)
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7);torch.cuda.reset_peak_memory_stats()
    dataset=Dataset('train');valid=Dataset('validation')
    vloader=torch.utils.data.DataLoader(valid,batch_size=16,num_workers=0)
    torch.manual_seed(7);np.random.seed(7)
    model,config=make_model('bc_transformer',max_steps=cfg['optimizer_updates'])
    write(directory/'config.json',json.loads(config.dump()))
    initial={k:v.detach().cpu().clone() for k,v in model.nets.named_parameters() if v.requires_grad}
    torch.save(initial,directory/'initial_trainable.pt')
    state=dict(status='RUNNING',actual_optimizer_updates=0,sample_positions=0,seed=7,batch=16,
               parameter_count=sum(p.numel() for p in model.nets.parameters()),validation_history=[],
               initial_trainable_sha256=digest(directory/'initial_trainable.pt'),selected=None)
    import atexit
    def save_on_exit():
        if state['status']!='COMPLETE':
            state['status']='TECHNICAL_FAILURE'
            state['failure_note']='Exited before frozen budget completed; see original training log.'
        write(directory/'status.json',state)
    atexit.register(save_on_exit)
    state['initial_validation']=validation(model,'bc_transformer',vloader)
    valid.close();write(directory/'status.json',state)
    loader=torch.utils.data.DataLoader(dataset,batch_size=16,num_workers=4,pin_memory=True,
        generator=torch.Generator().manual_seed(9981),drop_last=True,prefetch_factor=2)
    original=TorchUtils.backprop_for_loss
    def measured(*args,**kwargs):
        value=float(kwargs['loss'].detach())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite loss before backward')
        result=original(*args,**kwargs)
        state['actual_optimizer_updates']+=1;state['last_loss']=value
        return result
    TorchUtils.backprop_for_loss=measured
    start=time.monotonic();model.set_train()
    with (directory/'updates.csv').open('x',newline='') as f,(directory/'consumed_anchors.csv').open('x',newline='') as trace:
        w=csv.writer(f,lineterminator='\n');w.writerow(['update','sample_positions','loss','lr','elapsed_seconds'])
        tw=csv.writer(trace,lineterminator='\n');tw.writerow(['index','origin','task','episode_id','frame'])
        for step,batch in enumerate(loader,1):
            if time.monotonic()-start>cfg['max_wall_hours']*3600:raise TimeoutError('Frozen training wall cap')
            if step%100==1 and __import__('shutil').disk_usage(ROOT).free<25*1024**3:raise OSError('Frozen disk floor')
            previous=state['actual_optimizer_updates']
            model.train_on_batch(model.process_batch_for_training(batch),epoch=step);model.on_gradient_step()
            assert state['actual_optimizer_updates']==previous+1==step
            state['sample_positions']=step*16
            for j,(origin,task,eid,frame) in enumerate(zip(batch['_origin'],batch['_task'],batch['_episode_id'],batch['_anchor'])):
                tw.writerow([(step-1)*16+j,origin,task,eid,int(frame)])
            w.writerow([step,step*16,state['last_loss'],model.optimizers['policy'].param_groups[0]['lr'],time.monotonic()-start])
            if step*16%12000==0:
                metric=validation(model,'bc_transformer',vloader);metric.update(step=step,sample_positions=step*16)
                state['validation_history'].append(metric)
                print(json.dumps(metric),flush=True)
            if step*16 in cfg['candidate_anchor_positions']:
                metric=state['validation_history'][-1]
                checkpoint=directory/f'positions-{step*16:06d}.pt'
                torch.save(dict(method='bc_transformer',group='B',config=json.loads(config.dump()),model=model.serialize(),
                    target_batch=16,optimizer_updates=step,anchor_positions=step*16),checkpoint)
                candidate=dict(path=str(checkpoint),step=step,anchor_positions=step*16,
                               validation_expected_action_mse=metric['value'],validation_nll=metric['nll'])
                if state['selected'] is None or (candidate['validation_expected_action_mse'],step)<(state['selected']['validation_expected_action_mse'],state['selected']['step']):
                    state['selected']=candidate
            if step%100==0:
                f.flush();trace.flush();state.update(elapsed_seconds=time.monotonic()-start,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
                write(directory/'status.json',state)
                overall=read(OUT/'status.json')
                overall.update(actual_optimizer_updates=step,actual_training_sample_positions=step*16,
                               training_status_path=str(directory/'status.json'))
                write(OUT/'status.json',overall)
    assert state['actual_optimizer_updates']==cfg['optimizer_updates'] and state['sample_positions']==cfg['anchor_positions']
    changes=[]
    for name,value in model.nets.named_parameters():
        if name in initial:
            now=value.detach().cpu();assert torch.isfinite(now).all()
            changes.append(dict(name=name,changed=not torch.equal(now,initial[name]),max_abs_delta=float((now-initial[name]).abs().max())))
    assert any(r['changed'] for r in changes)
    write(directory/'parameter_changes.json',changes)
    state.update(status='COMPLETE',elapsed_seconds=time.monotonic()-start,
                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write(directory/'status.json',state)
    selected=dict(state['selected'],sha256=digest(state['selected']['path']),used_physical_success=False,
                  actual_run_updates=state['actual_optimizer_updates'],actual_run_sample_positions=state['sample_positions'],
                  selection_rule='minimum held-out human expected-action MSE among eight frozen candidates; ties earlier')
    write(OUT/'selection.json',selected)
    dataset.close();valid.close()


if __name__=='__main__':main()
