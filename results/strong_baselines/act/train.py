"""Actual official ACT optimizer calls, frozen budget, validation-only selection."""
import argparse
import csv
import math
import time
import traceback
import os
import shutil
from pathlib import Path
from baselines.strong.common import ROOT,out,read,write,digest,resource,now
from baselines.strong.data import Dataset
from results.strong_baselines.act.model import make

def validation(model,loader):
    import torch,numpy as np
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();torch.manual_seed(20261112)
    model.eval();rows=[]
    with torch.no_grad():
        for b in loader:
            result=model(b['language'].cuda(),b['image'].cuda(),b['actions'].cuda(),b['is_pad'].cuda())
            rows.append({k:float(v) for k,v in result.items()})
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda);model.train()
    result={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    assert all(math.isfinite(x) for x in result.values());return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['target_only','source_target'],required=True);p.add_argument('--probe',action='store_true');a=p.parse_args()
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7);torch.cuda.reset_peak_memory_stats()
    cfg=read(out('act')/('candidate_protocol.json' if a.probe else 'protocol.json'))
    if not a.probe:
        frozen=read(out('act')/'freeze.json');assert frozen['protocol_sha256']==digest(out('act')/'protocol.json')
    directory=ROOT/'act'/('probe' if a.probe else 'training')/a.group;directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'status.json';resume=directory/'latest.pt'
    if state_path.exists() and read(state_path)['status']=='COMPLETE':print('Reusing completed stage',directory);return
    if state_path.exists() and not resume.exists():raise RuntimeError('An incomplete non-resumable attempt must be preserved and reviewed')
    target_batch=cfg['target_batch'];budget=1 if a.probe else cfg['optimizer_updates_per_group']
    state=dict(status='RUNNING',group=a.group,role='DISCARDED_PROBE' if a.probe else 'FROZEN_FORMAL_TRAINING',actual_optimizer_updates=0,target_anchor_exposures=0,source_anchor_exposures=0,started_utc=now(),validation_history=[],best=None)
    model=make();optimizer=model.configure_optimizers();stats=read(ROOT/'act/data'/(a.group+'_stats.json'))
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);state=saved['state']
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.cuda()
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
    dataset=Dataset('act',a.group,offset=state['actual_optimizer_updates']*target_batch,limit=(budget-state['actual_optimizer_updates'])*target_batch)
    valid=Dataset('act',a.group,split='validation');vloader=torch.utils.data.DataLoader(valid,batch_size=16,num_workers=0)
    batch_size=target_batch*(2 if a.group=='source_target' else 1)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=4 if not a.probe else 0,pin_memory=True,
        generator=torch.Generator().manual_seed(9981),drop_last=True)
    start=time.monotonic();write(state_path,state);write(directory/'resource_start.json',resource())
    initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()} if a.probe else None
    def save():
        state.update(elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        obj=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),stats=stats,state=state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
        tmp=directory/'latest.tmp';torch.save(obj,tmp);tmp.replace(resume);write(state_path,state)
    mode='a' if state['actual_optimizer_updates'] else 'x'
    try:
        with (directory/'updates.csv').open(mode,newline='') as log,(directory/'consumed_anchors.csv').open(mode,newline='') as trace:
            w=csv.writer(log,lineterminator='\n');tw=csv.writer(trace,lineterminator='\n')
            if state['actual_optimizer_updates']==0:w.writerow(['update','loss','l1','kl','seconds']);tw.writerow(['update','domain','episode_id','frame'])
            model.train()
            for b in loader:
                if time.monotonic()-start>cfg['max_wall_hours_per_training_group']*3600:raise TimeoutError('Frozen training wall cap')
                if shutil.disk_usage(ROOT).free<cfg['disk_floor_gib']*1024**3:raise OSError('Frozen disk floor')
                t=time.monotonic();optimizer.zero_grad(set_to_none=True)
                values=model(b['language'].cuda(),b['image'].cuda(),b['actions'].cuda(),b['is_pad'].cuda())
                assert all(torch.isfinite(x).all() for x in values.values())
                values['loss'].backward()
                assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
                optimizer.step();torch.cuda.synchronize();state['actual_optimizer_updates']+=1;step=state['actual_optimizer_updates']
                state['target_anchor_exposures']=step*target_batch;state['source_anchor_exposures']=step*target_batch if a.group=='source_target' else 0
                w.writerow([step,float(values['loss']),float(values['l1']),float(values['kl']),time.monotonic()-t])
                for domain,eid,frame in zip(b['domain'],b['episode_id'],b['frame']):tw.writerow([step,domain,eid,int(frame)])
                if not a.probe and step%cfg['validation_every_updates']==0:
                    metric=validation(model,vloader);metric['step']=step;state['validation_history'].append(metric)
                    if state['best'] is None or (metric['loss'],step)<(state['best']['loss'],state['best']['step']):
                        torch.save(dict(model=model.state_dict(),stats=stats,step=step,group=a.group),directory/'best.pt');state['best']=metric
                    print(__import__('json').dumps(dict(group=a.group,**metric)),flush=True)
                if step%250==0 or a.probe:
                    log.flush();trace.flush();save();print('SAVED',a.group,step,state['elapsed_seconds'],flush=True)
            assert state['actual_optimizer_updates']==budget
            if a.probe:
                changed=[k for k,v in model.named_parameters() if not torch.equal(initial[k],v.detach().cpu())]
                assert changed;state['changed_parameter_tensors']=len(changed)
                write(directory/'parameter_changes.json',dict(changed_tensors=changed,all_parameters_finite=all(torch.isfinite(v).all().item() for v in model.parameters())))
            state['status']='COMPLETE';save()
            if not a.probe:
                write(out('act')/(a.group+'_selection.json'),dict(checkpoint=str(directory/'best.pt'),sha256=digest(directory/'best.pt'),selected=state['best'],actual_optimizer_updates=budget,target_anchor_exposures=state['target_anchor_exposures'],source_anchor_exposures=state['source_anchor_exposures'],used_physical_success=False))
    except BaseException as e:
        state.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc());write(directory/'error.json',state)
        save();raise
    finally:
        dataset.close();valid.close();write(state_path,state)
        write(out('act')/(('probe_' if a.probe else 'training_')+a.group+'.json'),dict(state,server_status=str(state_path),checkpoint=str(resume),checkpoint_sha256=digest(resume) if resume.exists() else None))

if __name__=='__main__':main()
