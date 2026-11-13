"""One measured native BC training run, fixed final budget, resumable checkpoints."""
import argparse
import csv
import math
import shutil
import signal
import time
import traceback
from baselines.strong.common import PROJECT,ROOT,out,read,write,digest,resource,now
from results.strong_baselines.mimicgen_v2.data import Dataset
from results.strong_baselines.mimicgen_v2.model import make

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7);torch.cuda.reset_peak_memory_stats()
    folder=out('mimicgen_v2');cfg=read(folder/('candidate_protocol.json' if args.probe else 'protocol.json'))
    assert str(ROOT)==cfg['server_root']
    if not args.probe:
        freeze=read(folder/'freeze.json');assert digest(folder/'protocol.json')==freeze['protocol_sha256']
        for p,sha in freeze['code_sha256'].items():assert digest(PROJECT/p)==sha,p
    directory=ROOT/'mimicgen_v2'/('probe' if args.probe else 'training');directory.mkdir(parents=True,exist_ok=True)
    latest=directory/'latest.pt';status_path=directory/'status.json'
    if status_path.exists() and read(status_path)['status']=='COMPLETE':print('REUSE COMPLETE',directory);return
    if status_path.exists() and not latest.exists():raise RuntimeError('Preserve uncheckpointed attempt; no automatic repeat')
    budget=1 if args.probe else cfg['optimizer_updates'];batch_size=cfg['batch']
    state=dict(status='RUNNING',role='DISCARDED_PROBE' if args.probe else 'FROZEN_FORMAL_TRAINING',actual_optimizer_updates=0,
        anchor_exposures=0,human_anchor_exposures=0,generated_anchor_exposures=0,validation_history=[],saved_elapsed_seconds=0.,started_utc=now())
    dataset=Dataset(offset=0,limit=batch_size);valid=Dataset(split='validation')
    # Random initialization is copied from the saved original control before any update.
    torch.manual_seed(7);np.random.seed(7);model,config=make()
    initial={k:v.detach().cpu().clone() for k,v in model.nets.named_parameters() if v.requires_grad} if args.probe else None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);model.deserialize(saved['model'],load_optimizers=True);state=saved['state']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
        with (directory/'updates.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==state['actual_optimizer_updates'],'Uncheckpointed log suffix requires explicit review'
    dataset.close();dataset=Dataset(offset=state['actual_optimizer_updates']*batch_size,limit=(budget-state['actual_optimizer_updates'])*batch_size)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=0 if args.probe else 4,pin_memory=True,
        generator=torch.Generator().manual_seed(9981),drop_last=True)
    vloader=torch.utils.data.DataLoader(valid,batch_size=16,num_workers=0)
    from baselines.common.train import validation
    import robomimic.utils.torch_utils as TorchUtils
    original=TorchUtils.backprop_for_loss;native_calls=[]
    def measured(*a,**kw):
        assert torch.isfinite(kw['loss']);result=original(*a,**kw);native_calls.append(float(kw['loss'].detach()));return result
    TorchUtils.backprop_for_loss=measured
    start=time.monotonic();prior=state['saved_elapsed_seconds'];requested=[]
    signal.signal(signal.SIGTERM,lambda *_:requested.append('SIGTERM'))
    write(directory/'resource_start.json',resource());write(status_path,state)
    write(directory/'config.json',__import__('json').loads(config.dump()))
    def save(path):
        state.update(elapsed_seconds=prior+time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        state['saved_elapsed_seconds']=state['elapsed_seconds']
        obj=dict(method='bc_transformer',group='MIMICGEN_V2_AUGMENTATION',config=__import__('json').loads(config.dump()),model=model.serialize(),state=dict(state),
            optimizer_updates=state['actual_optimizer_updates'],anchor_positions=state['anchor_exposures'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
        tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
    mode='a' if state['actual_optimizer_updates'] else 'x'
    try:
        model.set_train()
        with (directory/'updates.csv').open(mode,newline='') as log,(directory/'consumed_anchors.csv').open(mode,newline='') as trace:
            writer=csv.writer(log,lineterminator='\n');tw=csv.writer(trace,lineterminator='\n')
            if not state['actual_optimizer_updates']:
                writer.writerow(['update','native_nll_loss','human_exposures','generated_exposures','step_seconds']);tw.writerow(['update','origin','task','episode_id','frame'])
            for b in loader:
                if prior+time.monotonic()-start>cfg['max_training_wall_hours']*3600:raise TimeoutError('Frozen training wall cap')
                if shutil.disk_usage(ROOT).free<cfg['disk_floor_gib']*1024**3:raise OSError('Frozen disk floor')
                tick=time.monotonic();before=len(native_calls);step=state['actual_optimizer_updates']+1
                model.train_on_batch(model.process_batch_for_training(b),epoch=step);model.on_gradient_step()
                assert len(native_calls)==before+1 and math.isfinite(native_calls[-1]);torch.cuda.synchronize()
                state['actual_optimizer_updates']=step;state['anchor_exposures']=step*batch_size
                for origin,task,eid,frame in zip(b['origin'],b['task'],b['episode_id'],b['frame']):
                    state[origin+'_anchor_exposures']+=1;tw.writerow([step,origin,task,eid,int(frame)])
                writer.writerow([step,native_calls[-1],state['human_anchor_exposures'],state['generated_anchor_exposures'],time.monotonic()-tick])
                if not args.probe and step%cfg['validation_every_updates']==0:
                    metric=validation(model,'bc_transformer',vloader);metric['step']=step;state['validation_history'].append(metric)
                    print(__import__('json').dumps(metric),flush=True)
                if step%250==0 or args.probe or requested:
                    log.flush();trace.flush();save(latest);write(status_path,state)
                    print('SAVED',step,state['elapsed_seconds'],flush=True)
                if requested:raise InterruptedError('Saved requested stop')
            assert state['actual_optimizer_updates']==budget and state['human_anchor_exposures']+state['generated_anchor_exposures']==budget*batch_size
            assert all(torch.isfinite(p).all() for p in model.nets.parameters())
            if args.probe:
                changed=[k for k,v in model.nets.named_parameters() if k in initial and not torch.equal(v.detach().cpu(),initial[k])]
                assert changed and state['generated_anchor_exposures']>0 and state['human_anchor_exposures']>0
                write(directory/'parameter_changes.json',dict(changed=changed,changed_parameter_tensors=len(changed),exact_old_random_initialization=True))
                state['changed_parameter_tensors']=len(changed)
            state['status']='COMPLETE';save(latest);write(status_path,state)
            if not args.probe:
                final=directory/'final.pt'
                if final.exists():raise FileExistsError('Preserve prior final checkpoint')
                final.hardlink_to(latest)
                write(folder/'selection.json',dict(checkpoint=str(final),sha256=digest(final),selected_step=budget,actual_optimizer_updates=budget,
                    total_anchor_exposures=state['anchor_exposures'],human_anchor_exposures=state['human_anchor_exposures'],generated_anchor_exposures=state['generated_anchor_exposures'],
                    selection='fixed final48000 matched to old human control; not chosen by validation or reference success',used_physical_success=False))
    except BaseException as e:
        state.update(status='CHECKPOINTED_STOP' if requested else 'TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc());save(latest);write(directory/'error.json',state);raise
    finally:
        dataset.close();valid.close();write(status_path,state)
        write(folder/('probe_training.json' if args.probe else 'training.json'),dict(state,server_status=str(status_path),checkpoint=str(latest),checkpoint_sha256=digest(latest) if latest.exists() else None))

if __name__=='__main__':main()
