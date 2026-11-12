"""Two native optimizer stages, fixed budget, validation-only selection, resume."""
import argparse
import csv
import math
import os
import shutil
import signal
import time
import traceback
from baselines.strong.common import PROJECT,ROOT,out,read,write,digest,resource,now
from results.strong_baselines.vqbet.data import Dataset
from results.strong_baselines.vqbet.model import make,make_vq

def validate(vq,model,loader):
    import torch,numpy as np
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();torch.manual_seed(20261112)
    vq.vq_layer.eval()
    if model is not None:model.eval()
    records=[]
    with torch.no_grad():
        for b in loader:
            a=b['actions'].cuda()
            if model is None:
                _,code,reconstruction,_=vq.get_code(a,required_recon=True)
                records.append(dict(loss=float((reconstruction-a[:,0]).abs().mean()),mse=float((reconstruction-a[:,0]).square().mean())))
            else:
                _,loss,metrics=model(b['features'].cuda(),b['language'].cuda(),a)
                records.append({k:float(v) for k,v in metrics.items()})
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda)
    if model is None:vq.vq_layer.train()
    else:model.train();model.resnet.eval()
    result={k:float(np.mean([r[k] for r in records])) for k in records[0]}
    if model is not None:result['loss']=result['total_loss']
    assert all(math.isfinite(v) for v in result.values())
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['target_only','source_target'],required=True)
    parser.add_argument('--stage',choices=['vqvae','bet'],required=True);parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    import torch,numpy as np
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7);torch.cuda.reset_peak_memory_stats()
    cfg=read(out('vqbet')/('candidate_protocol.json' if args.probe else 'protocol.json'))
    if not args.probe:
        frozen=read(out('vqbet')/'freeze.json');assert frozen['protocol_sha256']==digest(out('vqbet')/'protocol.json')
        for relative,sha in frozen['code_sha256'].items():assert digest(out('vqbet')/relative)==sha,relative
        for relative,sha in frozen['shared_code_sha256'].items():assert digest(PROJECT/relative)==sha,relative
        assert digest(ROOT/'vqbet/data/complete.json')==frozen['data_sha256']
    root=ROOT/'vqbet'/('probe' if args.probe else 'training');directory=root/args.group/args.stage;directory.mkdir(parents=True,exist_ok=True)
    statepath=directory/'status.json';latest=directory/'latest.pt'
    if statepath.exists() and read(statepath)['status']=='COMPLETE':print('REUSE COMPLETE',directory);return
    if statepath.exists() and not latest.exists():raise RuntimeError('Preserve incomplete uncheckpointed attempt before any retry')
    stage=args.stage;group=args.group;batch=cfg[stage+'_target_batch'];budget=1 if args.probe else cfg[stage+'_updates']
    state=dict(status='RUNNING',group=group,stage=stage,role='DISCARDED_PROBE' if args.probe else 'FROZEN_FORMAL_TRAINING',
        actual_stage_updates=0,actual_optimizer_calls=0,started_utc=now(),target_anchor_exposures=0,source_anchor_exposures=0,best=None,validation_history=[],saved_elapsed_seconds=0.)
    state['update_log_loss_definition']='native reconstruction L1 +5*commitment; auxiliary=commitment' if stage=='vqvae' else 'native total focal-code +100*offset L1; auxiliary=offset L1'
    vq=make_vq(training=stage=='vqvae');model=None;optimizers=None
    if stage=='bet':
        ae=root/group/'vqvae'/('latest.pt' if args.probe else 'best.pt')
        saved_ae=torch.load(ae,map_location='cpu',weights_only=False);vq.load_state_dict(saved_ae['vqvae'])
        for module in [vq.encoder,vq.decoder,vq.vq_layer]:module.requires_grad_(False)
        vq.vq_layer.eval();model=make(vq)
        optimizers=model.configure_optimizers(weight_decay=cfg['bet']['weight_decay'],learning_rate=cfg['bet']['lr'],betas=tuple(cfg['bet']['betas']))
        state.update(vqvae_checkpoint=str(ae),vqvae_checkpoint_sha256=digest(ae))
        state['actual_optimizer_groups']={k:[dict(lr=g['lr'],weight_decay=g['weight_decay'],betas=list(g['betas']),parameter_tensors=len(g['params']),parameter_elements=sum(p.numel() for p in g['params'])) for g in opt.param_groups] for k,opt in optimizers.items()}
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False);vq.load_state_dict(saved['vqvae']);state=saved['state']
        if model is not None:
            model.load_state_dict(saved['model'])
            for k,opt in optimizers.items():opt.load_state_dict(saved['optimizers'][k])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
        # Reject an uncheckpointed suffix instead of silently repeating optimizer calls.
        lines=list(csv.DictReader((directory/'updates.csv').open()))
        assert len(lines)==state['actual_stage_updates'],'Uncheckpointed update suffix; explicit review required'
    if stage=='vqvae':vq.vq_layer.train()
    if model is not None:model.train();model.resnet.eval()
    dataset=Dataset(group,stage,offset=state['actual_stage_updates']*batch,limit=(budget-state['actual_stage_updates'])*batch)
    valid=Dataset(group,stage,split='validation');vloader=torch.utils.data.DataLoader(valid,batch_size=16,num_workers=0)
    loader=torch.utils.data.DataLoader(dataset,batch_size=batch*(2 if group=='source_target' else 1),num_workers=0,pin_memory=True,drop_last=True)
    initial={k:v.detach().cpu().clone() for k,v in (model.named_parameters() if model is not None else vq.encoder.named_parameters())} if args.probe else None
    start=time.monotonic();prior=float(state.get('saved_elapsed_seconds',0));stop_requested=[]
    signal.signal(signal.SIGTERM,lambda *_:stop_requested.append('SIGTERM'))
    write(directory/'resource_start.json',resource());write(statepath,state)
    def snapshot(path):
        state.update(elapsed_seconds=prior+time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        state['saved_elapsed_seconds']=state['elapsed_seconds']
        obj=dict(vqvae=vq.state_dict(),state=dict(state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
        if model is not None:obj.update(model=model.state_dict(),optimizers={k:v.state_dict() for k,v in optimizers.items()})
        tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
    mode='a' if state['actual_stage_updates'] else 'x'
    try:
        with (directory/'updates.csv').open(mode,newline='') as log,(directory/'consumed_anchors.csv').open(mode,newline='') as trace:
            writer=csv.writer(log,lineterminator='\n');tw=csv.writer(trace,lineterminator='\n')
            if not state['actual_stage_updates']:
                writer.writerow(['stage_update','loss','auxiliary','seconds']);tw.writerow(['stage_update','domain','episode_id','frame'])
            for b in loader:
                if prior+time.monotonic()-start>cfg['max_wall_hours_per_training_group']*3600:raise TimeoutError('Frozen wall cap')
                if shutil.disk_usage(ROOT).free<cfg['disk_floor_gib']*1024**3:raise OSError('Frozen disk floor')
                tick=time.monotonic();step=state['actual_stage_updates']+1
                if model is None:
                    l1,commitment,codes,mse=vq.vqvae_update(b['actions'].cuda());loss=float(l1+5*commitment);aux=float(commitment)
                    assert math.isfinite(loss) and math.isfinite(aux) and math.isfinite(mse)
                    assert all(torch.isfinite(p).all() for m in [vq.encoder,vq.decoder,vq.vq_layer] for p in m.parameters())
                    state['actual_optimizer_calls']+=1
                else:
                    offset_only=not args.probe and step>cfg['bet_full_optimizer_updates']
                    if offset_only and not state.get('non_offset_frozen'):
                        for name,p in model.named_parameters():p.requires_grad_(name.startswith('_map_to_cbet_preds_offset.'))
                        state['non_offset_frozen']=True
                    # On resumed offset-only training restore requires_grad flags too.
                    if offset_only:
                        for name,p in model.named_parameters():p.requires_grad_(name.startswith('_map_to_cbet_preds_offset.'))
                    for opt in optimizers.values():opt.zero_grad(set_to_none=True)
                    _,value,metrics=model(b['features'].cuda(),b['language'].cuda(),b['actions'].cuda())
                    assert torch.isfinite(value);value.backward()
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    if not offset_only:optimizers['optimizer1'].step();state['actual_optimizer_calls']+=1
                    optimizers['optimizer2'].step();state['actual_optimizer_calls']+=1
                    loss=float(value);aux=float(metrics['offset_loss'])
                torch.cuda.synchronize();state['actual_stage_updates']=step
                state['target_anchor_exposures']=step*batch;state['source_anchor_exposures']=step*batch if group=='source_target' else 0
                writer.writerow([step,loss,aux,time.monotonic()-tick])
                for domain,eid,frame in zip(b['domain'],b['episode_id'],b['frame']):tw.writerow([step,domain,eid,int(frame)])
                if not args.probe and step%cfg['validation_every_'+stage+'_updates']==0:
                    metric=validate(vq,model,vloader);metric['step']=step;state['validation_history'].append(metric)
                    if state['best'] is None or (metric['loss'],step)<(state['best']['loss'],state['best']['step']):
                        state['best']=metric;snapshot(directory/'best.pt')
                    print(__import__('json').dumps(dict(group=group,stage=stage,**metric)),flush=True)
                if step%250==0 or args.probe or stop_requested:
                    log.flush();trace.flush();snapshot(latest);write(statepath,state)
                    print('SAVED',group,stage,step,state['elapsed_seconds'],flush=True)
                if stop_requested:raise InterruptedError('Checkpointed requested stop')
            assert state['actual_stage_updates']==budget
            if args.probe:
                named=model.named_parameters() if model is not None else vq.encoder.named_parameters()
                changed=[k for k,v in named if not torch.equal(initial[k],v.detach().cpu())];assert changed
                if model is not None:
                    assert any(k.startswith('_gpt_model.') for k in changed)
                    assert any(k.startswith('_map_to_cbet_preds_bin.') for k in changed)
                    assert any(k.startswith('_map_to_cbet_preds_offset.') for k in changed)
                    assert any(k.startswith('_language_header.') for k in changed)
                state['changed_parameter_tensors']=len(changed);write(directory/'parameter_changes.json',dict(changed=changed))
            state['status']='COMPLETE';snapshot(latest);write(statepath,state)
            if not args.probe:
                write(out('vqbet')/(group+'_'+stage+'_selection.json'),dict(checkpoint=str(directory/'best.pt'),sha256=digest(directory/'best.pt'),
                    selected=state['best'],actual_stage_updates=budget,actual_optimizer_calls=state['actual_optimizer_calls'],target_anchor_exposures=state['target_anchor_exposures'],source_anchor_exposures=state['source_anchor_exposures'],used_physical_success=False))
    except BaseException as e:
        state.update(status='CHECKPOINTED_STOP' if stop_requested else 'TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc())
        snapshot(latest);write(directory/'error.json',state);write(statepath,state);raise
    finally:
        dataset.close();valid.close();write(statepath,state)
        write(out('vqbet')/(('probe_' if args.probe else 'training_')+group+'_'+stage+'.json'),dict(state,server_status=str(statepath),checkpoint=str(latest),checkpoint_sha256=digest(latest) if latest.exists() else None))

if __name__=='__main__':main()
