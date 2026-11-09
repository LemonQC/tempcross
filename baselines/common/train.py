"""Measured training with the pinned official algorithms and frozen anchor budget."""
import argparse,collections,csv,json,math,time,traceback,shutil
from pathlib import Path
import numpy as np
from baselines.common.paths import ROOT,PROJECT,VENDOR,digest,write_json
from baselines.common.probe import validate_freeze
from baselines.common.dataset_adapter import AnchorDataset,window_indices
from baselines.common.model_factory import make_model


def validation(model,method,loader):
 import torch
 import torch.nn.functional as F
 import robomimic.utils.tensor_utils as TensorUtils
 cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all();torch.manual_seed(140011);np_state=np.random.get_state();np.random.seed(140011)
 model.set_eval();scores=[];nlls=[]
 try:
  with torch.no_grad():
   for batch in loader:
    p=model.process_batch_for_training(batch)
    if method=='diffusion_policy':
     nets=model.ema.averaged_model if model.ema is not None else model.nets;nets.eval()
     features=TensorUtils.time_distributed(dict(obs=p['obs'],goal=None),nets['policy']['obs_encoder'],inputs_as_kwargs=True)
     actions=p['actions'];noise=torch.randn_like(actions);t=torch.randint(0,model.noise_scheduler.config.num_train_timesteps,(len(actions),),device=actions.device)
     prediction=nets['policy']['noise_pred_net'](model.noise_scheduler.add_noise(actions,noise,t),t,global_cond=features.flatten(1));scores.append(float(F.mse_loss(prediction,noise)))
    else:
     kw=dict(obs_dict=p['obs'],goal_dict=None)
     if method=='bc_transformer':kw.update(actions=None,low_noise_eval=False)
     dist=model.nets['policy'].forward_train(**kw);mean=dist.mean;target=p['actions']
     if method=='bc_transformer':
      scores.append(float(F.mse_loss(mean[:,-1],target)))
      nlls.append(float(-dist.log_prob(target[:,None].expand(-1,mean.shape[1],-1))[:,-1].mean()))
     else:
      scores.append(float(F.mse_loss(mean,target)));nlls.append(float(-dist.log_prob(target).mean()))
  assert all(math.isfinite(x) for x in scores+nlls)
  return dict(metric='ema_fixed_noise_mse' if method=='diffusion_policy' else 'expected_action_mse',value=float(np.mean(scores)),nll=float(np.mean(nlls)) if nlls else None,batches=len(scores))
 finally:
  torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu);np.random.set_state(np_state);model.set_train()


def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--group',required=True,choices=['A','B','C']);p.add_argument('--target-batch',type=int,default=16);p.add_argument('--end-positions',type=int,default=96000);a=p.parse_args()
 marker=validate_freeze();label=f'{a.group}_{a.method}';directory=ROOT/'training'/label;directory.mkdir(parents=True,exist_ok=True)
 import torch
 torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
 ledger=directory/'status.json';resume=directory/'latest.pt';state=json.loads(ledger.read_text()) if ledger.exists() else dict(status='INITIALIZING',label=label,method=a.method,group=a.group,completed_optimizer_steps=0,nonfinite_losses=0,anchor_positions=0,actual_target_positions=0,actual_source_positions=0,gpu_seconds=0.,batch_target=a.target_batch,actual_batch=a.target_batch*(2 if a.group=='C' else 1),frozen_commit=marker['remote_commit'],validation_history=[])
 if state['status']=='TECHNICAL_FAILURE':raise RuntimeError('Preserve and diagnose failed training before resume')
 if state['anchor_positions']>=a.end_positions:
  assert state['status']=='TIER_COMPLETE';print('Exact completed tier reused');return
 assert a.end_positions in [96000,192000,384000,768000] and a.end_positions%a.target_batch==0
 dataset=AnchorDataset(a.method,a.group);valid=AnchorDataset(a.method,'A' if a.group=='A' else 'B',split='validation',max_positions=256)
 validation_loader=torch.utils.data.DataLoader(valid,batch_size=16,num_workers=0)
 torch.manual_seed(7);np.random.seed(7);model,config=make_model(a.method,max_steps=768000//a.target_batch)
 params=sum(p.numel() for p in model.nets.parameters());write_json(directory/'config.json',json.loads(config.dump()))
 initial_path=directory/'initial_trainable.pt'
 if resume.exists():
  saved=torch.load(resume,map_location='cpu');assert saved['anchor_positions']==state['anchor_positions'];model.deserialize(saved['model'],load_optimizers=True)
  if a.method=='diffusion_policy':model.ema.optimization_step=saved.get('ema_optimization_step',state['completed_optimizer_steps'])
  torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng'])
 else:
  assert state['completed_optimizer_steps']==0
  initial={k:v.detach().cpu().clone() for k,v in model.nets.named_parameters() if v.requires_grad};torch.save(initial,initial_path)
  state.update(parameter_count=params,initial_trainable_sha256=digest(initial_path),official_model_class=type(model).__name__,executed_trainer_sha256=digest(Path(__file__)))
  initial_metric=validation(model,a.method,validation_loader);state['initial_validation']=initial_metric
 # No HDF5 handle is inherited when DataLoader starts its worker processes.
 valid.close()
 start_index=state['anchor_positions']*(2 if a.group=='C' else 1);end_index=a.end_positions*(2 if a.group=='C' else 1)
 subset=torch.utils.data.Subset(dataset,range(start_index,end_index))
 generator=torch.Generator();generator.manual_seed(9981)
 loader=torch.utils.data.DataLoader(subset,batch_size=state['actual_batch'],num_workers=4,pin_memory=True,prefetch_factor=2,generator=generator,drop_last=True)
 import robomimic.utils.torch_utils as TorchUtils
 original=TorchUtils.backprop_for_loss
 def measured(*args,**kwargs):
  loss=kwargs['loss'];value=float(loss.detach().item())
  if not math.isfinite(value):state['nonfinite_losses']+=1;raise FloatingPointError('Nonfinite loss before backward')
  result=original(*args,**kwargs);torch.cuda.synchronize();state['completed_optimizer_steps']+=1;state['last_loss']=value;return result
 TorchUtils.backprop_for_loss=measured
 logs=(directory/'optimizer_metrics.csv').open('a',newline='');writer=csv.writer(logs,lineterminator='\n')
 if start_index==0:writer.writerow(['optimizer_step','anchor_positions','target_positions','source_positions','raw_loss','learning_rate','step_seconds'])
 trace=(directory/'consumed_anchors.csv').open('a',newline='');tw=csv.writer(trace,lineterminator='\n')
 if start_index==0:tw.writerow(['domain','episode_id','anchor_frame','input_frame_count','label_frame_count'])
 state.update(status='TRAINING',prescribed_tier_positions=a.end_positions);write_json(ledger,state);model.set_train();start=time.monotonic()
 if 'trainer_segments' not in state:
  state['trainer_segments']=[] if state['anchor_positions']==0 else [dict(from_positions=0,to_positions=state['anchor_positions'],source_sha256=state['executed_trainer_sha256'])]
 source_snapshot=directory/f"executed_train_from_{state['anchor_positions']:06d}.py";shutil.copyfile(Path(__file__),source_snapshot)
 state['trainer_segments'].append(dict(from_positions=state['anchor_positions'],to_positions=a.end_positions,source_sha256=digest(source_snapshot),source_path=str(source_snapshot)))
 try:
  for batch in loader:
   before=state['completed_optimizer_steps'];t=time.monotonic();model.train_on_batch(model.process_batch_for_training(batch),epoch=before+1);model.on_gradient_step();assert state['completed_optimizer_steps']==before+1
   n=a.target_batch;state['anchor_positions']+=n;state['actual_target_positions']+=n if a.group!='A' else 0;state['actual_source_positions']+=n if a.group in ['A','C'] else 0
   for d,e,frame,nobs,nlabels in zip(batch['_domain'],batch['_episode_id'],batch['_anchor'],batch['_input_frame_count'],batch['_label_frame_count']):tw.writerow([d,e,int(frame),int(nobs),int(nlabels)])
   writer.writerow([state['completed_optimizer_steps'],state['anchor_positions'],state['actual_target_positions'],state['actual_source_positions'],state['last_loss'],model.optimizers['policy'].param_groups[0]['lr'],time.monotonic()-t])
   if state['anchor_positions']%12000==0:
    metric=validation(model,a.method,validation_loader);metric.update(anchor_positions=state['anchor_positions'],optimizer_step=state['completed_optimizer_steps']);state['validation_history'].append(metric);print(json.dumps(dict(label=label,**metric)),flush=True)
   if state['completed_optimizer_steps']%100==0:
    logs.flush();trace.flush();state.update(peak_gpu_allocated_bytes=max(state.get('peak_gpu_allocated_bytes',0),torch.cuda.max_memory_allocated()),peak_gpu_reserved_bytes=max(state.get('peak_gpu_reserved_bytes',0),torch.cuda.max_memory_reserved()),running_segment_seconds=time.monotonic()-start);write_json(ledger,state)
  assert state['anchor_positions']==a.end_positions and state['completed_optimizer_steps']*a.target_batch==state['anchor_positions']
  state['gpu_seconds']+=time.monotonic()-start
  initial=torch.load(initial_path,map_location='cpu');changes=[]
  for key,value in model.nets.named_parameters():
   if key not in initial:continue
   value=value.detach().cpu();assert torch.isfinite(value).all();delta=(value-initial[key]).float();changes.append(dict(key=key,changed=not torch.equal(value,initial[key]),max_abs_delta=float(delta.abs().max())))
  assert any(x['changed'] for x in changes)
  write_json(directory/'parameter_changes.json',dict(trainable_tensors=len(changes),changed_tensors=sum(x['changed'] for x in changes),all_final_trainable_parameters_finite=True,tensors=changes))
  checkpoint=directory/f'positions-{a.end_positions:06d}.pt'
  saved=dict(method=a.method,group=a.group,config=json.loads(config.dump()),model=model.serialize(),target_batch=a.target_batch,optimizer_updates=state['completed_optimizer_steps'],anchor_positions=state['anchor_positions'],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),ema_optimization_step=getattr(model.ema,'optimization_step',None) if a.method=='diffusion_policy' else None)
  torch.save(saved,checkpoint)
  # A hard link keeps exact checkpoints per tier without duplicating last weights.
  if resume.exists():resume.unlink()
  resume.hardlink_to(checkpoint)
  state.update(status='TIER_COMPLETE',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),changed_parameter_tensors=sum(x['changed'] for x in changes),peak_gpu_allocated_bytes=max(state.get('peak_gpu_allocated_bytes',0),torch.cuda.max_memory_allocated()),peak_gpu_reserved_bytes=max(state.get('peak_gpu_reserved_bytes',0),torch.cuda.max_memory_reserved()))
 except BaseException as exc:
  state.update(status='TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc());raise
 finally:
  logs.close();trace.close();dataset.close();valid.close();write_json(ledger,state);print(json.dumps({k:v for k,v in state.items() if k!='validation_history'},indent=2),flush=True)
if __name__=='__main__':main()
