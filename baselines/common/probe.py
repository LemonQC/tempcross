"""One actual official-model optimizer step, persisted and verified on real data."""
import argparse,json,time,traceback,subprocess
import numpy as np
from baselines.common.paths import ROOT,PROJECT,SOURCE,VENDOR,OFFICIAL_COMMIT,ROBOMIMIC_COMMIT,digest,write_json
from baselines.common.dataset_adapter import AnchorDataset
from baselines.common.model_factory import make_model


def validate_freeze():
 p=json.loads((ROOT/'frozen_publication.json').read_text());assert p['github_readback_verified']
 assert p['protocol_sha256']==digest(PROJECT/'configs/baselines/protocol.json')
 assert p['condition_manifest_sha256']==digest(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
 for root,commit in [(SOURCE,OFFICIAL_COMMIT),(VENDOR,ROBOMIMIC_COMMIT)]:
  assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==commit
  assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain']).strip()
 return p


def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--group',default='B',choices=['A','B','C']);p.add_argument('--target-batch',type=int,default=16);a=p.parse_args()
 marker=validate_freeze();label=f'{a.group}_{a.method}_b{a.target_batch}';directory=ROOT/'probes'/label
 if directory.exists():raise FileExistsError(directory)
 directory.mkdir(parents=True);status=dict(status='STARTING',method=a.method,group=a.group,target_batch=a.target_batch,completed_optimizer_steps=0,nonfinite_losses=0,frozen_commit=marker['remote_commit']);path=directory/'status.json'
 write_json(path,status)
 try:
  import torch
  torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
  dataset=AnchorDataset(a.method,a.group,max_positions=a.target_batch)
  loader=torch.utils.data.DataLoader(dataset,batch_size=len(dataset),num_workers=0);batch=next(iter(loader))
  torch.manual_seed(7);np.random.seed(7)
  model,config=make_model(a.method)
  initial={k:v.detach().cpu().clone() for k,v in model.nets.named_parameters() if v.requires_grad}
  import robomimic.utils.torch_utils as TorchUtils
  original=TorchUtils.backprop_for_loss
  def measured(*args,**kwargs):
   loss=kwargs['loss'];value=float(loss.detach().item())
   if not np.isfinite(value):status['nonfinite_losses']+=1;raise FloatingPointError(value)
   result=original(*args,**kwargs);torch.cuda.synchronize();status['completed_optimizer_steps']+=1;status['loss']=value;return result
  TorchUtils.backprop_for_loss=measured
  model.set_train();start=time.monotonic();status['initial_learning_rate']=model.optimizers['policy'].param_groups[0]['lr']
  model.train_on_batch(model.process_batch_for_training(batch),epoch=1);model.on_gradient_step()
  if a.method=='diffusion_policy' and status['initial_learning_rate']==0:
   # Keep the real single-step resource test. The official cosine warmup
   # intentionally makes that first update zero; a separately logged second
   # step proves parameter movement without changing the learning schedule.
   status['resource_probe_optimizer_steps']=1;status['additional_parameter_audit_steps']=1
   model.train_on_batch(model.process_batch_for_training(batch),epoch=2);model.on_gradient_step()
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  changes=[]
  for key,value in model.nets.named_parameters():
   if key not in initial:continue
   after=value.detach().cpu();assert torch.isfinite(after).all()
   delta=(after-initial[key]).float();changes.append(dict(key=key,changed=not torch.equal(after,initial[key]),max_abs_delta=float(delta.abs().max())))
  assert status['completed_optimizer_steps'] in [1,2] and any(x['changed'] for x in changes)
  checkpoint=directory/'probe.pt';torch.save(dict(method=a.method,group=a.group,config=json.loads(config.dump()),model=model.serialize(),target_batch=a.target_batch,optimizer_updates=status['completed_optimizer_steps'],role='DISCARDED_AFTER_INTERFACE_SMOKE_NOT_FORMAL_MODEL'),checkpoint)
  write_json(directory/'parameter_changes.json',dict(trainable_tensors=len(changes),changed_tensors=sum(x['changed'] for x in changes),tensors=changes))
  status.update(status='COMPLETE',actual_batch=len(dataset),target_samples=a.target_batch*status['completed_optimizer_steps'] if a.group!='A' else 0,source_samples=a.target_batch*status['completed_optimizer_steps'] if a.group in ['A','C'] else 0,parameter_count=sum(p.numel() for p in model.nets.parameters()),changed_parameter_tensors=sum(x['changed'] for x in changes),step_seconds=elapsed,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),official_model_class=type(model).__name__)
  dataset.close();write_json(directory/'config.json',json.loads(config.dump()))
 except BaseException as exc:
  status.update(status='TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc(),is_cuda_oom=isinstance(exc,__import__('torch').cuda.OutOfMemoryError));raise
 finally:write_json(path,status);print(json.dumps(status,indent=2),flush=True)
if __name__=='__main__':main()
