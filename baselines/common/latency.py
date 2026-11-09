"""Isolated policy timing on an old-development RGB frame; no physics step."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from baselines.common.paths import ROOT,SOURCE,write_json,digest,environment


def main():
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--kind',required=True,choices=['barx','robomimic']);p.add_argument('--checkpoint',required=True);a=p.parse_args()
 import os,sys,torch
 os.environ.update(environment());sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')];torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7)
 from baselines.common.policies import BarxPolicy,RobomimicPolicy
 out=ROOT/'latency'/f'{a.label}.json'
 if out.exists():
  data=json.loads(out.read_text());assert data['checkpoint_sha256']==digest(a.checkpoint);return
 frame=ROOT/'smoke/probe_B_bc_gmm/pnp_counter_to_sink/panda/condition-000.mp4';cap=cv2.VideoCapture(str(frame));ok,img=cap.read();cap.release();assert ok;rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 payload=json.loads((SOURCE/'evaluation/conditions/pnp_counter_to_sink/panda.json').read_text());instruction=payload['episodes'][0]['ep_meta']['lang']
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter();policy=(BarxPolicy if a.kind=='barx' else RobomimicPolicy)(a.checkpoint);load_seconds=time.perf_counter()-start
 observation=dict(rgb=rgb,instruction=instruction)
 for _ in range(16):policy.predict_action(observation)
 policy.reset();steps=[];queries=[]
 for i in range(80):
  torch.cuda.synchronize();t=time.perf_counter();result=policy.predict_action(observation);torch.cuda.synchronize();dt=time.perf_counter()-t;steps.append(dt)
  if result['new_query']:queries.append(dt)
  assert np.isfinite(result['action']).all()
 write_json(out,dict(label=a.label,checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),device=torch.cuda.get_device_name(0),role='ISOLATED_GPU_OLD_DEVELOPMENT_IMAGE_NO_PHYSICS',load_seconds=load_seconds,warmup_policy_calls=16,measured_policy_calls=len(steps),measured_queries=len(queries),mean_policy_step_seconds=float(np.mean(steps)),median_policy_step_seconds=float(np.median(steps)),p95_policy_step_seconds=float(np.quantile(steps,.95)),mean_fresh_query_seconds=float(np.mean(queries)),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),all_policy_step_seconds=steps,all_fresh_query_seconds=queries,parameter_count=sum(x.numel() for x in (policy.model.parameters() if a.kind=='barx' else policy.model.nets.parameters()))))
 print(json.dumps(dict(label=a.label,mean_seconds=np.mean(steps))),flush=True)
if __name__=='__main__':main()
