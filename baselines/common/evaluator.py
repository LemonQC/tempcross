"""One evaluator for all RGB policies, using the unchanged official tasks."""
import argparse,datetime,json,os,sys,time,traceback
from pathlib import Path
import cv2,numpy as np
from baselines.common.paths import ROOT,PROJECT,SOURCE,TASKS,OFFICIAL_COMMIT,digest,environment,write_json
from baselines.common.probe import validate_freeze
from baselines.common.action_adapter import canonical_action


def evaluate(label,kind,checkpoint,task,robot='panda',smoke=False,indices=None):
 import torch
 torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7)
 marker=validate_freeze();os.environ.update(environment());sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
 from barx.benchmark import TASKS as specs,EMBODIMENTS
 from barx.simulation import build_environment_config,initialize_observation_utils
 from barx.evaluation_conditions import load_bundle,restore_frozen_condition,validate_condition_bundle_for_evaluation
 from robocasa.utils.robomimic.robomimic_env_utils import create_env
 from simulation_validation.observe import Recorder
 from baselines.common.policies import BarxPolicy,RobomimicPolicy
 indices=([0] if smoke else list(range(60,100))) if indices is None else indices
 assert all(i<60 for i in indices) if smoke else indices==list(range(60,100))
 directory=ROOT/('smoke' if smoke else 'formal')/label/task/robot
 if directory.exists():
  previous=json.loads((directory/'job.json').read_text())
  if previous['status']=='COMPLETE' and previous['checkpoint_sha256']==digest(checkpoint) and previous['indices']==indices:return previous
  raise FileExistsError('Preserve and diagnose technical attempt '+str(directory))
 directory.mkdir(parents=True);checkpoint=Path(checkpoint)
 job=dict(status='LOADING_MODEL',label=label,kind=kind,task=task,robot=robot,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),indices=indices,role='DEVELOPMENT_SMOKE' if smoke else 'FROZEN_FINAL_TEST',frozen_commit=marker['remote_commit'],policy_information=['rgb','verbatim instruction'],technical_failures_not_task_failures=True)
 ledger=directory/'job.json';write_json(ledger,job);rec=None;env=None;writer=None;start=time.monotonic();records=[];phase='CHECKPOINT_LOADING'
 try:
  torch.cuda.reset_peak_memory_stats();policy=(BarxPolicy if kind=='barx' else RobomimicPolicy)(checkpoint)
  phase='FROZEN_ENVIRONMENT_INITIALIZATION'
  payload,states,models=load_bundle(SOURCE/'evaluation/conditions',task=task,embodiment=robot)
  validate_condition_bundle_for_evaluation(payload,states,models,task=task,camera_name=EMBODIMENTS[robot].camera,image_width=320,image_height=180,count=100)
  if not smoke:
   frozen=json.loads((PROJECT/'configs/evaluation/baseline_conditions_60_99.json').read_text())
   for i in indices:
    entry=next(e for e in frozen if e['task']==task and e['robot']==robot and e['official_index']==i)
    assert all(payload['episodes'][i][k]==v for k,v in entry['condition'].items())
  cfg=build_environment_config(task,robot);initialize_observation_utils(cfg);cfg['control_freq']=20
  env=create_env(**cfg,env_type=1,render=False,render_offscreen=True,use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(1000))
  rec=Recorder(directory/'telemetry');job.update(status='RUNNING',conditions=[{k:payload['episodes'][i][k] for k in ['episode','condition_id','seed','state_sha256','model_sha256','metadata_sha256']} for i in indices]);write_json(ledger,job)
  with (ROOT/'physical_launches.jsonl').open('a') as stream:stream.write(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),label=label,task=task,robot=robot,indices=indices,role=job['role'],frozen_commit=marker['remote_commit']))+'\n')
  stream=(directory/'episodes.jsonl').open('w',buffering=1)
  for trial,index in enumerate(indices):
   phase='FROZEN_CONDITION_RESTORE'
   policy.reset();obs,metadata=restore_frozen_condition(env,payload['episodes'][index],states[index],models[index]);rec.start(payload['episodes'][index],env,obs,metadata)
   rec.append('baseline/restored_initial_state',env.env.sim.get_state().flatten())
   raw=env.env;assert raw.control_freq==20
   phase='VIDEO_INITIALIZATION';video=directory/f'condition-{index:03d}.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,180));assert writer.isOpened()
   success=False;timings=[];clipped=0
   for step in range(specs[task].max_steps):
    # The wrapper already flips the MuJoCo image once. Do not flip again.
    rgb=(255*np.transpose(obs[EMBODIMENTS[robot].camera+'_image'],(1,2,0))).astype(np.uint8)
    instruction=metadata['lang'];public_observation=dict(rgb=rgb,instruction=instruction)
    phase='NETWORK_INFERENCE';torch.cuda.synchronize();t=time.perf_counter();prediction=policy.predict_action(public_observation);torch.cuda.synchronize();elapsed=time.perf_counter()-t;timings.append(elapsed)
    phase='ACTION_INTERFACE'
    action,conversion=canonical_action(prediction['action'],prediction['representation']);clipped+=conversion['clipped_dimensions']
    rec.append('baseline/native_action',prediction['action']);rec.append('baseline/canonical_before_clip',conversion['preclip_canonical']);rec.append('baseline/prediction_seconds',elapsed)
    if prediction['new_query']:
     rec.append('baseline/native_query',prediction['native_query']);rec.append('baseline/native_query_action_index',step)
    site=raw.robots[0].eef_site_id['right'];rec.append('baseline/pre_action_tcp_position',raw.sim.data.site_xpos[site]);rec.append('baseline/pre_action_tcp_rotation_matrix',raw.sim.data.site_xmat[site].reshape(3,3))
    writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));phase='SIMULATION_STEP';obs,_,_,info=env.step(action);phase='TELEMETRY_AND_SUCCESS_AUDIT';rec.snapshot(env,obs,action,info)
    from baselines.common.metrics import observe_subpredicates
    sub=observe_subpredicates(raw,task)
    for key,value in sub.items():rec.append('baseline/post_action_'+key,value)
    rec.append('baseline/post_action_tcp_position',raw.sim.data.site_xpos[site]);rec.append('baseline/post_action_tcp_rotation_matrix',raw.sim.data.site_xmat[site].reshape(3,3))
    label_success=info['is_success']['task'];assert isinstance(label_success,(bool,np.bool_));success=bool(label_success)
    assert success==(sub['placement'] and sub['retreat'])
    if success:break
   writer.release();writer=None;rec.close()
   record=dict(trial_index=trial,official_condition_index=index,condition_id=payload['episodes'][index]['condition_id'],seed=payload['episodes'][index]['seed'],success=success,steps=step+1,policy_calls=len(timings),mean_prediction_seconds=float(np.mean(timings)),total_prediction_seconds=float(sum(timings)),clipped_action_dimensions=clipped,video_path=str(video),telemetry_path=str(directory/f'telemetry/episode-{trial:03d}.h5'))
   records.append(record);stream.write(json.dumps(record)+'\n');job['completed_episodes']=len(records);write_json(ledger,job);print(json.dumps(record),flush=True)
  stream.close();job.update(status='COMPLETE',episodes=len(records),successes=sum(r['success'] for r in records),duration_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
 except BaseException as exc:
  job.update(status='TECHNICAL_FAILURE',technical_failure_phase=phase,error=repr(exc),traceback=traceback.format_exc(),completed_episodes=len(records),invalid_partial_episode_not_counted=True);raise
 finally:
  if writer is not None:writer.release()
  if rec is not None:rec.close()
  if env is not None:env.env.close()
  write_json(ledger,job)
 return job

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--kind',choices=['barx','robomimic'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--robot',default='panda',choices=['panda','ur5e']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 evaluate(a.label,a.kind,a.checkpoint,a.task,a.robot,a.smoke)
