"""Episode-resumable evaluation, unchanged frozen states/OSC/official task goal."""
import argparse
import importlib
import json
import time
import traceback
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,SOURCE,TASKS,out,read,write,digest,setup,resource,now

def evaluate(method,group,checkpoint,task,smoke=False):
    setup()
    import torch,numpy as np,cv2
    torch.set_num_threads(4);torch.manual_seed(7);np.random.seed(7);torch.cuda.reset_peak_memory_stats()
    from baselines.common.probe import validate_freeze
    validate_freeze()
    if not smoke:
        freeze=read(out(method)/'freeze.json');assert freeze['protocol_sha256']==digest(out(method)/'protocol.json')
    from barx.benchmark import TASKS as specs,EMBODIMENTS
    from barx.simulation import build_environment_config,initialize_observation_utils
    from barx.evaluation_conditions import load_bundle,restore_frozen_condition,validate_condition_bundle_for_evaluation
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    from simulation_validation.observe import Recorder
    from baselines.common.action_adapter import canonical_action
    from baselines.common.metrics import observe_subpredicates,gripper_pad_distance
    indices=[0] if smoke else list(range(60,100));role='DEVELOPMENT_SMOKE' if smoke else 'EXPOSED_REFERENCE'
    directory=ROOT/method/'physical'/('smoke' if smoke else 'formal')/group/task;directory.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(checkpoint);sha=digest(checkpoint);ledger=directory/'job.json'
    if ledger.exists() and read(ledger)['status']=='COMPLETE':
        previous=read(ledger);assert previous['checkpoint_sha256']==sha and previous['indices']==indices;return previous
    job=dict(status='LOADING',method=method,group=group,task=task,robot='panda',role=role,indices=indices,
             checkpoint=str(checkpoint),checkpoint_sha256=sha,completed_episodes=0,policy_information=['RGB','instruction'],
             original_benchmark_commit='0f435089f48b1f16d225f868dbef855a5e75d1e0',start_resources=resource())
    write(ledger,job);env=None;rec=None;writer=None;phase='MODEL_LOAD';records=[];start=time.monotonic()
    try:
        policy=importlib.import_module('results.strong_baselines.'+method+'.model').Policy(checkpoint)
        job['runtime']=policy.runtime_metadata
        payload,states,models=load_bundle(SOURCE/'evaluation/conditions',task=task,embodiment='panda')
        validate_condition_bundle_for_evaluation(payload,states,models,task=task,camera_name=EMBODIMENTS['panda'].camera,image_width=320,image_height=180,count=100)
        frozen=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
        if not smoke:
            for i in indices:
                e=next(x for x in frozen if x['task']==task and x['robot']=='panda' and x['official_index']==i)
                assert all(payload['episodes'][i][k]==v for k,v in e['condition'].items())
        cfg=build_environment_config(task,'panda');initialize_observation_utils(cfg);cfg['control_freq']=20
        phase='ENVIRONMENT_CREATE';env=create_env(**cfg,env_type=1,render=False,render_offscreen=True,use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(1000))
        assert specs[task].max_steps==(600 if task==TASKS[0] else 650)
        for i in indices:
            episode_dir=directory/f'condition-{i:03d}';closed=episode_dir/'episode.json'
            if closed.exists():
                r=read(closed);assert r['checkpoint_sha256']==sha and r['condition_id']==payload['episodes'][i]['condition_id'];records.append(r);continue
            if episode_dir.exists():raise FileExistsError('Preserve incomplete physical attempt; no automatic repeat: '+str(episode_dir))
            episode_dir.mkdir();write(episode_dir/'launch.json',dict(utc=now(),method=method,group=group,task=task,index=i,role=role,checkpoint_sha256=sha))
            with (ROOT/method/'physical/physical_launches.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=now(),group=group,task=task,index=i,role=role,checkpoint_sha256=sha))+'\n')
            phase='CONDITION_RESTORE';policy.reset();obs,metadata=restore_frozen_condition(env,payload['episodes'][i],states[i],models[i])
            rec=Recorder(episode_dir/'telemetry');rec.start(payload['episodes'][i],env,obs,metadata)
            raw=env.env;initial=raw.sim.get_state().flatten();rec.append('baseline/restored_initial_state',initial)
            if not smoke:
                expected=next(x for x in frozen if x['task']==task and x['robot']=='panda' and x['official_index']==i)
                assert np.array_equal(initial,np.asarray(expected['initial_state'],np.float64))
            rec.append('baseline/gripper_pad_center_distance',gripper_pad_distance(raw))
            assert raw.control_freq==20
            video=episode_dir/'rollout.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,180));assert writer.isOpened()
            success=False;times=[];clipped=0
            for step in range(specs[task].max_steps):
                rgb=(255*np.transpose(obs[EMBODIMENTS['panda'].camera+'_image'],(1,2,0))).astype(np.uint8)
                phase='POLICY_INFERENCE';torch.cuda.synchronize();t=time.perf_counter()
                prediction=policy.predict_action(dict(rgb=rgb,instruction=metadata['lang']));torch.cuda.synchronize();times.append(time.perf_counter()-t)
                action,conversion=canonical_action(prediction['action'],prediction['representation']);clipped+=conversion['clipped_dimensions']
                rec.append('baseline/native_action',prediction['action']);rec.append('baseline/canonical_before_clip',conversion['preclip_canonical'])
                rec.append('baseline/prediction_seconds',times[-1])
                if prediction['new_query']:
                    rec.append('baseline/native_query',prediction['native_query']);rec.append('baseline/native_query_action_index',step)
                site=raw.robots[0].eef_site_id['right'];rec.append('baseline/pre_action_tcp_position',raw.sim.data.site_xpos[site]);rec.append('baseline/pre_action_tcp_rotation_matrix',raw.sim.data.site_xmat[site].reshape(3,3))
                writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));phase='PHYSICS_STEP'
                obs,_,_,info=env.step(action);rec.snapshot(env,obs,action,info)
                sub=observe_subpredicates(raw,task)
                for key,value in sub.items():rec.append('baseline/post_action_'+key,value)
                rec.append('baseline/gripper_pad_center_distance',gripper_pad_distance(raw))
                rec.append('baseline/post_action_tcp_position',raw.sim.data.site_xpos[site]);rec.append('baseline/post_action_tcp_rotation_matrix',raw.sim.data.site_xmat[site].reshape(3,3))
                assert isinstance(info['is_success']['task'],(bool,np.bool_));success=bool(info['is_success']['task'])
                assert success==bool(sub['placement'] and sub['retreat'])
                if success:break
            writer.release();writer=None;rec.close();rec=None
            r=dict(method=method,group=group,task=task,robot='panda',role=role,official_condition_index=i,condition_id=payload['episodes'][i]['condition_id'],seed=payload['episodes'][i]['seed'],success=success,steps=step+1,
                   checkpoint_sha256=sha,policy_calls=len(times),mean_prediction_seconds=float(np.mean(times)),clipped_dimensions=clipped,
                   video_path=str(video),telemetry_path=str(episode_dir/'telemetry/episode-000.h5'),contact_log=str(episode_dir/'telemetry/episode-000.jsonl'))
            write(closed,r);records.append(r);job.update(status='RUNNING',completed_episodes=len(records));write(ledger,job);print(json.dumps(r),flush=True)
        job.update(status='COMPLETE',episodes=len(records),successes=sum(r['success'] for r in records),duration_seconds=time.monotonic()-start,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        job.update(status='TECHNICAL_FAILURE',phase=phase,error=repr(e),traceback=traceback.format_exc(),completed_episodes=len(records));write(directory/'error.json',job);raise
    finally:
        if writer is not None:writer.release()
        if rec is not None:rec.close()
        if env is not None:env.env.close()
        write(ledger,job)
    return job

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--group',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    evaluate(a.method,a.group,a.checkpoint,a.task,a.smoke)
