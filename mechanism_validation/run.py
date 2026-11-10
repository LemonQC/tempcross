"""At most 24 physical interventions. Never replay/retry a launched episode."""
import datetime
import fcntl
import hashlib
import json
import os
import pickle
import random
import sys
import traceback
import numpy as np
from mechanism_validation.common import *
from mechanism_validation.control import Prefix, Postgrasp, command12
from mechanism_validation.contact import physical_bilateral
from p3_validation.geometry import transform, error
from p3_validation.intervene import pose, contact_diagnostics
from p2_validation.replay_intervention import controller_state


def snapshot(raw,obs):
    import mujoco
    sim=raw.sim;spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(sim.model._model,spec))
    mujoco.mj_getState(sim.model._model,sim.data._data,state,spec)
    result={'mjSTATE_INTEGRATION':state,'qacc_warmstart':sim.data.qacc_warmstart.copy()}
    for key,value in controller_state(raw.robots[0]).items():result['controller::'+key]=value
    for key,value in obs.items():
        if 'image' not in key and np.asarray(value).dtype.kind in 'biuf':result['observation::'+key]=np.asarray(value).copy()
    for key in ['timestep','cur_time']:
        if hasattr(raw,key):result['environment::'+key]=np.asarray(getattr(raw,key))
    return result


def compare(a,b):
    assert set(a)==set(b),'Snapshot field mismatch'
    return {k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) for k in a if a[k].size}


def validate_freeze():
    c=load_config();marker=json.loads((ROOT/'frozen_publication.json').read_text())
    assert marker['github_readback_verified'] is True
    assert marker['protocol_sha256']==digest(CONFIG)
    for path,sha in c['integrity'].items():assert digest(PROJECT/path)==sha,path
    for path,sha in c['external_source_trace_integrity'].items():assert digest(path)==sha,path
    assert digest(c['checkpoint'])==c['checkpoint_sha256']
    return c,marker


def main():
    import cv2,h5py,torch
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c,marker=validate_freeze();os.environ.update(environment());sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    torch.set_num_threads(4)
    from barx.simulation import build_environment_config, initialize_observation_utils
    from barx.evaluation_conditions import load_bundle,restore_frozen_condition
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    from simulation_validation.observe import Recorder
    from baselines.common.policies import BarxPolicy
    from baselines.common.action_adapter import canonical_action
    from baselines.common.metrics import observe_subpredicates,gripper_pad_distance
    bank=json.loads((OUT/'source_contact_bank.json').read_text())
    payload,states,models=load_bundle(SOURCE/'evaluation/conditions',task=TASK,embodiment='panda')
    ledger=ROOT/'physical_launches.jsonl'
    launches=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    assert len(launches)<=24
    if any(not (ROOT/f"condition-{r['condition']:03d}"/r['method']/'result.json').exists() for r in launches):
        raise RuntimeError('Incomplete launched attempt must be audited; it cannot be retried automatically')
    cfg=build_environment_config(TASK,'panda');initialize_observation_utils(cfg);cfg['control_freq']=20
    policy=BarxPolicy(c['checkpoint']);print('MODEL_LOADED '+json.dumps(policy.runtime_metadata),flush=True)
    env=create_env(**cfg,env_type=1,render=False,render_offscreen=True,use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(1000))
    try:
        for template in bank:
            index=template['official_index'];pair_prefix=None;pair_snapshot=None;pair_count=None
            for method in c['methods']:
                directory=ROOT/f'condition-{index:03d}'/method
                if (directory/'result.json').exists():
                    with np.load(directory/'prefix.npz') as z:prefix_data={k:z[k] for k in z.files}
                    hand=directory/'handoff.npz'
                    if hand.exists():
                        with np.load(hand) as z:hand_data={k:z[k] for k in z.files}
                    else:hand_data=None
                    if pair_prefix is None:pair_prefix=prefix_data;pair_snapshot=hand_data
                    else:assert max(compare(pair_prefix,prefix_data).values(),default=0)<=1e-8
                    continue
                assert not directory.exists(),'Existing attempt is never overwritten'
                directory.mkdir(parents=True)
                torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed']);policy.reset()
                obs,metadata=restore_frozen_condition(env,payload['episodes'][index],states[index],models[index])
                raw=env.env;sim=raw.sim;robot=raw.robots[0];site=robot.eef_site_id['right'];oid=raw.obj_body_id['obj']
                assert raw.control_freq==20
                controller=robot.composite_controller.part_controllers['right']
                initial_obj=transform(sim.data.body_xpos[oid],sim.data.body_xquat[oid]);initial_tcp=pose(sim,site)
                prefix=Prefix(initial_tcp,initial_obj,np.asarray(template['T_object_tcp']),c['control'])
                rec=Recorder(directory/'telemetry');rec.start(payload['episodes'][index],env,obs,metadata)
                rec.append('mechanism/restored_initial_state',sim.get_state().flatten())
                write_json(directory/'initial.json',dict(condition=payload['episodes'][index],metadata=metadata,model_xml_sha256=hashlib.sha256(sim.model.get_xml().encode()).hexdigest(),
                    control_frequency=raw.control_freq,physics_timestep=float(sim.model.opt.timestep),controller_config=cfg,policy_runtime=policy.runtime_metadata))
                writer=cv2.VideoWriter(str(directory/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,180));assert writer.isOpened()
                history=[];handoff=None;post=None;failure=None;success=False;prefix_states=[sim.get_state().flatten().copy()];prefix_actions=[]
                launch=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),condition=index,method=method,frozen_commit=marker['remote_commit'],role=c['role'])
                assert len(launches)<24
                with ledger.open('a') as stream:stream.write(json.dumps(launch)+'\n');stream.flush();os.fsync(stream.fileno())
                launches.append(launch)
                print('START '+json.dumps(launch),flush=True)
                try:
                    for t in range(c['max_episode_steps']):
                        tcp=pose(sim,site);obj=transform(sim.data.body_xpos[oid],sim.data.body_xquat[oid]);contact=physical_bilateral(raw)
                        in_prefix=handoff is None
                        cmd=prefix.command(tcp,obj,contact) if in_prefix else None
                        if in_prefix and prefix.failed:
                            failure='PREFIX_'+prefix.failed;break
                        if in_prefix and prefix.done:
                            handoff=t;policy.reset();assert len(policy.queue)==0
                            numeric=snapshot(raw,obs);np.savez_compressed(directory/'handoff.npz',**numeric)
                            with (directory/'handoff_runtime.pkl').open('wb') as stream:
                                pickle.dump(dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                                    policy_queue=list(policy.queue),policy_config=vars(policy.cfg),model_training=policy.model.training,prefix=vars(prefix),checkpoint=c['checkpoint'],checkpoint_sha256=c['checkpoint_sha256']),stream)
                            write_json(directory/'handoff.json',dict(action_index=t,stable_confirmed=True,policy_queue_length=0,policy_inputs=['RGB','verbatim instruction'],
                                checkpoint_sha256=c['checkpoint_sha256'],model_state='UNCHANGED_CHECKPOINT; generation prompt recreated per query; persistent action queue cleared',
                                numerical_fields=list(numeric),prefix_phases_reached=prefix.reached_phases))
                            post=Postgrasp(tcp,obj,template,c['control'],method)
                            write_json(directory/'postgrasp_goals.json',dict(object_goals=[v.tolist() for v in post.object_goals],fixed_handoff_attachment=post.relative.tolist()))
                            in_prefix=False
                            print('HANDOFF '+json.dumps(dict(condition=index,method=method,action_index=t)),flush=True)
                        rgb=(255*np.transpose(obs['barx_panda_agentview_image'],(1,2,0))).astype(np.uint8)
                        if handoff is not None and method=='M1':
                            prediction=policy.predict_action(dict(rgb=rgb,instruction=metadata['lang']))
                            action,conversion=canonical_action(prediction['action'],prediction['representation']);phase='barx_postgrasp';goal=tcp
                            rec.append('mechanism/native_action',prediction['action']);rec.append('mechanism/canonical_before_clip',conversion['preclip_canonical'])
                            if prediction['new_query']:
                                rec.append('mechanism/native_query',prediction['native_query']);rec.append('mechanism/native_query_action_index',t)
                        else:
                            if handoff is not None:
                                cmd=post.command(tcp,obj)
                                if post.failed:failure='POSTGRASP_'+post.failed;break
                            phase,goal,grip=cmd
                            action=command12(controller,tcp,goal,grip,c['control'])
                        assert np.array_equal(action[7:],np.r_[np.zeros(4),-1.])
                        writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                        rec.append('mechanism/goal_tcp_world',goal);rec.append('mechanism/pre_action_tcp_world',tcp)
                        obs,_,_,info=env.step(action);rec.snapshot(env,obs,action,info)
                        after_tcp=pose(sim,site);after_obj=transform(sim.data.body_xpos[oid],sim.data.body_xquat[oid]);bilateral=physical_bilateral(raw)
                        sub=observe_subpredicates(raw,TASK)
                        success=bool(info['is_success']['task']);assert success==(sub['placement'] and sub['retreat'])
                        pad_contact=bool(raw._check_grasp(gripper=robot.gripper['right'],object_geoms=raw.objects['obj'].contact_geoms))
                        rec.append('mechanism/post_action_tcp_world',after_tcp);rec.append('mechanism/physical_bilateral_contact',bilateral)
                        rec.append('mechanism/gripper_pad_distance',gripper_pad_distance(raw))
                        for k,v in sub.items():rec.append('mechanism/post_action_'+k,v)
                        h=dict(action_index=t,phase=phase,bilateral_contact=bilateral,legacy_pad_only_contact=pad_contact,rise_m=float(after_obj[2,3]-initial_obj[2,3]),
                            object_horizontal_displacement_m=float(np.linalg.norm(after_obj[:2,3]-initial_obj[:2,3])),
                            tcp_position_error_m=error(after_tcp,goal)[0],tcp_rotation_error_rad=error(after_tcp,goal)[1],official_task_success=success,
                            gripper_command=float(action[6]),**sub,**contact_diagnostics(raw))
                        history.append(h)
                        if in_prefix:
                            prefix_actions.append(action.copy());prefix_states.append(sim.get_state().flatten().copy())
                        if success:break
                    if not success and failure is None:failure='GLOBAL_600_STEP_BUDGET_AT_'+history[-1]['phase']
                    prefix_data=dict(actions=np.asarray(prefix_actions),states=np.asarray(prefix_states))
                    np.savez_compressed(directory/'prefix.npz',**prefix_data)
                    pair_errors=compare(pair_prefix,prefix_data) if pair_prefix is not None else {k:0. for k in prefix_data}
                    numeric=None
                    if handoff is not None:
                        with np.load(directory/'handoff.npz') as z:numeric={k:z[k] for k in z.files}
                    if pair_prefix is None:pair_prefix=prefix_data;pair_snapshot=numeric;pair_count=handoff
                    else:
                        assert (pair_snapshot is None)==(numeric is None),'Handoff occurrence differs'
                        if numeric is not None:pair_errors.update({'handoff::'+k:v for k,v in compare(pair_snapshot,numeric).items()})
                    trusted=max(pair_errors.values(),default=0)<=1e-8
                    write_json(directory/'prefix_validation.json',dict(trusted=trusted,comparison='M0 reference; independently executed same original initial state',errors=pair_errors))
                    assert trusted,'Numerical prefix/handoff mismatch; stop before other conditions'
                    from p2_validation.offline_audit import longest_run
                    stable=longest_run([h['bilateral_contact'] and h['rise_m']>=.03 for h in history])>=10
                    result=dict(condition=index,method=method,condition_id=template['target_condition_id'],status='COMPLETE',role=c['role'],steps=len(history),handoff_action_index=handoff,
                        approach_reached='clearance' in prefix.reached_phases,pregrasp_reached='pregrasp' in prefix.reached_phases,grasp_pose_reached='descend' in prefix.reached_phases,
                        bilateral_contact=any(h['bilateral_contact'] for h in history),stable_lift=stable,handoff_confirmed=handoff is not None,
                        transport_10cm_in_contact=any(h['bilateral_contact'] and h['rise_m']>=.03 and h['object_horizontal_displacement_m']>=.10 for h in history),
                        transport_reached=post is not None and 'transport' in post.reached_phases if method!='M1' else False,
                        placement=any(h['placement'] for h in history),placement_and_retreat=any(h['placement'] and h['retreat'] for h in history),
                        task_success=success,failure_stage='OFFICIAL_SUCCESS' if success else failure,postgrasp_phases_reached=post.reached_phases if post is not None and method!='M1' else [],
                        prefix_trusted=trusted,video_path=str(directory/'rollout.mp4'),telemetry_path=str(directory/'telemetry/episode-000.h5'),history_path=str(directory/'history.json'))
                    write_json(directory/'history.json',history);write_json(directory/'result.json',result)
                    print('RESULT '+json.dumps(result),flush=True)
                except BaseException as exc:
                    write_json(directory/'technical_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),executed_steps=len(history),automatic_retry_forbidden=True))
                    raise
                finally:writer.release();rec.close()
        write_json(ROOT/'completion.json',dict(status='COMPLETE',physical_episodes=len(launches),frozen_commit=marker['remote_commit']))
    finally:env.env.close()


if __name__=='__main__':main()
