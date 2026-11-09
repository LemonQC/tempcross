"""Exactly 24 frozen A1/A2/A3 MuJoCo diagnostics; no state teleportation."""
import datetime
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np
from p3_validation.common import ROOT,PROJECT,SOURCE,environment
from p3_validation.evaluate import validate_freeze
from p3_validation.geometry import transform,error,bounded_delta
from p3_validation.plan import WaypointPlan
from p2_validation.replay_intervention import controller_state
from p2_validation.offline_audit import longest_run,sha256


def pose(sim,site):
    t=np.eye(4);t[:3,:3]=sim.data.site_xmat[site].reshape(3,3);t[:3,3]=sim.data.site_xpos[site];return t


def branch_snapshot(raw,obs):
    sim=raw.sim;robot=raw.robots[0]
    return dict(qpos=sim.data.qpos.copy(),qvel=sim.data.qvel.copy(),ctrl=sim.data.ctrl.copy(),
       qacc_warmstart=sim.data.qacc_warmstart.copy(),tcp=pose(sim,robot.eef_site_id['right']),
       object_position=sim.data.body_xpos[raw.obj_body_id['obj']].copy(),object_quaternion=sim.data.body_xquat[raw.obj_body_id['obj']].copy(),
       gripper=np.asarray(obs['robot0_gripper_qpos']).copy(),controllers=controller_state(robot))


def flatten_snapshot(snapshot):
    return {**{k:np.asarray(v) for k,v in snapshot.items() if k!='controllers'},
            **{'controller::'+k:np.asarray(v) for k,v in snapshot['controllers'].items()}}


def compare_snapshots(first,second):
    if first.keys()!=second.keys():raise ValueError('Branch state/controller fields changed')
    return {k:float(np.max(np.abs(first[k].astype(float)-second[k].astype(float)))) for k in first if first[k].size}


def osc_action(controller,current,goal,c):
    if controller.input_type!='delta' or controller.control_dim!=6:raise ValueError('Unexpected original controller')
    if controller._goal_update_mode!='achieved':raise ValueError('Original arm-mode reference convention changed')
    if controller.ref_name!='gripper0_right_grip_site':raise ValueError('Unexpected controller TCP')
    origin=controller.origin_ori if controller.input_ref_frame=='base' else np.eye(3)
    d=bounded_delta(current,goal,np.asarray(origin),c['gain_position'],c['gain_rotation'],c['max_position_step_m'],c['max_rotation_step_rad'])
    normalized=(d-controller.action_output_transform)/controller.action_scale+controller.action_input_transform
    assert np.allclose(controller.scale_action(normalized),d,atol=1e-12)
    return normalized


def contact_diagnostics(raw):
    sim=raw.sim;target=set(raw.objects['obj'].contact_geoms);unwanted=0;self_contacts=0;gripper_target=0
    for i in range(sim.data.ncon):
        contact=sim.data.contact[i];a=sim.model.geom_id2name(contact.geom1) or '';b=sim.model.geom_id2name(contact.geom2) or ''
        ra=a.startswith(('robot0_','gripper0_'));rb=b.startswith(('robot0_','gripper0_'))
        if (a in target and b.startswith('gripper0_')) or (b in target and a.startswith('gripper0_')):gripper_target+=1
        elif ra and rb:self_contacts+=1
        elif ra or rb:unwanted+=1
    margins=[]
    for j in range(sim.model.njnt):
        name=sim.model.joint_id2name(j) or ''
        if name.startswith('robot0_') and sim.model.jnt_limited[j] and sim.model.jnt_type[j] in [2,3]:
            x=sim.data.qpos[sim.model.jnt_qposadr[j]];lo,hi=sim.model.jnt_range[j];margins.append(min(x-lo,hi-x))
    return dict(unwanted_contacts=unwanted,robot_self_contacts=self_contacts,gripper_target_contacts=gripper_target,
                minimum_joint_margin=float(min(margins)) if margins else None)


def main():
    import h5py,cv2
    config,marker=validate_freeze();os.environ.update(environment());np.random.seed(config['seed']);sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    from barx.simulation import build_environment_config,initialize_observation_utils
    from barx.evaluation_conditions import load_bundle,restore_frozen_condition
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    from simulation_validation.observe import Recorder
    bank=json.loads((PROJECT/'evidence/p3/frozen/source_contact_bank.json').read_text())
    payload,states,models=load_bundle(SOURCE/'evaluation/conditions',task='pnp_counter_to_sink',embodiment='panda')
    output=ROOT/'diagnostics';output.mkdir(parents=True,exist_ok=True)
    ledger=output/'results.json';results=[]
    cfg=build_environment_config('pnp_counter_to_sink','panda');initialize_observation_utils(cfg)
    env=create_env(**cfg,env_type=1,render=False,render_offscreen=True,use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(1000))
    def save(status='RUNNING'):
        temporary=ledger.with_suffix('.tmp');temporary.write_text(json.dumps(dict(status=status,role='PRIVILEGED_DIAGNOSTIC_ONLY',frozen_commit=marker['remote_commit'],results=results),indent=2)+'\n');temporary.replace(ledger)
    try:
        for template in bank:
            index=template['official_index'];branch=template['branch_action_index'];base_snapshot=None
            with h5py.File(template['panda_telemetry'],'r') as f:
                original=f['actions/environment'][:];legacy_q=f['sim/qpos'][branch];legacy_v=f['sim/qvel'][branch]
            assert payload['episodes'][index]['condition_id']==template['panda_condition']['condition_id']
            for method in ['A1','A2','A3']:
                directory=output/f'condition-{index:03d}'/method
                if directory.exists():raise FileExistsError('No automatic repeat of an existing physical intervention: '+str(directory))
                directory.mkdir(parents=True);recorder=Recorder(directory/'telemetry');writer=None;history=[];snapshot=None
                with (ROOT/'physical_launches.jsonl').open('a') as f:f.write(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='DEVELOPMENT_INTERVENTION',condition=index,method=method,frozen_commit=marker['remote_commit']))+'\n')
                try:
                    obs,metadata=restore_frozen_condition(env,payload['episodes'][index],states[index],models[index])
                    recorder.start(payload['episodes'][index],env,obs,metadata)
                    raw=env.env;sim=raw.sim;robot=raw.robots[0];site=robot.eef_site_id['right'];oid=raw.obj_body_id['obj']
                    gripper=robot.gripper['right'];controller=robot.composite_controller.part_controllers['right']
                    initial_z=float(sim.data.body_xpos[oid,2]);close_latched=False;dwell=0;plan=None
                    success=False;video=directory/'rollout.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,180))
                    assert writer.isOpened()
                    for t in range(config['max_episode_steps']):
                        command=original[t].copy() if t<len(original) else np.r_[np.zeros(6),original[-1,6],np.zeros(4),-1.]
                        original_command=command.copy();phase='original_prefix' if t<branch else 'original_arm'
                        current=pose(sim,site);goal=current.copy();endpoint=current.copy();grasp_error=None
                        if t==branch:
                            snapshot=branch_snapshot(raw,obs);flat=flatten_snapshot(snapshot)
                            np.savez_compressed(directory/'branch_snapshot.npz',**flat)
                            legacy_errors=dict(qpos=float(np.abs(snapshot['qpos']-legacy_q).max()),qvel=float(np.abs(snapshot['qvel']-legacy_v).max()))
                            pair_errors=compare_snapshots(base_snapshot,flat) if base_snapshot is not None else {k:0. for k in flat}
                            if base_snapshot is None:base_snapshot=flat
                            trusted=max([*legacy_errors.values(),*pair_errors.values()])<=config['branch']['prefix_replay_tolerance']
                            (directory/'branch_validation.json').write_text(json.dumps(dict(trusted=trusted,legacy_errors=legacy_errors,pair_errors=pair_errors),indent=2)+'\n')
                            if not trusted:raise ValueError('Untrusted counterfactual branch; no intervention executed')
                            object_world=transform(sim.data.body_xpos[oid],sim.data.body_xquat[oid]);grasp=object_world@np.array(template['T_object_tcp'])
                            place=np.array(template['T_world_sink_target'])@np.array(template['T_sink_release_tcp'])
                            plan=WaypointPlan(current,grasp,place,config)
                            (directory/'world_goals.json').write_text(json.dumps(dict(grasp=grasp.tolist(),release=place.tolist(),initial_tcp=current.tolist()),indent=2)+'\n')
                        if t>=branch:
                            if method=='A1':
                                c=template['original_closest_action_index']
                                release=branch+config['controls']['A1']['release_offset']
                                if c<=t<release:command[6]=1.
                                elif t>=release:command[6]=-1.
                            else:
                                phase,goal,endpoint=plan.goal(t-branch)
                                command[:6]=osc_action(controller,current,goal,config['controls'])
                                grasp_error=error(current,plan.grasp)
                                if method=='A3':
                                    within=grasp_error[0]<=config['controls']['position_tolerance_m'] and grasp_error[1]<=np.deg2rad(config['controls']['rotation_tolerance_degrees'])
                                    dwell=dwell+1 if phase in ['descend','hold_grasp'] and within else 0
                                    if dwell>=config['controls']['pose_dwell_states']:close_latched=True
                                    command[6]=-1. if phase in ['release','retreat','post_retreat'] else 1. if close_latched else -1.
                        assert np.isfinite(command).all()
                        if method=='A1':assert np.array_equal(command[:6],original_command[:6])
                        assert np.array_equal(command[7:],original_command[7:])
                        if method=='A2':assert command[6]==original_command[6]
                        image=(np.transpose(obs['barx_panda_agentview_image'],(1,2,0))*255).astype(np.uint8)
                        writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                        recorder.append('p3/original_environment_action',original_command)
                        recorder.append('p3/goal_tcp_world',goal);recorder.append('p3/phase_endpoint_world',endpoint)
                        obs,_,_,info=env.step(command);recorder.snapshot(env,obs,command,info)
                        label=info['is_success']['task'];assert isinstance(label,(bool,np.bool_));success=bool(label)
                        contact=bool(raw._check_grasp(gripper=gripper,object_geoms=raw.objects['obj'].contact_geoms))
                        rise=float(sim.data.body_xpos[oid,2]-initial_z);current_after=pose(sim,site)
                        endpoint_error=error(current_after,endpoint);extra=contact_diagnostics(raw)
                        history.append(dict(action_index=t,phase=phase,success=success,bilateral_contact=contact,rise_m=rise,
                            endpoint_position_error_m=endpoint_error[0],endpoint_rotation_error_rad=endpoint_error[1],
                            grasp_position_error_m=grasp_error[0] if grasp_error else None,grasp_rotation_error_rad=grasp_error[1] if grasp_error else None,
                            close_latched=close_latched,gripper_command=float(command[6]),**extra))
                        if success:break
                    mask=[h['bilateral_contact'] and h['rise_m']>=.03 and h['action_index']>=branch for h in history]
                    phases={name:any(h['phase']==name and h['endpoint_position_error_m']<=.015 and h['endpoint_rotation_error_rad']<=np.deg2rad(10) for h in history) for name,_ in config['controls']['phases']}
                    stable=longest_run(mask)>=10
                    failure=('OFFICIAL_SUCCESS' if success else 'POST_GRASP_TASK_FAILURE' if stable else 'CONTACT_WITHOUT_STABLE_LIFT' if any(h['bilateral_contact'] for h in history[branch:]) else 'POSE_NOT_REACHED' if method!='A1' and not phases['hold_grasp'] else 'NO_BILATERAL_CONTACT')
                    result=dict(status='COMPLETE',role='PRIVILEGED_DIAGNOSTIC_ONLY',condition=index,condition_id=template['panda_condition']['condition_id'],method=method,
                        branch_action_index=branch,branch_trusted=True,steps=len(history),task_success=success,stable_contact_lift=stable,
                        longest_contact_lift_states=longest_run(mask),bilateral_contact_states=sum(h['bilateral_contact'] for h in history[branch:]),
                        maximum_rise_m=max(h['rise_m'] for h in history),phase_endpoints_reached=phases,
                        unwanted_contact_state_steps=sum(h['unwanted_contacts']>0 for h in history[branch:]),
                        joint_limit_state_steps=sum(h['minimum_joint_margin'] is not None and h['minimum_joint_margin']<=.01 for h in history[branch:]),
                        close_latched=close_latched if method=='A3' else None,observed_failure_stage=failure,causal_failure_reason='NOT_AUTOMATICALLY_INFERRED',
                        video_path=str(video),telemetry_path=str(directory/'telemetry/episode-000.h5'),history_path=str(directory/'history.json'))
                    (directory/'history.json').write_text(json.dumps(history)+'\n');(directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
                    results.append(result);save();print(json.dumps(result),flush=True)
                except BaseException as exc:
                    (directory/'technical_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),executed_steps=len(history),count_as_task_failure=False),indent=2)+'\n')
                    save('TECHNICAL_FAILURE');raise
                finally:
                    if writer is not None:writer.release()
                    recorder.close()
        save('COMPLETE')
    finally:env.env.close()


if __name__=='__main__':main()
