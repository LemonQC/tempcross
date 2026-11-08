"""Bounded physical B2 development probes from original frozen initial states.

Replays full original prefixes, compares state/controller snapshots at the branch,
then holds only the gripper closed for 32 steps while replaying original arm
actions. This is a privileged physical diagnostic, not a learned-policy baseline.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np


def controller_state(robot):
    result={}
    composite=getattr(robot,'composite_controller',None)
    controllers=getattr(composite,'part_controllers',{}) or getattr(robot,'part_controllers',{})
    def capture(prefix,obj):
        if not hasattr(obj,'__dict__'):return
        for key,value in vars(obj).items():
            if isinstance(value,(float,int,bool,np.ndarray)):
                a=np.asarray(value)
                if a.dtype.kind in 'biuf' and a.size<10000:result[prefix+key]=a.copy()
    if isinstance(controllers,dict):
        for name,controller in controllers.items():capture(name+'/',controller)
    grippers=robot.gripper if isinstance(robot.gripper,dict) else {'right':robot.gripper}
    for name,gripper in grippers.items():capture('gripper/'+name+'/',gripper)
    if not result:raise RuntimeError('Controller state introspection unavailable')
    return result


def main():
    import h5py
    import cv2
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--legacy',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.source).resolve();legacy=Path(a.legacy).resolve();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(source),str(source/'policy')]
    from barx.simulation import build_environment_config,initialize_observation_utils
    from barx.evaluation_conditions import load_bundle,restore_frozen_condition
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    cfg=build_environment_config('pnp_counter_to_sink','panda');initialize_observation_utils(cfg)
    env=create_env(**cfg,env_type=1,render=False,render_offscreen=True,use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(1000))
    payload,states,models=load_bundle(source/'evaluation/conditions',task='pnp_counter_to_sink',embodiment='panda')
    # Fixed representatives selected before any intervention outcome.
    specs=[('joint_reps',0,117),('joint_reps',0,125),('joint_reps',2,156),('joint_reps',2,164),
           ('joint_reps',12,143),('joint_reps',12,151),('no_reps',4,167)]
    results=[];ledger=out/'results.json'
    def save():ledger.write_text(json.dumps(dict(status='RUNNING',role='PRIVILEGED_DIAGNOSTIC_ONLY',probes=results),indent=2))
    try:
        for method,index,branch in specs:
            stem=f'{method}-{index:03d}-step-{branch}'
            path=legacy/'p1'/method/'panda/telemetry'/f'episode-{index:03d}.h5'
            with h5py.File(path,'r') as f:
                actions=f['actions/environment'][:];legacy_q=f['sim/qpos'][:];legacy_eef=f['obs/robot0_eef_pos'][:]
                legacy_obj=f['target/position'][:];legacy_grip=f['obs/robot0_gripper_qpos'][:]
            snapshots=[];pair=[]
            for intervene in [False,True]:
                obs,_=restore_frozen_condition(env,payload['episodes'][index],states[index],models[index])
                raw=env.env;sim=raw.sim;target=raw.obj_body_id['obj'];robot=raw.robots[0]
                gripper=robot.gripper['right'] if isinstance(robot.gripper,dict) else robot.gripper
                initial_z=float(sim.data.body_xpos[target,2]);contact_steps=[];lift_steps=[];official_success=False
                label='close32' if intervene else 'original';video=out/f'{stem}-{label}.mp4'
                writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,180))
                history=[];snapshot=None
                try:
                    for t,original_action in enumerate(actions):
                        if t==branch:
                            snapshot=dict(qpos=sim.data.qpos.copy(),qvel=sim.data.qvel.copy(),eef=np.asarray(obs['robot0_eef_pos']).copy(),
                              object=sim.data.body_xpos[target].copy(),gripper=np.asarray(obs['robot0_gripper_qpos']).copy(),ctrl=sim.data.ctrl.copy(),
                              qacc_warmstart=sim.data.qacc_warmstart.copy(),controllers=controller_state(robot))
                        image=(255*np.transpose(obs['barx_panda_agentview_image'],(1,2,0))).astype(np.uint8)
                        writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                        command=original_action.copy()
                        if intervene and branch<=t<branch+32:command[6]=1.
                        obs,_,_,info=env.step(command)
                        contact=bool(raw._check_grasp(gripper=gripper,object_geoms=raw.objects['obj'].contact_geoms))
                        rise=float(sim.data.body_xpos[target,2]-initial_z)
                        if contact:contact_steps.append(t+1)
                        if contact and rise>=.03:lift_steps.append(t+1)
                        history.append(dict(step=t+1,gripper_command=float(command[6]),bilateral_contact=contact,rise_m=rise,
                                            object=sim.data.body_xpos[target].tolist(),eef=np.asarray(obs['robot0_eef_pos']).tolist(),
                                            success=bool(info['is_success']['task'])))
                        if info['is_success']['task']:official_success=True;break
                finally:writer.release()
                if snapshot is None:raise RuntimeError('Branch was not reached')
                np.savez_compressed(out/f'{stem}-{label}-branch.npz',
                    **{k:v for k,v in snapshot.items() if k!='controllers'},
                    **{'controller::'+k:v for k,v in snapshot['controllers'].items()})
                snapshots.append(snapshot)
                (out/f'{stem}-{label}.json').write_text(json.dumps(history))
                from p2_validation.offline_audit import longest_run
                mask=[h['bilateral_contact'] and h['rise_m']>=.03 for h in history]
                pair.append(dict(branch=label,success=official_success,steps=len(history),contact_state_steps=len(contact_steps),
                                 maximum_rise_m=max(h['rise_m'] for h in history),contact_lift_consecutive_steps=longest_run(mask),video_path=str(video)))
            first,second=snapshots
            errors={key:float(np.abs(first[key]-second[key]).max()) for key in first if key!='controllers'}
            if first['controllers'].keys()!=second['controllers'].keys():raise RuntimeError('Controller keys differ across branches')
            control_errors={key:float(np.abs(first['controllers'][key].astype(float)-second['controllers'][key].astype(float)).max())
                            for key in first['controllers'] if first['controllers'][key].size}
            legacy_errors=dict(qpos=float(np.abs(first['qpos']-legacy_q[branch]).max()),
                               eef=float(np.abs(first['eef']-legacy_eef[branch]).max()),
                               object=float(np.abs(first['object']-legacy_obj[branch]).max()),
                               gripper=float(np.abs(first['gripper']-legacy_grip[branch]).max()))
            trusted=max([*errors.values(),*control_errors.values(),*legacy_errors.values()])<1e-8
            positive=method=='no_reps'
            results.append(dict(case_id=f'{method}/panda/{index:03d}',condition_id=payload['episodes'][index]['condition_id'],
                                branch_action_index=branch,hold_steps=32,positive_control=positive,
                                state_pair_max_errors=errors,controller_pair_max_errors=control_errors,
                                legacy_branch_max_errors=legacy_errors,branch_trusted=trusted,outcomes=pair))
            save();print(json.dumps(results[-1]),flush=True)
        positive_ok=all(r['branch_trusted'] and all(v['success'] for v in r['outcomes']) for r in results if r['positive_control'])
        rescued=[r for r in results if not r['positive_control'] and r['branch_trusted'] and
                 not r['outcomes'][0]['success'] and r['outcomes'][1]['success']]
        ledger.write_text(json.dumps(dict(status='COMPLETE',role='PRIVILEGED_DIAGNOSTIC_ONLY',probes=results,
            positive_control_passed=positive_ok,rescued_development_probes=len(rescued),
            formal_oracle_status='ORACLE_NOT_VALIDATED',
            explanation='A finite scheduled gripper probe is not a validated general closed-loop grasp-opportunity oracle.'),indent=2))
    finally:env.env.close()


if __name__=='__main__':main()
