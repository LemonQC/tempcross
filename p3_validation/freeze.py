"""Extract existing successful contact poses and freeze P3 before new physics."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from p3_validation.common import PROJECT,SOURCE,P2,ROOT,CHECKPOINT,OLD_ADAPTED
from p3_validation.geometry import transform,inverse,error,site_definition
from p2_validation.offline_audit import longest_run,sha256,fingerpad_specs,world_pad,center_prism

CASES=[22,24,26,29,34,35,37,39]


def extract():
    import h5py
    output=PROJECT/'evidence/p3/frozen';output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    from barx.evaluation_conditions import load_bundle,validate_condition_bundle_for_evaluation
    from barx.benchmark import EMBODIMENTS
    bundle_audit=[]
    for arm in ['ur5e','panda']:
        payload,states,xmls=load_bundle(SOURCE/'evaluation/conditions',task='pnp_counter_to_sink',embodiment=arm)
        validate_condition_bundle_for_evaluation(payload,states,xmls,task='pnp_counter_to_sink',camera_name=EMBODIMENTS[arm].camera,image_width=320,image_height=180,count=100)
        bundle_audit.append(dict(robot=arm,complete_official_validation_passed=True,
            files=[dict(path=str(SOURCE/f'evaluation/conditions/pnp_counter_to_sink/{arm}{suffix}'),sha256=sha256(SOURCE/f'evaluation/conditions/pnp_counter_to_sink/{arm}{suffix}')) for suffix in ['.json','.npz']]))
    (output/'official_condition_bundle_integrity.json').write_text(json.dumps(bundle_audit,indent=2)+'\n')
    jobs={arm:json.loads((P2/f'formal/original-joint_reps-{arm}/job.json').read_text()) for arm in ['ur5e','panda']}
    models={}
    for arm in jobs:
        with np.load(SOURCE/f'evaluation/conditions/pnp_counter_to_sink/{arm}.npz',allow_pickle=False) as f:models[arm]=f['model_xmls'].tolist()
    bank=[]
    for index in CASES:
        trial=jobs['ur5e']['indices'].index(index);path=P2/f'formal/original-joint_reps-ur5e/telemetry/episode-{trial:03d}.h5'
        records=[json.loads(x) for x in next((path.parents[1]/'official').rglob('episodes.jsonl')).read_text().splitlines()]
        assert records[trial]['success'] is True
        with h5py.File(path) as f:
            positions=f['target/position'][:];quats=f['target/quaternion'][:];actions=f['actions/environment'][:]
            eef=f['obs/robot0_eef_pos'][:];site_quat=f['obs/robot0_eef_quat_site'][:]
            names=json.loads(f.attrs['body_names']);meta=json.loads(f.attrs['ep_meta'])
            body,local=site_definition(models['ur5e'][index]);bid=names.index(body)
            reconstructed=np.stack([transform(p,q)@local for p,q in zip(f['sim/body_xpos'][:,bid],f['sim/body_xquat'][:,bid])])
            observed=np.stack([transform(p,q,'xyzw') for p,q in zip(eef,site_quat)])
            # Sensor observations are cached and can lag the stored simulator
            # poses by a substep. Use synchronous body+site geometry as truth;
            # site observations only audit the quaternion/frame convention.
            observation_position_error=np.max(np.linalg.norm(reconstructed[:,:3,3]-observed[:,:3,3],axis=1))
            observation_rotation_error=max(error(p,q)[1] for p,q in zip(reconstructed,observed))
            assert observation_position_error<.005 and observation_rotation_error<np.deg2rad(2),'Wrong TCP/quaternion convention'
            sink_body=meta['fixture_refs']['sink']+'_main';sink_id=names.index(sink_body)
            sink_source=transform(f['sim/body_xpos'][0,sink_id],f['sim/body_xquat'][0,sink_id])
            specs=fingerpad_specs(models['ur5e'][index]);pads={s:world_pad(f['sim/body_xpos'][:,names.index(v['body'])],f['sim/body_xquat'][:,names.index(v['body'])],v) for s,v in specs.items()}
            _,gaps,_,_=center_prism(pads['left'],pads['right'],positions,specs)
            contacts=[json.loads(x)['grasp_contact'] is True for x in path.with_suffix('.jsonl').read_text().splitlines()]
            contact=np.array(contacts);rise=positions[:,2]-positions[0,2]
            valid=[t for t in range(len(contact)-10) if contact[t:t+10].all() and longest_run(contact[t:t+100]&(rise[t:t+100]>=.03))>=10]
            if not valid:raise ValueError(f'No source confirmed grasp at {index}')
            t=valid[0];lift_start=next(k for k in range(t,len(contact)-9) if (contact[k:k+10]&(rise[k:k+10]>=.03)).all())
            open_indices=np.flatnonzero((actions[:,6]<0)&(np.arange(len(actions))>=lift_start+9))
            release=int(open_indices[0]) if len(open_indices) else min(int(np.flatnonzero(contact)[-1])+1,len(contact)-1)
            relative=inverse(transform(positions[t],quats[t]))@reconstructed[t]
            sink_release=inverse(sink_source)@reconstructed[release]
            source_geometry=dict(inner_pad_gap_m=float(gaps[t]),left_pad_position=pads['left'][0][t].tolist(),right_pad_position=pads['right'][0][t].tolist(),gripper_qpos=f['obs/robot0_gripper_qpos'][t].tolist())
        ptrial=jobs['panda']['indices'].index(index);ppath=P2/f'formal/original-joint_reps-panda/telemetry/episode-{ptrial:03d}.h5'
        with h5py.File(ppath) as f:
            distance=np.linalg.norm(f['obs/robot0_eef_pos'][:]-f['target/position'][:],axis=1)
            closest=int(np.argmin(distance[40:min(351,len(distance))])+40);branch=min(200,max(40,closest-16))
            target_names=json.loads(f.attrs['body_names']);target_meta=json.loads(f.attrs['ep_meta'])
            assert sink_body==target_meta['fixture_refs']['sink']+'_main'
            target_sink_id=target_names.index(sink_body)
            sink_target=transform(f['sim/body_xpos'][0,target_sink_id],f['sim/body_xquat'][0,target_sink_id])
            assert np.max(np.abs(f['actions/environment'][:,7:11]))==0
            assert np.all(f['actions/environment'][:,-1]==-1)
        source_obj=next(o for o in meta['object_cfgs'] if o['name']=='obj')
        target_obj=next(o for o in target_meta['object_cfgs'] if o['name']=='obj')
        assert source_obj['info']['mjcf_path']==target_obj['info']['mjcf_path'],'Object asset mismatch'
        bank.append(dict(official_index=index,source_trial_index=trial,panda_trial_index=ptrial,
            source_condition=jobs['ur5e']['conditions'][trial],panda_condition=jobs['panda']['conditions'][ptrial],
            source_telemetry=str(path),source_telemetry_sha256=sha256(path),panda_telemetry=str(ppath),panda_telemetry_sha256=sha256(ppath),
            contact_state_index=t,contact_confirmation_states=10,lift_confirmation_start_state=lift_start,
            release_state_index=release,release_rule='first open command after confirmed lift, else last contact plus one',
            T_object_tcp=relative.tolist(),T_sink_release_tcp=sink_release.tolist(),T_world_sink_target=sink_target.tolist(),
            source_tcp_world=reconstructed[t].tolist(),source_object_world=transform(positions[t],quats[t]).tolist(),
            source_grasp_geometry=source_geometry,sensor_vs_simulator_max_position_error_m=float(observation_position_error),
            sensor_vs_simulator_max_rotation_error_rad=float(observation_rotation_error),
            original_closest_action_index=closest,branch_action_index=branch,
            object_asset=source_obj['info']['mjcf_path'],source_and_target_object_asset_identical=True))
    (output/'source_contact_bank.json').write_text(json.dumps(bank,indent=2)+'\n')
    selected={}
    for arm in ['ur5e','panda']:
        payload=json.loads((SOURCE/f'evaluation/conditions/pnp_counter_to_sink/{arm}.json').read_text())
        selected[arm]=[{k:e[k] for k in ['episode','condition_id','seed','state_sha256','model_sha256','metadata_sha256']} for e in payload['episodes'][40:60]]
    (output/'independent_conditions_40_59.json').write_text(json.dumps(selected,indent=2)+'\n')
    config=dict(version='P3_202611091028',baseline_commit='702005202d84ba0f39d213d3af72cffeaf30cc41',source_commit='0f435089f48b1f16d225f868dbef855a5e75d1e0',
        development_indices=CASES,independent_indices=list(range(40,60)),seed=7,max_episode_steps=600,
        source_pose=dict(tcp_site='gripper0_right_grip_site',quaternion_convention='site obs xyzw; body/object wxyz; verified against XML',stable_bilateral_states=10,lift_m=.03,lift_states=10,confirmation_window_states=100),
        branch=dict(rule='min(200,max(40,argmin B0 EEF-object distance on states40..350 minus16))',frozen_indices={str(b['official_index']):b['branch_action_index'] for b in bank},prefix_replay_tolerance=1e-8),
        controls=dict(A1=dict(arm='original recorded full Panda actions',close_at='original minimum-distance state index',close_hold_steps=32,then='original gripper schedule'),
          A2=dict(arm='same fixed OSC waypoint schedule as A3',gripper='original B0 recorded binary command at same absolute episode action index; hold final command after B0 ends'),
          A3=dict(arm='same fixed OSC waypoint schedule as A2',gripper='open until 5 consecutive states within grasp tolerance during descend/hold; close until scheduled release; then open'),
          gain_position=.4,gain_rotation=.4,max_position_step_m=.012,max_rotation_step_rad=.12,
          position_tolerance_m=.015,rotation_tolerance_degrees=10,pose_dwell_states=5,
          approach_up_m=.08,pregrasp_up_m=.08,lift_m=.08,preplace_up_m=.12,retreat_up_m=.12,
          phases=[['clearance',30],['pregrasp',45],['descend',45],['hold_grasp',20],['lift',35],['preplace',65],['place',35],['release',16],['retreat',35]],
          sink_target='source release TCP transformed by source/target sink fixture frames; frozen source rule, no search',
          grasp_target='object branch world pose times source T_object_tcp, anchored for all subsequent waypoints',
          controller='unchanged official OSC_POSE delta control, gain/dynamics untouched; world error expressed in actual controller reference'),
        diagnostics=dict(stable_grasp='bilateral contact and object rise>=.03m for>=10 consecutive post-action states',joint_limit_margin=.01,collision='robot/fixture or non-target contacts; intended gripper-target contact separate',task_success='unchanged official info.is_success.task',proxies_not_task_success=True),
        training=dict(method='joint_reps',initial_checkpoint=str(CHECKPOINT),sample_positions=96000,batch_probe_candidates=[8,4,2],probe='one real forward/backward/optimizer step per candidate until first success; probe weights discarded',
          updates_by_batch={'8':12000,'4':24000,'2':48000},sample_seed=20261109,sampling='fresh official mixed RLDS stream; full published train split100 target/1800 source, not old6000 cache repetition',
          prior_weight=.05,target_weight=1.,balance_by_transitions=True,normalization='mg_pnp_lite prior statistics and VQ',shuffle_buffer=10000,learning_rate=2e-5,lora_rank=32,lora_alpha=16,lora_dropout=0,checkpoint_selection='only final prescribed update',
          target_revision='6e297509924c831c722180212d13f21686676e8e',source_revision='a2b5256a48428ea56064b0afb3dc184a045b10ae',coverage='raw target demo mapping by action/instruction/length fingerprint; per-episode/frame identities and actual consumed sequence logged',
          limitation='enhanced restricted LoRA, 1/8 official sample exposure; full-train coverage differs from P2 loader default95%, so budget-only causal attribution is not claimed'),
        evaluation=dict(cells=['original-joint_reps-ur5e','original-joint_reps-panda','b1-joint_reps-panda','p2-6000-joint_reps-panda','p3-96000-joint_reps-panda'],episodes_per_cell=20,b1_threshold=.85,b1_hysteresis=.03,b1_minimum_hold=8,old_adapter=str(OLD_ADAPTED),optional_privileged_20=False),
        frozen_assets=dict(source_bank_sha256=sha256(output/'source_contact_bank.json'),independent_conditions_sha256=sha256(output/'independent_conditions_40_59.json'),bundle_integrity_sha256=sha256(output/'official_condition_bundle_integrity.json')))
    (PROJECT/'configs/p3_protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(dict(source_templates=len(bank),branches=config['branch']['frozen_indices'],independent_conditions_per_robot=20,config_sha256=sha256(PROJECT/'configs/p3_protocol.json')),indent=2))


if __name__=='__main__':extract()
