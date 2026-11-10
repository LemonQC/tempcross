"""Read-only analysis of 160 saved BARX Joint trajectories; zero inference/physics.

Run with the original BARX Python environment (h5py/numpy/scipy installed).
Writes only evidence/interaction_initiation and a new external diagnostic root.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import h5py

REPO=Path(__file__).resolve().parents[2]
BASE=Path('/home/users/lemon/baseline-benchmark/run-20261109')
SOURCE=Path('/home/users/lemon/coft-p2/barx')
HOST=Path('/home/users/lemon/interaction-initiation-20261110')
OUT=REPO/'evidence/interaction_initiation'
TASKS=['pnp_counter_to_sink','pnp_sink_to_counter']
INPUTS={}
CONFIG=dict(role='POSTHOC_DESCRIPTIVE_AUDIT_NOT_NEW_BENCHMARK',control_freq=20,chunk=8,
    near_thresholds_m=[.04,.06,.08,.10,.12],primary_near_m=.08,primary_near_is_not_a_graspability_certificate=True,
    history_window_actions=10,spatial_position_caliper_m=.02,spatial_rotation_caliper_degrees=20,
    history_time_caliper_actions=40,history_path_caliper_m=.10,history_radial_speed_caliper_m_s=.04,
    history_jaw_caliper_m=.01,history_near_dwell_caliper_actions=8,
    matching='Same task/index and source first initiation query. Both preinteraction risk sets. Prefer candidates passing spatial AND history calipers, then spatial-only, then unsupported nearest. One target per source condition; descriptive conditional-on-source-initiation coverage, never causal pairing.',
    native_gripper='Stored post-decoding/unnormalization action, not token logits/probability. g<.5 -> close +1 via -sign(2g-1).',
    newly_executed_inference_calls=0,new_physics_steps=0,new_training_updates=0)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def record(path):
    path=Path(path);INPUTS[str(path)]=digest(path);return path


def read(path):return json.loads(record(path).read_text())


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def table(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    directory=HOST if name=='query_progress.csv' else OUT
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def first(mask):
    indices=np.flatnonzero(mask);return int(indices[0]) if len(indices) else None


def longest(mask):
    result=run=0
    for x in mask:run=run+1 if x else 0;result=max(result,run)
    return result


def physical_contact(records,geoms):
    result=[]
    for row in records:
        sides=set()
        for c in row['contacts']:
            if c['distance']>0:continue
            for a,b in [(c['geom1'],c['geom2']),(c['geom2'],c['geom1'])]:
                if b not in geoms:continue
                for side in ['left','right']:
                    if a in {f'gripper0_right_{side}_fingerpad_collision',f'gripper0_right_{side}_fingertip_collision'}:sides.add(side)
        result.append(len(sides)==2)
    return np.asarray(result,dtype=bool)


def describe(t,episode):
    return dict(action_index=int(t),distance_m=float(episode['distance'][t]),tcp_relative_object_xyz=episode['relative'][t].tolist(),
        object_relative_tcp_xyz=episode['object_in_tcp'][t].tolist(),
        tcp_world_xyz=episode['tcp_world'][t].tolist(),object_world_xyz=episode['object_world'][t].tolist(),
        tcp_relative_object_rotation=episode['rotation'][t].reshape(-1).tolist(),
        object_relative_tcp_rotation=episode['rotation'][t].T.reshape(-1).tolist(),object_rise_m=float(episode['rise'][t]),
        object_displacement_m=float(episode['object_displacement'][t]),jaw_gap_m=float(episode['jaw'][t]),
        cumulative_tcp_path_m=float(episode['path'][t]),recent_radial_speed_m_s=float(episode['speed'][t]),near_dwell_actions=int(episode['dwell'][t]),
        gripper_qpos=episode['gripper'][t].tolist(),native_g=float(episode['native'][t,6]),actual_g=float(episode['actions'][t,6]),
        recorded_native_action=episode['native'][t].tolist(),recorded_environment_action=episode['actions'][t].tolist(),
        query_start_action=int(8*(t//8)),query_offset=int(t%8))


def main():
    assert not (HOST/'policy_inference_calls.jsonl').exists(),'Do not merge a different inference attempt into this zero-inference audit'
    for path in [BASE/'physical_launches.jsonl',Path('/home/users/lemon/mechanism-20261110/physical_launches.jsonl'),REPO/'results/mechanism/decision.json',REPO/'baselines/common/policies.py',REPO/'baselines/common/action_adapter.py',REPO/'baselines/common/evaluator.py',SOURCE/'policy/experiments/robot/robot_utils.py',SOURCE/'policy/experiments/robot/robocasa_x/evaluate.py']:
        record(path)
    checkpoint=record('/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt')
    cfg=read('/home/users/lemon/coft-p2/artifacts/runs/joint_reps/config.json')
    assert cfg['image_aug'] is False and cfg['vla']['image_sequence_len']==1 and cfg['vla']['use_wrist_image'] is False
    # Never serialize the full training config: unrelated private credential fields may exist.
    safe_config={k:cfg[k] for k in ['image_aug','normalize_data','seed']}
    safe_config.update(image_sequence_len=cfg['vla']['image_sequence_len'],use_wrist_image=cfg['vla']['use_wrist_image'],center_crop_default=False,act_horizon=8,unnorm_key='mg_pnp_lite')
    write('analysis_spec.json',CONFIG)
    prior=read(REPO/'results/baselines/analysis.json')
    assert len(prior['episodes'])==1280
    selected=[r for r in prior['episodes'] if r['label']=='A_barx_joint' and r['task'] in TASKS and r['robot'] in ['ur5e','panda']]
    assert len(selected)==160
    bundles={(t,r):read(SOURCE/f'evaluation/conditions/{t}/{r}.json')['episodes'] for t in TASKS for r in ['ur5e','panda']}
    episodes=[];events=[];queries=[];queue_rows=[];all_data={};runtime_inputs=set()
    for entry in selected:
        task=entry['task'];robot=entry['robot'];index=entry['official_condition_index'];key=(task,robot,index)
        path=record(entry['telemetry_path']);job=read(path.parents[1]/'job.json');assert job['kind']=='barx' and job['label']=='A_barx_joint'
        assert job['checkpoint']==str(checkpoint) and job['checkpoint_sha256']==INPUTS[str(checkpoint)]
        runtime=job['policy_runtime']['algorithm_source_path']
        if runtime not in runtime_inputs:record(runtime);runtime_inputs.add(runtime)
        assert INPUTS[runtime]==job['policy_runtime']['algorithm_source_sha256']
        logs=[json.loads(x) for x in record(path.with_suffix('.jsonl')).read_text().splitlines()]
        with h5py.File(path,'r') as f:
            native=f['baseline/native_action'][:];action=f['actions/environment'][:];raw_query=f['baseline/native_query'][:];qt=f['baseline/native_query_action_index'][:].astype(int)
            n=len(action);assert native.shape==(n,7) and raw_query.shape[1:]==(8,7)
            assert np.array_equal(qt,np.arange(0,n,8))
            flattened=raw_query.reshape(-1,7);queued=flattened[:n]
            queue_error=float(np.max(np.abs(queued-native)));assert queue_error==0
            canonical=native.copy();canonical[:,6]=-np.sign(2*native[:,6]-1)
            before=f['baseline/canonical_before_clip'][:]
            pre_error=float(np.max(np.abs(canonical-before)));assert pre_error==0
            expected=np.c_[np.clip(canonical,-1,1),np.zeros((n,4)),-np.ones(n)]
            interface_error=float(np.max(np.abs(expected-action)));assert interface_error==0
            meta=json.loads(f.attrs['ep_meta']);condition=json.loads(f.attrs['condition']);official=bundles[task,robot][index]
            objects={o['name']:o for o in meta['object_cfgs']};cat=objects['obj']['info']['cat']
            expected_text=f'pick the {cat} from the counter and place it in the sink' if task==TASKS[0] else f"pick the {cat} from the sink and place it on the {objects['container']['info']['cat']} located on the counter"
            assert meta['lang']==official['instruction']==expected_text
            assert condition['condition_id']==official['condition_id']==entry['condition_id']
            tcp=f['baseline/pre_action_tcp_position'][:];tcp_rot=f['baseline/pre_action_tcp_rotation_matrix'][:]
            obj=f['target/position'][:];obj_rot=Rotation.from_quat(f['target/quaternion'][:][:,[1,2,3,0]]).as_matrix()
            names=json.loads(f.attrs['body_names']);geom_names=json.loads(f.attrs['geom_names'])
            geoms={g for g,b in zip(geom_names,f['model/geom_bodyid'][:]) if names[b].startswith('obj_')}
            contact=physical_contact(logs,geoms);assert len(contact)==n+1
            jaw=f['baseline/gripper_pad_center_distance'][:];gripper=f['obs/robot0_gripper_qpos'][:]
            placement=f['baseline/post_action_placement'][:].astype(bool)
            relative=np.einsum('nji,nj->ni',obj_rot[:-1],tcp-obj[:-1]);rotation=np.swapaxes(obj_rot[:-1],1,2)@tcp_rot
            distance=np.linalg.norm(relative,axis=1);rise=obj[:,2]-obj[0,2];disp=np.linalg.norm(obj-obj[0],axis=1)
            path_length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(tcp,axis=0),axis=1))]
            lag=np.maximum(np.arange(n)-10,0);dt=(np.arange(n)-lag)/20
            speed=np.divide(distance-distance[lag],dt,out=np.zeros(n),where=dt>0)
            dwell=[];run=0
            for b in distance<=.08:run=run+1 if b else 0;dwell.append(run)
            frustum=np.array([r.get('coft_static_projection',{}).get('in_frustum') is True if r.get('coft_static_projection') else False for r in logs])
            e=dict(distance=distance,relative=relative,object_in_tcp=np.einsum('nji,nj->ni',tcp_rot,obj[:-1]-tcp),rotation=rotation,rise=rise,object_displacement=disp,jaw=jaw,gripper=gripper,path=path_length,speed=speed,dwell=np.asarray(dwell),native=native,actions=action,tcp_world=tcp,object_world=obj)
            e.update(contact=contact,placement=placement,query_starts=qt,raw_query=raw_query,video=entry['video_path'],instruction=meta['lang'])
            close=native[:,6]<.5;actual=action[:,6]>0;assert np.array_equal(close,actual)
            fc=first(actual);first_query=next((int(t) for k,t in enumerate(qt) if np.any(raw_query[k,:min(8,n-t),6]<.5)),None)
            first_full_prediction=next((int(t) for k,t in enumerate(qt) if np.any(raw_query[k,:,6]<.5)),None)
            assert first_full_prediction==first_query,'A closure exists only in an unexecuted chunk tail'
            e['first_close']=fc;e['first_query']=first_query;all_data[key]=e
            valid_contact=contact[fc+1:] if fc is not None else np.asarray([],dtype=bool)
            stable=longest(contact & (rise>=.03))>=10
            bucket=('NO_CLOSE_OUTSIDE_NEAR_PROXY' if distance.min()>.08 else 'NEAR_PROXY_BUT_NO_CLOSE') if fc is None else 'CLOSE_WITHOUT_TWO_SIDED_CONTACT' if not valid_contact.any() else 'CONTACT_WITHOUT_STABLE_LIFT' if not stable else 'STABLE_LIFT'
            gap_reduction=float(max(jaw[:fc+1].max()-jaw[fc+1:].min(),0)) if fc is not None and fc<n else 0.
            row=dict(task=task,robot=robot,condition=index,condition_id=entry['condition_id'],category=cat,instruction=meta['lang'],steps=n,official_success=entry['success'],
                first_query_predicting_executable_close=first_query,first_full_query_predicting_close_including_unexecuted_tail=first_full_prediction,
                first_scheduled_and_executed_close=fc,first_execution_seconds=fc/20 if fc is not None else None,queue_delay_actions=fc-first_query if fc is not None else None,
                predicted_close=bool(close.any()),executed_close=fc is not None,minimum_native_g=float(native[:,6].min()),maximum_native_g=float(native[:,6].max()),
                native_g_exactly_half_count=int((native[:,6]==.5).sum()),native_g_outside_unit_interval_count=int(((native[:,6]<0)|(native[:,6]>1)).sum()),
                arm_actions_clipped_count=int((np.abs(native[:,:6])>1).any(axis=1).sum()),
                minimum_tcp_object_distance_m=float(distance.min()),first_near8_action=first(distance<=.08),near8_actions=int((distance<=.08).sum()),maximum_near8_dwell_actions=int(max(dwell)),
                near8_and_center_in_frustum_actions=int(((distance<=.08)&frustum[:-1]).sum()),near8_and_center_in_frustum_query_inputs=int(((distance[qt]<=.08)&frustum[qt]).sum()),
                contact_after_close=bool(valid_contact.any()),stable_lift=stable,maximum_jaw_gap_reduction_after_close_m=gap_reduction,
                stage_bucket=bucket,unexecuted_last_chunk_actions=len(flattened)-n,unexecuted_tail_close_actions=int((flattened[n:,6]<.5).sum()),
                telemetry_path=str(path),video_path=entry['video_path'])
            episodes.append(row)
            queue_rows.append(dict(task=task,robot=robot,condition=index,steps=n,queries=len(qt),query_schedule_every8=True,query_to_native_max_error=queue_error,native_to_preclip_max_error=pre_error,canonical_to_actual_max_error=interface_error,
                instruction_matches_official=True,checkpoint_sha256=job['checkpoint_sha256'],policy_source_sha256=job['policy_runtime']['algorithm_source_sha256']))
            for label,t in [('closest',int(np.argmin(distance))),('first_near8',first(distance<=.08)),('first_prediction_query',first_query),('first_executed_close',fc)]:
                if t is not None:events.append(dict(task=task,robot=robot,condition=index,event=label,**describe(t,e)))
            for k,t in enumerate(qt):
                # Preinteraction risk set: no earlier close/contact/placement or 3cm object motion.
                risk=(fc is None or t<=first_query) and not contact[:t+1].any() and not placement[:t].any() and disp[t]<.03
                chunk=raw_query[k,:min(8,n-t)]
                queries.append(dict(task=task,robot=robot,condition=index,query_action_index=int(t),distance_m=float(distance[t]),relative_x_m=float(relative[t,0]),relative_y_m=float(relative[t,1]),relative_z_m=float(relative[t,2]),
                    jaw_gap_m=float(jaw[t]),object_displacement_m=float(disp[t]),object_rise_m=float(rise[t]),prior_contact=bool(contact[:t+1].any()),preinteraction_risk_set=bool(risk),
                    object_center_in_frustum=bool(frustum[t]),
                    predicts_close_in_executed_prefix=bool((chunk[:,6]<.5).any()),minimum_chunk_native_g=float(chunk[:,6].min()),first_chunk_native_g=float(chunk[0,6]),
                    cumulative_tcp_path_m=float(path_length[t]),recent_radial_speed_m_s=float(speed[t]),near_dwell_actions=int(dwell[t])))
    table('episodes.csv',episodes);table('event_states.csv',events);table('query_progress.csv',queries);table('interface_queue_checks.csv',queue_rows)
    progress=HOST/'query_progress.csv'
    write('query_progress_index.json',dict(absolute_path=str(progress),bytes=progress.stat().st_size,sha256=digest(progress),rows=len(queries),role='DERIVED_SAVED_QUERY_INPUT_PROGRESS_NOT_NEW_INFERENCE'))
    groups=[];sensitivity=[]
    for task in TASKS:
        for robot in ['ur5e','panda']:
            rr=[r for r in episodes if r['task']==task and r['robot']==robot]
            groups.append(dict(task=task,robot=robot,episodes=40,raw_predicted_close=sum(r['predicted_close'] for r in rr),executed_close=sum(r['executed_close'] for r in rr),
                reached_near8=sum(r['minimum_tcp_object_distance_m']<=.08 for r in rr),near8_never_close=sum(r['minimum_tcp_object_distance_m']<=.08 and not r['executed_close'] for r in rr),
                near8_never_close_dwell20=sum(r['maximum_near8_dwell_actions']>=20 and not r['executed_close'] for r in rr),
                near8_never_close_center_in_frame_at_query=sum(r['near8_and_center_in_frustum_query_inputs']>0 and not r['executed_close'] for r in rr),
                contact_after_close=sum(r['contact_after_close'] for r in rr),stable_lift=sum(r['stable_lift'] for r in rr),success=sum(r['official_success'] for r in rr),
                median_minimum_distance_m=float(np.median([r['minimum_tcp_object_distance_m'] for r in rr])),stage_counts=dict(Counter(r['stage_bucket'] for r in rr))))
            for threshold in CONFIG['near_thresholds_m']:
                sensitivity.append(dict(task=task,robot=robot,near_threshold_m=threshold,episodes=40,ever_near=sum(r['minimum_tcp_object_distance_m']<=threshold for r in rr),near_never_closed=sum(r['minimum_tcp_object_distance_m']<=threshold and not r['executed_close'] for r in rr)))
    table('group_summary.csv',groups);table('near_threshold_sensitivity.csv',sensitivity)
    timing=[]
    for task in TASKS:
        for robot in ['ur5e','panda']:
            closing=[r for r in episodes if r['task']==task and r['robot']==robot and r['executed_close']]
            opening=[r for r in episodes if r['task']==task and r['robot']==robot and not r['executed_close']]
            timing.append(dict(task=task,robot=robot,closing_episodes=len(closing),
                median_first_prediction_query_action=float(np.median([r['first_query_predicting_executable_close'] for r in closing])),
                median_first_execution_action=float(np.median([r['first_scheduled_and_executed_close'] for r in closing])),
                queue_delays_observed_actions=sorted({r['queue_delay_actions'] for r in closing}),
                closed_episodes_with_jaw_reduction_at_least_5mm=sum(r['maximum_jaw_gap_reduction_after_close_m']>=.005 for r in closing),
                no_close_episodes=len(opening),median_minimum_native_g_if_no_close=float(np.median([r['minimum_native_g'] for r in opening])),
                minimum_native_g_if_no_close=float(min(r['minimum_native_g'] for r in opening)),maximum_minimum_native_g_if_no_close=float(max(r['minimum_native_g'] for r in opening))))
    table('timing_summary.csv',timing)
    matches=[]
    qmap={(t,r,i):[q for q in queries if (q['task'],q['robot'],q['condition'])==(t,r,i) and q['preinteraction_risk_set']] for t in TASKS for r in ['ur5e','panda'] for i in range(60,100)}
    for task in TASKS:
        for i in range(60,100):
            src=all_data[task,'ur5e',i];trg=all_data[task,'panda',i];s=src['first_query']
            assert src['instruction']==trg['instruction'],'Source/target instruction mismatch'
            if s is None:continue
            source_eligible=any(q['query_action_index']==s for q in qmap[task,'ur5e',i])
            candidates=qmap[task,'panda',i]
            row=dict(task=task,condition=i,source_query_action=s,source_risk_eligible=source_eligible,source_query_distance_m=float(src['distance'][s]),source_predicts_close=True,target_query_action=None,
                position_difference_m=None,orientation_difference_degrees=None,spatial_support=False,history_calipers_pass=False,target_predicts_close=None,time_difference_actions=None,path_difference_m=None,
                radial_speed_difference_m_s=None,jaw_gap_difference_m=None,near_dwell_difference_actions=None,spatial_candidate_count=0,history_candidate_count=0,reason='NO_TARGET_PREINTERACTION_QUERIES')
            if candidates:
                ts=np.array([c['query_action_index'] for c in candidates])
                dp=np.linalg.norm(trg['relative'][ts]-src['relative'][s],axis=1)
                dr=np.rad2deg(Rotation.from_matrix(trg['rotation'][ts]@src['rotation'][s].T).magnitude())
                # Prefer a spatially admissible candidate; then nearest position, orientation as tie cost.
                spatial=(dp<=.02)&(dr<=20)
                times=np.abs(ts-s);paths=np.abs(trg['path'][ts]-src['path'][s]);speeds=np.abs(trg['speed'][ts]-src['speed'][s]);jaws=np.abs(trg['jaw'][ts]-src['jaw'][s]);dwells=np.abs(trg['dwell'][ts]-src['dwell'][s])
                history_mask=source_eligible & spatial & (times<=40)&(paths<=.10)&(speeds<=.04)&(jaws<=.01)&(dwells<=8)
                score=dp/.02+dr/20;score+=np.where(spatial,0,1e6)+np.where(history_mask,0,1e3)
                k=int(np.argmin(score));t=int(ts[k]);q=candidates[k]
                time=abs(t-s);path=abs(trg['path'][t]-src['path'][s]);speed=abs(trg['speed'][t]-src['speed'][s]);jaw=abs(trg['jaw'][t]-src['jaw'][s]);dwell=abs(int(trg['dwell'][t])-int(src['dwell'][s]))
                history=bool(source_eligible and spatial[k] and time<=40 and path<=.10 and speed<=.04 and jaw<=.01 and dwell<=8)
                row.update(target_query_action=t,position_difference_m=float(dp[k]),orientation_difference_degrees=float(dr[k]),spatial_support=bool(source_eligible and spatial[k]),history_calipers_pass=history,
                    target_predicts_close=q['predicts_close_in_executed_prefix'],time_difference_actions=int(time),path_difference_m=float(path),radial_speed_difference_m_s=float(speed),jaw_gap_difference_m=float(jaw),near_dwell_difference_actions=int(dwell),
                    spatial_candidate_count=int(spatial.sum()) if source_eligible else 0,history_candidate_count=int(history_mask.sum()),
                    reason='DESCRIPTIVE_HISTORY_CALIPERS_PASS_NOT_CAUSAL_PAIR' if history else 'SPATIAL_ONLY_HISTORY_MISMATCH' if source_eligible and spatial[k] else 'NO_SPATIAL_COMMON_SUPPORT_OR_SOURCE_NOT_AT_RISK')
            matches.append(row)
    table('source_initiation_query_matches.csv',matches)
    for p,h in INPUTS.items():assert digest(p)==h,'Input changed: '+p
    assert len({r['checkpoint_sha256'] for r in queue_rows})==1 and len({r['policy_source_sha256'] for r in queue_rows})==1
    write('runtime_checks.json',dict(checkpoint_path=str(checkpoint),checkpoint_sha256=INPUTS[str(checkpoint)],checkpoint_rehashed_this_audit=True,training_config_safe_subset=safe_config,
        runtime_source_sha256=[INPUTS[p] for p in sorted(runtime_inputs)],same_condition_source_target_instructions_identical=True,
        full_training_config_deliberately_not_copied=True,native_values_are_postdecoded_unnormalized_not_logits=True,token_ids_and_network_input_lossless_pixels_not_saved=True,
        interface_verification_scope='Exact recorded postnetwork query -> FIFO -> canonical gripper mapping -> clipping and padding -> recorded environment commands. Physical arm response, true grasp feasibility and unrecorded preprocessing internals are not certified.'))
    write('summary.json',dict(groups=groups,episodes=160,total_executed_actions=sum(r['steps'] for r in episodes),total_saved_network_queries=sum(r['queries'] for r in queue_rows),
        native_g_exactly_half_actions=sum(r['native_g_exactly_half_count'] for r in episodes),native_g_outside_unit_interval_actions=sum(r['native_g_outside_unit_interval_count'] for r in episodes),
        arm_actions_clipped=sum(r['arm_actions_clipped_count'] for r in episodes),unexecuted_tail_close_entries=sum(r['unexecuted_tail_close_actions'] for r in episodes),episodes_with_close_only_in_unexecuted_tail=0,
        interface_mismatched_actions=0,queue_mismatched_actions=0,wrong_instructions=0,new_policy_inference_calls=0,new_physics_steps=0,new_training_updates=0,
        matching_by_task=[dict(task=t,source_initiation_queries=sum(r['task']==t for r in matches),source_risk_eligible=sum(r['task']==t and r['source_risk_eligible'] for r in matches),
            spatial_support=sum(r['task']==t and r['spatial_support'] for r in matches),history_support=sum(r['task']==t and r['history_calipers_pass'] for r in matches),
            spatial_target_not_closing=sum(r['task']==t and r['spatial_support'] and not r['target_predicts_close'] for r in matches),
            history_target_not_closing=sum(r['task']==t and r['history_calipers_pass'] and not r['target_predicts_close'] for r in matches)) for t in TASKS],
        metrics_are_descriptive_not_causal=True))
    write('input_manifest.json',dict(base_commit='b84bb7e9ad12f582e87d74bc267ef50da14e3c08',inputs=[dict(absolute_path=p,bytes=Path(p).stat().st_size,sha256=h) for p,h in sorted(INPUTS.items())],all_input_hashes_unchanged=True,simulator_and_policy_modules_loaded=False))
    print(json.dumps(dict(groups=groups,queries=len(queries),matching=read_output('summary.json')['matching_by_task']),indent=2))


def read_output(name):return json.loads((OUT/name).read_text())


if __name__=='__main__':main()
