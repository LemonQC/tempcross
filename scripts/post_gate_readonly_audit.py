"""Read existing JSON/CSV/H5 only. No simulator, policy, training, or controller calls.

Inputs: committed baseline/mechanism results and their indexed server telemetry.
Outputs: evidence/post_gate/*.json,csv. No experiment file is written.
Reproduce: /home/users/lemon/coft-p2/barx/.venv/bin/python scripts/post_gate_readonly_audit.py
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import h5py
import numpy as np

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/home/users/lemon/mechanism-20261110')
OUT=REPO/'evidence/post_gate'
CASES=[64,68,71,72,76,88,93,94]
INPUTS={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def track(path):
    path=Path(path);INPUTS[str(path)]=sha(path);return path


def read_json(path):return json.loads(track(path).read_text())


def output(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def table(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def longest(mask):
    best=run=0
    for b in mask:
        run=run+1 if b else 0;best=max(best,run)
    return best


def two_sided(records,target):
    values=[]
    for record in records:
        sides=set()
        for c in record['contacts']:
            if c['distance']>0:continue
            for finger,other in [(c['geom1'],c['geom2']),(c['geom2'],c['geom1'])]:
                if other not in target:continue
                for side in ['left','right']:
                    if finger in {f'gripper0_right_{side}_fingerpad_collision',f'gripper0_right_{side}_fingertip_collision'}:sides.add(side)
        values.append(len(sides)==2)
    return np.asarray(values,dtype=bool)


def raw_baseline(path):
    path=track(path);records=[json.loads(x) for x in track(path.with_suffix('.jsonl')).read_text().splitlines()]
    with h5py.File(path,'r') as f:
        names=json.loads(f.attrs['body_names']);geoms=json.loads(f.attrs['geom_names'])
        target={g for g,b in zip(geoms,f['model/geom_bodyid'][:]) if names[b].startswith('obj_')}
        c=two_sided(records,target);obj=f['target/position'][:];rise=obj[:,2]-obj[0,2]
        return dict(bilateral_states=int(c.sum()),longest_bilateral_run=longest(c),stable_lift=longest(c&(rise>=.03))>=10,
                    legacy_pad_contact_states=sum(r['grasp_contact'] is True for r in records),max_body_origin_rise_m=float(rise.max()))


def rotation(q):
    # MuJoCo body quaternion order wxyz; only numeric conversion, no MuJoCo import.
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))


def main():
    guard_files=[ROOT/'physical_launches.jsonl',ROOT/'completion.json',REPO/'configs/mechanism_protocol.json',REPO/'results/mechanism/decision.json',REPO/'results/baselines/status.json']
    for p in guard_files:track(p)
    for p in sorted((REPO/'mechanism_validation').glob('*.py')):track(p)
    for name in ['BASELINE_SCIENTIFIC_ASSESSMENT.md','P3_SPATIOTEMPORAL_CAUSAL_REPORT.md','MECHANISM_PROTOCOL.md','MECHANISM_RESULTS.md','MECHANISM_PRIOR_ART.md','P2_INTERVENTION_REPORT.md','P3_STRONGER_ADAPTATION.md']:
        track(REPO/'docs'/name)
    track(REPO/'results/mechanism/episodes.csv');track(REPO/'results/mechanism/recorded_tracking_diagnostics.csv')
    launches=read_json(ROOT/'completion.json')['physical_episodes'];assert launches==24
    b=read_json(REPO/'results/baselines/analysis.json')
    rows=[r for r in b['episodes'] if r['label']=='A_barx_joint' and r['task']=='pnp_counter_to_sink']
    source={r['official_condition_index']:r for r in rows if r['robot']=='ur5e'}
    target={r['official_condition_index']:r for r in rows if r['robot']=='panda'}
    assert source.keys()==target.keys() and len(source)==40
    for r in target.values():
        with h5py.File(track(r['telemetry_path']),'r') as f:
            assert bool(np.any(f['actions/environment'][:,6]>0))==r['issued_close']
    pairing={name:[i for i in source if source[i]['success']==s and target[i]['success']==t] for name,s,t in [('both_success',True,True),('source_only_success',True,False),('target_only_success',False,True),('both_failure',False,False)]}
    original=dict(source_success=sum(r['success'] for r in source.values()),target_success=sum(r['success'] for r in target.values()),episodes_per_robot=40,
        paired_indices=pairing,source_success_target_fail_observed_stages=dict(Counter(target[i]['observed_failure_stage'] for i in pairing['source_only_success'])),
        all_target_failure_observed_stages=dict(Counter(r['observed_failure_stage'] for r in target.values() if not r['success'])),
        source_success_target_fail_never_closed=[i for i in pairing['source_only_success'] if not target[i]['issued_close']],
        source_success_target_fail_placed_without_retreat=[i for i in pairing['source_only_success'] if target[i]['observed_failure_stage']=='PLACEMENT_WITHOUT_RETREAT'],
        all40_target_gripper_commands_verified_from_H5=True,
        stage_warning='Historical pad-only proxies are descriptive; case94 source exposes a fingertip false negative. Not a visual/kinematic/decoder causal classifier.')
    output('original_joint_pairing.json',original)
    scenes=[]
    for i in source:
        with h5py.File(track(source[i]['telemetry_path']),'r') as f,h5py.File(track(target[i]['telemetry_path']),'r') as g:
            ma=json.loads(f.attrs['ep_meta']);mb=json.loads(g.attrs['ep_meta'])
            assets=lambda meta:{o['name']:o['info']['mjcf_path'] for o in meta['object_cfgs']}
            na=json.loads(f.attrs['body_names']);nb=json.loads(g.attrs['body_names'])
            sa=na.index(ma['fixture_refs']['sink']+'_main');sb=nb.index(mb['fixture_refs']['sink']+'_main')
            cam_error=float(np.abs(f['camera/T_world_optical'][0]-g['camera/T_world_optical'][0]).max())
            qa=f['target/quaternion'][0];qb=g['target/quaternion'][0]
            angle=2*np.arccos(np.clip(abs(float(np.dot(qa,qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)))),-1,1))
            scenes.append(dict(condition=i,all_object_assets_equal=assets(ma)==assets(mb),instructions_equal=ma['lang']==mb['lang'],
                initial_target_object_position_difference_m=float(np.linalg.norm(f['target/position'][0]-g['target/position'][0])),initial_target_object_angle_difference_rad=float(angle),
                sink_position_difference_m=float(np.linalg.norm(f['sim/body_xpos'][0,sa]-g['sim/body_xpos'][0,sb])),
                sink_quaternion_maximum_difference=float(np.abs(f['sim/body_xquat'][0,sa]-g['sim/body_xquat'][0,sb]).max()),initial_camera_transform_maximum_difference=cam_error))
    table('joint40_scene_comparison.csv',scenes)
    new=read_json(REPO/'results/mechanism/analysis.json');methods={(r['condition'],r['method']):r for r in new['episodes']}
    bank={r['official_index']:r for r in read_json(REPO/'results/mechanism/source_contact_bank.json')}
    source_target=[];budgets=[];timeouts=[];attachments=[];contact94=[];command_ranges=[]
    for i in CASES:
        a=raw_baseline(source[i]['telemetry_path']);b=raw_baseline(target[i]['telemetry_path'])
        source_target.append(dict(condition=i,source_official_success=source[i]['success'],original_target_official_success=target[i]['success'],
            source_bilateral_states=a['bilateral_states'],source_legacy_pad_states=a['legacy_pad_contact_states'],source_stable_lift=a['stable_lift'],
            original_target_bilateral_states=b['bilateral_states'],original_target_stable_lift=b['stable_lift'],
            corrected_prefix_stable=methods[i,'M0']['handoff_confirmed'],M1_final_official_success=methods[i,'M1']['task_success']))
        histories={m:read_json(methods[i,m]['history_path']) for m in ['M0','M1','M2']}
        handoff=methods[i,'M0']['handoff_action_index']
        for m in ['M0','M1','M2']:
            result=methods[i,m];hh=histories[m]
            with h5py.File(track(result['telemetry_path']),'r') as f:
                n=result['steps'];assert len(f['actions/environment'])==n and len(f['target/position'])==n+1
                if handoff is not None:
                    with np.load(track(Path(result['telemetry_path']).parents[1]/'handoff.npz')) as snapshot:
                        scale=snapshot['controller::right/action_scale'];offset=snapshot['controller::right/action_output_transform'];center=snapshot['controller::right/action_input_transform']
                    common_end=min(methods[i,k]['steps'] for k in ['M0','M1','M2'])
                    delta=(f['actions/environment'][handoff:common_end,:6]-center)*scale+offset
                    norms=np.linalg.norm(delta[:,:3],axis=1)
                    command_ranges.append(dict(condition=i,method=m,common_post_action_count=len(norms),median_command_translation_delta_m=float(np.median(norms)),maximum_command_translation_delta_m=float(norms.max()),commands_above_12mm=int((norms>.012+1e-10).sum()),
                        meaning='Commanded OSC delta norm, NOT achieved motion, speed, or physical action limit'))
                if handoff is not None and m!='M1':
                    actions=f['actions/environment'][:];last=next((t for t in range(handoff,n) if actions[t,6]<0),n)
                    indices=[t for t in range(handoff,min(last,n)+1) if t==handoff or (t>0 and hh[t-1]['bilateral_contact'])]
                    pos=f['target/position'][:];rot=rotation(f['target/quaternion'][:]);tcp=f['mechanism/pre_action_tcp_world'][:]
                    if last==n:indices=[t for t in indices if t<n]
                    attach_pos=np.einsum('nji,nj->ni',rot[indices],tcp[indices,:3,3]-pos[indices])
                    attach_rot=np.swapaxes(rot[indices],1,2)@tcp[indices,:3,:3]
                    drift=np.linalg.norm(attach_pos-attach_pos[0],axis=1)
                    angle=np.arccos(np.clip((np.trace(attach_rot@attach_rot[0].T,axis1=1,axis2=2)-1)/2,-1,1))
                    attachments.append(dict(condition=i,method=m,contact_states_in_preopen_window=len(indices),maximum_attachment_position_change_m=float(drift.max()),maximum_attachment_angle_change_degrees=float(np.rad2deg(angle.max()))))
            if i==94:
                for phase in dict.fromkeys(h['phase'] for h in hh):
                    ph=[h for h in hh if h['phase']==phase]
                    contact94.append(dict(condition=i,method=m,phase=phase,steps=len(ph),bilateral_states=sum(h['bilateral_contact'] for h in ph),
                        maximum_body_origin_rise_m=max(h['rise_m'] for h in ph),terminal_TCP_goal_error_m=ph[-1]['tcp_position_error_m']))
            if m!='M1' and 'TIMEOUT' in result['failure_stage'] and hh[-1]['phase']!='lift':
                phase=hh[-1]['phase'];ph=[h for h in hh if h['phase']==phase]
                timeouts.append(dict(condition=i,method=m,phase=phase,executed_phase_steps=len(ph),terminal_goal_position_error_m=ph[-1]['tcp_position_error_m'],
                    final20_position_error_improvement_m=ph[max(0,len(ph)-20)]['tcp_position_error_m']-ph[-1]['tcp_position_error_m'],terminal_unwanted_contacts=ph[-1]['unwanted_contacts'],terminal_joint_margin_rad=ph[-1]['minimum_joint_margin']))
        if handoff is not None:
            ordinary_end=min(methods[i,'M0']['steps'],methods[i,'M2']['steps'])
            common_end=min(ordinary_end,methods[i,'M1']['steps'])
            budgets.append(dict(condition=i,handoff_action_index=handoff,M0_end_action_count=methods[i,'M0']['steps'],M2_end_action_count=methods[i,'M2']['steps'],
                ordinary_post_actions_at_first_end=ordinary_end-handoff,M1_allowed_post_actions=600-handoff,M1_executed_post_actions=methods[i,'M1']['steps']-handoff,
                M1_final_official_success=methods[i,'M1']['task_success'],M1_actions_beyond_first_ordinary_end=max(0,methods[i,'M1']['steps']-ordinary_end),
                common_observed_end_action_count=common_end,M0_success_by_common_end=any(h['official_task_success'] for h in histories['M0'][:common_end]),
                M1_success_by_common_end=any(h['official_task_success'] for h in histories['M1'][:common_end]),M2_success_by_common_end=any(h['official_task_success'] for h in histories['M2'][:common_end]),
                M1_placement_at_common_end=histories['M1'][common_end-1]['placement'],M1_bilateral_at_common_end=histories['M1'][common_end-1]['bilateral_contact']))
    table('source_target_and_intervention.csv',source_target);table('observed_budget_windows.csv',budgets)
    table('ordinary_timeout_progress.csv',timeouts);table('attachment_changes.csv',attachments);table('condition094_contact.csv',contact94)
    table('common_window_command_ranges.csv',command_ranges)
    output('metric_semantics.json',dict(M1_transport_reached='Hard-coded False outside scripted controllers; NOT an observation of failed transport',
        M1_tcp_position_error_m='Post-action TCP vs pre-action TCP placeholder: single-step displacement, NOT task goal error',
        M1_tcp_rotation_error_rad='Single-step rotation vs pre-action TCP, NOT task goal error',
        valid_cross_method_outcomes=['unchanged official task success','raw body-origin position/quaternion','physical bilateral contacts','same transport_10cm_in_contact diagnostic'],
        contact_label='Raw geometry contact is not contact force, force closure, friction margin or learned RGB state estimation',
        fixed_vs_feedback='M0 and M2 share object goals and gains but use different attachment updates AND arrival errors (fixed TCP vs actual object); closed-loop stage durations can differ',
        no_new_step=True))
    assert 'mujoco' not in sys.modules and 'robosuite' not in sys.modules
    for path,expected in INPUTS.items():assert sha(path)==expected,'Input changed: '+path
    output('readonly_manifest.json',dict(review_base_commit='576f4caee172c489f4507e87d55f097bf740833e',directive_commit='a0e8dfb0a62d66cacdbb6ddabc57209590a66302',
        input_files=[dict(absolute_path=p,sha256=s,bytes=Path(p).stat().st_size) for p,s in sorted(INPUTS.items())],input_hashes_unchanged=True,
        simulator_modules_imported=False,new_physics_steps=0,new_training_updates=0,controller_changes=0,physical_launches_before=launches,physical_launches_after=len((ROOT/'physical_launches.jsonl').read_text().splitlines()),
        old_physical_gate=read_json(REPO/'results/mechanism/decision.json')['algorithm_development'],counterfactual_continuation_inferred=False))
    print(json.dumps(dict(original=original,matched_handoffs=len(budgets),source_stable=sum(r['source_stable_lift'] for r in source_target),original_target_stable=sum(r['original_target_stable_lift'] for r in source_target),
        new_physics_steps=0,guarded_inputs=len(INPUTS)),indent=2))


if __name__=='__main__':main()
