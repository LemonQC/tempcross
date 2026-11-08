"""Analyze official physical labels and read-only telemetry, never relabel success.

Distance/lift thresholds describe stages only. They are not grasp validity,
reachability, collision causes, visibility scores or new task-success criteria.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;d=1+z*z/n;c=(k/n+z*z/(2*n))/d
    r=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/d
    return [max(0.,c-r),min(1.,c+r)]


def episode_metrics(directory,record):
    import h5py
    i=record['episode'];path=directory/'telemetry'/f'episode-{i:03d}.h5'
    row=dict(record,case_id='/'.join(directory.parts[-2:])+f'/{i:03d}',
             reported_seed=record['seed'],success_once=record['success'],
             telemetry_path=str(path),failure_category=None if record['success'] else 'UNKNOWN',
             classification_status='SUCCESS' if record['success'] else 'VIDEO_REVIEW_REQUIRED')
    videos=list((directory/'official').rglob(f'*--episode={i+1}--success=*--task=xpnpcountertosink.mp4'))
    row['video_path']=str(videos[0]) if len(videos)==1 else None
    with h5py.File(path) as f:
        condition=json.loads(f.attrs['condition']);row['actual_seed']=condition.get('seed')
        row['condition_id_observed']=condition.get('condition_id')
        assert row['condition_id_observed']==record['condition_id']
        obj=f['target/position'][:];eef=f['obs/robot0_eef_pos'][:]
        distance=np.linalg.norm(eef-obj,axis=1);rise=obj[:,2]-obj[0,2]
        row.update(min_eef_object_center_m=float(distance.min()),closest_step=int(distance.argmin()),
                   final_eef_object_center_distance_m=float(distance[-1]),
                   max_object_rise_m=float(rise.max()),max_rise_step=int(rise.argmax()),
                   final_object_displacement_m=float(np.linalg.norm(obj[-1]-obj[0])),
                   final_object_position=obj[-1].tolist(),initial_object_position=obj[0].tolist(),
                   eef_within_10cm=bool(np.any(distance<=.1)),object_rose_5cm=bool(np.any(rise>=.05)))
        actions=f['actions/environment'][:];row['environment_action_count']=len(actions)
        row['nonfinite_action_values']=int((~np.isfinite(actions)).sum())
        row['padded_action_error_max']=float(np.abs(actions[:,7:11]).max())
        row['mode_error_max']=float(np.abs(actions[:,11]+1).max())
        chunks=f['actions/policy_chunks'][:];starts=f['actions/policy_chunk_step'][:]
        errors=[]
        for chunk,start in zip(chunks,starts):
            start=int(start)-1;length=min(len(chunk),len(actions)-start)
            expected=chunk[:length].copy();expected[:,6]=-np.sign(2*expected[:,6]-1)
            errors.append(float(np.abs(actions[start:start+length,:7]-expected).max()))
        row['policy_to_environment_mapping_max_error']=max(errors) if errors else None
        row['last30_mean_arm_action']=actions[-30:,:6].mean(axis=0).tolist()
        row['last30_eef_delta_m']=(eef[-1]-eef[max(0,len(eef)-31)]).tolist()
        row['gripper_close_command_fraction']=float((actions[:,6]>0).mean())
        if 'model/jnt_range' in f:
            limited=f['model/jnt_limited'][:].astype(bool);ranges=f['model/jnt_range'][:][limited]
            addresses=f['model/jnt_qposadr'][:][limited];q=f['sim/qpos'][:,addresses]
            margins=np.minimum(q-ranges[:,0],ranges[:,1]-q)
            # All limited simulator joints, including fingers/fixtures; report names
            # rather than mislabeling the minimum as an arm reachability failure.
            idx=np.unravel_index(margins.argmin(),margins.shape)
            names=json.loads(f.attrs['joint_names']);limited_names=[name for name,yes in zip(names,limited) if yes]
            row['minimum_limited_joint_margin']=float(margins[idx])
            row['minimum_margin_joint']=limited_names[idx[1]]
            row['minimum_margin_step']=int(idx[0])
            arm=[j for j,name in enumerate(limited_names) if name and name.startswith('robot0_')]
            if arm:
                arm_margins=margins[:,arm];ai=np.unravel_index(arm_margins.argmin(),arm_margins.shape)
                row['minimum_arm_joint_margin_rad']=float(arm_margins[ai])
                row['minimum_arm_margin_joint']=limited_names[arm[ai[1]]]
                row['minimum_arm_margin_step']=int(ai[0])
        camera=f['camera/T_world_optical'][:]
        row['initial_camera_T_world_optical']=camera[0].tolist()
    telemetry=[json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
    grasps=[r['step'] for r in telemetry if r['grasp_contact'] is True]
    row['grasp_contact_steps']=len(grasps);row['first_grasp_step']=min(grasps) if grasps else None
    row['last_grasp_step']=max(grasps) if grasps else None
    row['contact_geometry_pairs']=sorted({tuple(sorted((c['geom1'] or '',c['geom2'] or ''))) for r in telemetry for c in r['contacts']})
    row['telemetry_errors']=sorted({e for r in telemetry for e in r['telemetry_errors']})
    row['posthoc_center_in_frustum_fraction']=float(np.mean([
        r['coft_static_projection']['in_frustum'] for r in telemetry
        if r.get('coft_static_projection') is not None
    ])) if any(r.get('coft_static_projection') is not None for r in telemetry) else None
    return row


def analyze(root,out,reviews=None):
    root=Path(root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review_map={r['case_id']:r for r in json.loads(Path(reviews).read_text())} if reviews else {}
    episodes=[];cells=[]
    for method in ('no_reps','joint_reps'):
        for arm in ('ur5e','panda'):
            directory=root/'p1'/method/arm;paths=list((directory/'official').rglob('summary.json'))
            records=[];summary=None
            if len(paths)==1:
                summary=json.loads(paths[0].read_text())
            episode_paths=list((directory/'official').rglob('episodes.jsonl'))
            if len(episode_paths)==1:
                text=episode_paths[0].read_text()
                records=[json.loads(x) for x in text.splitlines() if x] if text.endswith('\n') else [json.loads(x) for x in text.splitlines()[:-1] if x]
            complete=bool(summary and summary['status']=='complete' and len(records)==20 and summary['episodes']==20)
            if any(type(r.get('success')) is not bool for r in records):
                raise ValueError('Official success labels must be JSON booleans')
            k=sum(r['success'] for r in records);n=len(records)
            if complete and (summary['successes']!=k or not math.isclose(summary['success_rate'],k/n)):
                raise ValueError('Official summary and episode labels disagree')
            cells.append(dict(method=method,robot=arm,expected=20,completed=n,status='COMPLETE' if complete else 'INCOMPLETE',successes=k if n else None,
                              success_rate=k/n if complete else None,wilson95=wilson(k,n) if complete else None))
            # Active telemetry files remain owned by the evaluator. Analyze them
            # only after the cell completes, retaining partial episode counts above.
            if not complete:continue
            for record in records:
                row=episode_metrics(directory,record)
                if row['case_id'] in review_map:
                    review=review_map[row['case_id']]
                    assert review['video_path']==row['video_path']
                    assert review['evidence'] and review['reviewed_frame_steps']
                    row['video_review']=review
                    if not row['success']:
                        row['failure_category']=review['category'];row['classification_status']='VIDEO_AND_STATE_REVIEWED'
                episodes.append(row)
    comparisons=[]
    if all(c['status']=='COMPLETE' for c in cells):
        from scipy.stats import fisher_exact
        for method in ('no_reps','joint_reps'):
            source=next(c for c in cells if c['method']==method and c['robot']=='ur5e')
            target=next(c for c in cells if c['method']==method and c['robot']=='panda')
            table=[[source['successes'],20-source['successes']],[target['successes'],20-target['successes']]]
            odds,pvalue=fisher_exact(table,alternative='two-sided')
            comparisons.append(dict(method=method,source_minus_target=source['success_rate']-target['success_rate'],
                                    fisher_two_sided_p=float(pvalue),bonferroni_two_contrasts_p=min(1.,float(pvalue)*2),
                                    scope='descriptive non-strictly-paired comparison; two model-specific source-target contrasts; not a powered confirmatory experiment'))
    payload=dict(physical_success_source='unmodified official info[is_success][task]',
                 success_once_equals_success='official evaluator terminates at first success',
                 strict_cross_robot_pairing_verified=False,stage_metrics_are_diagnostics_only=True,cells=cells,comparisons=comparisons,episodes=episodes)
    (out/'analysis.json').write_text(json.dumps(payload,indent=2))
    with (out/'cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
    compact=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in episodes]
    if compact:
        keys=list(dict.fromkeys(k for row in compact for k in row))
        with (out/'episodes.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(compact)
    print(json.dumps(cells,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--reviews');a=p.parse_args()
    analyze(a.root,a.out,a.reviews)
