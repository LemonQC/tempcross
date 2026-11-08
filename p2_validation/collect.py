"""Collect real completed P2 policy runs; never turn system errors into failure."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(stream,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def collect(root,out):
    import h5py
    from simulation_validation.analyze_rollouts import wilson
    from p2_validation.offline_audit import longest_run, fingerpad_specs, world_pad, center_prism
    root=Path(root).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    cells=[];episodes=[];video_index=[];model_xmls={}
    for job_file in sorted(root.rglob('job.json')):
        job=json.loads(job_file.read_text());directory=job_file.parent
        stage='development' if 'development' in directory.parts else 'formal'
        if job['status']!='COMPLETE':
            partial=[]
            for record_path in (directory/'official').rglob('episodes.jsonl'):
                text=record_path.read_text();lines=text.splitlines()
                if not text.endswith('\n'):lines=lines[:-1]
                partial.extend(json.loads(line) for line in lines if line)
            assert all(type(r['success']) is bool for r in partial)
            cells.append(dict(cell=directory.name,stage=stage,status=job['status'],episodes=len(partial),
                              successes=sum(r['success'] for r in partial) if partial else None,success_rate=None));continue
        official=list((directory/'official').rglob('episodes.jsonl'));assert len(official)==1
        records=[json.loads(x) for x in official[0].read_text().splitlines()];assert len(records)==len(job['indices'])
        k=sum(r['success'] for r in records);assert k==job['summary']['successes']
        cells.append(dict(cell=directory.name,stage=stage,status='COMPLETE',episodes=len(records),successes=k,
                          success_rate=k/len(records),wilson95=wilson(k,len(records)),
                          checkpoint=job['checkpoint'],checkpoint_sha256=job['checkpoint_sha256'],
                          peak_gpu_allocated_bytes=job['peak_gpu_allocated_bytes']))
        destination=out/'jobs'/stage/directory.name;destination.mkdir(parents=True,exist_ok=True)
        for name in ['episodes.jsonl','summary.json','config.json']:shutil.copyfile(official[0].parent/name,destination/name)
        shutil.copyfile(job_file,destination/'job.json')
        for record,condition in zip(records,job['conditions']):
            assert type(record['success']) is bool and record['condition_id']==condition['condition_id']
            index=record['episode'];path=directory/'telemetry'/f'episode-{index:03d}.h5'
            with h5py.File(path,'r') as f:
                obj=f['target/position'][:];eef=f['obs/robot0_eef_pos'][:];actions=f['actions/environment'][:]
                raw=f['p2/raw_gripper'][:];chunks=f['actions/policy_chunks'][:];starts=f['actions/policy_chunk_step'][:]
                predicted=np.concatenate([chunk[:min(8,len(actions)-(int(start)-1)),:6] for chunk,start in zip(chunks,starts)])
                assert predicted.shape==actions[:,:6].shape and np.max(np.abs(predicted-actions[:,:6]))==0
                widths=f['p2/pre_action_pad_center_distance'][:];close=actions[:,6]>0
                arm=job['robot']
                if arm not in model_xmls:
                    with np.load('/home/users/lemon/coft-p2/barx/evaluation/conditions/pnp_counter_to_sink/'+arm+'.npz',allow_pickle=False) as archive:
                        model_xmls[arm]=archive['model_xmls'].tolist()
                specs=fingerpad_specs(model_xmls[arm][condition['episode']]);names=json.loads(f.attrs['body_names'])
                pads={side:world_pad(f['sim/body_xpos'][:,names.index(spec['body'])],
                                     f['sim/body_xquat'][:,names.index(spec['body'])],spec) for side,spec in specs.items()}
                _,inner_gap,_,_=center_prism(pads['left'],pads['right'],obj,specs)
                rise=obj[:,2]-obj[0,2];distance=np.linalg.norm(eef-obj,axis=1)
                nonfinite=int((~np.isfinite(actions)).sum());assert nonfinite==0
            telemetry=[json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
            contact=np.array([r['grasp_contact'] is True for r in telemetry]);proxy=longest_run(contact & (rise>=.03))>=10
            first_contact=int(np.flatnonzero(contact)[0]) if contact.any() else None
            releases=np.flatnonzero((actions[:-1,6]>0)&(actions[1:,6]<0))+1
            releases=[int(t) for t in releases if first_contact is not None and t>=first_contact]
            stage_label=('OFFICIAL_SUCCESS' if record['success'] else 'POST_ACQUISITION_TASK_FAILURE' if proxy else
                         'CONTACT_WITHOUT_SUSTAINED_LIFT' if contact.any() else 'NO_BILATERAL_CONTACT' if close.any() else 'NO_CLOSE_COMMAND')
            videos=list(official[0].parent.glob(f'*--episode={index+1}--success=*--task=xpnpcountertosink.mp4'));assert len(videos)==1
            row=dict(cell=directory.name,stage=stage,trial_index=index,official_condition_index=condition['episode'],
                condition_id=condition['condition_id'],actual_seed=condition['seed'],success=record['success'],steps=record['steps'],
                raw_gripper_min=float(raw.min()),raw_gripper_median=float(np.median(raw)),raw_gripper_max=float(raw.max()),
                close_command_steps=int(close.sum()),first_close_action_index=int(np.flatnonzero(close)[0]) if close.any() else None,
                pad_center_distance_min_m=float(widths.min()),pad_center_distance_max_m=float(widths.max()),
                inner_pad_gap_min_m=float(inner_gap.min()),inner_pad_gap_max_m=float(inner_gap.max()),
                bilateral_contact_state_steps=int(contact.sum()),sustained_contact_lift_proxy=proxy,
                max_object_rise_m=float(rise.max()),horizontal_object_displacement_m=float(np.linalg.norm(obj[-1,:2]-obj[0,:2])),
                release_after_contact_action_indices=releases,final_eef_object_distance_m=float(distance[-1]),
                final_10cm_retreat_condition=bool(distance[-1]>.1),object_inside_sink_subpredicate='NOT_SEPARATELY_RECORDED',
                observed_stage=stage_label,causal_failure_category=None if record['success'] else 'UNKNOWN',
                arm_policy_mapping_max_error=0.,nonfinite_actions=nonfinite,
                telemetry_path=str(path),video_path=str(videos[0]))
            episodes.append(row);video_index.append(dict(cell=directory.name,condition_id=condition['condition_id'],video_path=str(videos[0]),success=record['success']))
    comparisons=[];paired_stages=[]
    targets={c['cell']:c for c in cells if c['status']=='COMPLETE' and c['stage']=='formal'}
    for candidate in ['b1-joint_reps-panda','adapted-no_reps-panda','adapted-joint_reps-panda']:
        baseline='original-no_reps-panda' if candidate=='adapted-no_reps-panda' else 'original-joint_reps-panda'
        if candidate not in targets or baseline not in targets:continue
        first={r['condition_id']:r for r in episodes if r['cell']==baseline and r['stage']=='formal'}
        second={r['condition_id']:r for r in episodes if r['cell']==candidate and r['stage']=='formal'}
        assert first.keys()==second.keys()
        rescue=[key for key in first if not first[key]['success'] and second[key]['success']]
        regression=[key for key in first if first[key]['success'] and not second[key]['success']]
        comparisons.append(dict(method=candidate,baseline=baseline,episodes=len(first),rescued=len(rescue),regressed=len(regression),
                                net_success_change=len(rescue)-len(regression),rescue_condition_ids=rescue,regression_condition_ids=regression))
        for key in first:
            before,after=first[key],second[key]
            paired_stages.append(dict(method=candidate,baseline=baseline,condition_id=key,
                 official_condition_index=before['official_condition_index'],baseline_success=before['success'],candidate_success=after['success'],
                 baseline_stage=before['observed_stage'],candidate_stage=after['observed_stage'],
                 baseline_contact_steps=before['bilateral_contact_state_steps'],candidate_contact_steps=after['bilateral_contact_state_steps'],
                 baseline_lift_proxy=before['sustained_contact_lift_proxy'],candidate_lift_proxy=after['sustained_contact_lift_proxy']))
    payload=dict(cells=cells,comparisons=comparisons,episodes=episodes,
                 causal_failure_labels_require_further_evidence=True,stage_proxies_are_not_official_success=True)
    (out/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'video_index.json').write_text(json.dumps(video_index,indent=2)+'\n')
    write_csv(out/'cells.csv',cells);write_csv(out/'episodes.csv',episodes);write_csv(out/'paired_comparisons.csv',comparisons)
    write_csv(out/'formal_failures.csv',[row for row in episodes if row['stage']=='formal' and not row['success']])
    write_csv(out/'paired_stage_changes.csv',paired_stages)
    for path in [root/'frozen_protocol.json',root/'script-freeze.json',root/'adaptation_suite.json']:
        if path.exists():shutil.copyfile(path,out/path.name)
    # v3 repeats the same seven probes with full per-step telemetry. Retain v2
    # independently: repeated instrumentation runs are not new independent cases.
    for version in ['v2','v3']:
        path=root/f'b2-diagnostics-{version}'/'results.json'
        if path.exists():shutil.copyfile(path,out/f'b2_development_{version}.json')
    primary=root/'b2-diagnostics-v3/results.json'
    if primary.exists():shutil.copyfile(primary,out/'b2_development.json')
    print(json.dumps(dict(cells=cells,comparisons=comparisons),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();collect(a.root,a.out)
