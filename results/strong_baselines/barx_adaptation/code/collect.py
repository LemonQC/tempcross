"""Collect actual new results and immutable old BARX comparisons read-only."""
import csv
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import shutil
import numpy as np
import h5py
from common import OUT, PROJECT, ROOT, digest, read, write


def table(path,rows):
    with Path(path).open('w',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n')
        w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def analyze(row):
    spec=importlib.util.spec_from_file_location('saved_interaction_audit',PROJECT/'evidence/interaction_initiation/audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    p=Path(row['telemetry_path']);logs=[json.loads(s) for s in p.with_suffix('.jsonl').read_text().splitlines()]
    with h5py.File(p,'r') as f:
        action=f['actions/environment'][:];native=f['baseline/native_action'][:]
        closed=action[:,6]>0;indices=np.flatnonzero(closed);first=int(indices[0]) if len(indices) else None
        bodies=json.loads(f.attrs['body_names']);names=json.loads(f.attrs['geom_names'])
        geoms={g for g,b in zip(names,f['model/geom_bodyid'][:]) if bodies[b].startswith('obj_')}
        contact=audit.physical_contact(logs,geoms)
        if first is not None:contact[:first+1]=False
        rise=f['target/position'][:,2]-f['target/position'][0,2]
        stable=audit.longest(contact & (rise>=.03))>=10
        placement=bool(f['baseline/post_action_placement'][:].any())
        success=bool(row['success'])
        stage='SUCCESS' if success else 'NO_CLOSE_COMMAND' if first is None else 'CLOSE_WITHOUT_BILATERAL_CONTACT' if not contact.any() else 'CONTACT_WITHOUT_STABLE_LIFT' if not stable else 'PLACEMENT_WITHOUT_RETREAT' if placement else 'STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT'
        assert np.array_equal(native[:,6]<.5,closed)
    return dict(**row,issued_close=first is not None,first_close=first,bilateral_contact_after_close=bool(contact.any()),
                stable_lift_proxy=stable,ever_placement=placement,failure_stage=stage,
                causal_failure_reason='NOT_IDENTIFIED_BY_STAGE_PROXY')


def main():
    state=read(ROOT/'training/status.json');assert state['status']=='COMPLETE'
    assert read(ROOT/'physical/complete.json')['episodes']==160
    rows=[];jobs=[];artifacts=[]
    for p in sorted((ROOT/'physical/formal').rglob('job.json')):
        job=read(p);assert job['status']=='COMPLETE' and job['role']=='EXPOSED_FROZEN_REFERENCE_NOT_INDEPENDENT_TEST'
        validation=read(p.parent/'validation.json');assert validation['all_verified'] and validation['exact_original_frozen_initial_states_verified']
        jobs.append(dict(path=str(p),**job))
        for s in (p.parent/'episodes.jsonl').read_text().splitlines():
            row=json.loads(s);row.update(label='new_joint_adaptation_768000',task=job['task'],robot=job['robot'],
                                        evaluation_role='EXPOSED_REFERENCE',new_physical_run=True)
            rows.append(analyze(row))
        for r in validation['records']:
            artifacts.append(dict(kind='telemetry',path=r['telemetry_path'],sha256=r['telemetry_sha256']))
            artifacts.append(dict(kind='video',path=r['video_path'],sha256=r['video_sha256']))
    assert len(rows)==160
    old=read(PROJECT/'results/baselines/analysis.json')['episodes'];assert len(old)==1280
    labels={'A_barx_joint','A_barx_no_reps','C_barx_joint_96000','C_barx_no_reps_6000'}
    for r in old:
        if r['label'] in labels:
            r={k:v for k,v in r.items() if k not in ['issued_close','first_close','bilateral_contact_after_close','stable_lift_proxy','ever_placement','failure_stage','causal_failure_reason']}
            r.update(evaluation_role='HISTORICAL_EXPOSED_REFERENCE',new_physical_run=False)
            rows.append(analyze(r))
    table(OUT/'episodes.csv',rows)
    groups=defaultdict(list)
    for r in rows:groups[r['label'],r['task'],r['robot']].append(r)
    cells=[]
    for (label,task,robot),rr in sorted(groups.items()):
        cells.append(dict(label=label,task=task,robot=robot,episodes=len(rr),successes=sum(r['success'] for r in rr),
                          success_rate=sum(r['success'] for r in rr)/len(rr),issued_close=sum(r['issued_close'] for r in rr),
                          stable_lift=sum(r['stable_lift_proxy'] for r in rr),failure_counts=dict(Counter(r['failure_stage'] for r in rr)),
                          role=rr[0]['evaluation_role'],new_physical_run=rr[0]['new_physical_run']))
    table(OUT/'cells.csv',cells)
    pairs=[]
    for r in rows:
        if r['label']!='new_joint_adaptation_768000':continue
        for label in ['A_barx_joint','C_barx_joint_96000']:
            reference=[v for v in rows if v['label']==label and v['task']==r['task'] and v['robot']==r['robot'] and v['condition_id']==r['condition_id']]
            if not reference:continue
            assert len(reference)==1
            b=reference[0]
            pairs.append(dict(task=r['task'],robot=r['robot'],condition_id=r['condition_id'],official_index=r['official_condition_index'],baseline=label,
                              baseline_success=b['success'],candidate_success=r['success'],rescued=not b['success'] and r['success'],
                              regressed=b['success'] and not r['success'],baseline_video=b['video_path'],candidate_video=r['video_path'],
                              interpretation='DESCRIPTIVE_PAIRED_EXPOSED_CONDITIONS_DIFFERENT_TRAIN_SPLIT_AND_BUDGET'))
    table(OUT/'paired_cases.csv',pairs)
    trace=ROOT/'training/consumed_samples.csv';maps={r['episode_id']:r for r in __import__('common').manifests()}
    counts=Counter();covered=defaultdict(set);n=0
    with trace.open() as f:
        for r in csv.DictReader(f):
            assert int(r['index'])==n and maps[r['episode_id']]['split']=='train'
            counts[r['dataset'],r['task']]+=1;covered[r['dataset'],r['episode_id']].add(int(r['frame']));n+=1
    assert n==768000==state['sample_positions']
    write(OUT/'consumption.json',dict(actual_sample_positions=n,actual_optimizer_updates=state['completed_optimizer_updates'],
          actual_backwards=state['completed_backwards'],no_validation_demonstrations_consumed=True,
          counts=[dict(dataset=k[0],task=k[1],positions=v) for k,v in sorted(counts.items())],
          unique_demonstrations={d:sum(k[0]==d for k in covered) for d in ['mg_pnp_lite','panda_pnp']},
          unique_frame_positions=sum(len(v) for v in covered.values()),trace_path=str(trace),trace_sha256=digest(trace),trace_bytes=trace.stat().st_size))
    for name in ['training_status.json','selection.json']:
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,OUT/name)
    shutil.copyfile(ROOT/'training/updates.csv',OUT/'updates.csv')
    dest=OUT/'validation';dest.mkdir(exist_ok=True)
    for p in (ROOT/'validation').glob('*.json'):shutil.copyfile(p,dest/p.name)
    model=read(OUT/'model_index.json')
    for path in [Path(model['merged_checkpoint']),Path(model['adapter'])/'adapter_model.safetensors',trace]:
        artifacts.append(dict(kind='checkpoint_or_trace',path=str(path),sha256=digest(path),bytes=path.stat().st_size))
    write(OUT/'artifact_index.json',artifacts)
    write(OUT/'results.json',dict(status='COMPLETE',new_training=state,new_evaluation_episodes=160,
          old_physical_evaluations_rerun=0,independent_unseen_test=False,cells=cells,selected_model=model,
          old_joint_6000_context=dict(task='pnp_counter_to_sink',robot='panda',indices=[20,39],successes=3,episodes=20,
                                     role='HISTORICAL_DIFFERENT_CONDITIONS_NOT_DIRECTLY_COMPARABLE')))
    write(OUT/'physical_jobs.json',jobs)


if __name__=='__main__':main()
