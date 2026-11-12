"""Read-only acceptance of recorded physical trials, including ACT aggregation."""
import json
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest,TASKS

def validate_episode(path,method,smoke=False):
    import numpy as np,h5py,cv2
    from baselines.common.action_adapter import canonical_action
    r=read(path);telemetry=Path(r['telemetry_path']);video=Path(r['video_path'])
    with h5py.File(telemetry,'r') as f:
        n=r['steps'];assert len(f['sim/qpos'])==n+1 and len(f['actions/environment'])==n
        for key in ['sim/qpos','sim/qvel','target/position','target/quaternion','actions/environment','baseline/native_action','baseline/pre_action_tcp_position']:
            assert np.isfinite(f[key][:]).all(),(path,key)
        actual=f['actions/environment'][:];native=f['baseline/native_action'][:]
        assert np.array_equal(actual,np.stack([canonical_action(a)[0] for a in native]))
        placement=f['baseline/post_action_placement'][:].astype(bool);retreat=f['baseline/post_action_retreat'][:].astype(bool)
        assert bool((placement & retreat).any())==r['success']
        if not r['success']:assert n==(600 if r['task']==TASKS[0] else 650)
        entry=json.loads(f.attrs['condition']);assert entry['condition_id']==r['condition_id']
        if not smoke:
            frozen=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
            e=next(e for e in frozen if e['task']==r['task'] and e['robot']=='panda' and e['official_index']==r['official_condition_index'])
            assert np.array_equal(f['baseline/restored_initial_state'][0],np.asarray(e['initial_state'],np.float64))
        queries=f['baseline/native_query'][:];starts=f['baseline/native_query_action_index'][:]
        if method=='act':
            assert np.array_equal(starts,np.arange(n)) and queries.shape==(n,100,7)
            for t in range(n):
                indices=np.arange(max(0,t-99),t+1);candidates=queries[indices,t-indices]
                weights=np.exp(-.01*np.arange(len(indices)));weights/=weights.sum()
                assert np.allclose((candidates*weights[:,None]).sum(0),native[t],rtol=1e-7,atol=2e-7)
            query_audit='all forecasts and exponential aggregation independently reconstructed'
        else:
            assert len(queries)==len(starts) and starts[0]==0
            for q,t in zip(queries,starts):
                t=int(t);end=min(t+len(q),n);assert np.allclose(q[:end-t],native[t:end],rtol=1e-7,atol=1e-7)
            query_audit='all emitted action queues match saved native chunks'
        geoms=json.loads(f.attrs['geom_names']);bodies=json.loads(f.attrs['body_names']);object_geoms={g for g,b in zip(geoms,f['model/geom_bodyid'][:]) if bodies[b].startswith('obj_')}
        contact_rows=[json.loads(x) for x in Path(r['contact_log']).read_text().splitlines()];assert len(contact_rows)==n+1
        bilateral=[]
        for row in contact_rows:
            sides=set()
            for c in row['contacts']:
                if c['distance']>0:continue
                for a,b in [(c['geom1'],c['geom2']),(c['geom2'],c['geom1'])]:
                    if b not in object_geoms:continue
                    for side in ['left','right']:
                        if a in {f'gripper0_right_{side}_fingerpad_collision',f'gripper0_right_{side}_fingertip_collision'}:sides.add(side)
            bilateral.append(len(sides)==2)
        close=bool((native[:,6]>0).any());contact=bool(any(bilateral));positions=f['target/position'][:]
        rise=positions[:,2]-positions[0,2];longest=run=0
        for b in np.asarray(bilateral)&(rise>=.03):
            run=run+1 if b else 0;longest=max(longest,run)
        stage=('SUCCESS' if r['success'] else 'NO_CLOSE_COMMAND' if not close else 'CLOSE_WITHOUT_BILATERAL_CONTACT' if not contact else
               'CONTACT_WITHOUT_STABLE_LIFT_PROXY' if longest<10 else 'PLACEMENT_WITHOUT_RETREAT' if placement.any() else 'STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT')
    cap=cv2.VideoCapture(str(video));frames=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(180,320,3);frames+=1
    cap.release();assert frames==r['steps'] and frames>0
    result=dict(status='PASS',episode=r,decoded_frames=frames,video_sha256=digest(video),telemetry_sha256=digest(telemetry),contact_sha256=digest(r['contact_log']),
                exact_action_conversion=True,official_goal_verified=True,original_frozen_state_verified=not smoke,query_audit=query_audit,
                failure_stage=stage,failure_stage_is_descriptive_not_causal=True)
    write(Path(path).parent/'acceptance.json',result);return result

def validate_job(method,group,task,smoke=False):
    directory=ROOT/method/'physical'/('smoke' if smoke else 'formal')/group/task
    job=read(directory/'job.json');assert job['status']=='COMPLETE'
    rows=[]
    for i in job['indices']:
        p=directory/f'condition-{i:03d}/episode.json';cache=p.parent/'acceptance.json'
        if cache.exists():
            row=read(cache);assert digest(row['episode']['video_path'])==row['video_sha256'] and digest(row['episode']['telemetry_path'])==row['telemetry_sha256']
        else:row=validate_episode(p,method,smoke)
        rows.append(row)
    assert len(rows)==(1 if smoke else 40)
    result=dict(status='PASS',method=method,group=group,task=task,role='DEVELOPMENT_SMOKE' if smoke else 'EXPOSED_REFERENCE',episodes=len(rows),successes=sum(r['episode']['success'] for r in rows),rows=rows)
    write(directory/'acceptance.json',result);write(out(method)/(('smoke_' if smoke else 'evaluation_')+group+'_'+task+'.json'),result)
    return result
