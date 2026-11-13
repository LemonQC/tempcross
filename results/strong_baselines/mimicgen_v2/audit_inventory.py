"""Read all 177 existing records/lineages and videos; never creates a simulator."""
from collections import Counter
from pathlib import Path
import csv
import json
from baselines.strong.common import ROOT,PROJECT,SOURCE,OLD,TASKS,out,read,write,digest,resource,now

GENERATION=Path('/home/users/lemon/mimicgen-baseline-20261112')

def main():
    import h5py,numpy as np,cv2
    folder=out('mimicgen_v2');destination=ROOT/'mimicgen_v2';destination.mkdir(parents=True,exist_ok=True)
    human=read(PROJECT/'configs/datasets/baseline_target_manifest.json');by_id={r['episode_id']:r for r in human}
    seeds=read(out('mimicgen')/'source_seed_manifest.json')
    assert len(seeds)==10 and all(r['domain']=='target' and r['embodiment']=='panda' and r['split']=='train' for r in seeds)
    seed_ids={r['episode_id'] for r in seeds}
    reference=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json');reference_ids={r['condition']['condition_id'] for r in reference}
    paths=sorted((GENERATION/'augmentation/probe').glob('aug-*/episode.json'));assert len(paths)==177
    results=[];index=[]
    for j,path in enumerate(paths):
        r=read(path);assert r['status']=='COMPLETE' and r['role']=='MIMICGEN_GENERATION'
        assert r['id']==path.parent.name and r['task'] in TASKS and 0<=r['condition']<20 and r['condition_id'] not in reference_ids
        lineage_path=GENERATION/'augmentation/lineage'/(r['id']+'.json');lineage=read(lineage_path)
        assert lineage['selected_sources'] and lineage['condition']==r['condition'] and lineage['task']==r['task']
        assert all(p['episode_id'] in seed_ids and by_id[p['episode_id']]['split']=='train' and by_id[p['episode_id']]['task']==r['task'] for p in lineage['selected_sources'])
        for artifact in r['artifacts']:
            assert Path(artifact['path']).stat().st_size==artifact['bytes'] and digest(artifact['path'])==artifact['sha256']
            index.append(artifact)
        index.extend(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in [path,lineage_path])
        with h5py.File(r['trajectory'],'r') as f,h5py.File(r['telemetry'],'r') as t:
            actions=f['actions'][:];n=len(actions);assert n==r['steps'] and actions.shape==(n,12)
            assert len(f['rgb'])==n and len(f['states'])==n+1 and f['rgb'].shape[1:]==(180,320,3)
            assert np.isfinite(actions).all() and np.isfinite(f['states'][:]).all() and np.max(np.abs(actions[:,:7]))<=1.
            assert np.all(actions[:,7:11]==0) and np.all(actions[:,11]==-1)
            assert np.array_equal(actions,t['actions/environment'][:])
            placement=t['mimicgen/post_action_placement'][:].astype(bool);retreat=t['mimicgen/post_action_retreat'][:].astype(bool)
            assert bool((placement & retreat).any())==bool(r['success'])
            if r['success']:assert placement[-1] and retreat[-1]
            instruction=json.loads(f.attrs['ep_meta'])['lang']
            assert str(f.attrs['condition_id'])==r['condition_id']
        cap=cv2.VideoCapture(r['video']);frames=0
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.shape==(180,320,3);frames+=1
        cap.release();assert frames==n
        result=dict(r,lineage=str(lineage_path),generation_seed=lineage['seed'],selected_sources=lineage['selected_sources'],instruction=instruction,
            all_video_frames_decoded=True,decoded_frames=frames,official_goal_verified=True,canonical_actions_bounded=True)
        results.append(result)
        if (j+1)%25==0:print('AUDITED',j+1,'/177',flush=True)
    counts={task:dict(attempts=sum(r['task']==task for r in results),successes=sum(r['task']==task and r['success'] for r in results)) for task in TASKS}
    assert counts=={TASKS[0]:dict(attempts=77,successes=50),TASKS[1]:dict(attempts=100,successes=13)}
    write(destination/'all_records.json',results)
    write(folder/'existing_generation_index.json',index)
    with (folder/'existing_generation_records.csv').open('w',newline='') as stream:
        fields=['id','task','condition','condition_id','generation_seed','success','steps','trajectory','telemetry','video','lineage','decoded_frames']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in results)
    write(folder/'data_audit.json',dict(status='PASS',utc=now(),per_task=counts,existing_records=177,successful_training_eligible=63,
        parent_seed_ids=sorted(seed_ids),seed_embodiment='panda',seeds_per_task=5,all_parents_in_original_train_split=True,
        original_50_plus_50_gate='BLOCKED_PRESERVED',generation_condition_role='EXPOSED_DEVELOPMENT_0_19',reference_role='EXPOSED_REFERENCE',
        no_reference_conditions_in_generation=True,initial_probe_records_used_for_training=0,all_177_videos_fully_decoded=True,
        source_record_file=str(destination/'all_records.json'),source_record_sha256=digest(destination/'all_records.json'),
        old_mimicgen_status_sha256=digest(out('mimicgen')/'status.json'),new_optimizer_updates=0,new_physics_steps=0,new_generation_attempts=0))
    write(folder/'resources_preparation.json',resource());print('PASS all177 records,63 successes,10 Panda training parents; no new physics')

if __name__=='__main__':main()
