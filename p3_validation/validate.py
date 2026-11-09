"""Validate each completed P3 cell once, including every saved video frame."""
import json
from pathlib import Path
import numpy as np
from p3_validation.common import ROOT,PROJECT,SOURCE
from p2_validation.offline_audit import sha256


def main():
    import h5py,cv2
    from simulation_validation.validate_videos import validate
    directory=ROOT/'validation';directory.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((PROJECT/'evidence/p3/frozen/independent_conditions_40_59.json').read_text());cells=[]
    for path in sorted((ROOT/'formal').glob('*/job.json')):
        job=json.loads(path.read_text())
        if job['status']!='COMPLETE':continue
        assert job['indices']==list(range(40,60)) and job['conditions']==frozen[job['robot']]
        official=next((path.parent/'official').rglob('episodes.jsonl'));records=[json.loads(line) for line in official.read_text().splitlines()]
        assert len(records)==20 and all(type(x['success']) is bool for x in records)
        video_file=directory/(path.parent.name+'-videos.json')
        if not video_file.exists():validate(path.parent,video_file)
        video_rows=json.loads(video_file.read_text());changed=False
        for record in video_rows['records']:
            digest=sha256(record['video_path'])
            if 'sha256' in record:assert record['sha256']==digest
            else:record['sha256']=digest;changed=True
        if changed:video_file.write_text(json.dumps(video_rows,indent=2)+'\n')
        state_file=directory/(path.parent.name+'-states.json')
        if not state_file.exists():
            rows=[]
            for record,condition in zip(records,job['conditions']):
                assert record['condition_id']==condition['condition_id']
                p=path.parent/'telemetry'/f"episode-{record['episode']:03d}.h5"
                with h5py.File(p) as f:
                    entry=json.loads(f.attrs['condition']);assert entry['episode']==condition['episode'] and entry['condition_id']==condition['condition_id']
                    assert len(f['sim/qpos'])==record['steps']+1 and len(f['actions/environment'])==record['steps']
                    for key in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat','target/position','target/quaternion','actions/environment','p2/raw_gripper']:
                        assert np.isfinite(f[key][:]).all(),(p,key)
                rows.append(dict(condition_id=condition['condition_id'],original_condition_index=condition['episode'],trial_index=record['episode'],steps=record['steps'],finite_and_complete=True,telemetry_path=str(p),sha256=sha256(p)))
            state_file.write_text(json.dumps(dict(episodes=20,all_verified=True,records=rows),indent=2)+'\n')
        else:
            for record in json.loads(state_file.read_text())['records']:assert sha256(record['telemetry_path'])==record['sha256']
        cells.append(dict(cell=path.parent.name,episodes=20,window_and_hashes_match_frozen_protocol=True,states=str(state_file),videos=str(video_file)))
    # Beyond condition hashes, verify actual restored Panda states are identical
    # between methods. Cross-embodiment states naturally have different robots.
    matched=[];reference=ROOT/'formal/original-joint_reps-panda/telemetry'
    for cell in cells:
        if cell['cell']=='original-joint_reps-ur5e':continue
        for j in range(20):
            p=ROOT/'formal'/cell['cell']/'telemetry'/f'episode-{j:03d}.h5'
            with h5py.File(reference/f'episode-{j:03d}.h5') as a,h5py.File(p) as b:
                errors={k:float(np.abs(a[k][0]-b[k][0]).max()) for k in ['sim/qpos','sim/qvel','sim/body_xpos','sim/body_xquat']}
            assert max(errors.values())==0.,(cell['cell'],j,errors)
            matched.append(dict(cell=cell['cell'],original_condition_index=j+40,maximum_initial_state_error=0.))
    (directory/'panda_matched_initial_states.json').write_text(json.dumps(dict(records=matched,all_identical=True),indent=2)+'\n')
    diag=json.loads((ROOT/'diagnostics/results.json').read_text());dv=directory/'diagnostic-videos.json'
    if diag['status']=='COMPLETE' and not dv.exists():
        rows=[]
        for result in diag['results']:
            path=Path(result['video_path']);cap=cv2.VideoCapture(str(path));n=0
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
            cap.release();assert n==result['steps']
            rows.append(dict(video_path=str(path),expected_frames=result['steps'],decoded_frames=n,verified=True,sha256=sha256(path)))
        dv.write_text(json.dumps(dict(videos=len(rows),all_verified=True,records=rows),indent=2)+'\n')
    if dv.exists():
        for record in json.loads(dv.read_text())['records']:assert sha256(record['video_path'])==record['sha256']
    all_videos=[]
    for p in directory.glob('*-videos.json'):
        all_videos.extend(json.loads(p.read_text())['records'])
    result=dict(complete_formal_cells=len(cells),complete_formal_episodes=20*len(cells),cells=cells,diagnostic_rollouts=len(diag['results']),videos_verified=len(all_videos),decoded_frames=sum(r['decoded_frames'] for r in all_videos),complete_independent_suite=len(cells)==5,diagnostic_complete=diag['status']=='COMPLETE')
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
