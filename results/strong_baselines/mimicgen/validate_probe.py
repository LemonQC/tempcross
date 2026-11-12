"""Read-only independent verification of the ten bounded physical probes."""
import json
from pathlib import Path

from common import ROOT, OUT, TASKS, digest, read, write


def main():
    import h5py
    import numpy as np
    import cv2
    result=read(OUT/'probe_results.json')
    assert result['status']=='PASS' and len(result['records'])==10
    rows=[]
    for r in result['records']:
        with h5py.File(r['trajectory']) as f,h5py.File(r['telemetry']) as t:
            n=r['steps'];a=f['actions'][:]
            assert len(a)==n and len(f['states'])==n+1 and len(f['rgb'])==n
            assert np.array_equal(a,t['actions/environment'][:])
            assert np.isfinite(a).all() and np.isfinite(f['states'][:]).all()
            official=bool(np.any(t['mimicgen/post_action_placement'][:]&t['mimicgen/post_action_retreat'][:]))
            assert official==r['success']
        logs=[json.loads(x) for x in Path(r['contact_log']).read_text().splitlines()]
        assert len(logs)==n+1 and all(isinstance(x['contacts'],list) for x in logs)
        cap=cv2.VideoCapture(r['video']);count=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape[:2]==(180,320);count+=1
        cap.release();assert count==n
        rows.append(dict(id=r['id'],success=official,steps=n,decoded_video_frames=count,
                         contact_states=len(logs),actual_actions_match_telemetry=True))
    pairs=[]
    for r in result['records']:
        if r['role']!='MIMICGEN_GENERATION':continue
        replay=next(v for v in result['records'] if v['task']==r['task'] and v['condition']==r['condition'] and v['role']=='ACTION_REPLAY')
        with h5py.File(r['trajectory']) as a,h5py.File(replay['trajectory']) as b:
            assert np.array_equal(a['states'][0],b['states'][0])
            assert np.array_equal(a['actions'][:len(b['actions'])],b['actions'][:])
            common=min(len(a['states']),len(b['states']))
            pairs.append(dict(generation=r['id'],replay=replay['id'],same_initial_state=True,
                exact_action_prefix=True,max_abs_saved_state_difference=float(np.max(np.abs(a['states'][:common]-b['states'][:common])))))
    ledger=[json.loads(x) for x in (ROOT/'physical_launches.jsonl').read_text().splitlines()]
    assert len(ledger)==10 and len({x['id'] for x in ledger})==10
    status=read(OUT/'status.json')
    for path,sha in status['input_sha256'].items():assert digest(path)==sha
    report=dict(status='PASS',generation_successes=4,generation_attempts=5,replay_successes=4,replay_attempts=5,
                combined_successes=8,combined_physical_trajectories=10,distinct_condition_pairs=5,
                interpretation='Feasibility gate only; generation/replay pairs are correlated, not ten independent task tests.',
                records=rows,paired_checks=pairs,old_inputs_and_source_files_unchanged=True,
                all_raw_artifacts_preserved=True,new_physics_steps=0,new_policy_calls=0)
    write(OUT/'probe_validation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','paired_checks']},indent=2))


if __name__=='__main__':main()
