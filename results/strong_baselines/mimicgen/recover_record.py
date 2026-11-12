"""Recover a finished trajectory's missing JSON marker without any simulation."""
import json
from pathlib import Path
import shutil

from common import OUT, ROOT, digest, read, write


def main():
    import h5py
    import cv2
    import numpy as np
    evidence=OUT/'probe_logging_repair.json'
    if evidence.exists():
        print('Recovery already recorded; no files or trajectories repeated.')
        return
    state=read(OUT/'status.json')
    assert state['status']=='BLOCKED' and 'int64' in state['reason']
    write(ROOT/'technical_failure_status_before_recovery.json',state)
    shutil.copyfile(ROOT/'freeze.json',ROOT/'freeze_before_probe_logging_repair.json')
    key='pnp_counter_to_sink-gen-002';directory=ROOT/'probe'/key
    assert not (directory/'episode.json').exists()
    files=[p for p in directory.rglob('*') if p.is_file()]
    before={str(p):digest(p) for p in files}
    with h5py.File(directory/'trajectory.hdf5') as f, h5py.File(directory/'telemetry/episode-000.h5') as t:
        n=len(f['actions'])
        assert n==403 and len(f['states'])==n+1 and len(f['rgb'])==n
        assert np.array_equal(f['actions'][:],t['actions/environment'][:])
        assert np.isfinite(f['actions'][:]).all() and np.isfinite(f['states'][:]).all()
        success=bool(np.any(t['mimicgen/post_action_placement'][:] & t['mimicgen/post_action_retreat'][:]))
        assert not success and not bool(f.attrs['upstream_success'])
        condition_id=f.attrs['condition_id']
    cap=cv2.VideoCapture(str(directory/'video.mp4'));frames=0
    while True:
        ok,_=cap.read()
        if not ok:break
        frames+=1
    cap.release();assert frames==n
    record=dict(status='COMPLETE',id=key,role='MIMICGEN_GENERATION',task='pnp_counter_to_sink',
        condition=2,condition_id=condition_id,success=success,steps=n,
        trajectory=str(directory/'trajectory.hdf5'),video=str(directory/'video.mp4'),
        telemetry=str(directory/'telemetry/episode-000.h5'),contact_log=str(directory/'telemetry/episode-000.jsonl'),
        recovered_after_logging_error=True,source_demo_indices_available=False,
        source_demo_indices_unavailable_reason='Failed JSON serialization; not inferred after process exit',
        artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=before[str(p)]) for p in files])
    write(directory/'episode.json',record)
    assert all(digest(p)==h for p,h in before.items())
    try:
        json.dumps([np.int64(1),np.int64(3)])
        raise AssertionError('Expected original bug did not reproduce')
    except TypeError:
        pass
    assert json.loads(json.dumps([int(x) for x in [np.int64(1),np.int64(3)]]))==[1,3]
    repair=dict(status='COMPLETE',physical_probe_repair_attempt=1,
        authorization='202611121050.md: 首次探针失败可最多作一次有文档记录的实现修复',
        earlier_budget_error='First protocol merged preflight repair and physical-probe repair into a single cap; original user instruction separately allows both.',
        change='Cast upstream NumPy source-demo indices to Python int when writing JSON',
        raw_files_unchanged=True,raw_sha256=before,recovered_id=key,
        physics_steps=0,policy_calls=0,old_trajectories_repeated=0,
        seed=7,resume_rng_limitation='Original crash-time NumPy RNG state was not saved; skipped completed IDs do not consume RNG on restart. Remaining IDs restart the registered seed-7 process; not claimed bitwise identical to uninterrupted execution.',
        source_index_limitation='Selected per-subtask IDs unavailable for recovered failed probe; source pool remains fixed and known. All probe trajectories excluded from any downstream training.',
        no_more_probe_repairs=True)
    write(evidence,repair)
    state['generated_episode_ids'].append(key)
    state['recovered_episode_ids']=[key]
    state['repair_attempts'].append(dict(id=2,stage='PHYSICAL_PROBE_LOGGING',stage_attempt=1,
        evidence=str(evidence),physical_episodes_repeated=0,further_repairs_permitted=False))
    state.update(status='IN_PROGRESS',phase='REPAIR_COMPLETE_AWAITING_FROZEN_RESUME',
                 previous_technical_error=state.pop('reason'),automatic_retry_permitted=False)
    write(OUT/'status.json',state)
    print('Recovered existing 403-step failed trajectory; zero physics, no raw file changes.')


if __name__=='__main__':main()
