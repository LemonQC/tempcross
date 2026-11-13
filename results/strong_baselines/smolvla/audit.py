"""Read-only complete-goal/video, native-action and proprioception acceptance."""
import collections
import json
from pathlib import Path
import numpy as np
import h5py
from baselines.strong.common import ROOT,out,read,write,digest
from baselines.strong.audit import validate_job
from .data import DIRECTORY

def key(a):return np.asarray(a,np.float32).tobytes()

def audit_job(task,smoke=False):
    group='source_target_state';kind='smoke' if smoke else 'formal'
    directory=ROOT/'smolvla/physical'/kind/group/task
    accepted=validate_job('smolvla',group,task,smoke)
    job=read(directory/'job.json');ledger=Path(job['runtime']['raw_action_ledger'])
    stats=read(DIRECTORY/'statistics.json')['action'];mean=np.array(stats['mean'],np.float32);std=np.array(stats['std'],np.float32)
    expected=collections.Counter();states=0
    for r in accepted['rows']:
        with h5py.File(r['episode']['telemetry_path'],'r') as f:
            n=r['episode']['steps'];addresses=f['baseline/robot_state_qpos_addresses'][0].astype(int)
            assert addresses.shape==(9,) and len(set(addresses.tolist()))==9
            values=f['baseline/robot_state'][:]
            np.testing.assert_array_equal(values,f['sim/qpos'][:n,addresses].astype(np.float32))
            assert values.shape==(n,9);states+=n
            expected.update(key(a) for a in f['baseline/native_query'][:])
    actual=collections.Counter();queries=0
    for line in ledger.read_text().splitlines():
        r=json.loads(line);raw=np.asarray(r['normalized32'],np.float32);used=np.asarray(r['normalized7'],np.float32)
        canonical=np.asarray(r['canonical_chunk'],np.float32)
        assert raw.shape==(50,32) and used.shape==(50,7)
        np.testing.assert_array_equal(used,raw[:,:7])
        np.testing.assert_allclose(used*std+mean,canonical,rtol=1e-6,atol=2e-7)
        actual.update([key(canonical)]);queries+=1
    assert actual==expected, 'Every telemetry chunk must match one original native query'
    result=dict(status='PASS',task=task,role=accepted['role'],episodes=accepted['episodes'],successes=accepted['successes'],
        native_queries=queries,raw32_to7_crop_and_unnormalization_verified=True,
        native_queue_matches_all_executed_actions=True,proprioception_rows_verified=states,
        recorded_state_equals_pre_action_named_joint_qpos=True,object_state_is_not_policy_input=True,
        action_ledger=str(ledger),action_ledger_sha256=digest(ledger),new_physics_steps=0)
    write(directory/'state_action_acceptance.json',result)
    write(out('smolvla')/(('smoke_' if smoke else 'evaluation_')+'state_action_'+task+'.json'),result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();print(audit_job(a.task,a.smoke))
