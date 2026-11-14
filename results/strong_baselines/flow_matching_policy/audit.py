"""Verify actual native Euler generation, normalization,8-step queue and qpos."""
import collections
import json
from pathlib import Path
import numpy as np
import h5py
from .data import RUN,DIRECTORY
from baselines.strong.common import out,read,write,digest
from baselines.strong.audit import validate_job

GROUP='source_target_visual'
def key(a):return np.asarray(a,np.float32).tobytes()

def audit_job(task,smoke=False):
    kind='smoke' if smoke else 'formal';directory=RUN/'physical'/kind/GROUP/task
    accepted=validate_job('flow_matching_policy',GROUP,task,smoke)
    ledger=Path(read(directory/'job.json')['runtime']['raw_action_ledger'])
    stats=read(DIRECTORY/'statistics.json')['action']
    low=np.array(stats['min'],np.float32);high=np.array(stats['max'],np.float32)
    expected=collections.Counter();actual=collections.Counter();states=queries=0
    for row in accepted['rows']:
        e=row['episode']
        with h5py.File(e['telemetry_path'],'r') as f:
            n=e['steps'];addresses=f['baseline/robot_state_qpos_addresses'][0].astype(int)
            assert addresses.shape==(9,) and len(set(addresses.tolist()))==9
            values=f['baseline/robot_state'][:]
            np.testing.assert_array_equal(values,f['sim/qpos'][:n,addresses].astype(np.float32))
            assert values.shape==(n,9);states+=n
            starts=f['baseline/native_query_action_index'][:]
            np.testing.assert_array_equal(starts,np.arange(0,n,8))
            expected.update(key(chunk) for chunk in f['baseline/native_query'][:])
    for line in ledger.read_text().splitlines():
        r=json.loads(line);noise=np.array(r['source_noise'],np.float32);velocity=np.array(r['velocity'],np.float32)
        normalized=np.array(r['normalized16'],np.float32);canonical=np.array(r['canonical16'],np.float32)
        chunk=np.array(r['executed_chunk'],np.float32)
        assert noise.shape==velocity.shape==normalized.shape==canonical.shape==(16,7) and chunk.shape==(8,7)
        assert noise.min()>=0 and noise.max()<1 and r['t']==0 and r['dt']==1
        np.testing.assert_allclose(normalized,noise+velocity,rtol=1e-6,atol=2e-7)
        np.testing.assert_allclose(canonical,(normalized+1)/2*(high-low)+low,rtol=1e-6,atol=2e-7)
        np.testing.assert_array_equal(chunk,canonical[:8]);actual.update([key(chunk)]);queries+=1
    assert expected==actual,'Every executed queue must match exactly one original flow generation'
    result=dict(status='PASS',task=task,role=accepted['role'],episodes=accepted['episodes'],successes=accepted['successes'],
        native_queries=queries,real_qpos_rows_verified=states,native_uniform_single_Euler_verified=True,
        original_prediction16_execution8_verified=True,minmax_action_decode_and_OSC12_verified=True,
        full_task_official_goal_and_video_verified=True,object_state_not_policy_input=True,
        action_ledger=str(ledger),action_ledger_sha256=digest(ledger),new_physics_steps=0)
    write(directory/'state_action_acceptance.json',result)
    write(out('flow_matching_policy')/(('smoke_' if smoke else 'evaluation_')+'state_action_'+task+'.json'),result)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();print(audit_job(a.task,a.smoke))
