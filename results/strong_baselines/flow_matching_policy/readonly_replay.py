"""Replay saved raw/success commands through native normalization, no physics."""
import numpy as np
import h5py
from .native import normalize_data,unnormalize_data,sample_sequence
from .data import RUN,DIRECTORY
from baselines.strong.common import ROOT,out,read,write,digest
from baselines.common.action_adapter import canonical_action

def check(actions,stats):
    actions=np.asarray(actions,np.float32)
    restored=unnormalize_data(normalize_data(actions,stats),stats)
    np.testing.assert_allclose(restored,actions,rtol=1e-6,atol=2e-7)
    original=np.stack([canonical_action(a)[0] for a in actions]);mapped=np.stack([canonical_action(a)[0] for a in restored])
    np.testing.assert_allclose(mapped,original,rtol=1e-6,atol=2e-7)
    for t in [0,max(0,len(actions)-9),len(actions)-1]:
        end=min(t+16,len(actions));seq=sample_sequence({'action':actions},16,t,end,0,end-t)['action']
        np.testing.assert_array_equal(seq,actions[np.minimum(t+np.arange(16),len(actions)-1)])
        np.testing.assert_array_equal(seq[:8],actions[np.minimum(t+np.arange(8),len(actions)-1)])
    return dict(frames=len(actions),native_minmax_inverse=True,positive_close_preserved=True,
        OSC12_mode_and_extra_slots_unchanged=True,prediction16_execution8_verified=True)

def main():
    stats={k:np.array(v,np.float32) for k,v in read(DIRECTORY/'statistics.json')['action'].items()}
    selected={}
    for r in read(DIRECTORY/'manifest.json'):
        if r['split']=='train':selected.setdefault((r['domain'],r['task']),r)
    raw=[]
    for (domain,task),r in sorted(selected.items()):
        with h5py.File(r['raw_file'],'r') as f:
            g=f['data'][r['raw_demo']];a=g['actions'][:,:7].astype(np.float32);grip=g['obs/gripper_states'][:]
            a[-1]=a[-2];a[-1,:6]=0
        measured={}
        for sign,key in [(-1,'open'),(1,'close')]:
            mask=[t>=9 and np.all(np.sign(a[max(0,t-9):t+1,6])==sign) for t in range(len(a))]
            measured[key]=float(np.median(grip[mask,0]))
        assert measured['close']>measured['open']+.1
        raw.append(dict(check(a,stats),episode_id=r['episode_id'],domain=domain,task=task,
            gripper_qpos_medians=measured,raw_rewards_not_used_as_success_labels=True))
    certified=[]
    prior=read(ROOT/'smolvla/readonly_replay.json')
    for ref in prior['certified_success_references']:
        assert ref['official_goal_verified'] and digest(ref['trajectory'])==ref['sha256']
        with h5py.File(ref['trajectory'],'r') as f:a=f['actions'][:]
        certified.append(dict(check(a[:,:7],stats),trajectory=ref['trajectory'],sha256=ref['sha256'],
            official_success_previously_certified=True,included_in_FMP_training=False,new_generation=False))
    report=dict(status='PASS_READONLY_ACTIONS_AND_CERTIFIED_SUCCESS_REFERENCES',original_demos=raw,
        existing_certified_success_references=certified,physics_steps=0,new_physical_episodes=0,
        core_controller_modified=False,statistics_source='original630 training demos only')
    write(RUN/'readonly_replay.json',report);write(out('flow_matching_policy')/'readonly_replay.json',report)
    print('PASS4_RAW_PLUS2_EXISTING_SUCCESS_REFERENCES_NO_PHYSICS',flush=True)

if __name__=='__main__':main()
