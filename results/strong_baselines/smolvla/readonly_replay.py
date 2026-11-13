"""Read saved successful demo commands/states; never call environment.step."""
import collections
import numpy as np
import h5py
from baselines.strong.common import ROOT,out,read,write,digest
from baselines.common.action_adapter import canonical_action
from .data import DIRECTORY,robot_state

def main():
    selected={}
    for row in read(DIRECTORY/'manifest.json'):
        if row['split']=='train':selected.setdefault((row['domain'],row['task']),row)
    assert len(selected)==4
    reports=[]
    for group,row in sorted(selected.items()):
        with h5py.File(row['raw_file'],'r') as f:
            g=f['data'][row['raw_demo']];raw=g['actions'][:];state=g['states'][:];gripper=g['obs/gripper_states'][:]
            reward=g['rewards'][:];done=g['dones'][:]
        assert bool(np.any(done))
        canonical=raw[:,:7].astype(np.float32);canonical[-1]=canonical[-2];canonical[-1,:6]=0
        converted=np.stack([canonical_action(a)[0] for a in canonical])
        np.testing.assert_allclose(converted[:-1],raw[:-1],rtol=1e-6,atol=1e-7)
        assert np.all(converted[:,7:11]==0) and np.all(converted[:,11]==-1)
        values={}
        for sign,name in [(-1,'open'),(1,'close')]:
            mask=np.array([np.all(np.sign(canonical[max(0,t-9):t+1,6])==sign) and t>=9 for t in range(len(canonical))])
            assert mask.any();values[name]=float(np.median(gripper[mask,0]))
        assert values['close']>values['open']+.1, values
        chunks=[]
        for t in [0,max(0,len(canonical)-25),len(canonical)-1]:
            future=t+np.arange(50);pad=future>=len(canonical)
            expected=canonical[np.minimum(future,len(canonical)-1)]
            assert expected.shape==(50,7) and np.all(pad[future>=len(canonical)])
            chunks.append(dict(anchor=t,actual_frames=int((~pad).sum()),padded_frames=int(pad.sum())))
        q=robot_state(state,gripper,row['domain'])
        report=dict(episode_id=row['episode_id'],domain=group[0],task=group[1],raw_file=row['raw_file'],
            raw_demo=row['raw_demo'],frames=len(canonical),raw_rewards_are_zero=bool(np.all(reward==0)),saved_done=True,
            success_certification='raw rewards are not informative; these rows certify interface only',
            all_original_transition_actions_match_OSC12=True,terminal='original RLDS virtual terminal arm0/repeated gripper, not a new transition',
            gripper_positive_is_close=True,measured_driver_qpos_median=values,state_shape=list(q.shape),
            checked_chunks=chunks,physics_step_calls=0,new_policy_episodes=0)
        reports.append(report)
    # Existing accepted generated demos supply independently saved full-task
    # success. They are read-only interface references, NOT SmolVLA train data.
    generated=[];manifest=read(ROOT/'mimicgen_v2/data/manifest.json');records=read(ROOT/'mimicgen_v2/all_records.json')
    train_ids={r['episode_id'] for r in read(DIRECTORY/'manifest.json') if r['domain']=='target' and r['split']=='train'}
    for task in sorted({r['task'] for r in manifest}):
        m=next(r for r in manifest if r.get('origin')=='generated' and r['task']==task)
        r=next(r for r in records if r['trajectory']==m['original_file'])
        assert r['success'] and r['official_goal_verified'] and r['condition']<60
        assert all(parent['episode_id'] in train_ids for parent in m['source_lineage'])
        with h5py.File(r['trajectory'],'r') as f:actions=f['actions'][:]
        decoded=np.stack([canonical_action(a[:7])[0] for a in actions])
        np.testing.assert_allclose(decoded,actions,rtol=1e-7,atol=1e-7)
        artifact=next(a for a in r['artifacts'] if a['path']==r['trajectory'])
        assert digest(r['trajectory'])==artifact['sha256']
        generated.append(dict(id=r['id'],task=task,trajectory=r['trajectory'],sha256=artifact['sha256'],
            official_goal_verified=True,frames=len(actions),all_OSC12_actions_exact=True,
            included_in_smolvla_training=False,new_generation=False))
    result=dict(status='PASS_READ_ONLY_COMMANDS_AND_CERTIFIED_SUCCESS_REFERENCES',selection='first original train ID per domain/task; first existing accepted generated reference per task; no model/test-success selection',
        original_demos=reports,certified_success_references=generated,physics_step_calls=0,new_policy_episodes=0,core_control_interface_changed=False,
        raw_robot_state_audit=str(DIRECTORY/'state_audit_rows.json'),raw_robot_state_audit_sha256=digest(DIRECTORY/'state_audit_rows.json'))
    write(ROOT/'smolvla/readonly_replay.json',result);write(out('smolvla')/'readonly_replay.json',result)
    print('PASS4_RAW_INTERFACES_PLUS2_CERTIFIED_SUCCESS_REFERENCES_NO_PHYSICS',flush=True)

if __name__=='__main__':main()
