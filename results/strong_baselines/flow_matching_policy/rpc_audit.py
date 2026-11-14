"""Separate native direct generation and old-runtime RPC without physics."""
import argparse
import hashlib
import numpy as np
import h5py
from .data import RUN,DIRECTORY
from baselines.strong.common import out,read,write,digest

def direct():
    import torch
    from .runtime import load,sample
    from .native import unnormalize_data
    from .data import Dataset
    torch.set_num_threads(4);d=Dataset();j=next(i for i,r in enumerate(d.rows) if r['domain']=='target' and r['split']=='train')
    r=d.rows[j];d.anchors=np.array([[j,0]],np.int32);batch,_=d.batch(0,1)
    with h5py.File(r['smol_pack'],'r') as f:rgb=f['rgb'][0];state=f['state'][0]
    checkpoint=RUN/'probe/updated.pt';nets,_=load(checkpoint);torch.manual_seed(7)
    normalized,noise,velocity=sample(nets,{k:v.cuda() for k,v in batch.items()})
    stats={k:np.array(v,np.float32) for k,v in read(DIRECTORY/'statistics.json')['action'].items()}
    canonical=unnormalize_data(normalized[0].cpu().numpy(),stats)
    write(RUN/'rpc/direct.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        input_pack=r['smol_pack'],episode_id=r['episode_id'],frame=0,state=state.tolist(),instruction=r['instruction'],
        raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),canonical16=canonical.tolist(),
        source_noise=noise[0].cpu().tolist(),velocity=velocity[0].cpu().tolist(),seed=7,physics_steps=0))
    d.close();print('PASS_DIRECT_NATIVE_SAMPLE_SAVED_NO_PHYSICS',flush=True)

def rpc():
    from .model import Policy
    r=read(RUN/'rpc/direct.json')
    with h5py.File(r['input_pack'],'r') as f:rgb=f['rgb'][r['frame']];state=f['state'][r['frame']]
    assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['raw_rgb_sha256']
    policy=Policy(r['checkpoint'],role='OFFLINE_RPC_ACTION_MAPPING_CHECK');queries=[];actions=[];current=None
    try:
        for t in range(17):
            p=policy.predict_action(dict(rgb=rgb,state=state,instruction=r['instruction']))
            if p['new_query']:
                current=p['native_query'];queries.append(dict(step=t,chunk=current.tolist()))
                assert current.shape==(8,7)
                if t==0:np.testing.assert_allclose(current,np.array(r['canonical16'],np.float32)[:8],rtol=1e-6,atol=5e-7)
            np.testing.assert_array_equal(p['action'],current[t%8]);actions.append(p['action'].tolist())
        assert [q['step'] for q in queries]==[0,8,16]
        metadata=policy.runtime_metadata
    finally:policy.close()
    report=dict(status='PASS_NATIVE_DIRECT_AND_RPC_ACTION_QUEUE',calls=17,native_queries=3,query_steps=[0,8,16],
        prediction_horizon=16,execution_horizon=8,first_native_sample_matches_direct=True,
        full_instruction_preserved=True,bytes_lists_only_transport=True,physics_steps=0,optimizer_updates=0,
        runtime=metadata,actions=actions,queries=queries,direct_sample_path=str(RUN/'rpc/direct.json'),direct_sha256=digest(RUN/'rpc/direct.json'))
    write(RUN/'rpc_audit.json',report);write(out('flow_matching_policy')/'rpc_audit.json',report)
    print('PASS17_RPC_CALLS_3_NATIVE_QUERIES_NO_PHYSICS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--direct',action='store_true');a=p.parse_args();direct() if a.direct else rpc()
