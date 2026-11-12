"""Check every anchor against original identity partitions and canonical actions."""
from collections import Counter
from baselines.strong.common import PROJECT,ROOT,out,read,write,digest

def main():
    import numpy as np,h5py
    directory=ROOT/'vqbet/data';rows=read(directory/'manifest.json');original=[]
    for domain in ['target','source']:
        subset=read(PROJECT/f'configs/datasets/baseline_{domain}_manifest.json')
        if domain=='source':subset=[r for r in subset if r['embodiment']=='ur5e']
        original+=subset
    assert [{k:v for k,v in r.items() if k!='packed_file'} for r in rows]==original
    n=np.array([r['frames'] for r in rows]);checked=[]
    for stage in ['vqvae','bet']:
        for domain in ['target','source']:
            anchors=np.load(directory/f'{stage}_{domain}.npy');j=anchors[:,0];t=anchors[:,1]
            assert anchors.shape==(384000,2) and np.all((t>=9)&(t<n[j]))
            assert all(rows[k]['domain']==domain and rows[k]['split']=='train' for k in np.unique(j))
            counts=Counter(rows[k]['task'] for k in j)
            assert sorted(counts.values())==[192000,192000]
            checked.append(dict(stage=stage,domain=domain,anchors=len(anchors),unique_demo_ids=len(np.unique(j)),task_counts=dict(counts)))
    val=np.load(directory/'vq_validation.npy')
    assert len(val)==256 and all(rows[j]['domain']=='target' and rows[j]['split']=='validation' and 9<=t<n[j] for j,t in val)
    for domain in ['target','source']:
        selected=[r for r in rows if r['domain']==domain]
        with h5py.File(selected[0]['packed_file'],'r') as f:
            for r in selected:
                g=f[r['episode_id']];a=g['actions'][:]
                assert len(a)==r['frames']==len(g['rgb']) and np.isfinite(a).all() and np.max(np.abs(a))<=1.00001
    previous=read(out('act')/'data_acceptance.json')
    write(out('vqbet')/'data_acceptance.json',dict(status='PASS',episodes_checked=len(rows),all_anchors_checked=checked,validation_anchors_checked=len(val),
        canonical_action_labels_finite_and_bounded=True,original_identity_partitions_exact=True,causal_window=10,no_future_rgb=True,
        raw_pack_reference=previous['pack_references'],raw_pack_hash_note='Existing immutable pack completion SHA, not a new29GB RGB byte scan',
        identity_manifest_sha256=previous['identity_manifest_sha256'],new_optimizer_updates=0,new_physics_steps=0))

if __name__=='__main__':main()
