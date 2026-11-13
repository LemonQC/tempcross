"""Actual split, temporal/action alignment, and original-evidence protection."""
import numpy as np
from baselines.strong.common import ROOT,PROJECT,OLD,out,read,digest
from results.strong_baselines.mimicgen_v2.data import Dataset

def test_original_validation_identity_and_anchors_unchanged():
    assert digest(ROOT/'mimicgen_v2/data/validation.npy')==digest(OLD/'data/target_validation_anchors.npy')
    rows=read(ROOT/'mimicgen_v2/data/manifest.json');original=read(PROJECT/'configs/datasets/baseline_target_manifest.json')
    assert [{k:v for k,v in r.items() if k not in ['origin','packed_file']} for r in rows[:100]]==original
    assert len(rows)==163 and sum(r['origin']=='generated' for r in rows)==63

def test_causal_windows_and_generated_action_alignment():
    import h5py,cv2
    d=Dataset(limit=2)
    try:
        j=next(j for j,r in enumerate(d.rows) if r['origin']=='generated');r=d.rows[j];t=r['frames']-1
        d.anchors=np.array([[j,t],[j,0]],np.int32)
        with h5py.File(r['original_file'],'r') as f:
            b=d[0]
            assert np.array_equal(b['actions'][-1],f['actions'][t,:7].astype(np.float32))
            expected=cv2.resize(f['rgb'][t],(128,128),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/255
            assert np.array_equal(b['obs']['rgb'][-1],expected)
            first=d[1];assert np.array_equal(first['obs']['rgb'][0],first['obs']['rgb'][-1])
            assert b['obs']['rgb'].shape==(10,3,128,128) and b['actions'].shape==(10,7)
    finally:d.close()

def test_all_generated_parents_are_training_and_conditions_are_development():
    rows=read(ROOT/'mimicgen_v2/data/manifest.json');human={r['episode_id']:r for r in rows[:100]}
    reference=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json');ids={r['condition']['condition_id'] for r in reference}
    for r in rows[100:]:
        assert r['split']=='train' and r['condition_id'] not in ids
        assert all(human[p['episode_id']]['split']=='train' for p in r['source_lineage'])

def test_old_gate_preserved_and_not_relabelled_complete():
    audit=read(out('mimicgen_v2')/'data_audit.json')
    assert audit['existing_records']==177 and audit['successful_training_eligible']==63
    assert digest(out('mimicgen')/'status.json')==audit['old_mimicgen_status_sha256']
    assert read(out('mimicgen')/'status.json')['status']=='BLOCKED'
    assert read(out('mimicgen_v2')/'candidate_protocol.json')['planned_final_status']=='PARTIAL'
