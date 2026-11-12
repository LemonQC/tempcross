"""Temporal causality, identity isolation and canonical action alignment."""
import numpy as np
from baselines.strong.common import ROOT,read
from results.strong_baselines.vqbet.data import Dataset

def test_causal_window_and_exact_action_alignment():
    d=Dataset('source_target',limit=8)
    try:
        for i in range(len(d)):
            b=d[i];r=next(j for j,x in enumerate(d.rows) if x['episode_id']==b['episode_id']);t=b['frame']
            full=np.load(d.directory/'features'/f'{r:04d}.npy',mmap_mode='r')
            assert np.array_equal(b['features'],full[t-9:t+1])
            assert np.array_equal(b['actions'],d.actions[r][t-9:t+1])
            assert b['features'].shape==(10,512) and b['actions'].shape==(10,7)
    finally:d.close()

def test_validation_identity_isolation_and_target_domain():
    train=Dataset('source_target',limit=1);valid=Dataset('source_target',split='validation')
    try:
        ids={r['episode_id'] for r in train.rows if r['split']=='train'}
        for i in range(len(valid)):
            b=valid[i];assert b['episode_id'] not in ids and b['domain']=='target'
    finally:train.close();valid.close()

def test_residual_codebook_and_predictor_unchanged():
    from baselines.strong.common import out
    files=read(out('vqbet')/'official_source.json')['files']
    assert not files['vq_behavior_transformer/bet.py']['identical']
    assert all(v['identical'] for k,v in files.items() if k!='vq_behavior_transformer/bet.py')

def test_wrong_attempt_root_refused_before_model_allocation(monkeypatch):
    import pytest
    from pathlib import Path
    from results.strong_baselines.vqbet import model
    monkeypatch.setattr(model,'ROOT',Path('/invalid/attempt'))
    with pytest.raises(AssertionError,match='corrected'):
        model.imports()
