import csv,gzip,hashlib,json,pickle,sys
from pathlib import Path
import pytest
from p3_validation import train_runtime


def fixture_cache(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();rows=[]
    for i,frame in enumerate([0,1,1]):
        sample=dict(dataset_name=b'panda_pnp',p3_episode_id=b'example-demo',p3_frame_index=frame,p3_file_path=b'published/file.hdf5',p3_episode_length=2)
        raw=pickle.dumps(sample,protocol=4)
        with gzip.open(cache/f'sample-{i:06d}.pkl.gz','wb') as f:f.write(raw)
        rows.append(dict(index=i,dataset='panda_pnp',episode_id='example-demo',original_frame_index=frame,raw_sample_sha256=hashlib.sha256(raw).hexdigest()))
    with (cache/'sample_trace.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (cache/'complete.json').write_text(json.dumps(dict(samples=3)))
    return cache


def test_actual_consumption_preserves_exact_sequence_and_counts_unique_frames(tmp_path,monkeypatch):
    monkeypatch.setattr(train_runtime,'ROOT',tmp_path/'execution');monkeypatch.setattr(sys,'argv',['training','--out','matching-run'])
    cache=fixture_cache(tmp_path);stream=train_runtime.AuditedStream(cache)
    try:assert len(list(stream.as_numpy_iterator()))==3
    finally:stream.close()
    state=json.loads((stream.directory/'input_status.json').read_text())
    assert state['samples_consumed']==3 and state['unique_target_demonstrations']==1
    assert stream.frames[('panda_pnp','example-demo')]=={0,1}
    assert state['consumed_trace_sha256']==hashlib.sha256((cache/'sample_trace.csv').read_bytes()).hexdigest()


def test_corrupt_sample_is_rejected_before_counting_training_exposure(tmp_path,monkeypatch):
    monkeypatch.setattr(train_runtime,'ROOT',tmp_path/'execution');monkeypatch.setattr(sys,'argv',['training','--out','corrupt-run'])
    cache=fixture_cache(tmp_path)
    with gzip.open(cache/'sample-000000.pkl.gz','wb') as f:f.write(b'corrupt payload')
    stream=train_runtime.AuditedStream(cache)
    try:
        with pytest.raises(AssertionError):next(stream.as_numpy_iterator())
        assert stream.n==0
    finally:stream.close()


def test_input_tracking_survives_upstream_argv_rewrite(tmp_path,monkeypatch):
    monkeypatch.setattr(train_runtime,'ROOT',tmp_path/'execution')
    monkeypatch.setattr(train_runtime,'ACTIVE_RUN_LABEL','frozen-run')
    monkeypatch.setattr(sys,'argv',['official-train.py','--run_root_dir','other-path'])
    stream=train_runtime.AuditedStream(fixture_cache(tmp_path))
    try:
        assert stream.directory.name=='frozen-run'
        assert len(list(stream.as_numpy_iterator()))==3
    finally:stream.close()


def test_coverage_export_retains_target_demo_identity_after_source_row(tmp_path):
    from p3_validation.prepare_data import write_csv
    path=tmp_path/'coverage.csv'
    write_csv(path,[dict(dataset='source',episode_id='source-content-id'),
                    dict(dataset='target',episode_id='target-content-id',original_demo='demo_49',direction='CounterToSink')])
    with path.open() as f:rows=list(csv.DictReader(f))
    assert rows[0]['original_demo']==''
    assert rows[1]['original_demo']=='demo_49' and rows[1]['direction']=='CounterToSink'
