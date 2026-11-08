import pytest
import json
from simulation_validation.gate import complete_rollout

def test_zero_episode_technical_failure_is_not_valid_rollout(tmp_path):
    run=tmp_path/'official'/'run';run.mkdir(parents=True)
    (run/'summary.json').write_text(json.dumps(dict(status='failed',episodes=0,successes=0)))
    assert not complete_rollout(tmp_path,1)

@pytest.mark.parametrize('nested',[False,True])
def test_gate_requires_video_and_full_episode_not_success(tmp_path,monkeypatch,nested):
    run=tmp_path/'official'
    if nested:run=run/'run'
    run.mkdir(parents=True)
    (run/'summary.json').write_text(json.dumps(dict(status='complete',episodes=1,successes=0)))
    (run/'episodes.jsonl').write_text(json.dumps(dict(success=False,steps=600))+'\n')
    assert not complete_rollout(tmp_path,1)
    (run/'rollout.mp4').write_bytes(b'fixture placeholder, gate only tests indexing')
    import sys
    from types import SimpleNamespace
    class Decoder:
        def __init__(self,path):pass
        def read(self):return True,object()
        def release(self):pass
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(VideoCapture=Decoder))
    assert complete_rollout(tmp_path,1)
    assert not complete_rollout(tmp_path,20)


def test_existing_run_is_never_overwritten(tmp_path):
    from simulation_validation.gate import prepare_run_directory
    import pytest
    sentinel=tmp_path/'gate.json';sentinel.write_text('{"status":"P0_BLOCKED"}')
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):
        prepare_run_directory(tmp_path)
    assert sentinel.read_text()=='{"status":"P0_BLOCKED"}'
