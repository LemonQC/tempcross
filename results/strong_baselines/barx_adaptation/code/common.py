"""Paths and immutable-input helpers for the BARX target-adaptation baseline."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PROJECT = HERE.parents[3]
SOURCE = Path('/home/users/lemon/coft-p2/barx')
PYTHON = SOURCE / '.venv/bin/python'
ROOT = Path('/home/users/lemon/barx-target-adaptation-20261110')
DATA = Path('/home/users/lemon/coft-p2/p2/data')
CHECKPOINT = Path('/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt')
OLD_STATS = Path('/home/users/lemon/coft-p3/matched-samples-96000/dataset_statistics.json')
TASKS = ['pnp_counter_to_sink', 'pnp_sink_to_counter']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x:x.tolist()) + '\n')
    tmp.replace(path)


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(PROJECT), str(HERE), str(SOURCE), str(SOURCE / 'policy')]),
               HF_HOME='/home/users/lemon/coft-p2/artifacts/hf',
               BARX_VQ_ROOT='/home/users/lemon/coft-p2/artifacts/vq', HF_HUB_OFFLINE='1',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', TF_CPP_MIN_LOG_LEVEL='3')
    return env


def setup_source():
    import sys
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'policy'), str(PROJECT)]


def manifests():
    rows = []
    for domain in ['source', 'target']:
        rows.extend(read(PROJECT / f'configs/datasets/baseline_{domain}_manifest.json'))
    return rows
