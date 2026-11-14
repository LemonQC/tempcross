"""CPU-only durable state. Never writes to the historical baseline run roots."""
import datetime
import hashlib
import json
import os
import pickle
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('CROSS_BEHAVIOR_ROOT', '/home/users/lemon/cross-behavior-20261114'))
BRANCH = 'cross-behavior-task-expansion'
BASE = '92a6d212353f5e209d63534825b4763afe6893fd'
SOURCE = Path('/home/users/lemon/coft-p2/barx')
PYTHON = SOURCE / '.venv/bin/python'
TASKS = ('push', 'stack', 'drawer')
ROBOTS = ('ur5e', 'panda')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, data):
    atomic(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    atomic(path, pickle.dumps(data, protocol=5))


def environment():
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(PROJECT), str(SOURCE), str(SOURCE / 'policy')]),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1')
    return env


def episode_path(task, robot, condition):
    return ROOT / 'stage_a' / task / robot / f'dev-{condition:03d}'


def condition_seed(task, condition):
    # Separate new pool. Robot-independent condition IDs; robot-specific XML/state hashes.
    return 14180000 + TASKS.index(task) * 10000 + condition


def recover_step(directory):
    """Resolve a committed rename before pointer update; never replay an uncertain step."""
    directory = Path(directory)
    pointer = directory / 'cursor.json'
    step = read(pointer)['step'] if pointer.exists() else 0
    intent = directory / 'inflight.json'
    if intent.exists():
        pending = read(intent)['next_step']
        committed = directory / 'steps' / f'{pending:06d}.pkl'
        if committed.exists():
            assert pending in (step, step + 1)
            step = max(step, pending)
            write(pointer, {'step': step})
            intent.unlink()
        else:
            raise RuntimeError('AMBIGUOUS_INFLIGHT_PHYSICS: retain evidence; no automatic step replay')
    return step
