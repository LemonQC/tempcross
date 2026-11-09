import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/users/lemon/coft-p2/barx')
PYTHON=SOURCE/'.venv/bin/python'
ROOT=Path('/home/users/lemon/coft-p3/p3-20261109')
DATA=Path('/home/users/lemon/coft-p2/p2/data')
CACHE=Path('/home/users/lemon/coft-p3/matched-samples-96000')
P2=Path('/home/users/lemon/coft-p2/p2/physical-20261108')
CHECKPOINT=Path('/home/users/lemon/coft-p2/artifacts/runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt')
OLD_ADAPTED=P2/'adapted-checkpoints/joint_reps/checkpoints/step-003000-epoch-00-loss=restricted.pt'


def environment():
    env=os.environ.copy()
    env.update(HF_HOME='/home/users/lemon/coft-p2/artifacts/hf',BARX_VQ_ROOT='/home/users/lemon/coft-p2/artifacts/vq',
        HF_HUB_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(PROJECT)+os.pathsep+env.get('PYTHONPATH',''),TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='4')
    return env
