"""Offline-only paths, hashes, and an import guard against simulator creation."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'method_v0/results'
ROOT=Path('/home/users/lemon/method-v0-tocat-20261112')
BASE='571b1b0cf66eb20495a5cc3a6cdbb0da9d25834e'
TASKS=['pnp_counter_to_sink','pnp_sink_to_counter']


class NoPhysics(importlib.abc.MetaPathFinder):
    forbidden={'mujoco','mujoco_py','robosuite','robocasa','mimicgen'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in self.forbidden:
            raise RuntimeError('Method v0 forbids simulator/generator import: '+fullname)


def guard():
    assert not (set(sys.modules)&NoPhysics.forbidden)
    sys.meta_path.insert(0,NoPhysics())


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def ndarray_digest(x):
    import numpy as np
    a=np.ascontiguousarray(x)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()
