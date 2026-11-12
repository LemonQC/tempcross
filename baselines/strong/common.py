import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path('/home/users/lemon/strong-baselines-20261112')
OLD=Path('/home/users/lemon/baseline-benchmark/run-20261109')
SOURCE=Path('/home/users/lemon/coft-p2/barx')
PYTHON=SOURCE/'.venv/bin/python'
TASKS=['pnp_counter_to_sink','pnp_sink_to_counter']
METHODS=['act','vqbet','mimicgen_v2','kite','crosszero','object_flow']

def read(p):return json.loads(Path(p).read_text())
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def out(method):return PROJECT/'results/strong_baselines'/method
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def resource():
    import shutil
    p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return dict(utc=now(),gpu=p.stdout.strip(),disk_free_bytes=shutil.disk_usage(ROOT).free)
def setup():
    from baselines.common.paths import environment
    os.environ.update(environment());os.environ['USE_TF']='0'
    import sys
    sys.path[:0]=[str(PROJECT),str(SOURCE),str(SOURCE/'policy')]
def initialize():
    ROOT.mkdir(parents=True,exist_ok=True)
    for method in METHODS:
        p=out(method)/'status.json'
        if not p.exists():write(p,dict(status='NOT_RUN',method=method,actual_training_updates=0,physical_evaluation_episodes=0,success_rate=None,technical_repairs=0))
    frozen=ROOT/'preservation_freeze.json'
    if not frozen.exists():
        paths=[PROJECT/'results/baselines/analysis.json',PROJECT/'results/strong_baselines/barx_adaptation/results.json',
               PROJECT/'results/strong_baselines/mimicgen/status.json',OLD/'physical_launches.jsonl',
               Path('/home/users/lemon/barx-target-adaptation-20261110/physical/physical_launches.jsonl')]
        paths += [PROJECT/'configs/datasets'/('baseline_'+d+'_manifest.json') for d in ['source','target']]
        paths += [PROJECT/'configs/evaluation/baseline_conditions_60_99.json']
        write(frozen,dict(utc=now(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),files=[dict(path=str(p),sha256=digest(p)) for p in paths]))
    return read(frozen)

if __name__=='__main__':initialize();print(json.dumps(resource()))
