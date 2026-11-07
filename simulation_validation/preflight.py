"""Inspect fixed official protocol and hardware without importing a policy."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def inspect(source):
    source=Path(source).resolve()
    spec=importlib.util.spec_from_file_location('barx_benchmark',(source/'barx/benchmark.py'))
    # dataclasses inspect the registered module.
    import sys
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    assert mod.TASKS['pnp_counter_to_sink'].environment=='XPnPCounterToSink'
    assert mod.TASKS['pnp_counter_to_sink'].max_steps==600
    assert mod.ACTION_HORIZON==8
    files=[];conditions={}
    for arm in ('ur5e','panda'):
        paths=sorted((source/'evaluation/conditions/pnp_counter_to_sink').glob(arm+'.*'))
        assert paths, f'Official conditions missing for {arm}'
        for p in paths: files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
        p=next(p for p in paths if p.suffix=='.json');payload=json.loads(p.read_text())
        assert len(payload['episodes'])>=20
        conditions[arm]=dict(robot=mod.EMBODIMENTS[arm].robot,gripper=mod.EMBODIMENTS[arm].gripper,camera=mod.EMBODIMENTS[arm].camera,
                             first20=payload['episodes'][:20],provenance=payload.get('provenance',{}))
    output=dict(source=str(source),source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                conditions=conditions,files=files,hardware=subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
                paired_across_methods=True,strictly_paired_across_robots=False)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=inspect(a.source);Path(a.out).write_text(json.dumps(result,indent=2));print('Official protocol and first20 conditions verified')
