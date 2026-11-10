"""Finite, single-model execution; training -> selection -> one physical suite."""
import subprocess
import time
from common import HERE, OUT, PROJECT, PYTHON, ROOT, digest, environment, read, write


def main():
    assert read(ROOT/'frozen_publication.json')['github_readback_verified']
    ledger=ROOT/'execution.json'
    if ledger.exists():raise FileExistsError('Preserve prior execution; no implicit retraining: '+str(ledger))
    protected=[PROJECT/'results/baselines/analysis.json',
               __import__('pathlib').Path('/home/users/lemon/baseline-benchmark/run-20261109/physical_launches.jsonl'),
               __import__('pathlib').Path('/home/users/lemon/coft-p3/p3-20261109/physical_launches.jsonl')]
    original={str(p):digest(p) for p in protected}
    state=dict(status='RUNNING',started=time.time(),stages=[],protected_sha256=original)
    write(ledger,state)
    stages=[('train',[str(PYTHON),'-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','1',str(HERE/'train.py')]),
            ('merge',[str(PYTHON),str(HERE/'merge.py')]),
            ('physical_evaluation',[str(PYTHON),str(HERE/'evaluate.py')]),
            ('old_offline_validation',[str(PYTHON),str(HERE/'baseline_validation.py')]),
            ('collect',[str(PYTHON),str(HERE/'collect.py')])]
    for name,args in stages:
        state['active_stage']=name;write(ledger,state)
        log=ROOT/f'{name}.log'
        with log.open('x') as f:
            result=subprocess.run(args,cwd=PROJECT,env=environment(),stdout=f,stderr=subprocess.STDOUT)
        state['stages'].append(dict(name=name,returncode=result.returncode,log=str(log),log_sha256=digest(log)))
        if result.returncode:
            state.update(status='TECHNICAL_FAILURE',ended=time.time());write(ledger,state);return
        for p,h in original.items():assert digest(p)==h,'Protected old data changed: '+p
        write(ledger,state)
    state.update(status='COMPLETE',ended=time.time(),old_inputs_unchanged=True);write(ledger,state);write(OUT/'execution.json',state)


if __name__=='__main__':main()
