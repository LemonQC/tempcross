"""Finite stage supervisor; no automatic retries, retuning, or next method."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from common import OUT, ROOT, SOURCE, PROJECT, PYTHON, digest, read, write


def main():
    import fcntl
    lock=(ROOT/'execution.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'phase2_execution.json').exists():
        raise FileExistsError('Inspect PID/log/status; do not relaunch training or physics')
    assert read(OUT/'probe_validation.json')['status']=='PASS'
    marker=read(ROOT/'downstream_freeze.json')
    for path,sha in marker['sha256'].items():assert digest(path)==sha
    state=read(OUT/'status.json');assert state['status']=='IN_PROGRESS'
    for path,sha in state['input_sha256'].items():assert digest(path)==sha
    execution=dict(status='IN_PROGRESS',pid=os.getpid(),stages=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   freeze_commit=marker['remote_commit'])
    write(ROOT/'phase2_execution.json',execution)
    from baselines.common.paths import environment
    stages=[('augmentation',[str(PYTHON),str(OUT/'augment.py')],False),
            ('data_preparation',[str(PYTHON),'-c','from data import prepare; prepare()'],False),
            ('training',[str(PYTHON),str(OUT/'train.py')],True),
            ('evaluation',[str(PYTHON),str(OUT/'evaluate.py')],False),
            ('collection',[str(PYTHON),str(OUT/'report.py')],False)]
    try:
        for name,args,core in stages:
            state=read(OUT/'status.json');state['phase']=name;write(OUT/'status.json',state)
            execution['active_stage']=name;write(ROOT/'phase2_execution.json',execution)
            env=environment(core=core)
            env['PYTHONPATH']=str(OUT)+os.pathsep+env['PYTHONPATH']
            with (ROOT/(name+'.log')).open('x') as log:
                result=subprocess.run(args,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
            execution['stages'].append(dict(stage=name,returncode=result.returncode,
                log=str(ROOT/(name+'.log')),sha256=digest(ROOT/(name+'.log'))))
            write(ROOT/'phase2_execution.json',execution)
            if result.returncode:raise RuntimeError(f'{name} failed with return code {result.returncode}; no automatic repair/retry')
            state=read(OUT/'status.json')
            for path,sha in state['input_sha256'].items():assert digest(path)==sha
            if name not in state['completed_stages']:state['completed_stages'].append(name)
            write(OUT/'status.json',state)
        execution['status']='COMPLETE'
        state=read(OUT/'status.json');state.update(status='COMPLETE',phase='COMPLETE')
        training=read(ROOT/'downstream/training/status.json')
        state['actual_optimizer_updates']=training['actual_optimizer_updates']
        state['checkpoints']=[read(OUT/'selection.json')]
        state['actual_formal_evaluation_rollouts']=80
        state['actual_augmentation_rollouts']=read(ROOT/'augmentation/complete.json')['attempts']
        state['actual_new_physical_rollouts']=10+state['actual_augmentation_rollouts']+80
        write(OUT/'status.json',state)
    except BaseException as exc:
        execution.update(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc())
        state=read(OUT/'status.json');state.update(status='BLOCKED',reason=repr(exc))
        if (ROOT/'downstream/training/status.json').exists():
            training=read(ROOT/'downstream/training/status.json')
            state['actual_optimizer_updates']=training['actual_optimizer_updates']
        write(OUT/'status.json',state)
        env=environment(core=False);env['PYTHONPATH']=str(OUT)+os.pathsep+env['PYTHONPATH']
        with (ROOT/'blocked_report.log').open('x') as log:
            subprocess.run([str(PYTHON),str(OUT/'report.py')],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    finally:
        execution['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(ROOT/'phase2_execution.json',execution)


if __name__=='__main__':main()
