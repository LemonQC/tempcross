"""Finite memory probes, 96000-sample upstream LoRA, final-only evaluation."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time
from p3_validation.common import ROOT,SOURCE,PYTHON,PROJECT,CACHE,CHECKPOINT,environment
from p3_validation.evaluate import validate_freeze


def read(path):
    for _ in range(5):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:time.sleep(.1)
    raise ValueError('Unstable JSON '+str(path))


def command(args,log):
    with Path(log).open('w') as stream:
        return subprocess.run(list(map(str,args)),cwd=PROJECT,env=environment(),stdout=stream,stderr=subprocess.STDOUT).returncode


def training(label,batch,steps):
    directory=ROOT/'training'/label;directory.parent.mkdir(parents=True,exist_ok=True)
    if (directory/'training_status.json').exists():
        d=read(directory/'training_status.json')
        if d['status']=='COMPLETE' and d['completed_optimizer_steps']==steps and d['batch_size']==batch:return 0,directory
        if d['status']=='FAILED' and d.get('is_cuda_oom'):return 1,directory
        raise RuntimeError('Preserve/diagnose incomplete training rather than overwrite: '+str(directory))
    args=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','1','-m','p3_validation.train_runtime',
          '--source',SOURCE,'--checkpoint',CHECKPOINT,'--cache',CACHE,'--out',directory,'--method','joint_reps','--route','lora','--steps',steps,'--batch-size',batch]
    return command(args,ROOT/'training'/f'{label}.log'),directory


def merge(adapter,steps):
    args=[PYTHON,'-m','p3_validation.merge','--adapter',adapter,'--steps',steps]
    code=command(args,ROOT/'merge-adapter.log')
    if code:raise RuntimeError('Inspect '+str(ROOT/'merge-adapter.log'))
    return read(ROOT/'adapted/adapter_provenance.json')['merged_checkpoint']


def main():
    c,marker=validate_freeze();ROOT.mkdir(exist_ok=True,parents=True)
    audit_path=ROOT/'target_data_audit_40_59.json'
    if not audit_path.exists():
        from p2_validation.audit_data import audit
        from p3_validation.common import DATA
        audit(SOURCE,DATA,audit_path,indices=c['independent_indices'])
    audit_result=read(audit_path)
    assert audit_result['status']=='COMPLETE' and audit_result['frozen_test_indices']==[40,59]
    assert not audit_result['exact_physical_state_matches'],'Detected training/evaluation overlap requires resolution'
    state=dict(status='WAITING_FOR_CACHE_AND_GPU_RELEASE',frozen_commit=marker['remote_commit'],probes=[],training_attempts=[])
    ledger=ROOT/'training_suite.json'
    def save():
        state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp=ledger.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ledger)
    save()
    while True:
        data=(CACHE/'complete.json').exists();base=(ROOT/'baseline_suite_complete.json').exists()
        diagnostic=ROOT/'diagnostics/results.json'
        diag=read(diagnostic)['status'] if diagnostic.exists() else None
        if data and base and diag in ['COMPLETE','TECHNICAL_FAILURE']:break
        time.sleep(15)
    time.sleep(3) # allow terminating renderer contexts to release the GPU
    state['status']='ACTUAL_MEMORY_PROBES';save()
    successful=None
    for batch in c['training']['batch_probe_candidates']:
        code,directory=training(f'probe-b{batch}',batch,1)
        status=read(directory/'training_status.json') if (directory/'training_status.json').exists() else {}
        state['probes'].append(dict(batch=batch,returncode=code,training=status));save()
        if code:
            if not status.get('is_cuda_oom'):state['status']='PROBE_TECHNICAL_FAILURE';save();return
            continue
        steps=c['training']['sample_positions']//batch
        state.update(status='TRAINING_96000',selected_batch=batch,prescribed_steps=steps);save()
        code,directory=training(f'joint-reps-96000-b{batch}',batch,steps)
        status=read(directory/'training_status.json') if (directory/'training_status.json').exists() else {}
        state['training_attempts'].append(dict(batch=batch,returncode=code,training=status));save()
        if not code:successful=(directory,batch,steps);break
        # A real-training OOM is also a resource failure. Fall back only along
        # the already frozen 8/4/2 hierarchy, from the original checkpoint;
        # failed-attempt exposure is separately reported and never hidden.
        if not status.get('is_cuda_oom'):state['status']='TRAINING_TECHNICAL_FAILURE';save();return
    if successful is None:state['status']='ALL_RESOURCE_CANDIDATES_FAILED';save();return
    directory,batch,steps=successful
    adapters=[p for p in (directory/'runs').rglob(f'step-{steps:06d}-*') if p.is_dir() and (p/'adapter_config.json').exists()]
    if len(adapters)!=1:raise ValueError('Ambiguous final prescribed adapter')
    state.update(status='MERGING_AND_VERIFYING_FINAL_PARAMETERS',final_adapter=str(adapters[0]));save()
    checkpoint=merge(adapters[0],steps)
    state.update(status='EVALUATING_FINAL_CHECKPOINT',final_checkpoint=checkpoint);save()
    from p3_validation.evaluate import cell
    result=cell('p3-96000-joint_reps-panda',checkpoint)
    state.update(status='COMPLETE',final_evaluation=dict(episodes=result['summary']['episodes'],successes=result['summary']['successes']),actual_training_steps=steps,actual_sample_positions=96000);save()


if __name__=='__main__':main()
