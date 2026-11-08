"""Wait for verified data, train matched restricted adapters, evaluate physically."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/users/lemon/coft-p2/p2/physical-20261108')
SOURCE=Path('/home/users/lemon/coft-p2/barx')
PROJECT=Path('/home/users/lemon/CodexProject/tempcross')
PYTHON=SOURCE/'.venv/bin/python'


def call(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(dict(command=list(map(str,command)),log=str(log))),flush=True)
    with log.open('w') as stream:
        result=subprocess.run(list(map(str,command)),cwd=PROJECT,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT)
    return result.returncode


def train(method,route,steps,batch,label,cache):
    from p2_validation.execute import CHECKPOINTS
    directory=ROOT/'training'/label
    command=[PYTHON,'-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','1',
       '-m','p2_validation.train_adapter','--source',SOURCE,'--checkpoint',CHECKPOINTS[method],
       '--cache',cache,'--out',directory,'--method',method,'--route',route,'--steps',steps,'--batch-size',batch]
    code=call(command,ROOT/'training'/f'{label}.log')
    return code,directory


def merge(method,adapter_directory,destination,cache):
    from p2_validation.execute import CHECKPOINTS
    destination.mkdir(parents=True,exist_ok=False)
    script=PROJECT/'p2_validation/merge_adapter.py'
    command=[PYTHON,script,'--source',SOURCE,'--checkpoint',CHECKPOINTS[method],
             '--adapter',adapter_directory,'--out',destination,'--statistics',cache/'dataset_statistics.json']
    code=call(command,destination.parent/f'merge-{method}.log')
    if code:raise RuntimeError(f'Adapter merge failed {method}')
    return destination/'checkpoints/step-003000-epoch-00-loss=restricted.pt'


def main():
    os.environ.update(HF_HOME='/home/users/lemon/coft-p2/artifacts/hf',BARX_VQ_ROOT='/home/users/lemon/coft-p2/artifacts/vq',
        HF_HUB_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(PROJECT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    state=dict(status='WAITING_FOR_VERIFIED_DATA',completed_models=[],errors=[])
    ledger=ROOT/'adaptation_suite.json'
    def save():ledger.write_text(json.dumps(state,indent=2)+'\n')
    save();data=ROOT.parent/'data';cache=ROOT.parent/'matched-samples-6000'
    while not (data/'complete.json').exists():time.sleep(10)
    state['status']='PREPARING_MATCHED_SAMPLES';save()
    code=call([PYTHON,'-m','p2_validation.prepare_training','--source',SOURCE,'--data',data,'--out',cache,'--samples','6000'],ROOT/'prepare-training.log')
    if code:
        state.update(status='DATA_PREPARATION_FAILED');save();return
    state['status']='WAITING_FOR_POLICY_GPU_RELEASE';save()
    while not (ROOT/'policy_suite_complete.json').exists():time.sleep(10)
    state['status']='FULL_PARAMETER_STEP_PROBE';save()
    code,probe=train('joint_reps','full',1,1,'full-step-probe',cache)
    route='full'
    if code:
        status=json.loads((probe/'training_status.json').read_text()) if (probe/'training_status.json').exists() else {}
        state['full_probe']=status
        if not status.get('is_cuda_oom'):
            state.update(status='TRAINING_TECHNICAL_FAILURE',errors=[dict(log=str(ROOT/'training/full-step-probe.log'),status=status)]);save();return
        route='lora'
        code,lora_probe=train('joint_reps','lora',1,2,'lora-step-probe',cache)
        if code:
            state.update(status='LORA_TECHNICAL_FAILURE',errors=[dict(log=str(ROOT/'training/lora-step-probe.log'))]);save();return
    state.update(status='TRAINING_RESTRICTED_ADAPTATIONS',route=route);save()
    steps,batch=(3000,2) if route=='lora' else (6000,1)
    for method in ['no_reps','joint_reps']:
        code,directory=train(method,route,steps,batch,f'{method}-{route}',cache)
        if code:
            state['errors'].append(dict(method=method,log=str(ROOT/'training'/f'{method}-{route}.log')));save();continue
        if route=='lora':
            adapters=sorted((directory/'runs').rglob(f'step-{steps:06d}-*'))
            adapters=[p for p in adapters if p.is_dir() and (p/'adapter_config.json').exists()]
            if len(adapters)!=1:raise RuntimeError('Ambiguous prescribed final adapter')
            checkpoint=merge(method,adapters[0],ROOT/'adapted-checkpoints'/method,cache)
        else:
            checkpoints=list((directory/'runs').rglob(f'step-{steps:06d}-*.pt'))
            if len(checkpoints)!=1:raise RuntimeError('Ambiguous prescribed final checkpoint')
            checkpoint=checkpoints[0]
        state['completed_models'].append(dict(method=method,steps=steps,sample_positions=steps*batch,checkpoint=str(checkpoint)))
        save()
        from p2_validation.execute import run
        run(ROOT/'formal'/f'adapted-{method}-panda','original','panda',list(range(20,40)),checkpoint)
    state['status']='COMPLETE' if len(state['completed_models'])==2 and not state['errors'] else 'PARTIAL';save()


if __name__=='__main__':main()
