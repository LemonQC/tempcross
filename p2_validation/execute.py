"""Execute the bounded B1 development batch and frozen four-cell formal suite."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/home/users/lemon/coft-p2/p2/physical-20261108')
SOURCE=Path('/home/users/lemon/coft-p2/barx')
ARTIFACTS=Path('/home/users/lemon/coft-p2/artifacts')
CHECKPOINTS={'joint_reps':ARTIFACTS/'runs/joint_reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt',
             'no_reps':ARTIFACTS/'runs/no_reps/checkpoints/step-050000-epoch-15-loss=0.8424.pt'}
DEVELOPMENT=[0,1,2,4,6,10,12,18]


def run(path,method,robot,indices,checkpoint,threshold=.9):
    path=Path(path)
    if (path/'job.json').exists():
        j=json.loads((path/'job.json').read_text())
        if j['status']=='COMPLETE':return j
        raise RuntimeError(f'Existing incomplete/failed execution needs explicit repair: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(SOURCE/'.venv/bin/python'),'-m','p2_validation.run_policy','--source',str(SOURCE),
         '--checkpoint',str(checkpoint),'--robot',robot,'--indices',','.join(map(str,indices)),
         '--method',method,'--threshold',str(threshold),'--out',str(path)]
    print(json.dumps({'starting':str(path),'command':cmd}),flush=True)
    with path.with_suffix('.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    if result.returncode:raise RuntimeError(f'Failed subprocess {path}, exit {result.returncode}')
    j=json.loads((path/'job.json').read_text());assert j['status']=='COMPLETE';return j


def main():
    for key,value in dict(HF_HOME=str(ARTIFACTS/'hf'),BARX_VQ_ROOT=str(ARTIFACTS/'vq'),HF_HUB_OFFLINE='1',
                         MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false').items():os.environ[key]=value
    # The first candidate was launched as the initial integration run. Never duplicate it.
    first=ROOT/'development/b1-085/job.json'
    while first.exists() and json.loads(first.read_text())['status']=='RUNNING':time.sleep(5)
    candidates=[]
    for threshold in [.85,.90,.93]:
        label=f'b1-{round(threshold*100):03d}'
        candidates.append(run(ROOT/'development'/label,'b1','panda',DEVELOPMENT,CHECKPOINTS['joint_reps'],threshold))
    best=sorted(candidates,key=lambda j:(-j['summary']['successes'],j['threshold']))[0]
    frozen=dict(status='FROZEN_BEFORE_FORMAL_TEST',development_indices=DEVELOPMENT,
                candidate_successes=[dict(threshold=j['threshold'],successes=j['summary']['successes'],episodes=j['summary']['episodes']) for j in candidates],
                selection_rule='maximum official development successes, ties choose the lowest threshold',
                b1=dict(threshold=best['threshold'],hysteresis=.03,minimum_hold=8),
                formal_indices=list(range(20,40)),global_seed=7,
                adaptation=dict(route='full-parameter actual step probe; LoRA only if necessary',
                    final_sample_positions_per_model=6000,full_parameter_batch_size=1,lora_batch_size=2,
                    full_parameter_final_steps=6000,lora_final_steps=3000,learning_rate=2e-5,
                    prior_sampling_weight=.05,target_sampling_weight=1.,
                    target_revision='6e297509924c831c722180212d13f21686676e8e',
                    source_revision='a2b5256a48428ea56064b0afb3dc184a045b10ae',
                    checkpoint_selection='final prescribed step; no test-based selection',
                    equivalence='restricted single-GPU adaptation, not equivalent to official global-batch-256 exposure'))
    frozen_path=ROOT/'frozen_protocol.json'
    if frozen_path.exists():assert json.loads(frozen_path.read_text())==frozen
    else:frozen_path.write_text(json.dumps(frozen,indent=2)+'\n')
    for method,robot,checkpoint in [('original','ur5e','joint_reps'),('original','panda','joint_reps'),
                                    ('original','panda','no_reps'),('b1','panda','joint_reps')]:
        run(ROOT/'formal'/f'{method}-{checkpoint}-{robot}',method,robot,list(range(20,40)),
            CHECKPOINTS[checkpoint],best['threshold'])
    (ROOT/'policy_suite_complete.json').write_text(json.dumps(dict(status='COMPLETE',development_episodes=24,
          formal_episodes=80,b2_formal_status='PENDING_DEVELOPMENT_VALIDATION'),indent=2))


if __name__=='__main__':main()
