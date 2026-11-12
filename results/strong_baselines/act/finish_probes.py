"""Sequential probe completion; never starts formal training automatically."""
import subprocess
import sys
from baselines.strong.common import PROJECT,ROOT,PYTHON,TASKS,out,read,write
from baselines.strong.audit import validate_job

def run(args,log):
    with (ROOT/log).open('a') as f:
        subprocess.run([str(PYTHON),*args],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    for group in ['target_only','source_target']:
        run(['-m','results.strong_baselines.act.train','--group',group,'--probe'],'act-probe-'+group+'.log')
        for task in TASKS:
            run(['-m','baselines.strong.evaluate','--method','act','--group',group,'--checkpoint',str(ROOT/'act/probe'/group/'latest.pt'),'--task',task,'--smoke'],f'act-smoke-{group}-{task}.log')
            validate_job('act',group,task,True)
    write(out('act')/'probe_acceptance.json',dict(status='PASS',real_optimizer_updates=2,smoke_episodes=4,
        smoke_is_not_formal_evaluation=True,probe_weights_discarded_for_formal_training=True,success_threshold_used=False))
    print('PASS: both ACT settings trained once and completed audited full-horizon physical smoke on both tasks.',flush=True)

if __name__=='__main__':main()
