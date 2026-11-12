"""Run only ACT, sequentially and resumably; no following method is launched."""
import subprocess
import traceback
from baselines.strong.common import ROOT,PROJECT,PYTHON,TASKS,out,read,write

def run(args,log):
    with (ROOT/log).open('a') as f:subprocess.run([str(PYTHON),*args],cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    cfg=read(out('act')/'freeze.json')
    status=read(out('act')/'status.json');status.update(status='PARTIAL',phase='FORMAL_TRAINING',probe_optimizer_updates=2,physical_smoke_episodes=4)
    write(out('act')/'status.json',status)
    try:
        for group in ['target_only','source_target']:
            run(['-m','results.strong_baselines.act.train','--group',group],'act-training-'+group+'.log')
            selected=read(out('act')/(group+'_selection.json'));assert not selected['used_physical_success']
            status['phase']='FORMAL_EVALUATION_'+group;write(out('act')/'status.json',status)
            for task in TASKS:
                run(['-m','baselines.strong.evaluate','--method','act','--group',group,'--checkpoint',selected['checkpoint'],'--task',task],f'act-evaluation-{group}-{task}.log')
                # Validate saved files in the child interpreter without a second policy/GPU process.
                code=f"from baselines.strong.audit import validate_job; validate_job('act',{group!r},{task!r})"
                run(['-c',code],f'act-acceptance-{group}-{task}.log')
        run(['-m','results.strong_baselines.act.report'],'act-report.log')
    except BaseException as e:
        status.update(status='PARTIAL',phase='PRESERVED_TECHNICAL_ATTEMPT',error=repr(e),traceback=traceback.format_exc())
        write(out('act')/'status.json',status);raise

if __name__=='__main__':main()
