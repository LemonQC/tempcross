"""Last baseline only: finite native train ->80 trials ->accept ->stop."""
import fcntl
import os
import subprocess
import threading
from .data import RUN
from .audit import GROUP
from baselines.strong.common import ROOT,PROJECT,PYTHON,TASKS,out,read,write,digest,now

def command(args,path):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',USE_TF='0',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
    with path.open('a') as log:subprocess.run(args,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=read(out('flow_matching_policy')/'status.json')
        if status['status']=='COMPLETE_FAITHFUL_PORT':print('COMPLETE_NO_REPEAT');return
        assert read(PROJECT/'results/strong_baselines/research_scope.json')['current_round_only']=='flow_matching_policy'
        sha=digest(out('flow_matching_policy')/'protocol.json');assert read(out('flow_matching_policy')/'freeze.json')['protocol_sha256']==sha
        stop=threading.Event()
        def monitor():
            with (RUN/'resource_samples.csv').open('a') as stream:
                if stream.tell()==0:stream.write('utc,memory_used_MiB,utilization_percent,power_W\n')
                while not stop.is_set():
                    p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
                    stream.write(now()+','+p.stdout.strip()+'\n');stream.flush();stop.wait(5)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        progress=dict(phase='FLOW_MATCHING_FORMAL_TRAINING',pid=os.getpid(),active_method='flow_matching_policy',
            round_only='flow_matching_policy',protocol_sha256=sha,utc=now(),next='One frozen run,80 reference trials,accept,publish,STOP; no other baseline/Method')
        write(ROOT/'session_resume.json',progress);write(out('flow_matching_policy')/'status.json',dict(status,phase='FORMAL_TRAINING'))
        try:
            command([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.flow_matching_policy.train'],RUN/'formal-training.log')
            train=read(RUN/'training/source_target_visual/state.json');assert train['status']=='COMPLETE'
            command([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.flow_matching_policy.audit_optimizer'],RUN/'optimizer-audit.log')
            progress.update(phase='FLOW_MATCHING_FORMAL_EVALUATION',actual_training_updates=train['actual_training_updates'],utc=now())
            write(ROOT/'session_resume.json',progress)
            write(out('flow_matching_policy')/'status.json',dict(status,phase='FORMAL_EVALUATION',actual_training_updates=train['actual_training_updates']))
            for task,label in zip(TASKS,['counter','sink']):
                command([str(PYTHON),'-u','-m','results.strong_baselines.flow_matching_policy.evaluate',
                    '--method','flow_matching_policy','--group',GROUP,'--checkpoint',train['checkpoint'],'--task',task],RUN/f'formal-{label}.log')
            stop.set();thread.join(timeout=10)
            command([str(PYTHON),'-u','-m','results.strong_baselines.flow_matching_policy.results'],RUN/'acceptance.log')
            progress.update(phase='FLOW_MATCHING_ACCEPTED_AWAITING_GITHUB',formal_episodes=80,utc=now(),next='Publish verified evidence then STOP_WAIT_FOR_RESEARCHER; no next baseline/Method')
            write(ROOT/'session_resume.json',progress)
        except BaseException as e:
            progress.update(phase='FLOW_MATCHING_PIPELINE_ERROR',error=repr(e),utc=now())
            write(ROOT/'session_resume.json',progress);write(RUN/'pipeline_error.json',progress);raise
        finally:stop.set();thread.join(timeout=10)

if __name__=='__main__':main()
