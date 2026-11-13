"""One native model, frozen train ->80 physical trials ->accept ->stop."""
import fcntl
import os
import subprocess
import threading
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,PYTHON,out,read,write,digest,TASKS,now

RUN=ROOT/'smolvla'

def command(args,path):
    environment=dict(os.environ,HF_HOME=str(RUN/'huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        USE_TF='0',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
    with path.open('a') as log:subprocess.run(args,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=read(out('smolvla')/'status.json')
        if current['status']=='COMPLETE_FAITHFUL_PORT':print('COMPLETE_NO_REPEAT');return
        scope=read(PROJECT/'results/strong_baselines/research_scope.json');assert scope['current_round_only']=='smolvla'
        protocol=read(out('smolvla')/'protocol.json');assert read(out('smolvla')/'freeze.json')['protocol_sha256']==digest(out('smolvla')/'protocol.json')
        stopped=threading.Event()
        def monitor():
            with (RUN/'resource_samples.csv').open('a') as stream:
                if stream.tell()==0:stream.write('utc,memory_used_MiB,utilization_percent,power_W\n')
                while not stopped.is_set():
                    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True)
                    stream.write(now()+','+r.stdout.strip()+'\n');stream.flush();stopped.wait(5)
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
        progress=dict(phase='SMOLVLA_FORMAL_TRAINING',pid=os.getpid(),active_method='smolvla',round_only='smolvla',
            formal_budget_frozen=True,protocol_sha256=digest(out('smolvla')/'protocol.json'),
            next='Finish unique frozen model,80 reference episodes,accept,publish,stop; never launch next method',utc=now())
        write(ROOT/'session_resume.json',progress);write(out('smolvla')/'status.json',dict(current,phase='FORMAL_TRAINING'))
        try:
            command([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.smolvla.train'],RUN/'formal-training.log')
            train=read(RUN/'training/source_target_state/state.json')
            assert train['status']=='COMPLETE' and train['actual_training_updates']==protocol['training_updates']
            command([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.smolvla.audit_optimizer'],RUN/'optimizer-audit.log')
            progress.update(phase='SMOLVLA_FORMAL_EVALUATION',actual_training_updates=train['actual_training_updates'])
            write(ROOT/'session_resume.json',progress)
            write(out('smolvla')/'status.json',dict(current,phase='FORMAL_EVALUATION',actual_training_updates=train['actual_training_updates']))
            for task,label in zip(TASKS,['counter','sink']):
                command([str(PYTHON),'-u','-m','results.strong_baselines.smolvla.evaluate','--method','smolvla',
                    '--group','source_target_state','--checkpoint',train['checkpoint'],'--task',task],RUN/f'formal-{label}.log')
            stopped.set();monitor_thread.join(timeout=10)
            command([str(PYTHON),'-u','-m','results.strong_baselines.smolvla.results'],RUN/'acceptance.log')
            progress.update(phase='SMOLVLA_ACCEPTED_AWAITING_GITHUB',formal_episodes=80,
                next='Review and publish actual results,stop; no Flow Matching Policy or other method')
            write(ROOT/'session_resume.json',progress)
        except BaseException as e:
            progress.update(phase='SMOLVLA_PIPELINE_ERROR',error=repr(e),utc=now())
            write(ROOT/'session_resume.json',progress);write(RUN/'pipeline_error.json',progress);raise
        finally:stopped.set();monitor_thread.join(timeout=10)

if __name__=='__main__':main()
