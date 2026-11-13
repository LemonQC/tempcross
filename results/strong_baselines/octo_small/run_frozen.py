"""One bounded, sequential, restart-safe frozen train/evaluate/accept pipeline."""
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import threading
import time
from baselines.strong.common import ROOT,PROJECT,PYTHON,out,read,write,digest,TASKS,now

RUN=ROOT/'octo_small'

def command(args,log):
    with log.open('a') as stream:
        subprocess.run(args,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    with (RUN/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        current=read(out('octo_small')/'status.json')
        if current['status']=='COMPLETE_FAITHFUL_PORT':print('COMPLETE_NO_REPEAT');return
        cache=read(RUN/'data/pack_index.json');assert cache['status']=='PASS' and cache['completed']==640
        protocol=read(out('octo_small')/'protocol.json')
        assert read(out('octo_small')/'freeze.json')['protocol_sha256']==digest(out('octo_small')/'protocol.json')
        stopped=threading.Event()
        def monitor():
            with (RUN/'training_resources.csv').open('a') as stream:
                if stream.tell()==0:stream.write('utc,memory_used_MiB,utilization_percent,power_W\n')
                while not stopped.is_set():
                    r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True)
                    stream.write(now()+','+r.stdout.strip()+'\n');stream.flush();stopped.wait(5)
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        progress=dict(phase='OCTO_FORMAL_TRAINING',pid=os.getpid(),utc=now(),active_method='octo_small',
            formal_budget_frozen=True,formal_training_started=True,protocol_sha256=digest(out('octo_small')/'protocol.json'),
            no_other_model_training=True,next='Native training resumes automatically; evaluate80 frozen reference trials only after COMPLETE',
            last_verified_github='9b3aac451e169b6c90cbf14f1e2f5d950c6d9d66')
        write(ROOT/'session_resume.json',progress)
        write(out('octo_small')/'status.json',dict(current,phase='FORMAL_TRAINING'))
        try:
            command([str(RUN/'env/bin/python'),'-u','-m','results.strong_baselines.octo_small.launch','train'],RUN/'formal-training.log')
            train=read(RUN/'training/source_target/state.json')
            assert train['status']=='COMPLETE' and train['actual_training_updates']==protocol['training_updates']
            progress.update(phase='OCTO_FORMAL_EVALUATION',actual_formal_updates=train['actual_training_updates'])
            write(ROOT/'session_resume.json',progress)
            write(out('octo_small')/'status.json',dict(current,phase='FORMAL_EVALUATION',actual_training_updates=train['actual_training_updates']))
            for task,label in zip(TASKS,['counter','sink']):
                command([str(PYTHON),'-u','-m','baselines.strong.evaluate','--method','octo_small','--group','source_target',
                    '--checkpoint',train['checkpoint'],'--task',task],RUN/f'formal-{label}.log')
            command([str(PYTHON),'-u','-m','results.strong_baselines.octo_small.results'],RUN/'acceptance.log')
            progress.update(phase='OCTO_ACCEPTED_AWAITING_GITHUB_PUBLICATION',formal_episodes=80,next='Review actual results and publish current small files; do not start another model automatically')
            write(ROOT/'session_resume.json',progress)
        except BaseException as e:
            progress.update(phase='OCTO_PIPELINE_TECHNICAL_FAILURE',error=repr(e),utc=now())
            write(ROOT/'session_resume.json',progress)
            write(RUN/'pipeline_error.json',progress);raise
        finally:stopped.set();thread.join(timeout=10)

if __name__=='__main__':main()
