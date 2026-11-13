"""Only the bounded MimicGen v2 component, sequential and stage resumable."""
import argparse
import shutil
import subprocess
import traceback
from baselines.strong.common import PROJECT,ROOT,PYTHON,TASKS,out,read,write,digest,resource,now

def run(args,name):
    with (ROOT/('mimicgen-v2-'+name+'.log')).open('a') as stream:subprocess.run([str(PYTHON),*args],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def probes():
    run(['-m','results.strong_baselines.mimicgen_v2.train','--probe'],'probe')
    for task in TASKS:
        run(['-m','results.strong_baselines.mimicgen_v2.evaluate','--method','mimicgen_v2','--group','human_plus_generated','--checkpoint',str(ROOT/'mimicgen_v2/probe/latest.pt'),'--task',task,'--smoke'],'smoke-'+task)
        run(['-c',f"from baselines.strong.audit import validate_job; validate_job('mimicgen_v2','human_plus_generated',{task!r},True)"],'smoke-acceptance-'+task)
    write(out('mimicgen_v2')/'probe_acceptance.json',dict(status='PASS',actual_optimizer_updates=1,full_physical_smoke_episodes=2,
        probe_weights_discarded=True,task_success_threshold_used=False,new_generation_attempts=0))

def freeze():
    folder=out('mimicgen_v2');assert read(folder/'probe_acceptance.json')['status']=='PASS' and read(folder/'test_status.json')['status']=='PASS'
    if (folder/'freeze.json').exists():assert digest(folder/'protocol.json')==read(folder/'freeze.json')['protocol_sha256'];return
    shutil.copyfile(folder/'candidate_protocol.json',folder/'protocol.json')
    files=list(folder.glob('*.py'))+[PROJECT/'baselines/strong/common.py',PROJECT/'baselines/strong/evaluate.py',PROJECT/'baselines/strong/audit.py']
    write(folder/'freeze.json',dict(utc=now(),protocol_sha256=digest(folder/'protocol.json'),server_root=str(ROOT),
        code_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sorted(files)},data_freeze=read(ROOT/'mimicgen_v2/data/complete.json'),
        original_mimicgen_status_sha256=digest(out('mimicgen')/'status.json'),resources=resource(),physical_success_used_for_selection=False))

def formal():
    folder=out('mimicgen_v2');assert (folder/'freeze.json').exists()
    status=read(folder/'status.json');status.update(status='PARTIAL',phase='FORMAL_TRAINING');write(folder/'status.json',status)
    run(['-m','results.strong_baselines.mimicgen_v2.train'],'training')
    selected=read(folder/'selection.json');assert not selected['used_physical_success']
    status.update(actual_training_updates=selected['actual_optimizer_updates'],phase='FORMAL_EVALUATION');write(folder/'status.json',status)
    for task in TASKS:
        run(['-m','results.strong_baselines.mimicgen_v2.evaluate','--method','mimicgen_v2','--group','human_plus_generated','--checkpoint',selected['checkpoint'],'--task',task],'evaluation-'+task)
        run(['-c',f"from baselines.strong.audit import validate_job; validate_job('mimicgen_v2','human_plus_generated',{task!r})"],'acceptance-'+task)
    run(['-m','results.strong_baselines.mimicgen_v2.report'],'report')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['probes','freeze','formal']);args=parser.parse_args()
    try:globals()[args.phase]()
    except BaseException as exc:
        write(out('mimicgen_v2')/'controller_error.json',dict(phase=args.phase,error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise

if __name__=='__main__':main()
