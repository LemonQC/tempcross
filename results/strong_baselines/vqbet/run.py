"""Only VQ-BeT: probe, freeze, train, accept; subsequent algorithms not launched."""
import argparse
import shutil
import subprocess
import traceback
from baselines.strong.common import PROJECT,ROOT,PYTHON,TASKS,out,read,write,digest,resource,now

def run(args,log):
    path=ROOT/('vqbet-'+log+'.log')
    with path.open('a') as stream:subprocess.run([str(PYTHON),*args],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def probes():
    for group in ['target_only','source_target']:
        for stage in ['vqvae','bet']:run(['-m','results.strong_baselines.vqbet.train','--group',group,'--stage',stage,'--probe'],f'probe-{group}-{stage}')
        for task in TASKS:
            run(['-m','baselines.strong.evaluate','--method','vqbet','--group',group,'--checkpoint',str(ROOT/'vqbet/probe'/group/'bet/latest.pt'),'--task',task,'--smoke'],f'smoke-{group}-{task}')
            run(['-c',f"from baselines.strong.audit import validate_job; validate_job('vqbet',{group!r},{task!r},True)"],f'smoke-acceptance-{group}-{task}')
    write(out('vqbet')/'probe_acceptance.json',dict(status='PASS',real_stage_updates=4,real_optimizer_calls=6,physical_smoke_episodes=4,
        probe_weights_discarded_for_formal=True,success_threshold_used=False))

def freeze():
    folder=out('vqbet');assert read(folder/'probe_acceptance.json')['status']=='PASS'
    assert read(folder/'input_acceptance.json')['status']=='PASS'
    assert read(folder/'test_status.json')['status']=='PASS'
    if (folder/'freeze.json').exists():assert digest(folder/'protocol.json')==read(folder/'freeze.json')['protocol_sha256'];return
    shutil.copyfile(folder/'candidate_protocol.json',folder/'protocol.json')
    write(folder/'freeze.json',dict(utc=now(),protocol_sha256=digest(folder/'protocol.json'),code_sha256={p.name:digest(p) for p in sorted(folder.glob('*.py'))},
        shared_code_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sorted((PROJECT/'baselines/strong').glob('*.py')) if p.name not in ['matrix.py','matrix_details.py','publication_manifest.py']},
        data_sha256=digest(ROOT/'vqbet/data/complete.json'),source_sha256=digest(folder/'official_source.json'),resources=resource(),used_physical_success_for_selection=False,technical_repair_number=1,server_root=str(ROOT)))

def formal():
    assert (out('vqbet')/'freeze.json').exists()
    status=read(out('vqbet')/'status.json')
    status.update(status='PARTIAL',phase='FORMAL_TRAINING',server_root=str(ROOT),probe_stage_updates=4,probe_optimizer_calls=6,physical_smoke_episodes=4)
    write(out('vqbet')/'status.json',status)
    for group in ['target_only','source_target']:
        for stage in ['vqvae','bet']:
            status['phase']='FORMAL_TRAINING_'+group+'_'+stage;write(out('vqbet')/'status.json',status)
            run(['-m','results.strong_baselines.vqbet.train','--group',group,'--stage',stage],f'training-{group}-{stage}')
            completed=[read(p) for p in (ROOT/'vqbet/training').glob('*/*/status.json') if read(p)['status']=='COMPLETE']
            status['actual_training_updates']=sum(s['actual_stage_updates'] for s in completed)
            status['actual_optimizer_calls']=sum(s['actual_optimizer_calls'] for s in completed)
            write(out('vqbet')/'status.json',status)
        selected=read(out('vqbet')/(group+'_bet_selection.json'));assert not selected['used_physical_success']
        for task in TASKS:
            status['phase']='FORMAL_EVALUATION_'+group+'_'+task;write(out('vqbet')/'status.json',status)
            run(['-m','baselines.strong.evaluate','--method','vqbet','--group',group,'--checkpoint',selected['checkpoint'],'--task',task],f'evaluation-{group}-{task}')
            run(['-c',f"from baselines.strong.audit import validate_job; validate_job('vqbet',{group!r},{task!r})"],f'acceptance-{group}-{task}')
            status['physical_evaluation_episodes']=len(list((ROOT/'vqbet/physical/formal').glob('*/*/condition-*/episode.json')))
            write(out('vqbet')/'status.json',status)
    run(['-m','results.strong_baselines.vqbet.report'],'report')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['probes','freeze','formal']);args=parser.parse_args()
    try:globals()[args.phase]()
    except BaseException as exc:
        write(out('vqbet')/'controller_error.json',dict(phase=args.phase,error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise

if __name__=='__main__':main()
