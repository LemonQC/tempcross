"""Freeze the candidate once the actual training/physics probes are accepted."""
import shutil
from baselines.strong.common import PROJECT,ROOT,out,read,write,digest,resource,now

def main():
    folder=out('act');accepted=read(folder/'probe_acceptance.json');assert accepted['status']=='PASS'
    assert accepted['real_optimizer_updates']==2 and accepted['smoke_episodes']==4
    protocol=folder/'protocol.json'
    if (folder/'freeze.json').exists() and read(folder/'freeze.json').get('server_root')==str(ROOT):
        assert read(folder/'freeze.json')['protocol_sha256']==digest(protocol);return
    shutil.copyfile(folder/'candidate_protocol.json',protocol)
    paths=list((PROJECT/'baselines/strong').glob('*.py'))+list(folder.glob('*.py'))+[PROJECT/'tests/test_strong_baseline_data.py']
    write(folder/'freeze.json',dict(utc=now(),server_root=str(ROOT),technical_repair_number=1,protocol_sha256=digest(protocol),
        code_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sorted(paths)},
        data_freeze=read(ROOT/'act/data/complete.json'),probe_acceptance_sha256=digest(folder/'probe_acceptance.json'),
        resources=resource(),success_based_tuning=False,formal_reference_role='EXPOSED_REFERENCE'))
    for group in ['target_only','source_target']:
        if not (ROOT/'act/training'/group/'status.json').exists():
            write(folder/('training_'+group+'.json'),dict(status='NOT_RUN',group=group,server_root=str(ROOT),actual_optimizer_updates=0,technical_repair_number=1))
            write(folder/(group+'_selection.json'),dict(status='NOT_SELECTED',checkpoint=None,used_physical_success=False,server_root=str(ROOT)))
    print('ACT protocol and executable code frozen after four accepted physical smoke trials.')

if __name__=='__main__':main()
