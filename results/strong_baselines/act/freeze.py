"""Freeze the candidate once the actual training/physics probes are accepted."""
import shutil
from baselines.strong.common import PROJECT,ROOT,out,read,write,digest,resource,now

def main():
    folder=out('act');accepted=read(folder/'probe_acceptance.json');assert accepted['status']=='PASS'
    assert accepted['real_optimizer_updates']==2 and accepted['smoke_episodes']==4
    protocol=folder/'protocol.json'
    if (folder/'freeze.json').exists():
        assert read(folder/'freeze.json')['protocol_sha256']==digest(protocol);return
    shutil.copyfile(folder/'candidate_protocol.json',protocol)
    paths=list((PROJECT/'baselines/strong').glob('*.py'))+list(folder.glob('*.py'))+[PROJECT/'tests/test_strong_baseline_data.py']
    write(folder/'freeze.json',dict(utc=now(),protocol_sha256=digest(protocol),
        code_sha256={str(p.relative_to(PROJECT)):digest(p) for p in sorted(paths)},
        data_freeze=read(ROOT/'act/data/complete.json'),probe_acceptance_sha256=digest(folder/'probe_acceptance.json'),
        resources=resource(),success_based_tuning=False,formal_reference_role='EXPOSED_REFERENCE'))
    print('ACT protocol and executable code frozen after four accepted physical smoke trials.')

if __name__=='__main__':main()
