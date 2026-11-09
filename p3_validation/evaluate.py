"""Execute only the five frozen independent 40--59 evaluation cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
from p3_validation.common import ROOT,PROJECT,CHECKPOINT,OLD_ADAPTED,environment
from p2_validation.offline_audit import sha256
from p2_validation.execute import run


def validate_freeze():
    marker=json.loads((ROOT/'frozen_publication.json').read_text())
    assert marker['github_readback_verified']
    assert sha256(PROJECT/'configs/p3_protocol.json')==marker['config_sha256']
    c=json.loads((PROJECT/'configs/p3_protocol.json').read_text())
    assert sha256(PROJECT/'evidence/p3/frozen/source_contact_bank.json')==c['frozen_assets']['source_bank_sha256']
    assert sha256(PROJECT/'evidence/p3/frozen/independent_conditions_40_59.json')==c['frozen_assets']['independent_conditions_sha256']
    return c,marker


def cell(name,new_checkpoint=None):
    c,marker=validate_freeze();os.environ.update(environment())
    options={'original-joint_reps-ur5e':('original','ur5e',CHECKPOINT),
             'original-joint_reps-panda':('original','panda',CHECKPOINT),
             'b1-joint_reps-panda':('b1','panda',CHECKPOINT),
             'p2-6000-joint_reps-panda':('original','panda',OLD_ADAPTED),
             'p3-96000-joint_reps-panda':('original','panda',Path(new_checkpoint) if new_checkpoint else None)}
    method,robot,checkpoint=options[name]
    if checkpoint is None or not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    directory=ROOT/'formal'/name
    if (directory/'job.json').exists():
        previous=json.loads((directory/'job.json').read_text())
        assert previous['status']=='COMPLETE' and previous['indices']==c['independent_indices']
        assert previous['robot']==robot and previous['method']==method and previous['checkpoint']==str(checkpoint)
        return previous
    with (ROOT/'physical_launches.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='FORMAL_EVALUATION',cell=name,indices=c['independent_indices'],frozen_commit=marker['remote_commit'],frozen_protocol_sha256=marker['config_sha256']))+'\n')
    result=run(directory,method,robot,c['independent_indices'],checkpoint,.85)
    result['p3_frozen_protocol_commit']=marker['remote_commit'];result['p3_role']='FAIR_BASELINE'
    (directory/'job.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-suite',action='store_true');p.add_argument('--cell');p.add_argument('--new-checkpoint');a=p.parse_args()
    if a.baseline_suite:
        names=['original-joint_reps-ur5e','original-joint_reps-panda','b1-joint_reps-panda','p2-6000-joint_reps-panda']
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(cell,names))
        (ROOT/'baseline_suite_complete.json').write_text(json.dumps(dict(status='COMPLETE',formal_episodes=80,cells=[dict(cell=n,successes=r['summary']['successes'],episodes=20) for n,r in zip(names,results)]),indent=2)+'\n')
    else:cell(a.cell,a.new_checkpoint)


if __name__=='__main__':main()
