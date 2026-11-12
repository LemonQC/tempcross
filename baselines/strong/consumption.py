"""Independent read-only reconciliation of ACT's actual consumed identities.

This does not change training, choose a checkpoint or invoke a policy. During
training it reports only the atomically saved status frontier, not buffered
updates beyond that checkpoint. Final acceptance must reconcile all 24000.
"""
import csv
from collections import Counter
from baselines.strong.common import ROOT,out,read,write

def audit_act(group):
    directory=ROOT/'act/training'/group
    state=read(directory/'status.json');frontier=state['actual_optimizer_updates']
    manifest={r['episode_id']:r for r in read(ROOT/'act/data/manifest.json')}
    batch=16 if group=='target_only' else 32
    counts=Counter();labels=Counter();identities={'target':set(),'source':set()};steps=Counter()
    with (directory/'consumed_anchors.csv').open() as f:
        for row in csv.DictReader(f):
            step=int(row['update'])
            if step>frontier:continue
            r=manifest[row['episode_id']];frame=int(row['frame'])
            assert r['split']=='train' and r['domain']==row['domain']
            assert 0<=frame<r['frames'] and 1<=step<=frontier
            counts[r['domain']]+=1;labels[r['domain']]+=min(100,r['frames']-frame)
            identities[r['domain']].add(r['episode_id']);steps[step]+=1
    assert set(steps)==set(range(1,frontier+1)) and all(n==batch for n in steps.values())
    assert counts['target']==frontier*16==state['target_anchor_exposures']
    assert counts['source']==(frontier*16 if group=='source_target' else 0)==state['source_anchor_exposures']
    with (directory/'updates.csv').open() as f:
        updates=[r for r in csv.DictReader(f) if int(r['update'])<=frontier]
    assert [int(r['update']) for r in updates]==list(range(1,frontier+1))
    return dict(status='PASS',training_status=state['status'],group=group,atomically_saved_optimizer_updates=frontier,
        target_anchor_exposures=counts['target'],source_anchor_exposures=counts['source'],
        rgb_frame_exposures=sum(counts.values()),valid_future_action_label_exposures=dict(labels),
        distinct_training_demonstrations={d:len(ids) for d,ids in identities.items()},
        all_consumed_identities_in_original_training_split=True,all_update_and_anchor_counts_reconciled=True,
        actual_new_training_examples_not_independent_samples=True)

if __name__=='__main__':
    rows=[]
    for group in ['target_only','source_target']:
        if (ROOT/'act/training'/group/'status.json').exists():rows.append(audit_act(group))
    write(out('act')/'actual_exposure_reconciliation.json',rows)
    print(__import__('json').dumps(rows,indent=2))
