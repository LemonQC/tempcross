"""Actual consumed demonstration/frame identities, never planned-as-executed."""
import collections
import hashlib
import numpy as np
from baselines.strong.common import ROOT,out,read,write,digest
from .data import DIRECTORY

def main():
    run=ROOT/'octo_small';state=read(run/'training/source_target/state.json')
    protocol=read(out('octo_small')/'protocol.json')
    updates=state['actual_training_updates'];n=updates*protocol['batch_size']
    anchors=np.load(DIRECTORY/'train.npy',mmap_mode='r')[:n]
    rows=read(DIRECTORY/'manifest.json');groups=collections.Counter();counts=collections.Counter(map(int,anchors[:,0]))
    unique=collections.Counter(map(int,np.unique(anchors,axis=0)[:,0]))
    for j,count in counts.items():
        assert rows[j]['split']=='train'
        groups[(rows[j]['domain'],rows[j]['task'])]+=count
    actual=[dict(domain=d,task=t,sampled_anchors=count) for (d,t),count in sorted(groups.items())]
    result=dict(status='FINAL' if state['status']=='COMPLETE' else 'PARTIAL_SAVED_SNAPSHOT',
        actual_native_updates=updates,actual_sampled_anchors=n,unique_demo_frame_anchors=sum(unique.values()),
        whole_demonstrations_seen=len(counts),groups=actual,
        consumed_anchor_prefix_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
        full_anchor_file=dict(path=str(DIRECTORY/'train.npy'),sha256=digest(DIRECTORY/'train.npy')),
        manifest=dict(path=str(DIRECTORY/'manifest.json'),sha256=digest(DIRECTORY/'manifest.json')),
        validation_anchors_used_for_training=0,reference_conditions_used_for_training=0)
    write(out('octo_small')/'sample_exposure.json',result)
    csv=['episode_id,domain,task,original_frames,sampled_anchors,unique_frame_anchors']
    for j in sorted(counts):
        r=rows[j];csv.append(','.join(map(str,[r['episode_id'],r['domain'],r['task'],r['frames'],counts[j],unique[j]])))
    (out('octo_small')/'sample_exposure.csv').write_text('\n'.join(csv)+'\n')
    print(result,flush=True)

if __name__=='__main__':main()
