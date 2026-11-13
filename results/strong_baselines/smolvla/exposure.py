"""Actual consumed fixed anchor prefix; never label planned samples as done."""
import collections
import hashlib
import numpy as np
from baselines.strong.common import ROOT,out,read,write,digest
from .data import DIRECTORY

def main():
    state=read(ROOT/'smolvla/training/source_target_state/state.json');protocol=read(out('smolvla')/'protocol.json')
    n=state['actual_training_updates']*protocol['batch_size'];anchors=np.load(DIRECTORY/'train.npy',mmap_mode='r')[:n]
    rows=read(DIRECTORY/'manifest.json');counts=collections.Counter(map(int,anchors[:,0]));groups=collections.Counter()
    unique=collections.Counter(map(int,np.unique(anchors,axis=0)[:,0]));valid_futures=0
    for j,t in anchors:
        assert rows[int(j)]['split']=='train';valid_futures+=min(50,rows[int(j)]['frames']-int(t))
    for j,c in counts.items():groups[(rows[j]['domain'],rows[j]['task'])]+=c
    report=dict(status='FINAL' if state['status']=='COMPLETE' else 'PARTIAL_SAVED_SNAPSHOT',
        actual_native_updates=state['actual_training_updates'],actual_anchors=n,whole_training_demonstrations_seen=len(counts),
        unique_demo_frame_anchors=sum(unique.values()),real_supervised_future_vectors=valid_futures,
        total_future_slots_including_padding=n*50,core_native_loss_dimensions=32,real_action_dimensions=7,
        groups=[dict(domain=d,task=t,anchors=c) for (d,t),c in sorted(groups.items())],
        consumed_prefix_sha256=hashlib.sha256(anchors.tobytes()).hexdigest(),
        manifest_path=str(DIRECTORY/'manifest.json'),manifest_sha256=digest(DIRECTORY/'manifest.json'),
        anchor_path=str(DIRECTORY/'train.npy'),anchor_sha256=digest(DIRECTORY/'train.npy'),
        reference_rollout_files_used_for_training=0,generated_reference_demos_used_for_training=0)
    write(out('smolvla')/'sample_exposure.json',report)
    csv=['episode_id,domain,task,original_frames,sampled_anchors,unique_frame_anchors']
    for j in sorted(counts):
        r=rows[j];csv.append(','.join(map(str,[r['episode_id'],r['domain'],r['task'],r['frames'],counts[j],unique[j]])))
    (out('smolvla')/'sample_exposure.csv').write_text('\n'.join(csv)+'\n');return report

if __name__=='__main__':print(main())
