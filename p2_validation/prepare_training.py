"""Cache a fixed official mixed RLDS sample sequence shared by both methods."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys


class CachedStream:
    def __init__(self,root):self.root=Path(root)
    def as_numpy_iterator(self):
        metadata=json.loads((self.root/'complete.json').read_text())
        for i in range(metadata['samples']):
            with gzip.open(self.root/f'sample-{i:06d}.pkl.gz','rb') as stream:
                yield pickle.load(stream)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--samples',type=int,default=6000);a=p.parse_args()
    source=Path(a.source).resolve();out=Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);sys.path[:0]=[str(source),str(source/'policy')]
    from p2_validation.audit_data import audit
    audit(source,a.data,out.parent/'target-data-audit.json')
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.set_visible_devices([],'GPU');tf.random.set_seed(7);np.random.seed(7)
    from prismatic.vla.datasets.datasets import RLDSDataset
    dataset=RLDSDataset(Path(a.data),'panda_xp_900_pnp',[(lambda x:x,1.)],resize_resolution=(224,224),
       shuffle_buffer_size=10000,train=True,image_aug=False,future_action_window_size=7,
       future_obj_pose_window_size=40,future_2D_trace_window_size=40,obj_pose_stride=8,ee_pose_2D_stride=8,
       image_window_size=1,dataset_statistics_map={'panda_pnp':'mg_pnp_lite'})
    stats=dataset.dataset_statistics
    (out/'dataset_statistics.json').write_text(json.dumps(stats,default=lambda v:v.tolist(),indent=2))
    counts={};trace=[]
    for i,sample in enumerate(dataset.dataset.as_numpy_iterator()):
        if i>=a.samples:break
        data=pickle.dumps(sample,protocol=4);path=out/f'sample-{i:06d}.pkl.gz'
        with gzip.open(path,'wb',compresslevel=1) as stream:stream.write(data)
        name=sample['dataset_name'].decode();counts[name]=counts.get(name,0)+1
        trace.append(dict(index=i,dataset=name,raw_sample_sha256=hashlib.sha256(data).hexdigest(),path=str(path)))
        if (i+1)%100==0:print(json.dumps(dict(cached=i+1,counts=counts)),flush=True)
    if len(trace)!=a.samples:raise RuntimeError('Mixed sample stream ended early')
    with (out/'sample_trace.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trace[0]),lineterminator='\n');writer.writeheader();writer.writerows(trace)
    summary=dict(status='COMPLETE',samples=len(trace),dataset_length=len(dataset),dataset_counts=counts,
        source_sampling_weight=.05,target_sampling_weight=1.,seed=7,shuffle_buffer_size=10000,
        statistics_file=str(out/'dataset_statistics.json'),
        sample_trace_sha256=hashlib.sha256((out/'sample_trace.csv').read_bytes()).hexdigest(),
        limitation='Fixed finite mixed sequence for matched restricted single-GPU adaptation; not paper exposure budget.')
    (out/'complete.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
