"""Lossless native RGB256 cache from existing saved images; no simulator use."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import sys
sys.path.insert(0,'/home/users/lemon/baseline-benchmark/vendor/octo')
import argparse
from pathlib import Path
import shutil
import time
import h5py
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
import dlimp as dl
from baselines.strong.common import write,read,digest
from .data import DIRECTORY,resize_image

@tf.function(input_signature=[tf.TensorSpec([None,180,320,3],tf.uint8)])
def resize_batch(images):return dl.transforms.resize_image(images,size=(256,256))

def main(limit=None):
    rows=[r for r in read(DIRECTORY/'manifest.json') if r['split']=='train' or r['domain']=='target']
    if limit is not None:rows=rows[:limit]
    index=[];handles={};start=time.monotonic()
    try:
        for r in rows:
            p=Path(r['native_pack']);p.parent.mkdir(exist_ok=True,parents=True)
            if not p.exists():
                assert shutil.disk_usage(DIRECTORY).free>25*1024**3,'Stop before exhausting server storage'
                if r['raw_file'] not in handles:handles[r['raw_file']]=h5py.File(r['raw_file'],'r')
                g=handles[r['raw_file']]['data'][r['raw_demo']]
                a=g['actions'][:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
                temporary=p.with_suffix('.incomplete')
                with h5py.File(temporary,'w') as dest:
                    target=dest.create_dataset('rgb',(len(a),256,256,3),dtype='u1',chunks=(1,256,256,3),compression='lzf')
                    for offset in range(0,len(a),32):
                        raw=g['obs/agentview_rgb'][offset:offset+32]
                        resized=resize_batch(raw).numpy()
                        if offset==0:
                            np.testing.assert_array_equal(resized[0],resize_image(raw[0]))
                            np.testing.assert_array_equal(resized[-1],resize_image(raw[-1]))
                        target[offset:offset+len(raw)]=resized
                    dest.create_dataset('canonical_actions',data=a)
                    dest.attrs.update(complete=True,episode_id=r['episode_id'],raw_file=r['raw_file'],
                        raw_demo=r['raw_demo'],native_transform='dlimp.resize_image Lanczos3 antialias round clip uint8',
                        native_single_frame_parity=True)
                temporary.replace(p)
            with h5py.File(p,'r') as f:
                assert f.attrs['complete'] and f.attrs['episode_id']==r['episode_id']
                assert f['rgb'].shape==(r['frames'],256,256,3) and f.attrs['native_single_frame_parity']
            index.append(dict(episode_id=r['episode_id'],path=str(p),sha256=digest(p),bytes=p.stat().st_size,
                domain=r['domain'],task=r['task'],split=r['split'],frames=r['frames']))
            write(DIRECTORY/'pack_index.json',dict(status='RUNNING',completed=len(index),total=len(rows),files=index,
                seconds=time.monotonic()-start,new_simulator_steps=0,lossless=True))
            if len(index)%10==0:print('PACKED',len(index),'/',len(rows),'seconds',time.monotonic()-start,flush=True)
        write(DIRECTORY/'pack_index.json',dict(status='PASS',completed=len(index),total=len(rows),files=index,
            seconds=time.monotonic()-start,new_simulator_steps=0,lossless=True,source_manifest_sha256=digest(DIRECTORY/'manifest.json')))
    finally:
        for f in handles.values():f.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args();main(args.limit)
