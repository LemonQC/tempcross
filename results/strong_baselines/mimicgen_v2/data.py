"""Exact human split plus all63 audited successes; causal RGB windows."""
import os
from baselines.strong.common import ROOT,read

class Dataset:
    def __init__(self,split='train',offset=0,limit=None):
        import numpy as np
        self.directory=ROOT/'mimicgen_v2/data';self.rows=read(self.directory/'manifest.json');self.language=read(self.directory/'language.json')
        self.anchors=np.load(self.directory/(split+'.npy'),mmap_mode='r');self.split=split;self.offset=offset;self.limit=limit;self.handles={};self.pid=None
    def __len__(self):return len(self.anchors)-self.offset if self.limit is None else self.limit
    def __getitem__(self,index):
        import numpy as np,h5py
        j,t=map(int,self.anchors[self.offset+index]);r=self.rows[j];assert r['split']==self.split
        if self.pid!=os.getpid():self.handles={};self.pid=os.getpid()
        if r['packed_file'] not in self.handles:self.handles[r['packed_file']]=h5py.File(r['packed_file'],'r',rdcc_nbytes=32*1024*1024)
        g=self.handles[r['packed_file']][r['episode_id']];frames=np.maximum(np.arange(t-9,t+1),0);unique,inverse=np.unique(frames,return_inverse=True)
        rgb=np.transpose(g['rgb'][unique][inverse],(0,3,1,2)).astype(np.float32)/255
        actions=g['actions'][unique][inverse].astype(np.float32)
        language=np.repeat(np.asarray(self.language[r['instruction']],np.float32)[None],10,0)
        return dict(obs=dict(rgb=rgb,lang_emb=language),actions=actions,origin=r['origin'],episode_id=r['episode_id'],task=r['task'],frame=t)
    def close(self):
        for f in self.handles.values():f.close()
        self.handles={}
