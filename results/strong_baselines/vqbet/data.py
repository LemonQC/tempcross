"""Causal ten-frame windows; exact original demonstration identity partitions."""
import os
from baselines.strong.common import ROOT,read

class Dataset:
    def __init__(self,group,stage='bet',split='train',offset=0,limit=None):
        import numpy as np,h5py
        self.directory=ROOT/'vqbet/data';self.rows=read(self.directory/'manifest.json');self.language=read(self.directory/'language.json')
        self.group=group;self.stage=stage;self.split=split;self.offset=offset;self.limit=limit;self.features={}
        self.anchors={d:np.load(self.directory/f'{stage}_{d}.npy',mmap_mode='r') for d in ['target','source']} if split=='train' else {'target':np.load(self.directory/'vq_validation.npy')}
        self.actions={}
        for domain in ['target','source']:
            selected=[(i,r) for i,r in enumerate(self.rows) if r['domain']==domain and r['split']==split and (domain=='target' or group=='source_target')]
            if selected:
                with h5py.File(selected[0][1]['packed_file'],'r') as f:
                    for i,r in selected:self.actions[i]=f[r['episode_id']]['actions'][:].astype(np.float32)
    def __len__(self):
        n=len(self.anchors['target'])-self.offset if self.limit is None else self.limit
        return n*(2 if self.group=='source_target' and self.split=='train' else 1)
    def __getitem__(self,i):
        import numpy as np
        joint=self.group=='source_target' and self.split=='train';domain='source' if joint and i%2 else 'target'
        j,t=map(int,self.anchors[domain][self.offset+(i//2 if joint else i)]);r=self.rows[j]
        assert r['split']==self.split and r['domain']==domain and 9<=t<r['frames']
        result=dict(actions=self.actions[j][t:t+1].copy() if self.stage=='vqvae' else self.actions[j][t-9:t+1].copy(),domain=domain,episode_id=r['episode_id'],frame=t)
        if self.stage=='bet':
            if j not in self.features:self.features[j]=np.load(self.directory/'features'/f'{j:04d}.npy',mmap_mode='r')
            result.update(features=self.features[j][t-9:t+1].copy(),language=np.asarray(self.language[r['instruction']],np.float32)[None])
        return result
    def close(self):self.features={}
