"""Shared deterministic anchors, with explicit causal windows and raw action units."""
import collections,json,os
from pathlib import Path
import numpy as np
from baselines.common.paths import ROOT,PROJECT,TASKS,write_json,digest


def window_indices(method,anchor,length):
 if method=='bc_gmm':obs=np.array([anchor]);labels=np.array([anchor])
 elif method=='bc_rnn':obs=labels=np.minimum(np.arange(anchor,anchor+10),length-1)
 elif method=='bc_transformer':obs=np.maximum(np.arange(anchor-9,anchor+1),0);labels=obs.copy()
 elif method=='diffusion_policy':obs=np.maximum(np.arange(anchor-1,anchor+1),0);labels=np.clip(np.arange(anchor-1,anchor+15),0,length-1)
 else:raise ValueError(method)
 return obs,labels


def prepare_anchors():
 folder=ROOT/'data';length=768000
 for domain in ['target','source']:
  rows=json.loads((folder/f'{domain}_manifest.json').read_text())
  for split,n in [('train',length),('validation',256)]:
   dest=folder/f'{domain}_{split}_anchors.npy'
   if dest.exists():continue
   rng=np.random.default_rng(20261109+(17 if domain=='source' else 0)+(101 if split=='validation' else 0))
   groups=collections.defaultdict(list)
   for i,r in enumerate(rows):
    if r['split']==split:groups[(r['task'],r['embodiment'])].append(i)
   keys=sorted(groups);a=np.empty((n,2),np.int32)
   for i in range(n):
    ids=groups[keys[i%len(keys)]];j=int(rng.choice(ids));a[i]=j,int(rng.integers(rows[j]['frames']))
   np.save(dest,a);print('Saved anchors',str(dest),flush=True)
 write_json(folder/'anchor_sequence_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(folder.glob('*_anchors.npy'))])


class AnchorDataset:
 def __init__(self,method,group,split='train',max_positions=768000):
  self.method=method;self.group=group;self.split=split;self.handles={};self.pid=None
  domains=['source'] if group=='A' else ['target'] if group=='B' else ['target','source']
  self.rows={d:json.loads((ROOT/f'data/{d}_manifest.json').read_text()) for d in domains}
  for d in domains:assert digest(ROOT/f'data/{d}_manifest.json')==digest(PROJECT/f'configs/datasets/baseline_{d}_manifest.json')
  self.anchors={d:np.load(ROOT/f'data/{d}_{split}_anchors.npy',mmap_mode='r') for d in self.rows}
  self.n=min(max_positions,len(next(iter(self.anchors.values()))))
  from baselines.common.language import encode
  self.lang={s:encode(s).numpy().copy() for s in sorted({r['instruction'] for d in self.rows.values() for r in d})}
 def __len__(self):return self.n*(2 if self.group=='C' else 1)
 def __getitem__(self,index):
  import h5py
  domain=('target' if index%2==0 else 'source') if self.group=='C' else ('source' if self.group=='A' else 'target')
  position=index//2 if self.group=='C' else index
  row_index,anchor=map(int,self.anchors[domain][position]);row=self.rows[domain][row_index]
  assert row['split']==self.split
  if self.pid!=os.getpid():self.handles={};self.pid=os.getpid()
  if domain not in self.handles:self.handles[domain]=h5py.File(ROOT/f'data/{domain}.h5','r',rdcc_nbytes=64*1024*1024)
  g=self.handles[domain][row['episode_id']];obs,labels=window_indices(self.method,anchor,row['frames'])
  unique,inverse=np.unique(obs,return_inverse=True);images=g['rgb'][unique][inverse]
  unique,inverse=np.unique(labels,return_inverse=True);actions=g['actions'][unique][inverse]
  rgb=np.transpose(images,(0,3,1,2)).astype(np.float32)/255.
  lang=np.repeat(self.lang[row['instruction']][None],len(obs),axis=0)
  return dict(obs=dict(rgb=rgb,lang_emb=lang),actions=actions,_domain=domain,_episode_id=row['episode_id'],_anchor=anchor,_input_frame_count=len(obs),_label_frame_count=len(labels))
 def close(self):
  for h in self.handles.values():h.close()
  self.handles={}

if __name__=='__main__':prepare_anchors()
