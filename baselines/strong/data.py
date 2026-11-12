"""Frozen RGB/instruction/canonical-action data, original demo identities only."""
import os
from collections import defaultdict
from baselines.strong.common import ROOT,PROJECT,OLD,TASKS,out,read,write,digest

def prepare(method):
    import h5py,numpy as np
    dest=ROOT/method/'data';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        for p,sha in read(dest/'complete.json')['sha256'].items():assert digest(p)==sha
        return
    rows=[]
    for domain in ['target','source']:
        manifest=read(PROJECT/f'configs/datasets/baseline_{domain}_manifest.json')
        if domain=='source':manifest=[r for r in manifest if r['embodiment']=='ur5e']
        rows.extend(dict(r,packed_file=str(OLD/f'data/{domain}.h5')) for r in manifest)
    assert len(rows)==700 and len({r['episode_id'] for r in rows})==700
    from baselines.common.language import encode
    embeddings={s:encode(s).numpy().tolist() for s in sorted({r['instruction'] for r in rows})}
    write(dest/'language.json',embeddings);write(dest/'manifest.json',rows)
    counts=defaultdict(int)
    for r in rows:counts[(r['domain'],r['task'],r['split'])]+=1
    for domain in ['source','target']:
        groups={task:[j for j,r in enumerate(rows) if r['domain']==domain and r['split']=='train' and r['task']==task] for task in TASKS}
        rng=np.random.default_rng(20261112+(10 if domain=='source' else 0))
        anchors=np.empty((384000,2),np.int32)
        for i in range(len(anchors)):
            j=int(rng.choice(groups[TASKS[i%2]]));anchors[i]=j,int(rng.integers(rows[j]['frames']))
        np.save(dest/(domain+'_train.npy'),anchors)
    # Original target validation anchors preserved exactly in content identity.
    original=read(OLD/'data/target_manifest.json');index={r['episode_id']:i for i,r in enumerate(rows)}
    val=np.load(OLD/'data/target_validation_anchors.npy')
    va=np.asarray([[index[original[int(i)]['episode_id']],int(t)] for i,t in val],np.int32)
    assert all(rows[int(i)]['split']=='validation' and rows[int(i)]['domain']=='target' for i,t in va)
    np.save(dest/'validation.npy',va)
    for group in ['target_only','source_target']:
        arrays=[]
        for domain in (['target'] if group=='target_only' else ['source','target']):
            with h5py.File(OLD/f'data/{domain}.h5','r') as f:
                arrays.extend(f[r['episode_id']]['actions'][:] for r in rows if r['domain']==domain and r['split']=='train')
        a=np.concatenate(arrays)
        write(dest/(group+'_stats.json'),dict(mean=a.mean(0).tolist(),std=np.maximum(a.std(0),.01).tolist(),training_frames=len(a),validation_used=False))
    files=[p for p in dest.iterdir() if p.is_file()]
    write(dest/'complete.json',dict(status='COMPLETE',source_robot='ur5e',target_robot='panda',
        target_train_per_task=45,target_validation_per_task=5,source_train_per_task=270,source_validation_per_task=30,
        counts=[dict(domain=k[0],task=k[1],split=k[2],episodes=v) for k,v in sorted(counts.items())],
        sha256={str(p):digest(p) for p in files},additional_privileged_supervision=False))
    write(out(method)/'data_inventory.json',read(dest/'complete.json'))

class Dataset:
    def __init__(self,method,group,split='train',chunk=100,offset=0,limit=None):
        import numpy as np
        self.directory=ROOT/method/'data';self.rows=read(self.directory/'manifest.json');self.language=read(self.directory/'language.json')
        self.group=group;self.split=split;self.chunk=chunk;self.offset=offset;self.limit=limit
        self.stats=read(self.directory/(group+'_stats.json'));self.mean=np.asarray(self.stats['mean'],np.float32);self.std=np.asarray(self.stats['std'],np.float32)
        self.anchors={d:np.load(self.directory/(d+'_train.npy'),mmap_mode='r') for d in ['target','source']} if split=='train' else {'target':np.load(self.directory/'validation.npy')}
        self.handles={};self.pid=None
    def __len__(self):
        base=(len(self.anchors['target'])-self.offset) if self.limit is None else self.limit
        return base*(2 if self.group=='source_target' and self.split=='train' else 1)
    def __getitem__(self,i):
        import h5py,numpy as np
        joint=self.group=='source_target' and self.split=='train';domain='source' if joint and i%2 else 'target'
        j,t=map(int,self.anchors[domain][self.offset+(i//2 if joint else i)]);r=self.rows[j]
        assert r['split']==self.split and r['domain']==domain
        if self.pid!=os.getpid():self.handles={};self.pid=os.getpid()
        if domain not in self.handles:self.handles[domain]=h5py.File(r['packed_file'],'r',rdcc_nbytes=32*1024*1024)
        g=self.handles[domain][r['episode_id']];end=min(t+self.chunk,r['frames']);n=end-t
        raw=np.zeros((self.chunk,7),np.float32);raw[:n]=g['actions'][t:end]
        return dict(image=np.transpose(g['rgb'][t],(2,0,1)).astype(np.float32)[None]/255,
                    language=np.asarray(self.language[r['instruction']],np.float32),actions=(raw-self.mean)/self.std,
                    is_pad=np.arange(self.chunk)>=n,domain=domain,episode_id=r['episode_id'],frame=t)
    def close(self):
        for f in self.handles.values():f.close()
        self.handles={}
