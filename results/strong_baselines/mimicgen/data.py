"""Existing human split plus independent, traceable generated training episodes."""
import json
import os
from pathlib import Path
from collections import defaultdict
from common import PROJECT, ROOT, OUT, TASKS, digest, read, write

DATA=ROOT/'downstream/data'


def prepare():
    import h5py
    import cv2
    import numpy as np
    from p3_validation.prepare_data import fingerprint
    DATA.mkdir(parents=True,exist_ok=True)
    marker=DATA/'complete.json'
    if marker.exists():
        for p,sha in read(marker)['sha256'].items():assert digest(p)==sha
        return
    pack=DATA/'generated.h5'
    if pack.exists():raise FileExistsError('Preserve incomplete data pack '+str(pack))
    human=read(PROJECT/'configs/datasets/baseline_target_manifest.json')
    rows=[dict(r,origin='human',packed_file='/home/users/lemon/baseline-benchmark/run-20261109/data/target.h5') for r in human]
    existing={r['episode_id'] for r in rows}
    generation=read(ROOT/'augmentation/complete.json')
    formal=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
    formal_ids={r['condition_id'] for r in formal if 'condition_id' in r}
    if not formal_ids:formal_ids={r['condition']['condition_id'] for r in formal}
    with h5py.File(pack,'x') as out:
        for r in generation['rows']:
            if not r['success']:continue
            assert r['condition']<20 and r['condition_id'] not in formal_ids
            lineage=read(r['lineage']);assert lineage['selected_sources']
            assert all(next(h for h in human if h['episode_id']==p['episode_id'])['split']=='train'
                       for p in lineage['selected_sources'])
            with h5py.File(r['trajectory']) as f,h5py.File(r['telemetry']) as t:
                assert bool((t['mimicgen/post_action_placement'][:]&t['mimicgen/post_action_retreat'][:]).any())
                actions=f['actions'][:,:7].astype(np.float32)
                instruction=json.loads(f.attrs['ep_meta'])['lang']
                eid=fingerprint(actions,len(actions),instruction)
                if eid in existing:raise ValueError('Duplicate generated data identity '+eid)
                existing.add(eid)
                g=out.create_group(eid);g.create_dataset('actions',data=actions)
                rgb=g.create_dataset('rgb',shape=(len(actions),128,128,3),dtype='uint8',compression='lzf',chunks=(1,128,128,3))
                for i,image in enumerate(f['rgb']):rgb[i]=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
                rows.append(dict(domain='target',origin='generated',embodiment='panda',task=r['task'],
                    episode_id=eid,frames=len(actions),instruction=instruction,split='train',packed_file=str(pack),
                    original_file=r['trajectory'],original_demo=r['id'],condition_id=r['condition_id'],
                    generation_seed=r['generation_seed'],source_lineage=lineage['selected_sources']))
    write(DATA/'manifest.json',rows)
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split']=='train':groups[r['task']].append(i)
    cfg=read(OUT/'downstream_protocol.json');n=cfg['training']['anchor_positions']
    rng=np.random.default_rng(20261112);anchors=np.empty((n,2),np.int32)
    for i in range(n):
        row=int(rng.choice(groups[TASKS[i%2]]));anchors[i]=row,int(rng.integers(rows[row]['frames']))
    np.save(DATA/'train_anchors.npy',anchors)
    old=Path('/home/users/lemon/baseline-benchmark/run-20261109/data/target_validation_anchors.npy')
    validation=np.load(old);assert validation.shape==(256,2)
    assert all(rows[int(i)]['split']=='validation' and rows[int(i)]['origin']=='human' for i,_ in validation)
    np.save(DATA/'validation_anchors.npy',validation)
    files=[pack,DATA/'manifest.json',DATA/'train_anchors.npy',DATA/'validation_anchors.npy']
    write(marker,dict(status='COMPLETE',human_train=90,human_validation=10,generated_train=len(rows)-100,
        generated_probe_episodes_used=0,successful_generated_episode_ids=[r['episode_id'] for r in rows if r['origin']=='generated'],
        sha256={str(p):digest(p) for p in files},original_validation_anchors_sha256=digest(old),
        sampling='task-balanced, then uniform demonstration and frame, across human+generated pool',
        generated_terminal_rule='preserve each actual generated action including final success transition; human labels retain historical RLDS terminal convention'))


class Dataset:
    def __init__(self,split):
        import numpy as np
        self.rows=read(DATA/'manifest.json')
        self.anchors=np.load(DATA/(split+'_anchors.npy'),mmap_mode='r')
        self.split=split;self.handles={};self.pid=None
        from baselines.common.language import encode
        self.language={s:encode(s).numpy().copy() for s in sorted({r['instruction'] for r in self.rows})}
    def __len__(self):return len(self.anchors)
    def __getitem__(self,index):
        import h5py
        import numpy as np
        i,anchor=map(int,self.anchors[index]);r=self.rows[i];assert r['split']==self.split
        if self.pid!=os.getpid():self.handles={};self.pid=os.getpid()
        path=r['packed_file']
        if path not in self.handles:self.handles[path]=h5py.File(path,'r',rdcc_nbytes=64*1024*1024)
        g=self.handles[path][r['episode_id']]
        frames=np.maximum(np.arange(anchor-9,anchor+1),0);unique,inverse=np.unique(frames,return_inverse=True)
        rgb=np.transpose(g['rgb'][unique][inverse],(0,3,1,2)).astype(np.float32)/255.
        actions=g['actions'][unique][inverse]
        language=np.repeat(self.language[r['instruction']][None],10,axis=0)
        return dict(obs=dict(rgb=rgb,lang_emb=language),actions=actions,_episode_id=r['episode_id'],
                    _origin=r['origin'],_task=r['task'],_anchor=anchor)
    def close(self):
        for h in self.handles.values():h.close()
        self.handles={}
