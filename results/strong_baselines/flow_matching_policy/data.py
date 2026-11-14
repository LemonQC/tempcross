"""Frozen original identities and read-only, previously certified RGB/qpos data."""
import collections
from pathlib import Path
import shutil
import numpy as np
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest

RUN=ROOT/'flow_matching_policy';DIRECTORY=RUN/'data'
PANDA_STATE_JOINTS=tuple([f'robot0_joint{i}' for i in range(1,8)]+[
    'gripper0_right_finger_joint','gripper0_right_left_inner_finger_joint'])

def live_robot_state(raw):
    return np.array([raw.sim.data.get_joint_qpos(name) for name in PANDA_STATE_JOINTS],np.float32)

def prepare():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    source=ROOT/'smolvla/data';rows=read(source/'manifest.json')
    assert len(rows)==700 and len({r['episode_id'] for r in rows})==700
    for r in rows:
        if r['split']=='train' or r['domain']=='target':assert Path(r['smol_pack']).is_file()
    write(DIRECTORY/'manifest.json',rows)
    for name,src in [('train.npy',source/'train.npy'),('validation.npy',source/'validation.npy'),
        ('language.json',ROOT/'act/data/language.json')]:
        dest=DIRECTORY/name
        if not dest.exists():shutil.copyfile(src,dest)
        assert digest(dest)==digest(src)
    old=read(source/'statistics.json')
    statistics={name:{k:v for k,v in old[key].items() if k in ['min','max']}
        for name,key in [('action','action'),('state','observation.state')]}
    assert all(min(np.array(s['max'])-s['min'])>0 for s in statistics.values())
    write(DIRECTORY/'statistics.json',statistics)
    language=read(DIRECTORY/'language.json')
    assert set(language)=={r['instruction'] for r in rows} and all(len(v)==768 for v in language.values())
    train=np.load(DIRECTORY/'train.npy');val=np.load(DIRECTORY/'validation.npy')
    assert all(rows[int(i)]['split']=='train' for i in train[:,0])
    assert all(rows[int(i)]['split']=='validation' and rows[int(i)]['domain']=='target' for i in val[:,0])
    counts=collections.Counter((r['domain'],r['task'],r['split']) for r in rows)
    report=dict(status='PASS_ORIGINAL_IDENTITIES_AND_CERTIFIED_STATE_CACHE',demonstrations=700,
        source_train=540,source_validation=60,target_train=90,target_validation=10,target_train_per_task=45,
        counts=[dict(domain=k[0],task=k[1],split=k[2],episodes=v) for k,v in sorted(counts.items())],
        information_tier='RGB96_PLUS_REAL_QPOS9_PLUS_FROZEN_CLIP768',privileged_object_state_input=False,
        rgb_source='existing lossless original180x320 RGB, read only; cv2 default LINEAR resize to96 square',
        state_source='previously certified true arm/gripper qpos9; raw joint_states are velocities and are never used',
        language_source='unchanged existing CLIP-L/14 vectors for all16 complete object-specific instructions',
        state_action_statistics='min/max from original630 training demonstrations only',
        sampler='byte-identical original balanced1M train anchors and256 target validation anchors',
        future_padding='native repeat-last sample_sequence; permit15 tail slots to preserve all original anchor frames; native example permits7',
        loss_mask=False,new_demonstrations=0,new_physics_steps=0,
        files=[dict(path=str(DIRECTORY/n),sha256=digest(DIRECTORY/n)) for n in
            ['manifest.json','statistics.json','language.json','train.npy','validation.npy']])
    write(DIRECTORY/'ready.json',report);write(out('flow_matching_policy')/'data_identity.json',report)
    print('PASS700_ORIGINAL_IDENTITIES_630TRAIN_256VAL_NO_NEW_LABELS',flush=True)

class Dataset:
    def __init__(self,split='train'):
        from .native import normalize_data,sample_sequence
        self.normalize=normalize_data;self.sample=sample_sequence
        self.rows=read(DIRECTORY/'manifest.json');self.language=read(DIRECTORY/'language.json');self.split=split
        self.stats={k:{n:np.array(v,np.float32) for n,v in s.items()} for k,s in read(DIRECTORY/'statistics.json').items()}
        self.anchors=np.load(DIRECTORY/('train.npy' if split=='train' else 'validation.npy'),mmap_mode='r')
        self.handles=collections.OrderedDict()
    def item(self,index):
        import cv2,h5py
        j,t=map(int,self.anchors[index]);r=self.rows[j];assert r['split']==self.split
        path=r['smol_pack']
        if path not in self.handles:self.handles[path]=h5py.File(path,'r',rdcc_nbytes=256*1024)
        self.handles.move_to_end(path)
        while len(self.handles)>640:self.handles.popitem(last=False)[1].close()
        f=self.handles[path];end=min(t+16,r['frames'])
        actions=self.sample({'action':f['action']},16,t,end,0,end-t)['action']
        rgb=cv2.resize(f['rgb'][t],(96,96))
        state=f['state'][t]
        return dict(image=rgb.transpose(2,0,1).astype(np.float32)/255,
            state=self.normalize(state,self.stats['state']),action=self.normalize(actions,self.stats['action']),
            language=np.array(self.language[r['instruction']],np.float32),
            trace=dict(episode_id=r['episode_id'],frame=t,domain=r['domain'],task=r['task'],real_future_frames=end-t))
    def batch(self,offset,size):
        import torch
        rows=[self.item(i) for i in range(offset,offset+size)]
        return {k:torch.from_numpy(np.stack([r[k] for r in rows])).contiguous() for k in ['image','state','language','action']},[r['trace'] for r in rows]
    def close(self):
        for f in self.handles.values():f.close()
        self.handles.clear()

if __name__=='__main__':prepare()
