"""Existing demonstration identities, native Octo windows and reversible I/O.

No simulator, privileged robot state, goal image or generated demonstration is
used. Native RGB256 resize/augmentation runs in the isolated Octo environment.
"""
import collections
import functools
import os
from pathlib import Path
import numpy as np
from baselines.strong.common import ROOT,PROJECT,OLD,read,write,digest

DIRECTORY=ROOT/'octo_small/data'
SCALE=np.array([.05,.05,.05,.5,.5,.5],np.float32)


def encode_action(canonical):
    a=np.array(canonical,dtype=np.float32,copy=True)
    a[...,:6]*=SCALE
    a[...,6]=(1-a[...,6])/2  # native openness; canonical positive means close
    return a


def decode_action(native):
    a=np.array(native,dtype=np.float32,copy=True)
    a[...,:6]/=SCALE
    a[...,6]=1-2*a[...,6]
    return a


def normalized_action(a,stats):
    return np.where(np.asarray(stats['mask'],bool),(a-np.array(stats['mean'],np.float32))/(np.array(stats['std'],np.float32)+1e-8),a).astype(np.float32)


def prepare():
    import h5py
    from baselines.strong.raw_geometry_inventory import identity
    DIRECTORY.mkdir(exist_ok=True,parents=True)
    raw={r['episode_id']:r for r in read(ROOT/'shared_geometry/ur5e_raw_identity.json')}
    rows=[]
    for domain in ['target','source']:
        for original in read(PROJECT/f'configs/datasets/baseline_{domain}_manifest.json'):
            if domain=='source' and original['embodiment']!='ur5e':continue
            r=dict(original)
            if domain=='source':r.update(raw_file=raw[r['episode_id']]['raw_file'],raw_demo=raw[r['episode_id']]['raw_demo'])
            else:r.update(raw_file=r['original_file'],raw_demo=r['original_demo'])
            r['native_pack']=str(DIRECTORY/'rgb256'/f"{r['episode_id']}.h5")
            rows.append(r)
    assert len(rows)==700 and [r['episode_id'] for r in rows[:100]]==[r['episode_id'] for r in read(PROJECT/'configs/datasets/baseline_target_manifest.json')]
    assert collections.Counter((r['domain'],r['split']) for r in rows)=={('source','train'):540,('source','validation'):60,('target','train'):90,('target','validation'):10}
    train=[];file_handles={}
    try:
        for r in rows:
            if r['raw_file'] not in file_handles:file_handles[r['raw_file']]=h5py.File(r['raw_file'],'r')
            g=file_handles[r['raw_file']]['data'][r['raw_demo']]
            a=g['actions'][:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
            assert len(a)==r['frames'] and identity(a,r['instruction'])==r['episode_id']
            assert g['obs/agentview_rgb'].shape==(len(a),180,320,3)
            if r['split']=='train':train.append(encode_action(a))
    finally:
        for h in file_handles.values():h.close()
    values=np.concatenate(train).astype(np.float64)
    stats=dict(mean=values.mean(0).tolist(),std=values.std(0).tolist(),min=values.min(0).tolist(),max=values.max(0).tolist(),
        p01=np.quantile(values,.01,axis=0).tolist(),p99=np.quantile(values,.99,axis=0).tolist(),mask=[True]*6+[False],
        num_transitions=len(values),num_trajectories=630)
    assert min(stats['std'][:6])>0
    write(DIRECTORY/'manifest.json',rows);write(DIRECTORY/'action_statistics.json',stats)
    groups=collections.defaultdict(list)
    for j,r in enumerate(rows):
        if r['split']=='train':groups[(r['domain'],r['task'])].append(j)
    keys=sorted(groups);rng=np.random.default_rng(20261113)
    # Immutable prefix supports a resource-dependent finite budget frozen later.
    anchors=np.empty((1_000_000,2),np.int32)
    for i in range(len(anchors)):
        j=int(rng.choice(groups[keys[i%4]]));anchors[i]=j,int(rng.integers(rows[j]['frames']))
    np.save(DIRECTORY/'train.npy',anchors)
    val=np.load(OLD/'data/target_validation_anchors.npy');np.save(DIRECTORY/'validation.npy',val)
    assert digest(DIRECTORY/'validation.npy')==digest(OLD/'data/target_validation_anchors.npy')
    write(DIRECTORY/'ready.json',dict(status='PASS_IDENTITIES_AND_TRAIN_ONLY_STATS',demonstrations=700,train=630,validation=70,
        target_train_per_task=45,source_train_per_task=270,stats_use_validation=False,generated_demonstrations_used=0,
        rgb_resolution='raw180x320, native256 resize',input_fields=['image_primary','language_instruction'],
        action_convention='base-frame metres/radians plus0..1 openness; reversible into original OSC7',
        files=[dict(path=str(p),sha256=digest(p)) for p in DIRECTORY.iterdir() if p.is_file()]))
    print('Verified700 identities;630 training-only statistics; original target-validation anchors retained',flush=True)


@functools.lru_cache(maxsize=1)
def native_transforms():
    import sys
    sys.path.insert(0,'/home/users/lemon/baseline-benchmark/vendor/octo')
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    from octo.data.obs_transforms import decode_and_resize,augment
    from scripts.configs.finetune_config import get_config
    augmentation=get_config('full,language_conditioned').frame_transform_kwargs.image_augment_kwargs.primary.to_dict()
    @tf.function(input_signature=[tf.TensorSpec([None,None,3],tf.uint8)])
    def resize(image):return decode_and_resize({'image_primary':image},{'primary':(256,256)}, {})['image_primary']
    @tf.function(input_signature=[tf.TensorSpec([256,256,3],tf.uint8),tf.TensorSpec([2],tf.int32)])
    def aug(image,seed):
        return augment({'image_primary':image,'pad_mask_dict':{'image_primary':tf.constant(True)}},seed,augmentation)['image_primary']
    return resize,aug


def resize_image(image):return native_transforms()[0](image).numpy()


class Dataset:
    def __init__(self,split='train'):
        self.rows=read(DIRECTORY/'manifest.json');self.stats=read(DIRECTORY/'action_statistics.json')
        self.anchors=np.load(DIRECTORY/('train.npy' if split=='train' else 'validation.npy'),mmap_mode='r')
        self.split=split;self.handles=collections.OrderedDict();self.raw_cache=collections.OrderedDict();self.chunk_cache={}
    def handle(self,path):
        import h5py
        if path not in self.handles:self.handles[path]=h5py.File(path,'r',rdcc_nbytes=8*1024*1024)
        self.handles.move_to_end(path)
        while len(self.handles)>16:self.handles.popitem(last=False)[1].close()
        return self.handles[path]
    def item(self,index,augment_seed=None):
        j,t=map(int,self.anchors[index]);r=self.rows[j];assert r['split']==self.split
        history=np.array([t-1,t]);times=np.clip(history,0,r['frames']-1)
        if Path(r['native_pack']).exists():
            g=self.handle(r['native_pack']);assert g.attrs['complete']
            unique,inverse=np.unique(times,return_inverse=True);images=g['rgb'][unique][inverse];a=g['canonical_actions'][:]
        else:
            if r['episode_id'] not in self.raw_cache:
                g=self.handle(r['raw_file'])['data'][r['raw_demo']]
                a=g['actions'][:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
                self.raw_cache[r['episode_id']]=(g['obs/agentview_rgb'][:],a)
            self.raw_cache.move_to_end(r['episode_id'])
            while len(self.raw_cache)>4:self.raw_cache.popitem(last=False)
            rgb,a=self.raw_cache[r['episode_id']];images=np.stack([resize_image(rgb[i]) for i in times])
        if augment_seed is not None:
            images=np.stack([native_transforms()[1](image,np.array(augment_seed,np.int32)).numpy() for image in images])
        if r['episode_id'] not in self.chunk_cache:
            # Preserve the official implementation, including its exact terminal
            # task_completed convention; do not approximate native padding.
            import tensorflow as tf
            from octo.data.traj_transforms import chunk_act_obs
            native=chunk_act_obs(dict(observation={'frame':tf.range(len(a))},task={},
                action=tf.convert_to_tensor(normalized_action(encode_action(a),self.stats)),
                action_pad_mask=tf.ones((len(a),7),tf.bool)),window_size=2,action_horizon=4)
            self.chunk_cache[r['episode_id']]=(native['action'].numpy(),native['action_pad_mask'].numpy())
        labels,valid=self.chunk_cache[r['episode_id']]
        return dict(image=images,timestep_pad_mask=history>=0,action=labels[t],
            action_pad_mask=valid[t],instruction=r['instruction'],
            trace=dict(episode_id=r['episode_id'],domain=r['domain'],task=r['task'],frame=t,split=r['split']))
    def batch(self,offset,size,model,augment=False):
        items=[self.item(offset+i,[20261113,offset+i]) if augment else self.item(offset+i) for i in range(size)]
        task=model.create_tasks(texts=[x['instruction'] for x in items])
        batch=dict(observation=dict(image_primary=np.stack([x['image'] for x in items]),image_wrist=np.zeros((size,2,128,128,3),np.uint8),
            timestep_pad_mask=np.stack([x['timestep_pad_mask'] for x in items]),
            pad_mask_dict=dict(image_primary=np.ones((size,2),bool),image_wrist=np.zeros((size,2),bool))),
            task=task,action=np.stack([x['action'] for x in items]),action_pad_mask=np.stack([x['action_pad_mask'] for x in items]))
        return batch,[x['trace'] for x in items]
    def close(self):
        for h in self.handles.values():h.close()
        self.handles.clear();self.raw_cache.clear()


if __name__=='__main__':prepare()
