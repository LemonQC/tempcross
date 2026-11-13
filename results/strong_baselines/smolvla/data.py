"""Original identities, verified robot qpos and canonical OSC7 for SmolVLA.

No simulator step, generated label, object pose or reference rollout is used.
The original misleading `joint_states` field is audited as qvel, never qpos.
"""
import collections
import hashlib
from pathlib import Path
import shutil
import numpy as np
from baselines.strong.common import ROOT,PROJECT,OLD,read,write,digest

RUN=ROOT/'smolvla'
DIRECTORY=RUN/'data'
PANDA_STATE_JOINTS=tuple([f'robot0_joint{i}' for i in range(1,8)]+[
    'gripper0_right_finger_joint','gripper0_right_left_inner_finger_joint'])

def robot_state(flat,gripper,domain):
    n=6 if domain=='source' else 7
    q=np.asarray(flat[...,5:5+n],np.float32)
    if n==6:q=np.concatenate([q,np.zeros((*q.shape[:-1],1),np.float32)],axis=-1)
    return np.concatenate([q,np.asarray(gripper,np.float32)],axis=-1)

def live_robot_state(raw):
    return np.array([raw.sim.data.get_joint_qpos(name) for name in PANDA_STATE_JOINTS],np.float32)

def prepare():
    import h5py
    from baselines.strong.raw_geometry_inventory import identity
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    rows=read(ROOT/'octo_small/data/manifest.json')
    audit=[];statistics={'action':[],'observation.state':[]};handles={}
    try:
        for r in rows:
            if r['raw_file'] not in handles:handles[r['raw_file']]=h5py.File(r['raw_file'],'r')
            g=handles[r['raw_file']]['data'][r['raw_demo']]
            a=g['actions'][:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
            assert identity(a,r['instruction'])==r['episode_id'] and len(a)==r['frames']
            flat=g['states'][:];velocity=g['obs/joint_states'][:];gripper=g['obs/gripper_states'][:]
            n=6 if r['domain']=='source' else 7
            assert velocity.shape==(len(a),n) and gripper.shape==(len(a),2)
            assert np.isfinite(flat).all() and np.isfinite(velocity).all()
            # Old serialization: time, base(x,y,yaw), torso, arm, gripper...
            # Certify the hand-state columns and independently compare qdot.
            np.testing.assert_array_equal(flat[:,5+n:7+n],gripper)
            dt=np.diff(flat[:,0]);assert np.all(dt>0)
            positions=np.diff(flat[:,5:5+n],axis=0)/dt[:,None]
            qdot=(velocity[:-1]+velocity[1:])/2
            mse=float(np.mean((positions-qdot)**2))
            candidates={j:float(np.mean((np.diff(flat[:,j:j+n],axis=0)/dt[:,None]-qdot)**2)) for j in range(1,17-n)}
            assert min(candidates,key=candidates.get)==5,(r['episode_id'],candidates)
            # Locate qvel directly within the serialized state, independent of
            # changing scene joint counts. It must lie beyond the qpos block.
            hits=[j for j in range(flat.shape[1]//2-4,flat.shape[1]//2+14)
                if j+n<=flat.shape[1] and np.array_equal(flat[:,j:j+n],velocity)]
            assert len(hits)==1 and hits[0]>flat.shape[1]//2
            state=robot_state(flat,gripper,r['domain'])
            assert state.shape==(len(a),9)
            r['smol_pack']=str(DIRECTORY/'raw_rgb'/f"{r['episode_id']}.h5")
            audit.append(dict(episode_id=r['episode_id'],domain=r['domain'],task=r['task'],
                joint_field_is_velocity=True,flat_qvel_column=hits[0],flat_qpos_column=5,
                qpos_derivative_velocity_mse=mse,gripper_qpos_exact=True,
                non_20Hz_intervals=int(np.count_nonzero(~np.isclose(dt,.05))),
                max_observed_interval=float(dt.max()),terminal_identity_exact=True))
            if r['split']=='train':
                statistics['action'].append(a);statistics['observation.state'].append(state)
    finally:
        for h in handles.values():h.close()
    stats={}
    for name,values in statistics.items():
        value=np.concatenate(values).astype(np.float64)
        stats[name]=dict(mean=value.mean(0).tolist(),std=value.std(0).tolist(),min=value.min(0).tolist(),max=value.max(0).tolist())
        assert min(stats[name]['std'])>0
    write(DIRECTORY/'manifest.json',rows);write(DIRECTORY/'statistics.json',stats)
    # The immutable existing balanced sequence and validation IDs are reused.
    for src,dst in [(ROOT/'octo_small/data/train.npy',DIRECTORY/'train.npy'),
        (OLD/'data/target_validation_anchors.npy',DIRECTORY/'validation.npy')]:
        if not dst.exists():shutil.copyfile(src,dst)
        assert digest(src)==digest(dst)
    write(DIRECTORY/'state_audit_rows.json',audit)
    write(DIRECTORY/'ready.json',dict(status='PASS_IDENTITIES_AND_REAL_ROBOT_STATE',demonstrations=700,
        source_train=540,source_validation=60,target_train=90,target_validation=10,
        state_features=['arm_q1','arm_q2','arm_q3','arm_q4','arm_q5','arm_q6','arm_q7_or_zero_UR5e','gripper_q0','gripper_q1'],
        state_units='arm and Robotiq joint radians; UR5e seventh-arm slot zero padding',
        original_joint_states_are_velocities=True,object_state_passed_to_policy=False,
        preserved_non_20Hz_intervals=sum(r['non_20Hz_intervals'] for r in audit),
        time_handling='original frame order and action identities preserved; no resampling or interpolated labels',
        raw_state_schema='time,base3,torso1,arm6or7,gripper6,...; deployment uses named joint qpos, not this old flat offset',
        interface='canonical OSC7,positive close; unchanged OSC7->12',new_physics_steps=0,
        files=[dict(path=str(DIRECTORY/name),sha256=digest(DIRECTORY/name))
            for name in ['manifest.json','statistics.json','train.npy','validation.npy','state_audit_rows.json']]))
    print('PASS700_IDENTITIES_QPOS_QVEL_GRIPPER_AND_ORIGINAL_SPLITS',flush=True)

def pack():
    import h5py,time
    rows=[r for r in read(DIRECTORY/'manifest.json') if r['split']=='train' or r['domain']=='target']
    handles={};index=[];start=time.monotonic()
    try:
        for r in rows:
            p=Path(r['smol_pack']);p.parent.mkdir(exist_ok=True,parents=True)
            if not p.exists():
                assert shutil.disk_usage(DIRECTORY).free>30*1024**3,'Insufficient remaining disk for a bounded native runtime'
                if r['raw_file'] not in handles:handles[r['raw_file']]=h5py.File(r['raw_file'],'r')
                g=handles[r['raw_file']]['data'][r['raw_demo']]
                images=g['obs/agentview_rgb'][:]
                a=g['actions'][:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
                state=robot_state(g['states'][:],g['obs/gripper_states'][:],r['domain'])
                tmp=p.with_suffix('.incomplete')
                with h5py.File(tmp,'w') as f:
                    d=f.create_dataset('rgb',data=images,chunks=(1,180,320,3),compression='lzf')
                    f.create_dataset('state',data=state);f.create_dataset('action',data=a)
                    image_hash=hashlib.sha256(images.tobytes()).hexdigest()
                    assert hashlib.sha256(d[:].tobytes()).hexdigest()==image_hash
                    f.attrs.update(complete=True,episode_id=r['episode_id'],raw_rgb_sha256=image_hash,all_frames_lossless_verified=True)
                tmp.replace(p)
            with h5py.File(p,'r') as f:assert f.attrs['complete'] and f['rgb'].shape==(r['frames'],180,320,3)
            index.append(dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size,episode_id=r['episode_id']))
            write(DIRECTORY/'pack_index.json',dict(status='RUNNING',completed=len(index),total=len(rows),files=index))
            if len(index)%20==0:print('PACKED',len(index),len(rows),'seconds',time.monotonic()-start,flush=True)
        write(DIRECTORY/'pack_index.json',dict(status='PASS',completed=len(index),total=len(rows),files=index,
            lossless_raw_RGB=True,resize_performed=False,new_physics_steps=0,seconds=time.monotonic()-start))
    finally:
        for h in handles.values():h.close()

class Dataset:
    def __init__(self,split='train'):
        self.rows=read(DIRECTORY/'manifest.json');self.statistics=read(DIRECTORY/'statistics.json');self.split=split
        self.anchors=np.load(DIRECTORY/('train.npy' if split=='train' else 'validation.npy'),mmap_mode='r')
        self.handles=collections.OrderedDict()
    def item(self,index):
        import h5py
        j,t=map(int,self.anchors[index]);r=self.rows[j];assert r['split']==self.split
        if r['smol_pack'] not in self.handles:self.handles[r['smol_pack']]=h5py.File(r['smol_pack'],'r',rdcc_nbytes=1024**2)
        self.handles.move_to_end(r['smol_pack'])
        while len(self.handles)>16:self.handles.popitem(last=False)[1].close()
        f=self.handles[r['smol_pack']];future=t+np.arange(50);actions=f['action'][:]
        return dict(image=f['rgb'][t],state=f['state'][t],action=actions[np.minimum(future,len(actions)-1)],
            actions_id_pad=future>=len(actions),task=r['instruction'],
            trace=dict(episode_id=r['episode_id'],frame=t,domain=r['domain'],task=r['task']))
    def batch(self,offset,size):
        import torch
        rows=[self.item(i) for i in range(offset,offset+size)]
        batch={'observation.images.camera1':torch.from_numpy(np.stack([r['image'].transpose(2,0,1) for r in rows])).float()/255,
            'observation.state':torch.from_numpy(np.stack([r['state'] for r in rows])),
            'action':torch.from_numpy(np.stack([r['action'] for r in rows])),
            'actions_id_pad':torch.from_numpy(np.stack([r['actions_id_pad'] for r in rows])),
            'task':[r['task'] for r in rows]}
        return batch,[r['trace'] for r in rows]
    def close(self):
        for h in self.handles.values():h.close()
        self.handles.clear()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--pack',action='store_true');args=p.parse_args()
    pack() if args.pack else prepare()
