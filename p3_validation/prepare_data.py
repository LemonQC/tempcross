"""Fresh 96000-position official mixture, with original-demo/frame provenance.

Only the full train split and extra provenance fields differ from upstream data
loading. The immutable upstream source file is never edited.
"""
import csv
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import struct
import sys
from collections import Counter,defaultdict
import numpy as np
from p3_validation.common import SOURCE,DATA,CACHE,ROOT,PROJECT
from p2_validation.offline_audit import sha256


def fingerprint(actions,length,instruction):
    actions=np.ascontiguousarray(actions,dtype='<f4')
    if isinstance(instruction,bytes):instruction=instruction.decode()
    return hashlib.sha256(struct.pack('<Q',int(length))+actions.tobytes()+instruction.encode()).hexdigest()


def original_target_manifest():
    import h5py
    rows=[]
    for path in sorted((DATA/'raw_target/human/PandaOmron').glob('*/demo_gentex_im320.hdf5')):
        with h5py.File(path) as f:
            for name in sorted(f['data']):
                demo=f['data'][name];actions=demo['actions'][:,:7].astype(np.float32)
                actions[-1]=actions[-2];actions[-1,:6]=0
                instruction=json.loads(demo.attrs['ep_meta'])['lang']
                rows.append(dict(episode_id=fingerprint(actions,len(actions),instruction),
                   original_file=str(path),original_demo=name,direction=path.parent.name,frames=len(actions),instruction=instruction))
    assert len(rows)==100 and len({r['episode_id'] for r in rows})==100
    return rows


def instrument_loading(rd,tf):
    original=inspect.getsource(rd.make_dataset_from_rlds)
    anchor='        # apply a standardization function, if provided\n'
    assert original.count(anchor)==1
    patched=original.replace(anchor,'        p3_fields = _p3_capture_provenance(traj)\n'+anchor)
    anchor2='        if absolute_action_mask is not None:\n'
    assert patched.count(anchor2)==1
    patched=patched.replace(anchor2,'        traj.update(p3_fields)\n\n'+anchor2)
    anchor3='split = "train[:95%]" if train else "train[95%:]"'
    assert patched.count(anchor3)==1
    patched=patched.replace(anchor3,'split = "train" if train else "train[95%:]"')
    def capture(traj):
        n=tf.shape(traj['action'])[0]
        def identify(actions,length,instruction):
            return fingerprint(actions.numpy(),int(length.numpy()),instruction.numpy()).encode()
        identity=tf.py_function(identify,[traj['action'],n,traj['language_instruction'][0]],Tout=tf.string)
        identity=tf.ensure_shape(identity,[])
        return dict(p3_episode_id=tf.repeat(identity,n),p3_file_path=traj['traj_metadata']['episode_metadata']['file_path'],
                    p3_frame_index=traj['_frame_index'],p3_episode_length=traj['_len'])
    rd.__dict__['_p3_capture_provenance']=capture
    directory=ROOT/'data_preparation';directory.mkdir(parents=True,exist_ok=True)
    (directory/'upstream_loader.py').write_text(original)
    snapshot=directory/'instrumented_loader.py';snapshot.write_text(patched)
    exec(compile(patched,str(snapshot),'exec'),rd.__dict__)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    if (CACHE/'complete.json').exists():
        d=json.loads((CACHE/'complete.json').read_text());assert d['samples']==96000
        assert d['sample_trace_sha256']==sha256(CACHE/'sample_trace.csv')
        print('Verified existing complete P3 cache');return
    if CACHE.exists() and any(CACHE.iterdir()):raise FileExistsError('Preserve incomplete attempt before restarting: '+str(CACHE))
    CACHE.mkdir(parents=True,exist_ok=True)
    targets=original_target_manifest();target_map={r['episode_id']:r for r in targets}
    (CACHE/'original_target_demonstrations.json').write_text(json.dumps(targets,indent=2)+'\n')
    sys.path[:0]=[str(SOURCE),str(SOURCE/'policy')]
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.random.set_seed(20261109);np.random.seed(20261109)
    import prismatic.vla.datasets.rlds.dataset as rd
    instrument_loading(rd,tf)
    from prismatic.vla.datasets.datasets import RLDSDataset
    dataset=RLDSDataset(DATA,'panda_xp_900_pnp',[(lambda x:x,1.)],resize_resolution=(224,224),shuffle_buffer_size=10000,
        train=True,image_aug=False,future_action_window_size=7,future_obj_pose_window_size=40,future_2D_trace_window_size=40,
        obj_pose_stride=8,ee_pose_2D_stride=8,image_window_size=1,dataset_statistics_map={'panda_pnp':'mg_pnp_lite'})
    (CACHE/'dataset_statistics.json').write_text(json.dumps(dataset.dataset_statistics,default=lambda x:x.tolist(),indent=2)+'\n')
    counts=Counter();per_demo={};frames=defaultdict(set);seen_hash=set();duplicate_hashes=0
    with (CACHE/'sample_trace.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['index','dataset','episode_id','original_frame_index','raw_sample_sha256'],lineterminator='\n');writer.writeheader()
        for i,sample in enumerate(dataset.dataset.as_numpy_iterator()):
            if i==96000:break
            name=sample['dataset_name'].decode();eid=sample['p3_episode_id'].decode();frame=int(sample['p3_frame_index'])
            original_file=sample['p3_file_path'].decode();length=int(sample['p3_episode_length'])
            if name=='panda_pnp' and eid not in target_map:raise ValueError('Target TFDS trajectory does not match original HDF5: '+eid)
            data=pickle.dumps(sample,protocol=4);digest=hashlib.sha256(data).hexdigest()
            with gzip.open(CACHE/f'sample-{i:06d}.pkl.gz','wb',compresslevel=1) as f:f.write(data)
            writer.writerow(dict(index=i,dataset=name,episode_id=eid,original_frame_index=frame,raw_sample_sha256=digest))
            counts[name]+=1;key=(name,eid);frames[key].add(frame)
            if key not in per_demo:per_demo[key]=dict(dataset=name,episode_id=eid,source_file=original_file,original_frames=length,sampled_positions=0,**({k:target_map[eid][k] for k in ['original_file','original_demo','direction']} if name=='panda_pnp' else {}))
            per_demo[key]['sampled_positions']+=1
            if digest in seen_hash:duplicate_hashes+=1
            seen_hash.add(digest)
            if (i+1)%1000==0:
                progress=dict(status='PREPARING',samples=i+1,dataset_counts=dict(counts),unique_target_demos=sum(k[0]=='panda_pnp' for k in frames),unique_source_demos=sum(k[0]=='mg_pnp_lite' for k in frames))
                temporary=CACHE/'progress.tmp';temporary.write_text(json.dumps(progress));temporary.replace(CACHE/'progress.json')
                print(json.dumps(progress),flush=True)
    assert sum(counts.values())==96000
    coverage=[]
    for key,row in sorted(per_demo.items()):coverage.append(dict(**row,unique_frames=len(frames[key]),unique_frame_fraction=len(frames[key])/row['original_frames']))
    write_csv(CACHE/'demonstration_coverage.csv',coverage)
    summary=dict(status='COMPLETE',samples=96000,dataset_length=len(dataset),dataset_counts=dict(counts),sample_seed=20261109,
        sample_trace_sha256=sha256(CACHE/'sample_trace.csv'),statistics_sha256=sha256(CACHE/'dataset_statistics.json'),
        sampling='fresh full-train official RLDS mixture; not repeated P2 cache',target_pool=100,source_pool=1800,
        unique_target_demonstrations=sum(k[0]=='panda_pnp' for k in frames),unique_source_demonstrations=sum(k[0]=='mg_pnp_lite' for k in frames),
        unique_frame_positions=sum(len(s) for s in frames.values()),repeated_frame_positions=96000-sum(len(s) for s in frames.values()),
        exact_repeated_preprocessed_samples=duplicate_hashes,prior_weight=.05,target_weight=1.,balance_by_transitions=True,
        split='all published train episodes, explicit deviation from P2 train[:95%]',source_code_unmodified=True,
        dataset_counts_full_sequence=dict(counts))
    # Compatibility with the measured upstream training launcher.
    (CACHE/'complete.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
