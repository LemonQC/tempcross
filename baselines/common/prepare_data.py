"""Pack existing released data without new demonstrations or privileged inputs."""
import argparse,collections,csv,json,re
from pathlib import Path
import h5py,numpy as np,cv2
from baselines.common.paths import ROOT,DATA,DIRECTIONS,TASKS,digest,write_json
from p3_validation.prepare_data import fingerprint


def split_rows(rows):
 # Entire demonstrations, not neighboring frames, are held out.
 for domain in ['target','source']:
  for task in TASKS:
   robots=sorted({r['embodiment'] for r in rows if r['domain']==domain and r['task']==task})
   for robot in robots:
    group=sorted([r for r in rows if r['domain']==domain and r['task']==task and r['embodiment']==robot],key=lambda r:__import__('hashlib').sha256(('20261109:'+r['episode_id']).encode()).hexdigest())
    nval=max(1,len(group)//10)
    for i,r in enumerate(group):r['split']='validation' if i<nval else 'train'
 return rows


def store_episode(out,eid,images,actions,instruction,metadata):
 if eid in out:raise ValueError('Duplicate original episode identity '+eid)
 g=out.create_group(eid);n=len(actions);g.attrs['instruction']=instruction;g.attrs['provenance']=json.dumps(metadata)
 g.create_dataset('actions',data=np.asarray(actions,dtype=np.float32))
 d=g.create_dataset('rgb',shape=(n,128,128,3),dtype='uint8',chunks=(1,128,128,3),compression='lzf')
 # Released target images use large HDF5 chunks. Read once per trajectory,
 # rather than repeatedly decompressing the same chunk for each image.
 if isinstance(images,h5py.Dataset):images=images[()]
 for i,image in enumerate(images):
  if isinstance(image,(bytes,np.bytes_)):
   image=cv2.cvtColor(cv2.imdecode(np.frombuffer(image,np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
  assert image.ndim==3 and image.shape[-1]==3
  d[i]=cv2.resize(image,(128,128),interpolation=cv2.INTER_AREA)
 g.attrs['pack_complete']=True


def target():
 folder=ROOT/'data';folder.mkdir(parents=True,exist_ok=True);path=folder/'target.h5'
 if path.exists():raise FileExistsError('Preserve existing pack '+str(path))
 rows=[]
 with h5py.File(path,'w') as out:
  for p in sorted((DATA/'raw_target/human/PandaOmron').glob('*/demo_gentex_im320.hdf5')):
   task=DIRECTIONS[p.parent.name]
   with h5py.File(p) as f:
    for name in sorted(f['data']):
     demo=f['data'][name];raw=demo['actions'][:];assert raw.shape[1]==12
     assert np.all(raw[:,7:11]==0) and np.all(raw[:,11]==-1)
     a=raw[:,:7].astype(np.float32);a[-1]=a[-2];a[-1,:6]=0
     instruction=json.loads(demo.attrs['ep_meta'])['lang'];eid=fingerprint(a,len(a),instruction)
     row=dict(domain='target',embodiment='panda',task=task,episode_id=eid,original_file=str(p),original_demo=name,frames=len(a),instruction=instruction)
     store_episode(out,eid,demo['obs/agentview_rgb'],a,instruction,row);rows.append(row)
 assert len(rows)==100
 split_rows(rows);write_json(folder/'target_manifest.json',rows)
 write_json(folder/'target_complete.json',dict(status='COMPLETE',demonstrations=100,train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='validation' for r in rows),packed_file=str(path),bytes=path.stat().st_size,sha256=digest(path),image_resolution=[128,128],terminal_action_rule='official RLDS: repeat previous gripper; final arm delta zero',policy_observations=['rgb','verbatim language instruction']))
 print('Target data complete',flush=True)


def source():
 import tensorflow as tf,tensorflow_datasets as tfds
 tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
 folder=ROOT/'data';folder.mkdir(parents=True,exist_ok=True);path=folder/'source.h5'
 if path.exists():raise FileExistsError('Preserve existing pack '+str(path))
 builder=tfds.builder_from_directory(str(DATA/'mg_pnp_lite/1.0.0'))
 dataset=builder.as_dataset(split='train',shuffle_files=False,decoders={'steps':tfds.decode.SkipDecoding()})
 rows=[];counts=collections.Counter()
 with h5py.File(path,'w') as out:
  for x in tfds.as_numpy(dataset):
   s=x['steps'];a=s['action'];instruction=s['language_instruction'][0].decode();original=x['episode_metadata']['file_path'].decode()
   robot=next((r for r in ['IIWA','Kinova3','UR5e'] if '/'+r+'Omron/' in original),None)
   task=next((task for direction,task in DIRECTIONS.items() if '/'+direction+'/' in original),None)
   if robot is None or task is None:raise ValueError(original)
   assert a.shape[1]==7 and np.isfinite(a).all() and np.max(np.abs(a))<=1.000001
   # TFDS contains the actual canonical simulator gripper in [-1,+1],
   # before BARX's learning transform remaps it to [0,1]. Do not invert it.
   eid=fingerprint(a,len(a),instruction)
   row=dict(domain='source',embodiment=robot.lower(),task=task,episode_id=eid,original_file=original,original_demo='CONTENT_ID_'+eid,frames=len(a),instruction=instruction)
   store_episode(out,eid,s['observation']['image'],a,instruction,row);rows.append(row);counts[(robot,task)]+=1
   if len(rows)%20==0:
    out.flush();write_json(folder/'source_progress.json',dict(status='PACKING',episodes=len(rows),counts={str(k):v for k,v in counts.items()}));print('Packed source episodes',len(rows),flush=True)
 assert len(rows)==1800 and len(counts)==6 and set(counts.values())=={300},dict(counts)
 split_rows(rows);write_json(folder/'source_manifest.json',rows)
 write_json(folder/'source_complete.json',dict(status='COMPLETE',demonstrations=len(rows),train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='validation' for r in rows),counts={str(k):v for k,v in counts.items()},packed_file=str(path),bytes=path.stat().st_size,sha256=digest(path),canonical_gripper='raw TFDS [-1,+1]; no inverse of pretraining-normalized labels'))
 print('Source data complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['target','source']);a=p.parse_args()
 (target if a.kind=='target' else source)()
