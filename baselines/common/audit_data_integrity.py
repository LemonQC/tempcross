"""Read-only action/split and target raw-HDF versus published-TFDS image audit."""
import collections,json
import cv2,h5py,numpy as np
from baselines.common.paths import ROOT,DATA,PROJECT,write_json,digest
from p3_validation.prepare_data import fingerprint

def main():
 import tensorflow as tf,tensorflow_datasets as tfds
 tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
 result=dict(new_training_or_physics=False,domains={});target={}
 for domain in ['target','source']:
  rows=json.loads((ROOT/f'data/{domain}_manifest.json').read_text());lo=np.full(7,np.inf);hi=-lo;count=0;markers=0
  closed=json.loads((ROOT/f'data/{domain}_complete.json').read_text());assert closed['sha256']==digest(ROOT/f'data/{domain}.h5')
  with h5py.File(ROOT/f'data/{domain}.h5') as f:
   assert set(f)=={r['episode_id'] for r in rows}
   for r in rows:
    g=f[r['episode_id']];a=g['actions'][:];assert len(a)==len(g['rgb'])==r['frames'];assert np.isfinite(a).all() and np.abs(a).max()<=1.000001
    if 'pack_complete' in g.attrs:assert g.attrs['pack_complete'];markers+=1
    assert fingerprint(a,len(a),r['instruction'])==r['episode_id'];lo=np.minimum(lo,a.min(0));hi=np.maximum(hi,a.max(0));count+=len(a)
    if domain=='target':target[r['episode_id']]=r
  result['domains'][domain]=dict(demonstrations=len(rows),frames=count,action_min=lo.tolist(),action_max=hi.tolist(),all_finite=True,canonical_identity_actions=True,all_episode_fingerprints_match=True,pack_complete_group_markers=markers,complete_file_sha256_verified=True)
 builder=tfds.builder_from_directory(str(DATA/'panda_pnp/1.0.0'));pairs=[]
 with h5py.File(ROOT/'data/target.h5') as packed:
  for x in tfds.as_numpy(builder.as_dataset(split='train',shuffle_files=False,decoders={'steps':tfds.decode.SkipDecoding()})):
   steps=x['steps'];a=steps['action'];instruction=steps['language_instruction'][0].decode();eid=fingerprint(a,len(a),instruction);assert eid in target;g=packed[eid];assert np.array_equal(a,g['actions'][:])
   for i in [0,len(a)//2,len(a)-1]:
    rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(steps['observation']['image'][i],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB);official=cv2.resize(rgb,(128,128),interpolation=cv2.INTER_AREA).astype(np.float32);actual=g['rgb'][i].astype(np.float32)
    pairs.append(dict(episode_id=eid,frame=i,raw_vs_tfds_jpeg_mean_abs_pixel_error=float(np.abs(actual-official).mean()),vertically_flipped_error=float(np.abs(actual[::-1]-official).mean()),red_blue_swapped_error=float(np.abs(actual[...,::-1]-official).mean())))
 assert len(pairs)==300
 same=np.mean([x['raw_vs_tfds_jpeg_mean_abs_pixel_error'] for x in pairs]);flipped=np.mean([x['vertically_flipped_error'] for x in pairs]);swapped=np.mean([x['red_blue_swapped_error'] for x in pairs]);assert same<flipped and same<swapped
 result['target_published_tfds_comparison']=dict(episodes=100,frames=300,all_canonical_actions_exactly_equal=True,mean_raw_vs_lossy_jpeg_error=float(same),mean_wrong_vertical_flip_error=float(flipped),mean_wrong_red_blue_swap_error=float(swapped),orientation_and_color_order_verified=True,raw_hdf_images_preserve_original_quality=True,pairs=pairs)
 write_json(ROOT/'data_integrity_audit.json',result);write_json(PROJECT/'results/baselines/data_integrity_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='target_published_tfds_comparison'}));print('Target raw/TFDS mean absolute errors, correct/flip/RBswap:',same,flipped,swapped)
if __name__=='__main__':main()
