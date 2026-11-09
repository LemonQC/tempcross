"""Recover legacy source-demo identity by exact RGB hashes, without rerunning it."""
import csv,gzip,hashlib,json,pickle,collections
from pathlib import Path
import numpy as np
from baselines.common.paths import ROOT,DATA,PROJECT,write_json
from p3_validation.prepare_data import fingerprint


def main():
 import tensorflow as tf,tensorflow_datasets as tfds,dlimp as dl
 tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
 legacy=Path('/home/users/lemon/coft-p2/p2/matched-samples-6000');needed=collections.defaultdict(list)
 with (legacy/'sample_trace.csv').open() as f:
  for row in csv.DictReader(f):
   if row['dataset']!='mg_pnp_lite':continue
   raw=gzip.open(row['path'],'rb').read();assert hashlib.sha256(raw).hexdigest()==row['raw_sample_sha256'];sample=pickle.loads(raw)
   image=sample['observation']['image_primary'][0];instruction=sample['task']['language_instruction'].decode();key=(hashlib.sha256(image.tobytes()).hexdigest(),instruction);needed[key].append(int(row['index']))
 matched=collections.defaultdict(list);builder=tfds.builder_from_directory(str(DATA/'mg_pnp_lite/1.0.0'))
 @tf.function
 def transform(image):return dl.transforms.resize_image(tf.io.decode_image(image,expand_animations=False,dtype=tf.uint8),size=(224,224))
 for x in tfds.as_numpy(builder.as_dataset(split='train[:95%]',shuffle_files=False,decoders={'steps':tfds.decode.SkipDecoding()})):
  s=x['steps'];instruction=s['language_instruction'][0].decode();eid=fingerprint(s['action'],len(s['action']),instruction)
  ds=tf.data.Dataset.from_tensor_slices(s['observation']['image']).map(transform,num_parallel_calls=4).batch(64)
  frame=0
  for batch in ds.as_numpy_iterator():
   for image in batch:
    key=(hashlib.sha256(image.tobytes()).hexdigest(),instruction)
    if key in needed:matched[key].append((eid,frame))
    frame+=1
 resolved=[];unresolved=[]
 for key,indices in needed.items():
  candidates=matched.get(key,[]);ids={e for e,_ in candidates}
  for index in indices:
   if len(ids)==1:resolved.append(dict(cache_index=index,episode_id=next(iter(ids)),matching_frame_candidates=[i for _,i in candidates],exact_rgb_sha256=key[0]))
   else:unresolved.append(dict(cache_index=index,candidate_episode_ids=sorted(ids)))
 unique={r['episode_id'] for r in resolved};summary=dict(method='EXACT_ORIGINAL_TFDS_DECODE_AND_UPSTREAM_RESIZE_RGB_HASH_PLUS_INSTRUCTION',source_samples=2932,resolved_samples=len(resolved),unresolved_samples=len(unresolved),resolved_unique_source_demonstrations=len(unique),resolved_episode_ids=sorted(unique),all_matched_exact_rgb_and_raw_cache_sha256=len(resolved)==2932,target_demo_coverage='separate exact target audit',new_physics_steps=0)
 write_json(ROOT/'legacy_source_provenance.json',dict(summary=summary,resolved=resolved,unresolved=unresolved));write_json(PROJECT/'results/baselines/legacy_source_provenance_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
