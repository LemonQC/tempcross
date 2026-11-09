"""Audit legacy BARX sampling facts without retraining or rerunning old cases."""
import collections,csv,json
from pathlib import Path
from baselines.common.paths import ROOT,DATA,PROJECT,write_json
from p3_validation.prepare_data import fingerprint


def main():
 import tensorflow as tf,tensorflow_datasets as tfds
 tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
 manifests=json.loads((ROOT/'data/target_manifest.json').read_text());known={r['episode_id']:r for r in manifests}
 b=tfds.builder_from_directory(str(DATA/'panda_pnp/1.0.0'));pool=[]
 for x in tfds.as_numpy(b.as_dataset(split='train[:95%]',shuffle_files=False,decoders={'steps':tfds.decode.SkipDecoding()})):
  s=x['steps'];instruction=s['language_instruction'][0].decode();eid=fingerprint(s['action'],len(s['action']),instruction);assert eid in known
  pool.append({k:known[eid][k] for k in ['episode_id','original_file','original_demo','task']})
 assert len(pool)==95
 p2=Path('/home/users/lemon/coft-p2/p2/matched-samples-6000');p3=Path('/home/users/lemon/coft-p3/matched-samples-96000');counts=collections.Counter();demos=collections.Counter()
 for r in csv.DictReader((p3/'demonstration_coverage.csv').open()):
  if r['dataset']=='mg_pnp_lite':
   robot=next(n for n in ['IIWA','Kinova3','UR5e'] if '/'+n+'Omron/' in r['source_file']);counts[robot]+=int(r['sampled_positions']);demos[robot]+=1
 result=dict(status='LEGACY_RECIPE_AUDITED_NO_RETRAINING',no_reps_6000=dict(cache=json.loads((p2/'complete.json').read_text()),eligible_target_train_episodes=95,eligible_target_train_ids=pool,actual_unique_target_demo_coverage='NOT_IDENTIFIABLE_FROM_OLD_CACHE_METADATA; do not claim all95 visited',target_pool_total=100,matched_new90_10_split=False),joint_96000=dict(cache=json.loads((p3/'complete.json').read_text()),actual_target_unique_demonstrations=100,source_positions_by_embodiment=dict(counts),source_unique_demos_by_embodiment=dict(demos),matched_new90_10_split=False),ranking='LEGACY_DIFFERENT_SPLIT_AND_BUDGET_RESTRICTED_LORA; no causal comparison to new matched90/10 models')
 for domain in ['target','source']:
  path=PROJECT/f'results/baselines/legacy_{domain}_provenance_summary.json'
  if path.exists():
   summary=json.loads(path.read_text());result['no_reps_6000'][f'{domain}_exact_provenance_recovery']=summary
   if summary['unresolved_samples']==0:result['no_reps_6000'][f'actual_unique_{domain}_demo_coverage']=summary[f'resolved_unique_{domain}_demonstrations']
 write_json(ROOT/'legacy_data_settings.json',result);write_json(PROJECT/'results/baselines/legacy_data_settings.json',result);print(json.dumps(dict(legacy_eligible95=len(pool),joint96k_source_counts=dict(counts)),indent=2))
if __name__=='__main__':main()
