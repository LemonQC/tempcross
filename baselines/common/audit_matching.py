"""Audit actual B/C initialization and consumed target traces; no new model run."""
import csv,hashlib,json
from baselines.common.paths import ROOT,PROJECT,digest,write_json

def target_trace(path):
 h=hashlib.sha256();n=0
 with path.open() as f:
  for row in csv.DictReader(f):
   if row['domain']!='target':continue
   h.update((row['episode_id']+','+str(int(row['anchor_frame']))+'\n').encode());n+=1
 return dict(target_positions=n,ordered_episode_id_and_anchor_frame_sha256=h.hexdigest())

def main():
 pairs=[];blocked=[]
 for method in ['bc_transformer','diffusion_policy']:
  directories=[ROOT/'training'/f'{g}_{method}' for g in ['B','C']]
  if not all((d/'selected.json').exists() for d in directories):
   blocked.append(dict(method=method,reason='Matched selected checkpoints unavailable; no paired data-effect claim'));continue
  states=[json.loads((d/'selected.json').read_text()) for d in directories]
  initial=[digest(d/'initial_trainable.pt') for d in directories];configs=[digest(d/'config.json') for d in directories];traces=[target_trace(d/'consumed_anchors.csv') for d in directories]
  assert all(initial[i]==states[i]['initial_trainable_sha256'] for i in [0,1]);assert initial[0]==initial[1] and configs[0]==configs[1] and traces[0]==traces[1]
  assert states[0]['actual_target_positions']==states[1]['actual_target_positions']==traces[0]['target_positions'];assert states[0]['completed_optimizer_steps']==states[1]['completed_optimizer_steps']
  pairs.append(dict(method=method,initial_trainable_serialized_file_sha256=initial[0],initial_trainable_tensors_bitwise_identical=True,architecture_and_optimizer_config_sha256=configs[0],actual_ordered_target_trace=traces[0],optimizer_updates_per_model=states[0]['completed_optimizer_steps'],batch_B=states[0]['actual_batch'],batch_C=states[1]['actual_batch'],C_additional_source_positions=states[1]['actual_source_positions'],same_target_anchor_sequence_and_exposure=True,augmentation='Same preprocessing and stochastic-crop configuration, not identical per-sample crop realizations; mixed C batch has different random draws and batch statistics',interpretation='Joint1:1 source addition at matched target exposure and initialization, with larger C total data/compute; not equal-total-compute or a unique mechanistic attribution'))
 result=dict(status='VERIFIED_ACTUAL_MATCHED_INITIALIZATION_AND_TARGET_EXPOSURE' if not blocked else 'AVAILABLE_PAIRS_VERIFIED_WITH_BLOCKED_UNAVAILABLE_PAIRS',no_training_inference_or_physics=True,pairs=pairs,blocked_pairs=blocked)
 write_json(ROOT/'matched_data_audit.json',result);write_json(PROJECT/'results/baselines/matched_data_audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
