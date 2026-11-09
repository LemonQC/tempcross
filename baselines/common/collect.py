"""Collect only actual training and closed simulation results; never manufacture 0%."""
import argparse,collections,csv,json,math,shutil
from pathlib import Path
import numpy as np,h5py
from baselines.common.paths import ROOT,PROJECT,TASKS,digest,write_json
from baselines.common.dataset_adapter import window_indices
from baselines.common.metrics import summarize_states
from simulation_validation.analyze_rollouts import wilson


def csv_write(path,rows):
 if not rows:Path(path).write_text('');return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def copy_small(source,destination):
 destination.parent.mkdir(parents=True,exist_ok=True);text=source.read_text()
 # Active CSV logs can have an incomplete final line; never export that row.
 if source.suffix=='.csv' and not text.endswith('\n'):text=text[:text.rfind('\n')+1]
 destination.write_text(text)


def coverage(directory,state,out):
 path=directory/'consumed_anchors.csv';method=state['method'];group=state['group']
 if method=='bc_transformer':assert json.loads((directory/'config.json').read_text())['algo']['transformer']['supervise_all_steps'] is False
 prior=out/'consumption_summary.json'
 if prior.exists():
  cached=json.loads(prior.read_text())
  if cached.get('coverage_schema')==2 and cached['trace_bytes']==path.stat().st_size and cached['positions'].get('target',0)==state['actual_target_positions'] and cached['positions'].get('source',0)==state['actual_source_positions']:
   assert cached['trace_sha256']==digest(path);return cached
 manifests={d:json.loads((ROOT/f'data/{d}_manifest.json').read_text()) for d in ['source','target']}
 maps={d:{r['episode_id']:r for r in rows} for d,rows in manifests.items()}
 anchors={d:np.load(ROOT/f'data/{d}_train_anchors.npy',mmap_mode='r') for d in manifests};counts=collections.Counter();entries={};expanded=collections.Counter();position=collections.Counter()
 with path.open() as f:
  for r in csv.DictReader(f):
   d=r['domain'];eid=r['episode_id'];frame=int(r['anchor_frame']);j=position[d];row_index,expected_frame=anchors[d][j]
   assert manifests[d][int(row_index)]['episode_id']==eid and frame==int(expected_frame)
   row=maps[d][eid];assert row['split']=='train';position[d]+=1;counts[d]+=1
   key=d,eid
   if key not in entries:entries[key]=dict(row=row,n=0,anchor=set(),inputs=set(),labels=set(),loaded_labels=set())
   x=entries[key];x['n']+=1;x['anchor'].add(frame);obs,labels=window_indices(method,frame,row['frames'])
   assert int(r['input_frame_count'])==len(obs) and int(r['label_frame_count'])==len(labels)
   supervised=labels[-1:] if method=='bc_transformer' else labels
   x['inputs'].update(map(int,obs));x['labels'].update(map(int,supervised));x['loaded_labels'].update(map(int,labels));expanded[(d,'input')]+=len(obs);expanded[(d,'label')]+=len(supervised);expanded[(d,'loaded_label')]+=len(labels)
 assert counts['target']==state['actual_target_positions'] and counts['source']==state['actual_source_positions']
 rows=[]
 for (d,eid),x in sorted(entries.items()):
  r=x['row'];rows.append(dict(domain=d,episode_id=eid,task=r['task'],embodiment=r['embodiment'],original_file=r['original_file'],original_demo=r['original_demo'],original_frames=r['frames'],anchor_positions=x['n'],unique_anchor_frames=len(x['anchor']),unique_input_frames=len(x['inputs']),unique_supervised_label_frames=len(x['labels']),unique_loaded_action_frames=len(x['loaded_labels'])))
 csv_write(out/'demonstration_coverage.csv',rows)
 result=dict(coverage_schema=2,trace_path=str(path),trace_bytes=path.stat().st_size,trace_sha256=digest(path),exact_sequence_matches_frozen_anchors=True,positions=dict(counts),unique_demonstrations={d:sum(k[0]==d for k in entries) for d in manifests},expanded_input_positions={d:expanded[(d,'input')] for d in manifests},expanded_label_positions={d:expanded[(d,'label')] for d in manifests},expanded_loaded_action_positions={d:expanded[(d,'loaded_label')] for d in manifests},label_accounting='Transformer loads10 historical labels but the frozen supervise_all_steps=False core supervises only the last; GMM1, RNN10, DP16',unique_anchor_frames={d:sum(len(x['anchor']) for (domain,_),x in entries.items() if domain==d) for d in manifests})
 write_json(out/'consumption_summary.json',result);return result


def training(out,final):
 rows=[];checkpoints=[]
 for directory in sorted((ROOT/'training').glob('*')):
  path=directory/'status.json'
  if not path.exists():continue
  s=json.loads(path.read_text());dest=out/'training'/directory.name;dest.mkdir(parents=True,exist_ok=True);write_json(dest/'status.json',s)
  for name in ['config.json','optimizer_metrics.csv','parameter_changes.json','selected.json']:
   p=directory/name
   if p.exists():copy_small(p,dest/name)
  result=None
  if s['status']=='TIER_COMPLETE':
   result=coverage(directory,s,dest);assert s['nonfinite_losses']==0
   with (dest/'optimizer_metrics.csv').open() as f:metrics=list(csv.DictReader(f))
   assert len(metrics)==s['completed_optimizer_steps'] and all(int(r['optimizer_step'])==i+1 and np.isfinite(float(r['raw_loss'])) for i,r in enumerate(metrics))
   if final:assert s['checkpoint_sha256']==digest(s['checkpoint'])
   checkpoints.append(dict(label=directory.name,path=s['checkpoint'],bytes=Path(s['checkpoint']).stat().st_size,sha256=s['checkpoint_sha256'],optimizer_updates=s['completed_optimizer_steps'],target_positions=s['actual_target_positions'],source_positions=s['actual_source_positions']))
  rows.append(dict(label=directory.name,method=s['method'],group=s['group'],status=s['status'],optimizer_updates=s['completed_optimizer_steps'],target_positions=s['actual_target_positions'],source_positions=s['actual_source_positions'],total_positions=s['actual_target_positions']+s['actual_source_positions'],actual_batch=s['actual_batch'],anchor_batch_per_domain=s['batch_target'],parameter_count=s.get('parameter_count'),actual_unique_target_demonstrations=result['unique_demonstrations']['target'] if result else None,actual_unique_source_demonstrations=result['unique_demonstrations']['source'] if result else None,gpu_occupied_wall_seconds=s.get('gpu_seconds'),peak_gpu_allocated_bytes=s.get('peak_gpu_allocated_bytes'),selected=(directory/'selected.json').exists(),checkpoint=s.get('checkpoint'),checkpoint_sha256=s.get('checkpoint_sha256'),budget_stop_reason=json.loads((directory/'selected.json').read_text()).get('budget_stop_reason') if (directory/'selected.json').exists() else None))
 from baselines.barx.registry import LEGACY
 legacy=json.loads((ROOT/'legacy_data_settings.json').read_text())
 recovered=PROJECT/'results/baselines/legacy_target_provenance_summary.json'
 actual_unique=json.loads(recovered.read_text())['resolved_unique_target_demonstrations'] if recovered.exists() else None
 for label,path in LEGACY.items():
  adaptation='C_' in label;no_reps='no_reps' in label;key='no_reps_6000' if no_reps else 'joint_96000';cache=legacy[key]['cache'] if adaptation else {}
  counts=cache.get('dataset_counts',{});target=counts.get('panda_pnp',0);source=counts.get('mg_pnp_lite',0)
  metadata=dict(label=label,path=str(path),bytes=path.stat().st_size,sha256=digest(path),optimizer_updates=(3000 if no_reps else 12000) if adaptation else 'RELEASED_50000_SOURCE_PRETRAINING_UPDATES_NOT_RUN_HERE',target_positions=target,source_positions=source if adaptation else 'RELEASED_PRETRAINING_EXPOSURE_NOT_RECONSTRUCTED',actual_target_unique_demonstrations=(actual_unique if no_reps else 100) if adaptation else 0,budget_setting='LEGACY_DIFFERENT_SPLIT_AND_BUDGET_RESTRICTED_LORA' if adaptation else 'RELEASED_SOURCE_PRIOR',new_training_this_benchmark=False)
  checkpoints.append(metadata)
  rows.append(dict(label=label,group=label[0],method='barx',status='IMMUTABLE_EXISTING_CHECKPOINT_REUSED',optimizer_updates=metadata['optimizer_updates'],target_positions=target,source_positions=metadata['source_positions'],total_positions=target+source if adaptation else 'UNKNOWN_RELEASED_PRETRAINING',actual_batch=(2 if no_reps else 8) if adaptation else 'RELEASED_NOT_RUN_HERE',selected=True,actual_unique_target_demonstrations=metadata['actual_target_unique_demonstrations'],budget_setting=metadata['budget_setting'],checkpoint=str(path),checkpoint_sha256=metadata['sha256']))
 csv_write(out/'training_budgets.csv',rows);csv_write(out/'checkpoint_index.csv',checkpoints)
 return rows


def episodes(out,final):
 cells=[];rows=[];technical=[];pending=[]
 for path in sorted((ROOT/'formal').rglob('job.json')):
  job=json.loads(path.read_text());key=dict(label=job['label'],task=job['task'],robot=job['robot'])
  if job['status']!='COMPLETE':
   record=dict(**key,status=job['status'],completed_episodes=job.get('completed_episodes',0),technical_failure_phase=job.get('technical_failure_phase'),error=job.get('error'),job_path=str(path))
   (technical if job['status']=='TECHNICAL_FAILURE' else pending).append(record);continue
  records=[json.loads(x) for x in (path.parent/'episodes.jsonl').read_text().splitlines()];assert len(records)==40 and [r['official_condition_index'] for r in records]==list(range(60,100));assert all(type(r['success']) is bool for r in records)
  successes=sum(r['success'] for r in records);assert successes==job['successes']
  cell=dict(**key,status='COMPLETE',successes=successes,episodes=40,success_rate=successes/40,wilson95=wilson(successes,40),checkpoint=job['checkpoint'],checkpoint_sha256=job['checkpoint_sha256'],group=job['label'][0],legacy_setting='barx' in job['label'],model_kind=job['kind'],mean_policy_prediction_seconds=sum(r['total_prediction_seconds'] for r in records)/sum(r['policy_calls'] for r in records),peak_gpu_allocated_bytes=job['peak_gpu_allocated_bytes'])
  cells.append(cell);dest=out/'jobs'/job['label']/job['task']/job['robot'];dest.mkdir(parents=True,exist_ok=True)
  for name in ['job.json','episodes.jsonl']:shutil.copyfile(path.parent/name,dest/name)
  if final:
   from baselines.common.validate import validate_job
   validation=validate_job(path);shutil.copyfile(path.parent/'validation.json',dest/'validation.json')
  for r in records:
   telem=Path(r['telemetry_path']);contacts=[json.loads(line)['grasp_contact'] is True for line in telem.with_suffix('.jsonl').read_text().splitlines()]
   with h5py.File(telem) as f:
    actions=f['actions/environment'][:];post=f['baseline/post_action_tcp_position'][:];tcp=np.concatenate([f['baseline/pre_action_tcp_position'][0][None],post]);obj=f['target/position'][:];placement=f['baseline/post_action_placement'][:];retreat=f['baseline/post_action_retreat'][:]
    assert np.isfinite(actions).all() and len(obj)==len(contacts)==len(actions)+1
    states=summarize_states(tcp,obj,contacts,actions,placement,retreat,r['success'],f['baseline/gripper_pad_center_distance'][:])
   stage=('OFFICIAL_SUCCESS' if r['success'] else 'NO_CLOSE_COMMAND' if not states['issued_close'] else 'CLOSE_COMMAND_WITHOUT_5MM_JAW_REDUCTION' if not states['physical_jaw_closure_5mm'] else 'CLOSE_WITHOUT_BILATERAL_CONTACT' if not states['bilateral_contact'] else 'CONTACT_WITHOUT_STABLE_LIFT' if not states['stable_contact_lift'] else 'STABLE_LIFT_NO_10CM_TRANSPORT_PROXY' if not states['transport_10cm_contact_proxy'] else 'TRANSPORT_WITHOUT_OFFICIAL_PLACEMENT' if not states['placement_official_subpredicate'] else 'PLACEMENT_WITHOUT_RETREAT')
   rows.append(dict(**key,**r,**states,observed_failure_stage=stage,causal_failure_reason='NOT_IDENTIFIED_FROM_STAGE_PROXY'))
 csv_write(out/'cells.csv',cells);csv_write(out/'episodes.csv',rows);csv_write(out/'technical_status.csv',technical);csv_write(out/'pending_cells.csv',pending);csv_write(out/'failure_index.csv',[r for r in rows if not r['success']]);write_json(out/'analysis.json',dict(cells=cells,episodes=rows,technical=technical,pending=pending,formal_results_only=True))
 return cells,rows,technical


def paired(out,rows):
 by=collections.defaultdict(dict)
 for r in rows:
  if r['robot']=='panda':by[(r['label'],r['task'])][r['condition_id']]=r
 pairs=[];cases=[]
 for task in TASKS:
  specs=[('C_bc_transformer','B_bc_transformer'),('C_diffusion_policy','B_diffusion_policy'),('C_barx_joint_96000','A_barx_joint'),('C_barx_no_reps_6000','A_barx_no_reps')]
  for candidate,baseline in specs:
   a=by.get((candidate,task));b=by.get((baseline,task))
   if not a or not b:continue
   assert a.keys()==b.keys()
   rescued=[];regressed=[]
   for key in b:
    before,after=b[key],a[key]
    if not before['success'] and after['success']:rescued.append(before['official_condition_index'])
    if before['success'] and not after['success']:regressed.append(before['official_condition_index'])
    cases.append(dict(candidate=candidate,baseline=baseline,task=task,condition_id=key,official_index=before['official_condition_index'],baseline_success=before['success'],candidate_success=after['success'],baseline_stage=before['observed_failure_stage'],candidate_stage=after['observed_failure_stage']))
   pairs.append(dict(candidate=candidate,baseline=baseline,task=task,episodes=40,rescued=rescued,regressed=regressed,net_success_change=len(rescued)-len(regressed),interpretation='MATCHED_NEW_TARGET_SPLIT_AND_EXPOSURE' if not 'barx' in candidate else 'LEGACY_DIFFERENT_TRAINING_BUDGET_NOT_CONTROLLED_DATA_EFFECT'))
 csv_write(out/'paired_comparisons.csv',pairs);csv_write(out/'paired_cases.csv',cases);return pairs


def collect(final=False):
 out=PROJECT/'results/baselines';out.mkdir(parents=True,exist_ok=True)
 tr=training(out,final);cells,rows,technical=episodes(out,final);comparisons=paired(out,rows)
 for name in ['probe_suite.json','probe_suite_technical_attempt_01.json','smoke_suite.json','barx_smoke_suite.json','smoke_validation_summary.json','formal_validation_summary.json','closed_cell_audit.json','training_suite.json','formal_suite.json','frozen_publication.json','model_selection_publication.json','first_formal_interface_audit.json','canonical_control_semantics.json','legacy_data_settings.json','environment.json','data_integrity_audit.json','camera_calibration_audit.json']:
  p=ROOT/name
  if p.exists():shutil.copyfile(p,out/name)
 for p in (ROOT/'probes').glob('*/status.json'):
  dest=out/'probes'/p.parent.name;dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest/'status.json')
  for name in ['config.json','parameter_changes.json']:
   if (p.parent/name).exists():shutil.copyfile(p.parent/name,dest/name)
 latency=out/'latency'
 if (ROOT/'latency').exists():
  latency.mkdir(exist_ok=True)
  for p in (ROOT/'latency').glob('*.json'):shutil.copyfile(p,latency/p.name)
 new=[r for r in tr if r['method']!='barx']
 status=dict(status='COMPLETE' if final else 'ACTUAL_BENCHMARK_RUNNING',directive='202611091535.md',core_learning_algorithms=4,new_learning_settings=8,models_with_training_logs=len(new),models_with_completed_tiers=sum(r['status']=='TIER_COMPLETE' for r in new),models_with_selected_checkpoints=sum(r['selected'] for r in new),formal_complete_cells=len(cells),formal_episodes=sum(c['episodes'] for c in cells),formal_technical_cells=technical,training=tr,cells=cells,paired_comparisons=comparisons,execution_root=str(ROOT))
 if final:
  suite=json.loads((ROOT/'formal_suite.json').read_text());assert suite['status'].startswith('COMPLETE')
  status['blocked_methods_or_measurements']=suite.get('blocked',[])
  if technical or suite.get('blocked'):status['status']='COMPLETE_WITH_TECHNICAL_BLOCKERS'
  assert len(cells)==sum(c['status']=='COMPLETE' for c in suite['cells']) and all(c['episodes']==40 for c in cells)
  files=[]
  for p in sorted(ROOT.rglob('*')):
   if p.name=='delivery_ready.json' or p.name.startswith('delivery-step-'):continue
   if p.is_file() and p.suffix!='.tmp':files.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
  for r in csv.DictReader((out/'checkpoint_index.csv').open()):
   if not str(r['path']).startswith(str(ROOT)):files.append(dict(absolute_path=r['path'],bytes=int(r['bytes']),sha256=r['sha256']))
  from baselines.common.paths import DATA,ASSETS
  inputs=list((DATA/'mg_pnp_lite/1.0.0').glob('*'))+list((DATA/'panda_pnp/1.0.0').glob('*'))+list((DATA/'raw_target/human/PandaOmron').glob('*/demo_gentex_im320.hdf5'))+list((ASSETS/'clip-vit-large-patch14').glob('*'))+list(ROOT.parent.glob('*.log'))
  for p in sorted(inputs):
   if p.is_file():files.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
  csv_write(out/'external_files.csv',files)
  video=[];indexed={x['absolute_path']:x for x in files}
  for r in rows:
   v=indexed[r['video_path']];h=indexed[r['telemetry_path']];video.append(dict(label=r['label'],task=r['task'],robot=r['robot'],official_index=r['official_condition_index'],condition_id=r['condition_id'],success=r['success'],video_path=r['video_path'],video_bytes=v['bytes'],video_sha256=v['sha256'],telemetry_path=r['telemetry_path'],telemetry_bytes=h['bytes'],telemetry_sha256=h['sha256']))
  csv_write(out/'video_index.csv',video)
  representative=[];seen=set()
  for r in rows:
   key=(r['label'],r['task'],r['robot'],r['observed_failure_stage'])
   if key not in seen:
    seen.add(key);v=indexed[r['video_path']];representative.append(dict(label=r['label'],task=r['task'],robot=r['robot'],official_index=r['official_condition_index'],condition_id=r['condition_id'],success=r['success'],observed_stage=r['observed_failure_stage'],video_path=r['video_path'],video_sha256=v['sha256'],telemetry_path=r['telemetry_path'],selection_rule='FIRST_OFFICIAL_INDEX_WITH_THIS_OUTCOME_STAGE; descriptive posthoc only, no model selection'))
  csv_write(out/'representative_cases.csv',representative)
 write_json(out/'status.json',status);print(json.dumps({k:status[k] for k in ['status','models_with_training_logs','models_with_completed_tiers','models_with_selected_checkpoints','formal_complete_cells','formal_episodes']},indent=2))
 return status

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();collect(a.final)
