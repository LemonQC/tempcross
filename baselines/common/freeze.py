"""Freeze untouched official 60--99 conditions and all selection rules."""
import json,subprocess,sys,shutil
import numpy as np
from baselines.common.paths import PROJECT,SOURCE,ROOT,TASKS,OFFICIAL_COMMIT,ROBOMIMIC_COMMIT,digest,write_json


def main():
 if (ROOT/'physical_launches.jsonl').exists():raise RuntimeError('Freeze is immutable after any new benchmark physics')
 assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==OFFICIAL_COMMIT
 sys.path.insert(0,str(SOURCE))
 from barx.benchmark import TASKS as specs,EMBODIMENTS
 from barx.evaluation_conditions import load_bundle,validate_condition_bundle_for_evaluation
 rows=[];integrity=[]
 for task in TASKS:
  for robot in ['panda','ur5e']:
   payload,states,models=load_bundle(SOURCE/'evaluation/conditions',task=task,embodiment=robot)
   validate_condition_bundle_for_evaluation(payload,states,models,task=task,camera_name=EMBODIMENTS[robot].camera,image_width=320,image_height=180,count=100)
   for i in range(60,100):
    e=payload['episodes'][i];obj=next(o for o in e['ep_meta']['object_cfgs'] if o['name']=='obj')
    rows.append(dict(task=task,robot=robot,official_index=i,condition={k:e[k] for k in ['episode','condition_id','seed','state_sha256','model_sha256','metadata_sha256']},initial_state=np.asarray(states[i]).tolist(),object_identity=obj['info']['mjcf_path'],instruction=e['ep_meta']['lang'],max_steps=specs[task].max_steps,camera=EMBODIMENTS[robot].camera))
   integrity.append(dict(task=task,robot=robot,full100_validated_before_selection=True,json_sha256=digest(SOURCE/f'evaluation/conditions/{task}/{robot}.json'),npz_sha256=digest(SOURCE/f'evaluation/conditions/{task}/{robot}.npz')))
 out=PROJECT/'configs/evaluation';write_json(out/'baseline_conditions_60_99.json',rows);write_json(out/'baseline_full_bundle_integrity.json',integrity)
 data_manifests={}
 for domain in ['source','target']:
  p=ROOT/f'data/{domain}_manifest.json';assert p.is_file()
  q=PROJECT/f'configs/datasets/baseline_{domain}_manifest.json';q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  data_manifests[domain+'_manifest_sha256']=digest(q)
 config=dict(version='BASELINE_202611091535',baseline_commit='bd0e2864eece26fe27b2b4ccb4fae889cbdf7c82',official_source_commit=OFFICIAL_COMMIT,robomimic_commit=ROBOMIMIC_COMMIT,
  tasks={t:dict(max_steps=specs[t].max_steps,control_hz=20,target_robot='PandaOmron',gripper='Robotiq85Gripper',success='unchanged official info.is_success.task') for t in TASKS},
  development_indices=[0,1],final_indices=list(range(60,100)),per_cell_episodes=40,
  information=dict(policy_keys=['rgb','instruction'],no_proprioception=True,no_depth=True,no_point_cloud=True,no_object_truth=True,telemetry_never_passed_to_policy=True),
  data=dict(target_pool=100,target_train=90,target_validation=10,per_task_train=45,per_task_validation=5,split_rule='whole demos sorted sha256(20261109:episode_id); first 10% validation, separately by task/robot',source_pool=1800,source_embodiments=['iiwa','kinova3','ur5e'],source_per_task_per_robot=300,source_train=1620,source_validation=180,
   rgb=dict(size=[128,128],resize='OpenCV INTER_AREA, RGB, no added flip; same in train/inference',train_crop=[116,116],eval_crop='upstream center crop'),language='frozen official CLIP-L/14 projection, exact per-episode instruction; official max_length25 padding convention',actions='identity canonical 7D normalized base-frame OSC delta + actual [-1,+1] gripper; base/torso0 and arm-mode-1',terminal='official RLDS last action repeat prior gripper and zero arm'),
  sampling=dict(seed=20261109,target='task-balanced, demonstration-balanced, uniform anchor frame; identical target sequence shared by B/C',source='uniform over 3 source embodiments and 2 tasks, demonstration-balanced, uniform frame',joint='one target minibatch and one source minibatch per update; target exposure identical to matched B, twice total exposure'),
  budgets=dict(initial_anchor_positions=96000,extension_tiers=[192000,384000,768000],batch_candidates_target=[16,8,4],joint_batch='twice selected target batch',resource_selection='first successful actual forward/backward/optimizer step; discard probe and restart from identical seed',validation_every_target_anchors=12000,validation_anchors=256,
   stop='At each tier >=96000, stop if relative reduction between means of last3 and prior3 fixed-validation metrics is <=2%; otherwise next tier, capped at768000. B/C architecture pairs extend together until both plateau or cap.',selection='final checkpoint at common stopped tier, never final test; validation diagnoses convergence and determines predeclared extension only',source_only='same rule using source validation only, zero target demos/validation',metrics='BC fixed-seed expected-action MSE and NLL separately; DP fixed-noise denoising MSE',unconverged_at_cap='report BUDGET_LIMITED_NOT_CONVERGED, do not declare algorithm intrinsically worse'),
  algorithms=dict(bc_gmm='official BC_GMM, ResNet18+SpatialSoftmax64, GMM5, actor1024x1024',bc_rnn='official BC_RNN_GMM, LSTM400x2, context10, GMM5',bc_transformer='official BC_Transformer_GMM, 512d/6layers/8heads/context10, GMM5',diffusion_policy='official robomimic port, UNet[256,512,1024], obs2/action8/pred16, EMA.75, 100 training noise steps, 10-step DDIM inference'),
  optimizer=dict(bc_gmm='Adam1e-4 constant',bc_rnn='Adam1e-4 constant',bc_transformer='AdamW1e-4, wd.01 constant; declared fixed-step schedule deviation',diffusion_policy='AdamW1e-4 wd1e-6, cosine to cap, warmup500; upstream scheduler/EMA'),
  models=['A_barx_no_reps','A_barx_joint','A_source_bc_transformer','A_source_diffusion_policy','B_bc_gmm','B_bc_rnn','B_bc_transformer','B_diffusion_policy','C_barx_no_reps_6000','C_barx_joint_96000','C_bc_transformer','C_diffusion_policy'],
  reference_robots=dict(ur5e=['A_barx_no_reps','A_barx_joint']),smoke=dict(per_model_per_task_conditions=[0],full_rollout=True,probe_updates=1),
  legacy_barx='Reuse immutable source/LoRA weights; full100 target exposure and 224px preprocessing differ from new 90/10-128px models; separate legacy setting, not matched-data causal ranking',
  tests=dict(condition_manifest_sha256=digest(out/'baseline_conditions_60_99.json'),bundle_manifest_sha256=digest(out/'baseline_full_bundle_integrity.json'),**data_manifests))
 write_json(PROJECT/'configs/baselines/protocol.json',config);write_json(ROOT/'protocol.json',config)
 print(json.dumps(dict(frozen_conditions=len(rows),physics_started=False,config_sha256=digest(PROJECT/'configs/baselines/protocol.json'))))
if __name__=='__main__':main()
