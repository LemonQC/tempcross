"""Accept only actual native training and all80 original official-goal trials."""
import collections
import json
import subprocess
from pathlib import Path
import numpy as np
from .data import RUN,DIRECTORY
from .audit import GROUP,audit_job
from baselines.strong.common import ROOT,PROJECT,TASKS,out,read,write,digest,now

def exposure(train):
    manifest=read(DIRECTORY/'manifest.json');a=np.load(DIRECTORY/'train.npy')[:train['anchors_consumed']]
    count=collections.Counter(int(i) for i in a[:,0]);groups=collections.Counter();real=0
    for i,n in count.items():r=manifest[i];groups[(r['domain'],r['task'])]+=n
    lengths=np.array([manifest[int(i)]['frames'] for i in a[:,0]])
    real=int(np.minimum(16,lengths-a[:,1]).sum())
    result=dict(actual_optimizer_calls=train['actual_training_updates'],actual_anchors=len(a),
        unique_demonstrations=len(count),unique_demo_frame_anchors=len(np.unique(a,axis=0)),
        real_future_action_vectors=real,total_future_slots=len(a)*16,
        repeated_boundary_slots=len(a)*16-real,loss_masks_used=False,
        selected_checkpoint_anchors=train['selected_checkpoint_anchors_consumed'],
        groups=[dict(domain=k[0],task=k[1],anchors=v) for k,v in sorted(groups.items())],
        consumed_prefix_sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest(),
        train_anchor_path=str(DIRECTORY/'train.npy'),train_anchor_sha256=digest(DIRECTORY/'train.npy'),
        new_demonstrations=0,reference_rollouts_used_for_training=0)
    directory=out('flow_matching_policy');write(directory/'sample_exposure.json',result)
    lines=['episode_id,domain,task,anchors_seen']
    lines += [f"{manifest[i]['episode_id']},{manifest[i]['domain']},{manifest[i]['task']},{n}" for i,n in sorted(count.items())]
    (directory/'sample_exposure.csv').write_text('\n'.join(lines)+'\n');return result

def main():
    directory=out('flow_matching_policy');train=read(RUN/'training/source_target_visual/state.json')
    protocol=read(directory/'protocol.json');assert train['status']=='COMPLETE' and train['actual_training_updates']==protocol['training_updates']
    assert digest(train['checkpoint'])==train['checkpoint_sha256']
    optimizer=read(directory/'optimizer_audit.json');assert optimizer['saved_update']==protocol['training_updates']
    for r in protocol['code_files']+protocol['native_files']+protocol['data_files']:assert digest(r['path'])==r['sha256'],r['path']
    protection=read(RUN/'preservation_freeze.json')
    for r in protection['trees']:
        tree=subprocess.check_output(['git','rev-parse','HEAD:'+r['path']],cwd=PROJECT,text=True).strip();assert tree==r['tree']
    for r in protection['files']:assert digest(r['path'])==r['sha256'],r['path']
    write(directory/'prior_baselines_preserved.json',dict(status='PASS',base_commit=protection['base_commit'],
        old_result_trees_verified=len(protection['trees']),protected_files_verified=len(protection['files']),
        protected_smolvla_artifacts=protection['protected_smolvla_artifacts'],protected_octo_artifacts=protection['protected_octo_artifacts'],utc=now()))
    cells=[];rows=[];audits=[]
    for task in TASKS:
        audits.append(audit_job(task));a=read(RUN/'physical/formal'/GROUP/task/'acceptance.json')
        assert a['episodes']==40 and a['role']=='EXPOSED_REFERENCE'
        assert all(r['episode']['checkpoint_sha256']==train['checkpoint_sha256'] for r in a['rows'])
        rows.extend(a['rows']);cells.append(dict(group=GROUP,task=task,episodes=40,successes=a['successes'],
            success_rate=a['successes']/40,role='EXPOSED_REFERENCE',failure_counts=dict(collections.Counter(r['failure_stage'] for r in a['rows']))))
    launches=[json.loads(s) for s in (RUN/'physical/physical_launches.jsonl').read_text().splitlines()]
    assert len(launches)==82 and sum(r['role']=='DEVELOPMENT_SMOKE' for r in launches)==2
    metrics=[json.loads(s) for s in (RUN/'training/source_target_visual/training.jsonl').read_text().splitlines()]
    vals=[r for r in metrics if 'validation' in r];chosen=min(vals,key=lambda r:r['validation']['EMA_velocity_mse'])
    assert chosen['update']==train['best_step']
    result=dict(status='COMPLETE_FAITHFUL_PORT',scope='one joint official VISUAL flow-component port; target-only not run',
        original_robomimic_lowdim_recipe_reproduced=False,full_affordance_prompt_pipeline_reproduced=False,
        information_tier='RGB96_PLUS_REAL_QPOS9_PLUS_FROZEN_CLIP768',role='EXPOSED_REFERENCE',cells=cells,
        actual_training_updates=train['actual_training_updates'],discarded_probe_optimizer_calls=1,total_optimizer_calls=train['actual_training_updates']+1,
        anchors_consumed=train['anchors_consumed'],selected_update=train['best_step'],
        selected_checkpoint_effective_optimizer_calls=train['best_step'],selected_checkpoint_anchors_consumed=train['selected_checkpoint_anchors_consumed'],
        selected_validation_EMA_velocity_mse=train['best_validation'],checkpoint=train['checkpoint'],checkpoint_sha256=train['checkpoint_sha256'],
        actual_sample_exposure=exposure(train),native_queries=sum(r['native_queries'] for r in audits),
        official_control_steps=sum(r['episode']['steps'] for r in rows),formal_physical_episodes=80,development_smoke_episodes=2,
        original_network_topology_unchanged=True,interface_dimensions_changed=True,native_CFM_sigma0_and_velocity_loss_unchanged=True,
        native_uniform_single_Euler_generation_retained=True,Gaussian_training_uniform_deployment_mismatch_retained=True,
        official_goal='placement AND retreat',control_frequency_Hz=20,object_state_not_policy_input=True,
        prior_baseline_reruns=0,new_Method_runs=0,new_demonstrations=0,
        training_seconds=train['elapsed_seconds'],peak_training_allocated_bytes=train['peak_allocated_bytes'],utc=now())
    write(directory/'results.json',result);write(directory/'training_state.json',train)
    write(directory/'training_metrics.json',dict(records=metrics,source_path=str(RUN/'training/source_target_visual/training.jsonl'),source_sha256=digest(RUN/'training/source_target_visual/training.jsonl')))
    csv=['task,condition_index,success,steps,failure_stage,checkpoint_sha256,telemetry_path,video_path']
    for r in rows:
        e=r['episode'];csv.append(','.join(map(str,[e['task'],e['official_condition_index'],int(e['success']),e['steps'],r['failure_stage'],e['checkpoint_sha256'],e['telemetry_path'],e['video_path']])))
    (directory/'episodes.csv').write_text('\n'.join(csv)+'\n')
    (directory/'validation.csv').write_text('update,EMA_velocity_mse,selected\n'+'\n'.join(f"{r['update']},{r['validation']['EMA_velocity_mse']},{int(r['update']==train['best_step'])}" for r in vals)+'\n')
    files=[Path(train['checkpoint']),RUN/'training/source_target_visual/resume.pt',RUN/'training/source_target_visual/state.json',
        RUN/'training/source_target_visual/training.jsonl',RUN/'training/source_target_visual/initial_parameter_audit.json',RUN/'resource_samples.csv',RUN/'probe/updated.pt']
    for kind in ['formal','smoke']:
        for task in TASKS:
            a=read(RUN/'physical'/kind/GROUP/task/'acceptance.json')
            for row in a['rows']:
                e=row['episode'];files.extend(Path(e[k]) for k in ['video_path','telemetry_path','contact_log'])
                files.append(Path(e['telemetry_path']).parent.parent/'episode.json')
            files.append(Path(read(RUN/'physical'/kind/GROUP/task/'job.json')['runtime']['raw_action_ledger']))
    write(directory/'artifact_index.json',dict(large_files_remain_on_server=True,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))]))
    write(directory/'status.json',dict(status='COMPLETE_FAITHFUL_PORT',method='flow_matching_policy',phase='ACCEPTED',
        actual_training_updates=train['actual_training_updates'],physical_evaluation_episodes=80,physical_smoke_episodes=2,
        discarded_probe_updates=1,successes=sum(c['successes'] for c in cells),success_rate=sum(c['successes'] for c in cells)/80,
        information_tier=result['information_tier'],role='EXPOSED_REFERENCE',stop_this_round=True))
    lines=['# Official visual Flow Matching Policy port','',
        'COMPLETE_FAITHFUL_PORT for one source+target visual/state/language setting. This is the official visual flow-component port, not the original Robomimic object-state/absolute-action recipe or the full affordance-prompt framework.','',
        '| Task | Official successes/episodes |','|---|---:|']
    lines += [f"| {c['task']} | {c['successes']}/40 |" for c in cells]
    lines += ['',f"Actual formal updates:{train['actual_training_updates']}; discarded diagnostic update:1; full-run anchors:{train['anchors_consumed']}.",
        f"Selected native EMA update:{train['best_step']}, effective anchors:{train['selected_checkpoint_anchors_consumed']}, fixed-target-validation velocity MSE:{train['best_validation']:.8f}.",
        f"Checkpoint:`{train['checkpoint']}`; SHA256:`{train['checkpoint_sha256']}`.",
        '', 'Only validation chooses the checkpoint. Both development smokes completed600/650 steps but failed task success; they passed interface acceptance and are excluded from80 formal episodes.',
        'Author ResNet18/GN, original UNet widths/time embedding, TorchCFM sigma0 velocity MSE, native EMA and16-predict/8-execute mechanism are retained. Native Gaussian training versus uniform single-Euler deployment is preserved and disclosed, not silently corrected.',
        'RGB96 plus real qpos9 and frozen full-instruction CLIP768 are an explicitly separate information tier. No object poses, added camera or generated training data. Dataset/tensor-dimension, terminal-padding, finite-budget and script-driver differences are in PROTOCOL.md and OFFICIAL_AUDIT.md.','',
        '| Task | Descriptive failure category | Count |','|---|---|---:|']
    for c in cells:
        lines += [f"| {c['task']} | {k} | {v} |" for k,v in sorted(c['failure_counts'].items())]
    lines += ['', 'Contact/lift classifications are telemetry proxies, not causal explanations. Official task success is unchanged placement AND retreat. All60–99 conditions are EXPOSED_REFERENCE.',
        'Reproducibility: official_source.json, protocol.json, freeze.json, installed_runtime.txt, errors.json, optimizer_audit.json, sample_exposure.json, validation.csv, episodes.csv and artifact_index.json. Large weights, videos, telemetry and native action logs remain at the indexed absolute server paths.',
        'All previous baseline evidence is frozen and verified. No next baseline or new Method is started. Stop after GitHub publication and readback.']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix_current import main as matrix
    matrix();print(read(directory/'status.json'),flush=True)

if __name__=='__main__':main()
