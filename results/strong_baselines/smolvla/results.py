"""Only accept actual training, official complete tasks and immutable evidence."""
import collections
import json
import subprocess
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest,TASKS,now
from .audit import audit_job
from .exposure import main as exposure

RUN=ROOT/'smolvla'

def append_detail_tables(directory,result):
    """Render existing accepted statistics only; no training or physics."""
    path=directory/'RESULTS.md';marker='\n## Failure classification and frozen reference\n'
    lines=['', '## Failure classification and frozen reference','',
        'These are descriptive telemetry categories; contact/lift proxies do not establish failure causes.','',
        '| Task | Closed without bilateral contact | Contact without stable-lift proxy | Stable lift without official placement | Official success |',
        '|---|---:|---:|---:|---:|']
    for cell in result['cells']:
        f=cell['failure_counts']
        lines.append(f"| {cell['task']} | {f.get('CLOSE_WITHOUT_BILATERAL_CONTACT',0)} | {f.get('CONTACT_WITHOUT_STABLE_LIFT_PROXY',0)} | {f.get('STABLE_LIFT_WITHOUT_OFFICIAL_PLACEMENT',0)} | {f.get('SUCCESS',0)} |")
    prior=read(out('octo_small')/'results.json')
    lines += ['', '| Frozen/new joint baseline | Information | Counter to Sink | Sink to Counter |',
        '|---|---|---:|---:|',
        '| Octo-Small (frozen, no rerun) | RGB256 + language | '+ ' | '.join(f"{c['successes']}/{c['episodes']}" for c in prior['cells'])+' |',
        '| SmolVLA | RGB512 + language + real joint/gripper qpos9 | '+ ' | '.join(f"{c['successes']}/{c['episodes']}" for c in result['cells'])+' |',
        '', 'Both use EXPOSED_REFERENCE60–99. Input information, pretraining and budgets differ; this table does not establish an equal-input improvement. SmolVLA is a completed bounded faithful port, not the full official batch64/20000-update example recipe.',
        'This round ends here. No extra training, physics, next baseline or new Method is authorized by these results.']
    path.write_text(path.read_text().split(marker)[0]+'\n'.join(lines)+'\n')

def main():
    directory=out('smolvla');train=read(RUN/'training/source_target_state/state.json')
    protocol=read(directory/'protocol.json')
    assert train['status']=='COMPLETE' and train['actual_training_updates']==protocol['training_updates']
    assert digest(train['checkpoint'])==train['checkpoint_sha256']
    optimizer=read(directory/'optimizer_audit.json');assert optimizer['saved_update']==protocol['training_updates']
    assert optimizer['frozen_tensors_changed']==0
    for f in protocol['code_files']+protocol['native_files']+protocol['data_files']:assert digest(f['path'])==f['sha256'],f['path']
    protection=read(RUN/'preservation_freeze.json')
    for r in protection['trees']:
        current=subprocess.check_output(['git','rev-parse','HEAD:'+r['path']],cwd=PROJECT,text=True).strip();assert current==r['tree']
    for r in protection['protected_old_files']+protection['octo_artifacts']:assert digest(r['path'])==r['sha256'],r['path']
    write(directory/'prior_baselines_preserved.json',dict(status='PASS',base_commit=protection['base_commit'],trees=protection['trees'],
        protected_old_files_verified=len(protection['protected_old_files']),octo_artifact_files_verified=len(protection['octo_artifacts']),utc=now()))
    cells=[];accepted_rows=[];state_action=[]
    for task in TASKS:
        extra=audit_job(task,False);state_action.append(extra)
        acceptance=read(RUN/'physical/formal/source_target_state'/task/'acceptance.json')
        assert acceptance['episodes']==40 and acceptance['role']=='EXPOSED_REFERENCE'
        stages=collections.Counter(r['failure_stage'] for r in acceptance['rows'])
        cells.append(dict(group='source_target_state',task=task,episodes=40,successes=acceptance['successes'],
            success_rate=acceptance['successes']/40,role='EXPOSED_REFERENCE',failure_counts=dict(stages)))
        accepted_rows.extend(acceptance['rows'])
    launches=[json.loads(s) for s in (RUN/'physical/physical_launches.jsonl').read_text().splitlines()]
    assert len(launches)==82 and sum(s['role']=='DEVELOPMENT_SMOKE' for s in launches)==2
    samples=exposure()
    metrics=[json.loads(line) for line in (RUN/'training/source_target_state/training.jsonl').read_text().splitlines()]
    write(directory/'training_metrics.json',dict(records=metrics,source_path=str(RUN/'training/source_target_state/training.jsonl'),
        source_sha256=digest(RUN/'training/source_target_state/training.jsonl')))
    validations=[r for r in metrics if 'validation' in r]
    chosen=min(validations,key=lambda r:r['validation']['real7_valid_flow_mse'])
    assert chosen['update']==train['best_step']
    csv=['task,condition_index,success,steps,failure_stage,checkpoint_sha256,telemetry_path,video_path']
    files=[Path(train['checkpoint']),RUN/'training/source_target_state/resume.pt',RUN/'training/source_target_state/state.json',
        RUN/'training/source_target_state/training.jsonl',RUN/'data/pack_index.json',RUN/'resource_samples.csv']
    for a in accepted_rows:
        e=a['episode'];csv.append(','.join(map(str,[e['task'],e['official_condition_index'],int(e['success']),e['steps'],
            a['failure_stage'],e['checkpoint_sha256'],e['telemetry_path'],e['video_path']])))
        files.extend(Path(e[k]) for k in ['telemetry_path','video_path','contact_log'])
        files.append(Path(e['telemetry_path']).parent.parent/'episode.json')
    for r in state_action:files.append(Path(r['action_ledger']))
    result=dict(status='COMPLETE_FAITHFUL_PORT',scope='one joint UR5e+Panda source_target_state setting; target-only not run',
        information_tier='RGB_PLUS_NATIVE_PROPRIOCEPTION',role='EXPOSED_REFERENCE',cells=cells,
        actual_training_updates=train['actual_training_updates'],discarded_probe_updates=9,total_optimizer_calls=train['actual_training_updates']+9,
        actual_anchors=train['anchors_consumed'],actual_sample_exposure=samples,
        checkpoint=train['checkpoint'],checkpoint_sha256=train['checkpoint_sha256'],selected_update=train['best_step'],
        selected_checkpoint_training_updates=train['best_step'],
        selected_checkpoint_anchors_consumed=train['best_step']*protocol['batch_size'],
        selected_target_validation_real7_flow_mse=train['best_validation'],native_training_loss_changed=False,
        core_network_changed=False,extra_real_cameras=False,robot_state_used=True,object_state_used=False,
        official_full_task_episodes=80,development_smoke_episodes=2,official_control_steps=sum(a['episode']['steps'] for a in accepted_rows),
        native_queries=sum(r['native_queries'] for r in state_action),native_actions_and_qpos_verified=True,
        old_baseline_reruns=0,new_demonstrations=0,standalone_flow_matching_policy_runs=0,
        training_seconds=train['elapsed_seconds'],peak_training_allocated_bytes=train['peak_allocated_bytes'],utc=now())
    write(directory/'results.json',result);write(directory/'training_state.json',train)
    write(directory/'artifact_index.json',dict(large_files_remain_on_server=True,
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))]))
    (directory/'episodes.csv').write_text('\n'.join(csv)+'\n')
    (directory/'validation.csv').write_text('update,real7_valid_flow_mse,native_full32_masked_mean,selected\n'+
        '\n'.join(f"{r['update']},{r['validation']['real7_valid_flow_mse']},{r['validation']['native_full32_masked_mean']},{int(r['update']==train['best_step'])}" for r in validations)+'\n')
    status=dict(status='COMPLETE_FAITHFUL_PORT',method='smolvla',phase='ACCEPTED',information_tier='RGB_PLUS_NATIVE_PROPRIOCEPTION',
        actual_training_updates=train['actual_training_updates'],discarded_probe_optimizer_updates=9,
        physical_evaluation_episodes=80,physical_smoke_episodes=2,successes=sum(c['successes'] for c in cells),
        success_rate=sum(c['successes'] for c in cells)/80,role='EXPOSED_REFERENCE',ordinary_repairs_allowed=True)
    write(directory/'status.json',status);write(directory/'progress.json',dict(status,as_of_utc=now(),validation_history=validations))
    lines=['# SmolVLA state-conditioned joint baseline results','',
        'COMPLETE_FAITHFUL_PORT for one source_target_state setting. Native RGB+joint/gripper state; not RGB-only. Target-only was not run.',
        'Conditions60–99 are EXPOSED_REFERENCE; the original official placement AND retreat criterion is unchanged.','',
        '| Task | Official successes / episodes |','|---|---:|']
    lines += [f"| {c['task']} | {c['successes']}/40 |" for c in cells]
    lines += ['',f"Actual formal updates:{train['actual_training_updates']}; actual discarded probe updates:9; anchors:{train['anchors_consumed']}.",
        f"Selected update:{train['best_step']}; fixed target-validation real7 flow MSE:{train['best_validation']:.8f}. No test-based model selection.",
        f"The selected checkpoint has seen{train['best_step']*protocol['batch_size']}anchors; the full training run consumed{train['anchors_consumed']}. These are different budgets when selection precedes the last update.",
        f"Checkpoint:`{train['checkpoint']}`; SHA256:`{train['checkpoint_sha256']}`.",
        '', 'The official network,50-step native queue,10-step flow generation,32-dimensional training loss, native optimizer and frozen VLM are retained. Declared data, processor, validation and finite-budget differences are in PROTOCOL.md.',
        'Both development smokes failed task success but completed600/650 steps and passed interface/state/video acceptance; they are excluded from the80 formal episodes.',
        'All actual policy state vectors match saved pre-action named joint qpos. Raw native32 actions, crop7, native unnormalization,50-step queues and executed OSC12 were checked. No generated success reference enters training.',
        'Failure stages are descriptive sampled-contact/placement proxies, not causal explanations. No old baseline was retrained or evaluated. No new Method or next baseline started.',
        '', 'Code/config/provenance: official_source.json, protocol.json, freeze.json, errors.json. Evidence: training_state.json, training_metrics.json, optimizer_audit.json, sample_exposure.json, episodes.csv, per-task acceptance files, artifact_index.json. Large artifacts stay on the server.']
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    append_detail_tables(directory,result)
    from baselines.strong.matrix_current import main as matrix
    matrix();print(status,flush=True)

if __name__=='__main__':main()
