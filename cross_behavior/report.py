"""Compile existing committed episode steps; never performs physics or training."""
import csv
import io
import json
from pathlib import Path
from cross_behavior.common import ROOT, PROJECT, TASKS, ROBOTS, BASE, now, read, write, digest, condition_seed


def compile_results():
    import cv2
    state = read(ROOT/'status.json')
    assert state['status'] in ('BLOCKED_PHYSICAL_GATE','STAGE_A_PASSED')
    destination = PROJECT/'results/cross_behavior'
    destination.mkdir(parents=True,exist_ok=True)
    conditions, episodes, indexes = [], [], []
    for task in TASKS:
        for robot in ROBOTS:
            for directory in sorted((ROOT/'stage_a'/task/robot).glob('dev-*')):
                path = directory/'episode.json'
                assert path.exists(), f'Incomplete physical attempt must be resolved before delivery: {directory}'
                entry = read(path)
                condition = read(directory/'condition.json')
                telemetry = [json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
                assert [r['step'] for r in telemetry] == list(range(1,entry['steps']+1))
                assert len(list((directory/'steps').glob('*.pkl'))) == entry['steps']
                capture = cv2.VideoCapture(entry['video'])
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = capture.get(cv2.CAP_PROP_FPS)
                capture.release()
                assert count == entry['steps'] and fps == 20.
                metrics = [r['metrics'] for r in telemetry]
                summary = dict(entry)
                summary.update(ever_native_success=any(m['native_success'] for m in metrics),
                    max_stable_control_steps=max(m['stable_control_steps'] for m in metrics),
                    max_contact_steps=max(m['contact_steps'] for m in metrics),
                    max_expert_phase=max(r['decision'].get('phase',0) for r in telemetry),
                    verified_video_frames=count, verified_contiguous_committed_steps=True,
                    resume_records=[str(p) for p in sorted(directory.glob('resume-*.json'))],
                    launch_code_is_initial_attempt_only=True,
                    execution_code_records=[str(p) for p in sorted(directory.glob('execution-*.json'))])
                if task == 'drawer':
                    summary['maximum_normalized_opening'] = max(max(m['drawer_normalized_joint'].values()) for m in metrics)
                if entry['expert_error']:
                    summary['failure_class']='PERCEPTION_TARGET_UNIDENTIFIABLE'
                elif task=='stack' and summary['ever_native_success']:
                    summary['failure_class']='NATIVE_STACK_WITHOUT_FROZEN_RETREAT_STABILITY_CONTRACT'
                elif task=='push' and summary['max_contact_steps']:
                    summary['failure_class']='CONTACT_PUSH_OFF_GOAL'
                elif task=='drawer' and summary['max_contact_steps']:
                    summary['failure_class']='CONTACT_WITH_INSUFFICIENT_OPENING_OR_HORIZON'
                else:
                    summary['failure_class']='APPROACH_OR_TARGET_ACQUISITION_NOT_COMPLETED'
                full_index = directory/'artifact_index.json'
                if not full_index.exists():
                    files=[]
                    for p in sorted(directory.rglob('*')):
                        if p.is_file() and p.name not in ('artifact_index.json','worker.lock') and not p.name.endswith('.tmp'):
                            files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
                    write(full_index,dict(files=files,complete_episode=entry['steps'],utc=now()))
                index=read(full_index)
                assert all(Path(f['path']).stat().st_size==f['bytes'] for f in index['files'])
                mini=[]
                for name in ('episode.json','launch.json','condition.json','model.xml','initial.pkl',
                             'initial.png','initial_rgbd.npz','final.png','telemetry.jsonl','rollout.mp4'):
                    p=directory/name
                    mini.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
                indexes.append(dict(task=task,robot=robot,condition=entry['condition'],files=mini,
                    full_artifact_index=dict(path=str(full_index),sha256=digest(full_index),files=len(index['files']))))
                conditions.append(condition)
                episodes.append(summary)
    write(destination/'episodes.json',episodes)
    write(destination/'conditions.json',dict(role='DEVELOPMENT_FEASIBILITY',
          unique_task_condition_ids=len({c['condition_id'] for c in conditions}),robot_specific_restores=len(conditions),
          frozen_formal_conditions=0,training_demonstrations=0,conditions=conditions,
          note='Identical condition IDs/seeds across robots do not certify identical object poses or histories. No formal conditions inspected.'))
    write(destination/'artifact_index.json',dict(server_root=str(ROOT),episodes=indexes,large_files_in_git=False))
    for task in TASKS:
        write(destination/task/'gate.json',read(ROOT/'stage_a'/task/'gate.json'))
        write(destination/task/'repair.json',read(ROOT/'repairs'/f'{task}.json'))
    preserved = {}
    for name in ('preservation_initial.json','preservation_final.json','legacy_inventory_initial.json','legacy_inventory_final.json'):
        p=ROOT/name
        record=read(p)
        preserved[name]={k:v for k,v in record.items() if k!='files'}
        preserved[name]['server_manifest']=str(p)
        preserved[name]['server_manifest_sha256']=digest(p)
    write(destination/'preservation.json',preserved)
    records=[]
    for method, groups, tier in [
        ('bc_gmm',['source_only','target_only','source_target'],'RGB+instruction'),
        ('bc_rnn',['source_only','target_only','source_target'],'RGB+instruction'),
        ('bc_transformer',['source_only','target_only','source_target'],'RGB+instruction'),
        ('diffusion_policy',['source_only','target_only','source_target'],'RGB+instruction'),
        ('barx_no_reps',['source_only'],'native RGB+instruction'),
        ('barx_joint_reps',['source_only'],'native RGB+instruction'),
        ('barx_lora',['restricted_lora_1','restricted_lora_2','target_adaptation'],'native RGB+instruction'),
        ('act',['target_only','source_target'],'native input tier; not yet adapted'),
        ('vqbet',['target_only','source_target'],'native input tier; not yet adapted'),
        ('octo_small',['source_target'],'RGB+instruction'),
        ('smolvla',['source_target_state'],'RGB+instruction+qpos9'),
        ('flow_matching_policy',['source_target_visual'],'RGB+CLIP768+qpos9')]:
        for task in TASKS:
            for group in groups:
                records.append(dict(method=method,configuration=group,task=task,status='NOT_RUN',
                    information=tier,training_updates=0,physical_policy_evaluations=0,
                    reason='All three dual-robot physical task gates required before data/adapters/training'))
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (destination/'method_integration_matrix.csv').write_text(text.getvalue())
    write(PROJECT/'configs/tasks/cross_behavior_tasks.json',dict(base=BASE,status='NO_TASK_ACCEPTED' if not any(g['status']=='PASS' for g in state['gates'].values()) else 'PARTIAL_TASK_ACCEPTANCE',
        validated_tasks=[t for t in TASKS if state['gates'][t]['status']=='PASS'],
        candidates={t:dict(status=state['gates'][t]['status'],class_name=dict(push='CBPush',stack='CBStack',drawer='CBDrawer')[t],
                    report=f'results/cross_behavior/{t}/gate.json') for t in TASKS},
        contract_sha256=digest(PROJECT/'configs/tasks/cross_behavior_contracts.json'),
        learning_allowed=all(g['status']=='PASS' for g in state['gates'].values())))
    summary=dict(base=BASE,status=state['status'],actual_physical_episodes=len(episodes),
        actual_control_steps=sum(e['steps'] for e in episodes),successes=sum(e['success'] for e in episodes),
        gates=state['gates'],new_training_updates=0,old_baseline_reruns=0,new_formal_policy_evaluations=0,
        demonstration_pilots=0,formal_conditions_created=0,server_root=str(ROOT),
        resources=read(ROOT/'inventory.json')['resources'],
        interpretation='Bounded expert/perception feasibility failure, not evidence that the robots or native tasks are physically impossible. No policy comparison or new algorithm conclusion.',
        next='Researcher review of expert observability and declared trajectory horizon; no automatic further trials/training')
    write(destination/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    compile_results()
