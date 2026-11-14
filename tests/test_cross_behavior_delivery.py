"""Cross-file acceptance checks; a blocked task must never become a trained baseline."""
import json
from pathlib import Path
from cross_behavior.common import PROJECT, digest


def test_delivery_counts_and_learning_gate():
    root=PROJECT/'results/cross_behavior'
    summary=json.loads((root/'summary.json').read_text())
    episodes=json.loads((root/'episodes.json').read_text())
    registry=json.loads((PROJECT/'configs/tasks/cross_behavior_tasks.json').read_text())
    assert summary['actual_physical_episodes']==len(episodes)==19
    assert summary['actual_control_steps']==sum(e['steps'] for e in episodes)
    assert len({(e['task'],e['robot'],e['condition']) for e in episodes})==len(episodes)
    assert all(e['verified_contiguous_committed_steps'] and e['verified_video_frames']==e['steps'] for e in episodes)
    assert summary['new_training_updates']==summary['new_formal_policy_evaluations']==summary['old_baseline_reruns']==0
    assert registry['learning_allowed'] is False and registry['validated_tasks']==[]
    assert registry['contract_sha256']==digest(PROJECT/'configs/tasks/cross_behavior_contracts.json')
    for task,gate in summary['gates'].items():
        for robot,record in gate['robots'].items():
            rows=[e for e in episodes if e['task']==task and e['robot']==robot]
            assert record['episodes']==len(rows)
            assert record['successes']==sum(e['success'] for e in rows)
            assert record['not_run']+len(rows)==8
            assert gate['status']=='BLOCKED'


def test_native_predicate_is_not_reported_as_full_success():
    episodes=json.loads((PROJECT/'results/cross_behavior/episodes.json').read_text())
    native=[e for e in episodes if e['task']=='stack' and e['ever_native_success']]
    assert len(native)==5
    assert all(not e['success'] and 'FULL' not in e['status'] for e in native)
