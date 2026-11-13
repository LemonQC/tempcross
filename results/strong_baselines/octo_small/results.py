"""Accept only recorded official trials and actual native optimizer progress."""
import collections
import json
from pathlib import Path
import numpy as np
import h5py
from baselines.strong.common import ROOT,PROJECT,out,read,write,digest,TASKS,now
from baselines.strong.audit import validate_job
from .data import decode_action,DIRECTORY

RUN=ROOT/'octo_small'

def key(a):return np.asarray(a,np.float32).tobytes()

def native_actions(cells):
    stats=read(DIRECTORY/'action_statistics.json');expected=collections.Counter();paths=set()
    for cell in cells:
        job=read(RUN/'physical/formal/source_target'/cell['task']/'job.json')
        paths.add(Path(job['runtime']['raw_action_ledger']))
        for row in cell['rows']:
            with h5py.File(row['episode']['telemetry_path'],'r') as f:
                expected.update(key(a) for a in f['baseline/native_query'][:])
    actual=collections.Counter();records=0
    for p in paths:
        for line in p.read_text().splitlines():
            r=json.loads(line);a=np.asarray(r['normalized'],np.float32)
            physical=np.where(np.array(stats['mask'],bool),a*np.array(stats['std'])+np.array(stats['mean']),a).astype(np.float32)
            np.testing.assert_array_equal(physical,np.asarray(r['native_physical'],np.float32))
            canonical=decode_action(physical)
            np.testing.assert_array_equal(canonical,np.asarray(r['canonical_osc'],np.float32))
            actual.update([key(canonical)]);records+=1
    assert actual==expected,'Raw policy logs must cover every recorded action query exactly'
    return dict(status='PASS',native_queries=records,normalized_and_physical_and_canonical_verified=True,
        recorded_telemetry_query_multiset_exact=True,files=[dict(path=str(p),sha256=digest(p)) for p in sorted(paths)])

def main():
    train=read(RUN/'training/source_target/state.json');protocol=read(out('octo_small')/'protocol.json')
    assert train['status']=='COMPLETE' and train['actual_training_updates']==protocol['training_updates']
    assert digest(train['checkpoint'])==train['checkpoint_sha256']
    cells=[validate_job('octo_small','source_target',task) for task in TASKS]
    assert all(c['episodes']==40 and c['role']=='EXPOSED_REFERENCE' for c in cells)
    native=native_actions(cells);write(out('octo_small')/'native_action_audit.json',native)
    for r in read(ROOT/'preservation_freeze.json')['files']:assert digest(r['path'])==r['sha256']
    summary=[];csv=['task,condition_index,success,steps,failure_stage,checkpoint_sha256,telemetry_path,video_path']
    files=[Path(train['checkpoint']),RUN/'training/source_target/resume.msgpack',
        RUN/'training/source_target/state.json',RUN/'training/source_target/training.jsonl',DIRECTORY/'pack_index.json']
    for c in cells:
        stages=collections.Counter(r['failure_stage'] for r in c['rows'])
        summary.append(dict(group='source_target',task=c['task'],episodes=c['episodes'],successes=c['successes'],
            success_rate=c['successes']/c['episodes'],role='EXPOSED_REFERENCE',failure_counts=dict(stages)))
        for r in c['rows']:
            e=r['episode'];csv.append(','.join(map(str,[c['task'],e['official_condition_index'],int(e['success']),e['steps'],
                r['failure_stage'],e['checkpoint_sha256'],e['telemetry_path'],e['video_path']])))
            files.extend(Path(e[k]) for k in ['telemetry_path','contact_log','video_path'])
        (out('octo_small')/f"evaluation_source_target_{c['task']}.json").write_text(json.dumps(c,indent=2)+'\n')
    result=dict(status='COMPLETE_FAITHFUL_PORT',scope='one joint UR5e+Panda full-finetuned Octo-Small baseline',
        role='EXPOSED_REFERENCE',cells=summary,actual_training_updates=train['actual_training_updates'],
        discarded_probe_updates=6,total_optimizer_calls=train['actual_training_updates']+6,
        anchors_consumed=train['anchors_consumed'],checkpoint=train['checkpoint'],checkpoint_sha256=train['checkpoint_sha256'],
        selected_validation_ddpm_loss=train['best_validation'],selected_update=train['best_step'],
        official_full_task_episodes=80,smoke_episodes=2,prior_baselines_rerun=False,
        preservation_freeze_all_passed=True,native_action_audit=native,information_condition='RGB256+native T5 instruction; no extra state',
        elapsed_training_seconds=train['elapsed_seconds'],target_only_setting_run=False,utc=now())
    write(out('octo_small')/'results.json',result)
    (out('octo_small')/'episodes.csv').write_text('\n'.join(csv)+'\n')
    write(out('octo_small')/'training_state.json',train)
    files += [Path(f['path']) for f in native['files']]
    files += [p for p in [RUN/'training_resources.csv',RUN/'formal-training.log',RUN/'formal-counter.log',RUN/'formal-sink.log'] if p.exists()]
    write(out('octo_small')/'artifact_index.json',dict(files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(set(files))],large_files_stay_on_server=True))
    status=dict(status='COMPLETE_FAITHFUL_PORT',method='octo_small',phase='ACCEPTED',actual_training_updates=train['actual_training_updates'],
        discarded_probe_optimizer_updates=6,physical_evaluation_episodes=80,physical_smoke_episodes=2,
        successes=sum(c['successes'] for c in cells),success_rate=sum(c['successes'] for c in cells)/80,
        role='EXPOSED_REFERENCE',checkpoint_sha256=train['checkpoint_sha256'],ordinary_repairs_allowed=True)
    write(out('octo_small')/'status.json',status)
    lines=['# Octo-Small source+target results','',
        'COMPLETE_FAITHFUL_PORT for one joint UR5e+Panda setting. Target-only was not run.',
        'Original conditions60–99 are EXPOSED_REFERENCE, not independent unseen test. Official placement AND retreat is unchanged.','',
        '| Task | Official successes / episodes |','|---|---:|']
    lines += [f"| {c['task']} | {c['successes']}/{c['episodes']} |" for c in summary]
    lines += ['',f"Actual formal updates: {train['actual_training_updates']}; discarded diagnostic updates:6; anchors:{train['anchors_consumed']}.",
        f"Selected update:{train['best_step']}, minimum fixed target-validation DDPM loss:{train['best_validation']:.8f}. No test-based selection.",
        f"Checkpoint:`{train['checkpoint']}`; SHA256:`{train['checkpoint_sha256']}`.",'',
        'Official transformer, native diffusion head/loss, frozen T5, resize/augmentation and optimizer are unchanged. Differences from the generic official recipe are declared in PROTOCOL.md. Native RGB256/T5 pretraining differs from old RGB128 models; no additional robot state or real wrist camera is used.',
        'Both development smoke episodes failed task success but passed full-length interface acceptance. They are separate from the80 formal episodes. Failure-stage labels describe recorded contact/placement predicates; they do not establish causes.',
        'See results.json, episodes.csv, training_state.json, per-task acceptance files, native_action_audit.json and artifact_index.json. All large weights, raw telemetry and videos remain at their absolute server paths. Earlier baseline evidence and launches passed preservation checks.']
    (out('octo_small')/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    from baselines.strong.matrix_current import main as matrix
    matrix();print(json.dumps(status),flush=True)

if __name__=='__main__':main()
