"""Synthetic calculations and complete native-Git lifecycle, without physics."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from cross_behavior import handoff as h
from cross_behavior.offline_diagnostic import episode_diagnostic, projection_from_rgbd, project


def command(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,text=True).strip()


def row(step,task='push'):
    metrics=dict(step=step,tcp=[.1,0.,.1],native_success=False,stable_control_steps=0,
                 object_position=[.01*step,0.,.02],goal_center=[.1,0.,0.],displacement_m=.01*step,
                 gripper_object_contact=False)
    if task=='stack':metrics.update(object_B_position=[.1,0.,.025],released=step>=3,retreat=False,xy_error_m=0.,height_error_m=0.,native_success=step>=3)
    if task=='drawer':metrics.update(handle_contact=step in (2,3),drawer_normalized_joint={'door':step*.2})
    perceived=dict(start=[0.,0.,.02],target=[.1,0.,0.],direction=[1.,0.,0.],handle=[.1,0.,.5],normal=[0.,-1.,0.])
    return dict(step=step,action=[0.,0.,0.,0.,0.,0.,-1.],
                decision=dict(phase=0 if step<3 else 6,target_tcp=[.2,0.,.1],translation_error_m=.1,perceived=perceived),metrics=metrics)


def dataset(root,tasks=('push','stack','drawer')):
    episodes=[]
    for task in tasks:
        directory=root/task;directory.mkdir(parents=True,exist_ok=True)
        h.write(directory/'episode.json',dict(steps=4,success=False))
        h.write(directory/'condition.json',dict(task=task,role='SYNTHETIC_UNIT_TEST_ONLY'))
        h.atomic(directory/'telemetry.jsonl',''.join(json.dumps(row(i,task))+'\n' for i in range(1,5)).encode())
        files=[]
        for path in sorted(directory.iterdir()):
            files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=h.sha(path.read_bytes())))
        episodes.append(dict(task=task,robot='panda',condition=0,files=files))
    h.write(root/'artifact_index.json',dict(server_root=str(root),episodes=episodes))
    return episodes


@pytest.fixture
def system(tmp_path):
    repo=tmp_path/'repository';repo.mkdir();remote=tmp_path/'remote.git'
    subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
    subprocess.run(['git','init'],cwd=repo,check=True,capture_output=True)
    command(repo,'symbolic-ref','HEAD','refs/heads/cross-behavior-task-expansion')
    command(repo,'config','user.email','unit-test@example.invalid');command(repo,'config','user.name','Handoff tests')
    for name in ['cross_behavior/__init__.py','cross_behavior/handoff.py','cross_behavior/offline_diagnostic.py',
                 'scripts/cross_behavior_handoff.py','scripts/analyze_stage_a_offline.py','scripts/run_cross_behavior_handoff.sh',
                 'docs/diagnostics/HANDOFF_SYSTEM_GUIDE.md']:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(h.PROJECT/name,p)
    h.atomic(repo/'cross_behavior/expert.py',b'# Source context only; no policy execution\n')
    h.write(repo/'configs/tasks/cross_behavior_contracts.json',dict(max_steps=4,stack=dict(retreat_distance_m=.1)))
    h.atomic(repo/'.gitignore',b'results/cross_behavior/diagnostics/\n')
    h.atomic(repo/'unrelated.txt',b'original user content\n')
    data=tmp_path/'data';data.mkdir();dataset(data)
    cfg=copy.deepcopy(h.read(h.PROJECT/'configs/diagnostics/handoff_config.json'))
    cfg.update(data_root=str(data),artifact_index=str(data/'artifact_index.json'),python=sys.executable,
               source_context=['cross_behavior/expert.py','configs/tasks/cross_behavior_contracts.json'],
               allow_local_test_remote=True)
    cfg['publish_programs']=[p for p in cfg['publish_programs'] if p!='tests/test_cross_behavior_handoff.py']
    h.write(repo/'configs/diagnostics/handoff_config.json',cfg)
    command(repo,'add','.');command(repo,'commit','-m','Synthetic fixture, not experimental evidence')
    command(repo,'remote','add','origin',str(remote));command(repo,'push','-u','origin','HEAD')
    return SimpleNamespace(repo=repo,data=data,remote=remote,cfg=cfg)


def cli(system,*args,ok=True):
    p=subprocess.run([sys.executable,'scripts/cross_behavior_handoff.py',*args],cwd=system.repo,capture_output=True,text=True,timeout=45)
    if ok:assert p.returncode==0,(p.stdout,p.stderr)
    return p


def latest(system):
    root=Path(command(system.repo,'rev-parse','--absolute-git-dir'))/'cross_behavior_handoff'
    pointer=root/('latest-'+h.sha(b'cross-behavior-task-expansion')[:12]+'.json')
    return h.read(h.read(pointer)['state_path'])


def test_calculations_are_derived_from_synthetic_trajectories(tmp_path):
    entries=dataset(tmp_path)
    for e in entries:e['id']=e['task']+'-panda-000';e['files']={Path(f['path']).name:f for f in e['files']}
    push,phases,_=episode_diagnostic(entries[0],{'max_steps':4},20)
    assert push['observed_reference_displacement_xy_m']==pytest.approx(.03)
    assert push['goal_localization_xy_error_m']==pytest.approx(0.)
    assert push['object_localization_error_m']==pytest.approx(.01)
    assert push['push_direction_error_degrees']==pytest.approx(0.)
    assert [(p['start_step'],p['end_step']) for p in phases]==[(1,2),(3,4)]
    stack,_,_=episode_diagnostic(entries[1],{'max_steps':4,'stack':{'retreat_distance_m':.1}},20)
    assert stack['first_native_success_step']==3 and stack['first_release_step']==3
    assert stack['final_retreat_distance_m']==pytest.approx(np.linalg.norm([.06,0.,.08]))
    drawer,_,_=episode_diagnostic(entries[2],{'max_steps':4},20)
    assert drawer['maximum_normalized_opening']==pytest.approx(.8)
    assert drawer['first_handle_contact_step']==2
    assert drawer['longest_continuous_contact_seconds']==pytest.approx(.1)


def test_pixel_projection_from_saved_rays():
    y,x=np.indices((40,60));depth=1.+.01*x+.002*y*y
    xyz=np.stack(((x-30)*depth/80,(y-20)*depth/80,depth),-1)
    P,rms=projection_from_rgbd(xyz)
    assert P is not None and rms<1e-7
    np.testing.assert_allclose(project(P,xyz[17,24]),[24,17],atol=1e-6)


def test_doctor_dry_run_and_complete_native_publication_preserve_user_index(system):
    doctor=json.loads(cli(system,'doctor').stdout)
    assert doctor['ready_for_run'] and doctor['push_auth_status']=='READY'
    root=system.repo/'results/cross_behavior/diagnostics/runs'
    cli(system,'run','--dry-run');assert not root.exists()
    h.atomic(system.repo/'unrelated.txt',b'user staged change\n');command(system.repo,'add','unrelated.txt')
    h.atomic(system.repo/'unrelated.txt',b'user staged change\nuser unstaged change\n')
    h.atomic(system.repo/'not-related.env',b'PRIVATE_LOCAL_CONTEXT=never-copy\n')
    index=Path(command(system.repo,'rev-parse','--git-path','index'))
    if not index.is_absolute():index=system.repo/index
    before_index=index.read_bytes();before_work=(system.repo/'unrelated.txt').read_bytes()
    state=json.loads(cli(system,'run').stdout)
    assert state['analysis_status']==state['publish_status']=='COMPLETE'
    assert index.read_bytes()==before_index
    assert (system.repo/'unrelated.txt').read_bytes()==before_work
    commit=state['github_commit']
    assert command(system.repo,'show',commit+':unrelated.txt')=='original user content'
    files=command(system.repo,'diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    assert not any('unrelated' in p or '.env' in p for p in files)
    assert not any(p.endswith(('.mp4','.npz','.pkl')) for p in files)
    assert state['remote_files_verified']>0
    assert json.loads(cli(system,'doctor').stdout)['ready_for_run']
    assert json.loads(cli(system,'status').stdout)['github_commit']==commit
    prior=(Path(state['output'])/'episode_metrics.csv').stat().st_mtime_ns
    repeated=json.loads(cli(system,'run').stdout)
    assert repeated['run_id']==state['run_id'] and repeated['publication_already_complete']
    assert (Path(state['output'])/'episode_metrics.csv').stat().st_mtime_ns==prior


def test_clean_interruption_resume_and_different_inputs(system):
    p=cli(system,'analyze','--stop-after-episodes','1',ok=False)
    assert p.returncode==75
    initial=latest(system);assert initial['analysis_status']=='INTERRUPTED' and len(initial['episodes'])==1
    cached=Path(next(iter(initial['episodes'].values()))['path']);mtime=cached.stat().st_mtime_ns
    done=json.loads(cli(system,'analyze','--resume').stdout)
    assert done['analysis_status']=='COMPLETE' and len(done['episodes'])==3
    assert cached.stat().st_mtime_ns==mtime
    root=Path(done['output']);old_report=(root/'episode_metrics.csv').read_bytes()
    idx=h.read(system.data/'artifact_index.json')
    p=system.data/'push/telemetry.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
    rows[-1]['metrics']['object_position'][0]=.05
    h.atomic(p,''.join(json.dumps(r)+'\n' for r in rows).encode())
    for f in idx['episodes'][0]['files']:
        if f['path']==str(p):f.update(sha256=h.sha(p.read_bytes()),bytes=p.stat().st_size)
    h.write(system.data/'artifact_index.json',idx)
    new=json.loads(cli(system,'analyze').stdout)
    assert new['run_id']!=done['run_id'] and root.exists()
    assert (root/'episode_metrics.csv').read_bytes()==old_report


def test_missing_input_and_corrupted_package_fail_closed(system):
    p=system.data/'push/telemetry.jsonl';p.unlink()
    bad=cli(system,'analyze',ok=False)
    assert bad.returncode==1 and latest(system)['analysis_status']=='FAILED_INPUT_INTEGRITY'
    assert cli(system,'publish',ok=False).returncode==1
    doctor=cli(system,'doctor',ok=False)
    assert doctor.returncode==2 and json.loads(doctor.stdout)['data_ready'] is False


def test_resume_sealing_after_analyzer_completed_before_manifest(system):
    state=json.loads(cli(system,'analyze').stdout)
    manifest=Path(state['output'])/'snapshot_manifest.json';manifest.unlink()
    resumed=json.loads(cli(system,'analyze','--resume').stdout)
    assert resumed['analysis_status']=='COMPLETE' and manifest.exists()
    h.atomic(Path(state['output'])/'episode_metrics.csv',b'tampered\n')
    assert cli(system,'publish','--dry-run',ok=False).returncode==1


def test_publish_network_failure_retry_and_accepted_push_recovery(system):
    state=json.loads(cli(system,'analyze').stdout)
    hook=system.remote/'hooks/pre-receive';hook.write_text('#!/bin/sh\nexit 1\n');hook.chmod(0o755)
    failed=cli(system,'publish',ok=False)
    assert failed.returncode==1
    retained=latest(system);candidate=retained['candidate_commit']
    assert retained['analysis_status']=='COMPLETE' and retained['publish_status']=='FAILED_AUTH_OR_NETWORK'
    hook.unlink()
    # Simulate a received push whose response was lost before the ledger update.
    command(system.repo,'push','origin',candidate+':refs/heads/cross-behavior-task-expansion')
    done=json.loads(cli(system,'publish','--resume').stdout)
    assert done['github_commit']==candidate and done['publish_status']=='COMPLETE'


def test_remote_conflict_stops_without_force_push(system):
    state=json.loads(cli(system,'analyze').stdout)
    base=command(system.repo,'rev-parse','HEAD');tree=command(system.repo,'rev-parse','HEAD^{tree}')
    p=subprocess.run(['git','commit-tree',tree,'-p',base],input='external work\n',cwd=system.repo,capture_output=True,text=True,check=True)
    external=p.stdout.strip();command(system.repo,'push','origin',external+':refs/heads/cross-behavior-task-expansion')
    failed=cli(system,'publish',ok=False)
    assert failed.returncode==1 and latest(system)['publish_status']=='CONFLICT'
    assert command(system.repo,'ls-remote','origin','refs/heads/cross-behavior-task-expansion').split()[0]==external


def test_main_forbidden_and_secret_redaction(system):
    assert cli(system,'run','--branch','main',ok=False).returncode==1
    token='ghp_'+('a'*36)
    assert token not in h.safe_text('failed '+token)
    h.atomic(system.repo/'cross_behavior/expert.py',('access_token = "'+token+'"\n').encode())
    result=cli(system,'analyze',ok=False)
    assert result.returncode==1 and token not in result.stderr


def test_hash_cache_invalidated_on_content_change(tmp_path):
    p=tmp_path/'x';p.write_text('first');cache={}
    a=h.inspect_file(p,cache);assert a['read_success']
    p.write_text('other')
    b=h.inspect_file(p,cache);assert b['sha256']!=a['sha256'] and b['hash_source']=='SHA256_READ'
    import time
    time.sleep(2.05)
    h.inspect_file(p,cache)
    assert h.inspect_file(p,cache)['hash_source']=='VERIFIED_STAT_CACHE'


def test_malformed_trajectory_never_becomes_complete(system):
    idx=h.read(system.data/'artifact_index.json');p=system.data/'push/telemetry.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines()];rows[1]['step']=9
    h.atomic(p,''.join(json.dumps(r)+'\n' for r in rows).encode())
    for f in idx['episodes'][0]['files']:
        if f['path']==str(p):f.update(sha256=h.sha(p.read_bytes()),bytes=p.stat().st_size)
    h.write(system.data/'artifact_index.json',idx)
    assert cli(system,'analyze',ok=False).returncode==1
    state=latest(system)
    assert state['analysis_status']=='FAILED_ANALYSIS' and state['publish_status']=='NOT_RUN'
    assert not (Path(state['output'])/'snapshot_manifest.json').exists()


def test_data_root_override_relocates_indexed_paths(system):
    relocated=system.data.parent/'relocated-data';shutil.copytree(system.data,relocated)
    state=json.loads(cli(system,'analyze','--data-root',str(relocated)).stdout)
    manifest=h.read(Path(state['output'])/'input_manifest.json')
    indexed=[r for r in manifest['files'] if r.get('episode_id')]
    assert indexed and all(Path(r['path']).is_relative_to(relocated) for r in indexed)


def test_forbidden_raw_secret_and_size_guards(system):
    state=json.loads(cli(system,'analyze').stdout);output=Path(state['output'])
    cfg=dict(system.cfg)
    p=output/'raw.mp4';p.write_bytes(b'not-a-real-video');h.seal_package(state)
    with pytest.raises(h.HandoffError,match='FORBIDDEN_RAW_UPLOAD'):h.validate_package(cfg,state)
    p.unlink()
    p=output/'credentials.json';h.write(p,{'password':''.join(['synthetic','test','value'])});h.seal_package(state)
    with pytest.raises(h.HandoffError,match='SECRET_PATTERN|CREDENTIAL_KEY'):h.validate_package(cfg,state)
    p.unlink();h.seal_package(state)
    cfg['maximum_file_bytes']=10
    with pytest.raises(h.HandoffError,match='OUTPUT_FILE_TOO_LARGE'):h.validate_package(cfg,state)


def test_concurrent_runner_is_rejected(system):
    import fcntl
    root=Path(command(system.repo,'rev-parse','--absolute-git-dir'))/'cross_behavior_handoff';root.mkdir()
    with (root/'handoff.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=cli(system,'analyze',ok=False)
        assert result.returncode==1 and 'HANDOFF_ALREADY_RUNNING' in result.stderr
    assert not (system.repo/'results/cross_behavior/diagnostics/runs').exists()


def test_explicit_other_research_branch_has_its_own_publication_parent(system):
    first=json.loads(cli(system,'run').stdout)
    command(system.repo,'checkout','-b','another-research-branch')
    command(system.repo,'push','origin','HEAD:refs/heads/another-research-branch')
    second=json.loads(cli(system,'run','--branch','another-research-branch').stdout)
    assert second['publish_status']=='COMPLETE' and second['branch']=='another-research-branch'
    assert first['run_id']!=second['run_id']
