"""Terminal evidence collection and local publication snapshot; zero experiments."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import PROJECT, ROOT, OUT, SOURCE, digest, read, write
from report import table


def main():
    state=read(OUT/'status.json');assert state['status'] in ['COMPLETE','BLOCKED']
    execution=read(ROOT/'phase2_execution.json');assert execution['status'] in ['COMPLETE','BLOCKED']
    pid=read(ROOT/'phase2_pid.json')['pid']
    cmd=Path('/proc')/str(pid)/'cmdline'
    assert not cmd.exists() or b'phase2.py' not in cmd.read_bytes(), 'Supervisor still alive; wait without restarting it.'
    frozen=read(ROOT/'downstream_freeze.json')
    for path,sha in frozen['sha256'].items():assert digest(path)==sha,'Frozen execution input changed: '+path
    for path,sha in state['input_sha256'].items():assert digest(path)==sha,'Historical input changed: '+path
    small=[ROOT/'freeze.json',ROOT/'downstream_freeze.json',ROOT/'phase2_execution.json',
           ROOT/'technical_failure_status_before_recovery.json',ROOT/'dependency_and_hdf_inventory.json']
    for p in small:
        if p.exists():shutil.copyfile(p,OUT/p.name)
    launches=[]
    for stage,base in [('PROBE',ROOT),('AUGMENTATION',ROOT/'augmentation'),('FORMAL_EXPOSED_REFERENCE',ROOT/'evaluation')]:
        ledger=base/'physical_launches.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():launches.append(dict(stage=stage,**json.loads(line)))
    with (OUT/'physical_launches.jsonl').open('w') as f:
        for row in launches:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    completed=[]
    for stage,base in [('PROBE',ROOT/'probe'),('AUGMENTATION',ROOT/'augmentation/probe')]:
        for p in sorted(base.glob('*/episode.json')):
            r=read(p);completed.append(dict(stage=stage,id=r['id'],task=r['task'],role=r['role'],
                success=r['success'],steps=r['steps'],condition=r['condition'],condition_id=r['condition_id'],
                video=r['video'],trajectory=r['trajectory'],contact_log=r['contact_log'],
                recovered_after_logging_error=r.get('recovered_after_logging_error',False)))
    table(OUT/'generation_and_replay_episodes.csv',completed)
    launched_ids=[r['id'] for r in launches if 'id' in r]
    assert len(launched_ids)==len(set(launched_ids)), 'Duplicate physical ID launched'
    assert set(r['id'] for r in completed)==set(launched_ids)
    integrity=read(OUT/'generation_integrity.json')
    assert integrity['status']=='PASS'
    assert integrity['checked_trajectories']==sum(r['stage']=='AUGMENTATION' for r in completed)
    for p in (ROOT/'augmentation').glob('*-progress.json'):
        x=read(p)
        write(OUT/p.name,{k:v for k,v in x.items() if k!='rows'})
    index=[]
    for p in sorted(ROOT.rglob('*')):
        if p.name.startswith('finalize') and p.suffix=='.log':
            continue  # Publication's own log is still being written, not experimental evidence.
        if p.is_file() and p.suffix in ['.h5','.hdf5','.pt','.npy','.mp4','.json','.jsonl','.log','.csv','.png']:
            index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
    write(OUT/'external_file_index.json',index)
    if state['status']=='COMPLETE':
        result=read(OUT/'results.json')
        assert result['new_formal_episodes']==80 and result['augmentation_successes']==100
        assert result['training']['actual_optimizer_updates']==48000
        assert len([r for r in completed if r['stage']=='PROBE'])==10
        assert sum(r['success'] for r in completed if r['stage']=='AUGMENTATION')==100
        selected=read(OUT/'selection.json');assert digest(selected['path'])==selected['sha256']
    check=dict(status=state['status'],old_1280_plus_160_not_rerun=True,old_input_hashes_unchanged=True,
        frozen_execution_inputs_unchanged=True,source_checkout_clean=not bool(subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()),
        actual_probe_completed=sum(r['stage']=='PROBE' for r in completed),
        actual_augmentation_completed=sum(r['stage']=='AUGMENTATION' for r in completed),
        actual_augmentation_successes=sum(r['success'] for r in completed if r['stage']=='AUGMENTATION'),
        actual_optimizer_updates=state['actual_optimizer_updates'],new_policy_or_physics_calls_in_finalizer=0)
    check.update(unique_physical_ids=True,all_launched_generation_and_replay_ids_have_completed_saved_records=True)
    write(OUT/'publication_checks.json',check)
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=PROJECT,text=True).splitlines()
    assert all(p.startswith('results/strong_baselines/mimicgen/') or p in ['docs/MIMICGEN_BASELINE_PROTOCOL.md','docs/MIMICGEN_BASELINE_RESULTS.md'] for p in changed)
    files=([PROJECT/'docs/MIMICGEN_BASELINE_PROTOCOL.md',PROJECT/'docs/MIMICGEN_BASELINE_RESULTS.md']+
        [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.svg'] and '__pycache__' not in p.parts])
    subprocess.run(['git','add','-f','--',*[str(p.relative_to(PROJECT)) for p in files]],cwd=PROJECT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=PROJECT,check=True)
    message='Report completed bounded MimicGen visual-policy baseline' if state['status']=='COMPLETE' else 'Report bounded MimicGen stop and actual completed evidence'
    subprocess.run(['git','commit','-m',message],cwd=PROJECT,check=True)
    subprocess.run(['git','fetch','origin','main'],cwd=PROJECT,check=True)
    subprocess.run(['git','rebase','origin/main'],cwd=PROJECT,check=True)
    def g(*args):return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()
    rows=[]
    for path in g('diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines():
        raw=subprocess.check_output(['git','show','HEAD:'+path],cwd=PROJECT)
        rows.append(dict(path=path,sha=g('rev-parse','HEAD:'+path),characters=len(raw.decode()),bytes=len(raw)))
    manifest=dict(local_sha=g('rev-parse','HEAD'),parent=g('rev-parse','HEAD^'),base_tree=g('rev-parse','HEAD^^{tree}'),
        tree=g('rev-parse','HEAD^{tree}'),files=rows,message=message)
    write(ROOT/'final_delivery_manifest.json',manifest)
    print(json.dumps(dict(status=state['status'],local_sha=manifest['local_sha'],files=len(rows))))


if __name__=='__main__':main()
