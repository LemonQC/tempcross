"""Verify preserved inputs and index offline server outputs; no model or physics."""
import csv
import re
import sys
from pathlib import Path

from method_v0.common import BASE,PROJECT,OUT,ROOT,guard,read,write,digest
from method_v0.organize import load_rows


def entry(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path))


def main():
    guard()
    freeze=read(ROOT/'pre_extraction_freeze.json')
    assert freeze['base']==BASE
    checks=[]
    for row in freeze['input_files']:
        actual=entry(row['path']);assert actual['sha256']==row['sha256']
        checks.append(dict(**actual,reference='pre_extraction_freeze'))
    old=read(PROJECT/'results/strong_baselines/mimicgen/status.json')
    for path,expected in old['input_sha256'].items():
        actual=entry(path);assert actual['sha256']==expected
        checks.append(dict(**actual,reference='existing_mimicgen_input_freeze'))
    for row in read(PROJECT/'results/strong_baselines/mimicgen/external_file_index.json'):
        if Path(row['path']).name=='physical_launches.jsonl':
            actual=entry(row['path']);assert actual['sha256']==row['sha256']
            checks.append(dict(**actual,reference='existing_mimicgen_external_index'))
    with (PROJECT/'results/mechanism/external_files.csv').open() as f:
        for row in csv.DictReader(f):
            if Path(row['absolute_path']).name=='physical_launches.jsonl':
                actual=entry(row['absolute_path']);assert actual['sha256']==row['sha256']
                checks.append(dict(**actual,reference='existing_mechanism_external_index'))
    ledgers=[r for r in checks if Path(r['path']).name=='physical_launches.jsonl']
    assert len({r['path'] for r in ledgers})==5
    write(OUT/'preservation.json',dict(status='PASS',checks=checks,
        distinct_unchanged_physical_launch_ledgers=5,new_physics_steps=0,
        scope='Frozen project inputs, raw Panda HDF5 and five historical launch ledgers; read-only extraction code and import guard provide additional evidence.'))

    records=load_rows('factual_continuations')
    assert len(load_rows('input_index'))==3234
    arrays=[entry(r['extracted_arrays']) for r in records]
    assert len({r['path'] for r in arrays})==1617
    # Full array index remains one reviewable text file; no numeric arrays enter Git.
    misc=[entry(ROOT/name) for name in ['extraction.log','analysis.log','tests.log',
                                      'pre_extraction_freeze.json','factual_continuations.json','input_index.json']]
    write(OUT/'server_artifacts.json',dict(root=str(ROOT),arrays=arrays,other=misc,
        role='Existing-log extraction outputs only; not new robot trajectories or model checkpoints.'))

    matched=[]
    for group in read(OUT/'matched_continuation_witnesses.json'):
        for suffix in group['suffixes']:
            directory=Path(suffix['history_path']).parent
            paths=[directory/name for name in ['handoff.npz','prefix.npz','handoff_runtime.pkl','initial.json']]
            paths += [Path(suffix['history_path']),Path(suffix['telemetry_path']),Path(suffix['telemetry_path']).with_suffix('.jsonl')]
            matched.extend(entry(p) for p in paths)
    write(OUT/'matched_handoff_input_index.json',dict(files=matched,
        runtime_pickle_scope='Only declared RNG/queue/mode keys compared; no private policy configuration exported.'))

    inventory_paths=[PROJECT/'configs/datasets/baseline_source_manifest.json',
                     PROJECT/'configs/datasets/baseline_target_manifest.json',
                     Path('/home/users/lemon/coft-p2/p2/data/mg_pnp_lite/1.0.0/features.json')]
    write(OUT/'inventory_input_index.json',[entry(p) for p in inventory_paths])
    text=(ROOT/'tests.log').read_text()
    count=re.search(r'Ran (\d+) tests? in',text)
    assert count and text.rstrip().endswith('OK')
    write(OUT/'test_report.json',dict(status='PASS',tests=int(count.group(1)),
        python=sys.version,log=entry(ROOT/'tests.log'),command='python -m unittest discover -s method_v0/tests -v',
        new_training_updates=0,new_physics_steps=0))
    print('PASS: frozen inputs and five launch ledgers unchanged; 1617 extracted array files indexed.')


if __name__=='__main__':main()
