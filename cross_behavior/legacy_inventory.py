"""Cached broader historical artifact inventory; no old simulation or training."""
import csv
import json
import subprocess
from pathlib import Path
from cross_behavior.common import ROOT, PROJECT, digest, read, write, now


def references(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value['path'].startswith('/') and len(str(value.get('sha256',''))) == 64:
            yield value['path'], value['sha256']
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def inventory(final=False):
    dest = ROOT/('legacy_inventory_final.json' if final else 'legacy_inventory_initial.json')
    if dest.exists():
        return read(dest)
    source = ROOT/'legacy_catalog.json'
    if not source.exists():
        expected = {}
        indexes = []
        paths = subprocess.check_output(['git','ls-files','results'], cwd=PROJECT, text=True).splitlines()
        for name in paths:
            if name.endswith('.json') and any(s in name for s in ('sha_index','artifact_index','preservation')):
                indexes.append(name)
                for path, sha in references(read(PROJECT/name)):
                    expected.setdefault(path, set()).add(sha)
        for name in ('results/mechanism/source_artifact_index.csv',):
            with (PROJECT/name).open() as f:
                for row in csv.DictReader(f):
                    expected.setdefault(row['absolute_path'],set()).add(row['sha256'])
        # Original 1,280 formal episodes: enumerate once, never call the simulator.
        old_formal = Path('/home/users/lemon/baseline-benchmark/run-20261109/formal')
        for path in old_formal.rglob('*'):
            if path.is_file():
                expected.setdefault(str(path), set())
        write(source, dict(utc=now(), indexes=indexes,
                          expected={path:sorted(values) for path,values in sorted(expected.items())}))
    catalog = read(source)
    progress = ROOT/('legacy_inventory_final.progress.json' if final else 'legacy_inventory_initial.progress.json')
    records = read(progress)['files'] if progress.exists() else []
    verified = {f['path']:f for f in records}
    initial = {f['path']:f for f in read(ROOT/'legacy_inventory_initial.json')['files']} if final else {}
    for i,(name, expected) in enumerate(catalog['expected'].items()):
        path = Path(name)
        if not path.exists():
            record = dict(path=name, status='PREEXISTING_MISSING', expected_sha256=expected)
        else:
            stat = path.stat()
            previous = verified.get(name)
            if previous and previous.get('bytes') == stat.st_size and previous.get('mtime_ns') == stat.st_mtime_ns:
                continue
            actual = digest(path)
            record = dict(path=name, sha256=actual, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          published_hash_matches=not expected or actual in expected,
                          published_hash_count=len(expected), status='VERIFIED_PRESENT')
        if final:
            old = initial[name]
            assert record.get('sha256') == old.get('sha256'), 'Historical artifact changed during task: '+name
            assert record['status'] == old['status'], 'Historical artifact presence changed: '+name
        verified[name] = record
        if i % 40 == 0:
            write(progress, dict(files=list(verified.values()), utc=now(), last_index=i))
    records = list(verified.values())
    missing = [r for r in records if r['status']=='PREEXISTING_MISSING']
    mismatches = [r for r in records if r.get('published_hash_matches') is False]
    result = dict(status='UNCHANGED_FROM_INITIAL_SNAPSHOT' if final else 'SNAPSHOT_COMPLETE', utc=now(),
                  catalog_path=str(source), catalog_sha256=digest(source), files=records,
                  present=sum(r['status']=='VERIFIED_PRESENT' for r in records),
                  missing_before_current_work=len(missing), historical_checksum_discrepancies=len(mismatches),
                  bytes=sum(r.get('bytes',0) for r in records),
                  note='Old attempt/probe indexes can reference superseded mutable files. Discrepancies are preserved, not repaired or attributed to this task.')
    write(dest,result)
    return result


if __name__=='__main__':
    import sys
    result=inventory('--final' in sys.argv)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
