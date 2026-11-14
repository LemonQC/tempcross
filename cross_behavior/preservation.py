"""One initial scan; resume reuses the durable verified inventory."""
import subprocess
from pathlib import Path
from cross_behavior.common import ROOT, PROJECT, BASE, now, digest, write, read


def audit(final=False):
    cache = ROOT/('preservation_final.json' if final else 'preservation_initial.json')
    if cache.exists():
        result = read(cache)
        assert result['status'] == 'PASS'
        return result
    old = Path('/home/users/lemon/strong-baselines-20261112/flow_matching_policy/preservation_freeze.json')
    historical = read(old)
    indexed = {f['path']:f for f in historical['files']}
    for f in read(PROJECT/'results/strong_baselines/flow_matching_policy/artifact_index.json')['files']:
        indexed[f['path']] = f
    trees = []
    for folder in ('baselines', 'results/strong_baselines', 'results/baselines', 'configs/evaluation'):
        ref = subprocess.check_output(['git','rev-parse',f'{BASE}:{folder}'], cwd=PROJECT, text=True).strip()
        current = subprocess.check_output(['git','rev-parse',f'HEAD:{folder}'], cwd=PROJECT, text=True).strip()
        assert ref == current, f'Historical committed tree changed: {folder}'
        assert not subprocess.check_output(['git','diff',BASE,'--',folder], cwd=PROJECT), folder
        trees.append(dict(path=folder, tree=ref))
    records = []
    for path, f in indexed.items():
        p = Path(path)
        assert p.exists(), path
        actual = digest(p)
        assert actual == f['sha256'], f'Historical artifact changed: {path}'
        stat = p.stat()
        records.append(dict(path=path, sha256=actual, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    result = dict(status='PASS', utc=now(), base=BASE, phase='final' if final else 'initial',
                  historical_formal_episodes=2080, protection='Existing evidence manifests plus immutable Git trees; no old rollout rerun',
                  protected_artifacts=len(records), protected_bytes=sum(f['bytes'] for f in records),
                  trees=trees, files=records)
    write(cache, result)
    return result


if __name__ == '__main__':
    import json
    result = audit()
    print(json.dumps({k:v for k,v in result.items() if k not in ('files', 'trees')}), flush=True)
