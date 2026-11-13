"""Acquire the original UR5e XP900 raw files, without generating new data.

Used to test native geometry/state supervision. This is not a policy, a
training run, or a physics replay. Manifest and release are pinned by BARX.
Existing files are checked rather than overwritten.
"""
import csv
import json
import argparse
import os
import shutil
from pathlib import Path
from baselines.strong.common import ROOT, SOURCE, digest, write, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--direct-mirror-ranges', action='store_true',
        help='Use the already verified benchmark byte-range transport with identical published SHA256.')
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import hf_hub_download
    manifest = SOURCE/'dataset/subsets/robocasa-x-xp900-pnp.csv'
    release = json.loads((SOURCE/'configs/raw_dataset.json').read_text())
    records = [r for r in csv.DictReader(manifest.open()) if r['embodiment'] == 'UR5e']
    assert len(records) == 6 and sum(int(r['demonstrations']) for r in records) == 600
    folder = ROOT/'raw_source_ur5e'
    folder.mkdir(exist_ok=True)
    assert shutil.disk_usage(folder).free > sum(int(r['bytes']) for r in records) + 25*2**30
    verified = []
    for record in records:
        path = folder/record['relative_path']
        if not path.exists():
            if args.direct_mirror_ranges:
                import urllib.request
                from simulation_validation.download_ranges import download
                urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
                url = f"https://hf-mirror.com/datasets/{release['repo_id']}/resolve/{release['revision']}/{record['relative_path']}"
                download(url, path, int(record['bytes']), record['sha256'], workers=8)
            else:
                acquired = hf_hub_download(repo_id=release['repo_id'], repo_type='dataset',
                    revision=release['revision'], filename=record['relative_path'],
                    local_dir=str(folder))
                assert Path(acquired).resolve() == path.resolve()
        assert path.stat().st_size == int(record['bytes']) and digest(path) == record['sha256']
        verified.append(dict(**record, absolute_path=str(path)))
        write(folder/'acquisition.json', dict(utc=now(), status='COMPLETE' if len(verified)==6 else 'DOWNLOADING',
            release=release, manifest_sha256=digest(manifest), verified=verified,
            transport='direct_public_mirror_ranges' if args.direct_mirror_ranges else 'official_hf_client',
            new_demonstrations=0, new_physics_steps=0, new_training_updates=0,
            note='Original XP900 UR5e data only; identity mapping still must be verified before training.'))
        print('Verified', len(verified), '/6', str(path), flush=True)


if __name__ == '__main__':
    main()
