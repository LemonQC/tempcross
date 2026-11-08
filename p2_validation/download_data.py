"""Download only pinned BARX source/target RLDS and target provenance HDF5."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests
from huggingface_hub import HfApi
from simulation_validation.download_ranges import download

SPECS = [
    ('ajaysri/robocasa-x-target-panda-pnp', '6e297509924c831c722180212d13f21686676e8e', 'panda_pnp'),
    ('ajaysri/robocasa-x-xp900-pnp', 'a2b5256a48428ea56064b0afb3dc184a045b10ae', 'mg_pnp_lite'),
    ('ajaysri/barx-raw-hdf5', 'c8b3aba2dfbbfbafa67fa82466320945d560208f', 'raw_target'),
]


def main(root, file_workers=8, range_workers=4, direct_mirror=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    files = []
    for repo, revision, folder in SPECS:
        info = HfApi().repo_info(repo, repo_type='dataset', revision=revision, files_metadata=True)
        assert info.sha == revision
        for item in info.siblings:
            name = item.rfilename
            if folder == 'raw_target' and name not in [
                'human/PandaOmron/PnPCounterToSink/demo_gentex_im320.hdf5',
                'human/PandaOmron/PnPSinkToCounter/demo_gentex_im320.hdf5']:
                continue
            if folder != 'raw_target' and not (name.startswith('1.0.0/') or name == 'README.md'):
                continue
            lfs = item.lfs or {}
            files.append(dict(repo=repo, revision=revision, filename=name,
                              path=str(root/folder/name), size=item.size,
                              sha256=lfs.get('sha256'), git_blob=item.blob_id))
    files.sort(key=lambda r: 0 if '/panda_pnp/' in r['path'] else 1 if '/raw_target/' in r['path'] else 2)
    (root/'manifest.json').write_text(json.dumps(files, indent=2))
    if direct_mirror:
        import urllib.request
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    session=requests.Session()
    if direct_mirror:session.trust_env=False
    (root/'transport.json').write_text(json.dumps(dict(direct_public_mirror=direct_mirror,
        immutable_revisions_and_published_lfs_sha256_unchanged=True)))
    def fetch(row):
        p = Path(row['path']); p.parent.mkdir(parents=True, exist_ok=True)
        host='hf-mirror.com' if direct_mirror else 'huggingface.co'
        url = f"https://{host}/datasets/{row['repo']}/resolve/{row['revision']}/{row['filename']}"
        if row['sha256']:
            download(url, p, row['size'], row['sha256'], workers=range_workers)
        else:
            r = session.get(url, timeout=(30,120));r.raise_for_status();data = r.content
            assert len(data) == row['size']
            assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest() == row['git_blob']
            p.write_bytes(data)
        print(json.dumps({'file_complete':str(p),'bytes':p.stat().st_size}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=file_workers) as pool:
        list(pool.map(fetch, files))
    (root/'complete.json').write_text(json.dumps(dict(status='COMPLETE', files=len(files), bytes=sum(r['size'] for r in files)),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--file-workers',type=int,default=8);parser.add_argument('--range-workers',type=int,default=4)
    parser.add_argument('--direct-mirror',action='store_true')
    a=parser.parse_args();main(a.root,a.file_workers,a.range_workers,a.direct_mirror)
