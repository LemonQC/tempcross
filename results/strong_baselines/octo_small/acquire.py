"""Pinned official release acquisition; hashes remain the published hashes."""
import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from baselines.strong.common import ROOT,read,write,digest


def main():
    import requests
    from simulation_validation.download_ranges import download
    p=argparse.ArgumentParser();p.add_argument('--metadata-only',action='store_true');args=p.parse_args()
    folder=ROOT/'octo_small';manifest=read(folder/'pretrained_manifest.json')
    assert manifest['revision']=='dc9aa3019f764726c770814b27e4ab0fc6e32a58'
    assert subprocess.check_output(['git','-C','/home/users/lemon/baseline-benchmark/vendor/octo','rev-parse','HEAD'],text=True).strip()=='241fb3514b7c40957a86d869fecb7c7fc353f540'
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    session=requests.Session();session.trust_env=False;verified=[]
    for r in manifest['files']:
        if r['path']=='.gitattributes' or (args.metadata_only and r['lfs']):continue
        dest=folder/('official_metadata' if args.metadata_only else 'pretrained')/r['path']
        url=f"https://hf-mirror.com/{manifest['repository']}/resolve/{manifest['revision']}/{r['path']}"
        if r['lfs']:download(url,dest,r['size'],r['lfs']['sha256'],workers=8)
        else:
            cached=folder/'official_metadata'/r['path']
            if dest.exists():body=dest.read_bytes()
            elif cached.exists():body=cached.read_bytes()
            else:
                response=session.get(url,timeout=(15,30));response.raise_for_status();body=response.content
            assert len(body)==r['size'] and hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()==r['blob_id']
            dest.parent.mkdir(exist_ok=True,parents=True);dest.write_bytes(body)
        verified.append(dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest(dest)))
    write(folder/('metadata_acquisition.json' if args.metadata_only else 'official_acquisition.json'),dict(status='PASS',revision=manifest['revision'],files=verified))


if __name__=='__main__':main()
