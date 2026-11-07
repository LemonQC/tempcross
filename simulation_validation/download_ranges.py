"""Resumable public artifact transport; never substitutes content or ignores SHA256.

Uses fixed HTTP byte ranges because the host's single-stream downloads are slow.
Only use published URLs and published size/hash; saves all partial chunks externally.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def download(url,dest,size,sha256,workers=12):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    def sha(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        return h.hexdigest()
    if dest.exists():
        assert dest.stat().st_size==size and sha(dest)==sha256
        return
    parts=dest.parent/(dest.name+'.chunks');parts.mkdir(exist_ok=True)
    chunk=8*1024*1024
    def fetch(i):
        lo=i*chunk;hi=min(size,(i+1)*chunk)-1;p=parts/f'{i:06d}'
        if p.exists() and p.stat().st_size==hi-lo+1:return
        req=urllib.request.Request(url+('&' if '?' in url else '?')+f'barx_range={i}',headers={'Range':f'bytes={lo}-{hi}'})
        with urllib.request.urlopen(req,timeout=90) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':
                raise RuntimeError(f'Invalid range response {r.status}, {r.headers.get("Content-Range")}')
            with p.with_suffix('.part').open('wb') as f:
                while True:
                    b=r.read(1024*1024)
                    if not b:break
                    f.write(b)
        assert p.with_suffix('.part').stat().st_size==hi-lo+1
        p.with_suffix('.part').replace(p)
        print(json.dumps(dict(chunk=i,bytes=hi-lo+1)),flush=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:list(ex.map(fetch,range((size+chunk-1)//chunk)))
    temp=dest.with_suffix(dest.suffix+'.assembling')
    with temp.open('wb') as out:
        for i in range((size+chunk-1)//chunk):
            with (parts/f'{i:06d}').open('rb') as f:
                for b in iter(lambda:f.read(4*1024*1024),b''):out.write(b)
    assert temp.stat().st_size==size and sha(temp)==sha256,'Artifact SHA256 mismatch'
    temp.replace(dest)
    print(json.dumps(dict(path=str(dest),size_bytes=size,sha256=sha256,verified=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--dest',required=True);p.add_argument('--size',type=int,required=True);p.add_argument('--sha256',required=True);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    download(a.url,a.dest,a.size,a.sha256,a.workers)
