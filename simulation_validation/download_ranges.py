"""Resumable public artifact transport; never substitutes content or ignores SHA256.

Uses fixed HTTP byte ranges because the host's single-stream downloads are slow.
Only use published URLs and published size/hash; saves all partial chunks externally.
"""
import argparse
import concurrent.futures
import hashlib
import json
import threading
import time
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
    stopping=threading.Event()
    def fetch(i):
        start=i*chunk;hi=min(size,(i+1)*chunk)-1;p=parts/f'{i:06d}'
        if p.exists() and p.stat().st_size==hi-start+1:return
        partial=p.with_suffix('.part')
        for attempt in range(4):
            if stopping.is_set():raise RuntimeError('Download cancelled after peer failure')
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>hi-start+1:partial.unlink();offset=0
            lo=start+offset
            if lo>hi:partial.replace(p);return
            # A resolved presigned URL must retain its exact signed query.
            signed=any(key in url for key in ('X-Amz-Signature=', 'Signature=', 'sig='))
            request_url=url if signed else url+('&' if '?' in url else '?')+f'barx_range={i}&coft_offset={offset}'
            req=urllib.request.Request(request_url,headers={'Range':f'bytes={lo}-{hi}'})
            try:
                with urllib.request.urlopen(req,timeout=120) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{size}':
                        raise RuntimeError(f'Invalid range response {r.status}, {r.headers.get("Content-Range")}')
                    with partial.open('ab') as f:
                        while True:
                            if stopping.is_set():raise RuntimeError('Download cancelled after peer failure')
                            block=r.read(256*1024)
                            if not block:break
                            f.write(block)
                if partial.stat().st_size!=hi-start+1:raise RuntimeError('Incomplete range')
                partial.replace(p)
                print(json.dumps(dict(chunk=i,bytes=hi-start+1)),flush=True)
                return
            except Exception as error:
                print(json.dumps(dict(chunk=i,attempt=attempt+1,error=str(error))),flush=True)
                if attempt==3:raise
                time.sleep(min(2**attempt,8))
    pool=concurrent.futures.ThreadPoolExecutor(workers)
    futures=[pool.submit(fetch,i) for i in range((size+chunk-1)//chunk)]
    try:
        for future in concurrent.futures.as_completed(futures):future.result()
    except BaseException:
        stopping.set()
        for future in futures:future.cancel()
        raise
    finally:pool.shutdown(wait=True,cancel_futures=True)
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
