"""Resume verified public dependency wheels using bounded parallel HTTP ranges.

PyPI published SHA256 is mandatory. The official Google JAX artifact publishes
its MD5 ETag; verify that plus size and retain a local SHA256 for subsequent use.
This only changes artifact transport, never wheel contents or dependency APIs.
"""
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request
from baselines.strong.common import ROOT,read,write,digest

RUN=ROOT/'octo_small'
DIRECTORY=RUN/'wheels'

def get_json(url):
    with urllib.request.urlopen(url,timeout=30) as r:return json.load(r)

def download(record):
    destination=DIRECTORY/record['filename'];size=record['size']
    algorithm='sha256' if 'sha256' in record else 'md5'
    def valid(path):
        if not path.exists() or path.stat().st_size!=size:return False
        h=hashlib.new(algorithm)
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        return h.hexdigest()==record[algorithm]
    if valid(destination):return
    chunks=destination.with_suffix('.chunks');chunks.mkdir(parents=True,exist_ok=True)
    chunk=4*1024*1024
    def fetch(i):
        lo=i*chunk;hi=min(size,lo+chunk)-1;path=chunks/f'{i:06d}'
        if path.exists() and path.stat().st_size==hi-lo+1:return
        for attempt in range(4):
            try:
                u=record['url']+f'?octo_runtime_range={i}&attempt={attempt}'
                # Public PyPI/GCS direct CDN is considerably faster on this host.
                # No authentication is changed; publication hashes still gate use.
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(urllib.request.Request(u,headers={'Range':f'bytes={lo}-{hi}'}),timeout=90) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {lo}-{hi}/{size}'
                    content=r.read()
                assert len(content)==hi-lo+1
                temp=path.with_suffix('.tmp');temp.write_bytes(content);temp.replace(path)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(fetch,range(math.ceil(size/chunk))))
    tmp=destination.with_suffix('.assembling')
    with tmp.open('wb') as stream:
        for i in range(math.ceil(size/chunk)):stream.write((chunks/f'{i:06d}').read_bytes())
    assert valid(tmp),'Published wheel integrity check failed'
    tmp.replace(destination)
    print('VERIFIED',destination.name,size,digest(destination),flush=True)

def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    records=[]
    for line in (RUN/'resolved_runtime.txt').read_text().splitlines():
        match=re.fullmatch(r'(nvidia-[\w-]+|tensorflow)==([\w.]+)',line)
        if not match:continue
        package,version=match.groups()
        r=get_json(f'https://pypi.org/pypi/{package}/{version}/json')
        candidates=[x for x in r['urls'] if x['filename'].endswith('.whl') and 'x86_64' in x['filename']
            and ('cp310-cp310' in x['filename'] if package=='tensorflow' else 'py3-none' in x['filename'])]
        # Some NVIDIA releases publish byte-distinct manylinux1 and2014
        # retags. Both are author uploads; choose one explicit compatible tag.
        candidates.sort(key=lambda x:('manylinux2014' in x['filename'],x['filename']),reverse=True)
        assert candidates,package
        f=candidates[0]
        if f['size']<30_000_000:continue  # smaller packages already in uv cache
        records.append(dict(filename=f['filename'],size=f['size'],sha256=f['digests']['sha256'],url=f['url']))
    u=read(RUN/'jax_wheel_source.json')['matches'][0]
    with urllib.request.urlopen(urllib.request.Request(u,method='HEAD'),timeout=30) as r:
        records.append(dict(filename=u.rsplit('/',1)[1],size=int(r.headers['Content-Length']),
            md5=r.headers['ETag'].strip('"'),url=u))
    write(RUN/'runtime_wheel_sources.json',records)
    for record in records:
        print('START',record['filename'],record['size'],flush=True)
        download(record)
    write(RUN/'runtime_wheels_complete.json',dict(status='PASS',files=[dict(path=str(DIRECTORY/r['filename']),
        sha256=digest(DIRECTORY/r['filename']),bytes=r['size']) for r in records]))

if __name__=='__main__':main()
