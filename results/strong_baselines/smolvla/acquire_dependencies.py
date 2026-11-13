"""Official PyPI wheel hashes and bounded resumable CDN transfer."""
import concurrent.futures
import json
import re
import urllib.request
import hashlib
import math
from pathlib import Path
from baselines.strong.common import ROOT,out,write,digest
from simulation_validation.download_ranges import download

def direct_wheel(record,dest):
    if dest.exists():
        assert digest(dest)==record['sha256'];return
    directory=dest.with_suffix('.ranges4m');directory.mkdir(parents=True,exist_ok=True)
    size=record['size'];chunk=4*1024*1024
    def fetch(i):
        lo=i*chunk;hi=min(size,lo+chunk)-1;p=directory/f'{i:06d}'
        if p.exists() and p.stat().st_size==hi-lo+1:return
        for attempt in range(3):
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request=urllib.request.Request(record['url']+f'?smolvla_range={i}&attempt={attempt}',headers={'Range':f'bytes={lo}-{hi}'})
                with opener.open(request,timeout=30) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {lo}-{hi}/{size}'
                    value=r.read()
                assert len(value)==hi-lo+1;p.write_bytes(value);return
            except Exception as e:
                print('RANGE_RETRY',i,attempt,repr(e),flush=True)
                if attempt==2:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(fetch,range(math.ceil(size/chunk))))
    tmp=dest.with_suffix('.assembling')
    with tmp.open('wb') as stream:
        for i in range(math.ceil(size/chunk)):stream.write((directory/f'{i:06d}').read_bytes())
    assert tmp.stat().st_size==size and digest(tmp)==record['sha256'];tmp.replace(dest)
    print('VERIFIED',dest.name,flush=True)

def one(line):
    match=re.fullmatch(r'([\w-]+)==([\w.]+)',line.strip())
    if not match:return None
    package,version=match.groups()
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json',timeout=30) as r:info=json.load(r)
    files=[f for f in info['urls'] if f['filename'].endswith('.whl')
        and ('-py3-none-any.whl' in f['filename'] or 'x86_64' in f['filename'] and 'manylinux' in f['filename']
            and ('cp310-' in f['filename'] or '-py3-none-' in f['filename']
                or re.search(r'-cp3(?:[789]|10)-abi3-',f['filename'])))]
    if not files:return None
    files.sort(key=lambda f:('cp310-' in f['filename'],'manylinux_2_28' in f['filename'],f['filename']),reverse=True)
    f=files[0]
    if f['size']<1_000_000:return None
    return dict(filename=f['filename'],url=f['url'],size=f['size'],sha256=f['digests']['sha256'])

def main():
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    directory=ROOT/'smolvla/wheels';directory.mkdir(exist_ok=True,parents=True)
    source=ROOT/'smolvla/dependency_wheel_sources.json'
    if source.exists():
        records=json.loads(source.read_text())
        for line in ['opencv-python-headless==4.12.0.88','rerun-sdk==0.26.2','pyarrow==21.0.0','pandas==2.3.3']:
            r=one(line)
            if r and r['filename'] not in {x['filename'] for x in records}:records.append(r)
        write(source,records)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            records=[r for r in pool.map(one,(out('smolvla')/'requirements_runtime.txt').read_text().splitlines()) if r]
        write(source,records)
    for r in records:
        print('START',r['filename'],r['size'],flush=True)
        direct_wheel(r,directory/r['filename'])
    write(ROOT/'smolvla/dependency_wheels_complete.json',dict(status='PASS',files=[dict(path=str(directory/r['filename']),sha256=digest(directory/r['filename'])) for r in records]))

if __name__=='__main__':main()
