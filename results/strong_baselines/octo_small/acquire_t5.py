"""Acquire tokenizer/config only; frozen T5 weights already belong to Octo."""
import concurrent.futures
import hashlib
import json
import urllib.request
from baselines.strong.common import ROOT,write,digest

REVISION='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
RUN=ROOT/'octo_small'
DEST=RUN/'huggingface/hub/models--t5-base'

def main():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/google-t5/t5-base/tree/{REVISION}',timeout=30) as r:
        tree=json.load(r)
    tree={r['path']:r for r in tree}
    snapshot=DEST/'snapshots'/REVISION;snapshot.mkdir(parents=True,exist_ok=True)
    def one(name):
        f=tree[name];p=snapshot/name
        if not p.exists():
            with urllib.request.urlopen(f'https://huggingface.co/google-t5/t5-base/resolve/{REVISION}/{name}',timeout=120) as r:
                data=r.read()
            p.write_bytes(data)
        data=p.read_bytes();assert len(data)==f['size']
        if f.get('lfs'):assert hashlib.sha256(data).hexdigest()==f['lfs']['oid']
        else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==f['oid']
        return dict(path=str(p),bytes=len(data),sha256=digest(p),published_metadata=f)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files=list(pool.map(one,['config.json','tokenizer.json','spiece.model']))
    (DEST/'refs').mkdir(exist_ok=True);(DEST/'refs/main').write_text(REVISION)
    write(RUN/'t5_tokenizer_acquisition.json',dict(status='PASS',revision=REVISION,
        additional_model_weights_downloaded=False,files=files))
    print('PASS_T5_CONFIG_TOKENIZER_ONLY',flush=True)

if __name__=='__main__':main()
