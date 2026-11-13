"""Pinned official model/tokenizer artifacts with published hash checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request
from baselines.strong.common import ROOT,write,digest
from simulation_validation.download_ranges import download

RUN=ROOT/'smolvla'
POLICY_REVISION='5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6'

def metadata(repo,revision=None):
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+('/revision/'+revision if revision else ''),timeout=30) as r:
        info=json.load(r)
    revision=info['sha']
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=false',timeout=30) as r:
        tree=json.load(r)
    return revision,{x['path']:x for x in tree if x['type']=='file'}

def small(repo,revision,record,directory):
    p=directory/record['path'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{record['path']}",timeout=90) as r:
            p.write_bytes(r.read())
    value=p.read_bytes();assert len(value)==record['size']
    if 'lfs' in record:assert digest(p)==record['lfs']['oid']
    else:assert hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()==record['oid']
    return dict(path=str(p),bytes=len(value),sha256=digest(p))

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    revision,policy=metadata('lerobot/smolvla_base',POLICY_REVISION)
    base_repo='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
    base_revision,base=metadata(base_repo)
    directory=RUN/'pretrained';directory.mkdir(exist_ok=True)
    files=[]
    for name in ['config.json','README.md','policy_preprocessor.json','policy_postprocessor.json',
        'policy_preprocessor_step_5_normalizer_processor.safetensors','policy_postprocessor_step_0_unnormalizer_processor.safetensors']:
        files.append(small('lerobot/smolvla_base',revision,policy[name],directory))
    # A complete SmolVLA checkpoint already contains all VLM parameters it uses.
    # Native load_vlm_weights=False skips a redundant base-weight download.
    # Strict model loading must establish that no used weights remain random.
    cache=RUN/'huggingface/hub'/('models--'+base_repo.replace('/','--'))
    names=[name for name in base if name.endswith('.json') and not name.endswith('.index.json')
        or name.endswith(('.model','.txt','.jinja')) and not name.startswith('.')]
    snapshot=cache/'snapshots'/base_revision
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files.extend(pool.map(lambda name:small(base_repo,base_revision,base[name],snapshot),names))
    (cache/'refs').mkdir(exist_ok=True,parents=True);(cache/'refs/main').write_text(base_revision)
    write(RUN/'acquisition_manifest.json',dict(policy_repo='lerobot/smolvla_base',policy_revision=revision,
        backbone_repo=base_repo,backbone_revision=base_revision,policy_files=policy,base_metadata_files=names,
        published_policy_sha256=policy['model.safetensors']['lfs']['oid']))
    item=policy['model.safetensors'];dest=directory/'model.safetensors'
    if not dest.exists():
        url=f'https://hf-mirror.com/lerobot/smolvla_base/resolve/{revision}/model.safetensors?download=true'
        download(url,dest,item['size'],item['lfs']['oid'],workers=12)
    assert dest.stat().st_size==item['size'] and digest(dest)==item['lfs']['oid']
    files.append(dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest(dest)))
    write(RUN/'acquisition.json',dict(status='PASS',policy_revision=revision,backbone_revision=base_revision,
        files=files,additional_base_weights_downloaded=False))
    print('VERIFIED_OFFICIAL_SMOLVLA_ALL_ARTIFACTS',flush=True)

if __name__=='__main__':main()
