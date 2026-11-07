"""Only official inference artifacts; never downloads RLDS or base VLM checkpoint.

The base VLM checkpoint is unnecessary for load_vla: architecture comes from its
registry, full policy weights from the source-pretraining checkpoint. Runtime
backbones are still needed by the official constructor and are cached at pinned revisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

JOINT_REV='191da4c06c1050767a0379feb67a5c20e85bd703'
NO_REPS_REV='9ebc6f6fb8311f3c4394cd77811a4025b26d0a62'

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);p.add_argument('--include-no-reps',action='store_true');p.add_argument('--skip-joint',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();manifest=json.loads((Path(a.source)/'configs/public_artifacts.json').read_text())
    for item in manifest['artifacts']:
        if item['destination'].startswith(('data/','base-')):continue
        if 'joint-reps' in item['destination']:
            if a.skip_joint:continue
            dest=root/'runs/joint_reps'
        else:dest=root/item['destination']
        snapshot_download(item['repo_id'],revision=item['revision'],local_dir=dest,allow_patterns=item['allow_patterns'],token=False,max_workers=2)
        for f,v in item.get('expected_files',{}).items():
            path=dest/f
            assert path.stat().st_size==v['size_bytes']
            h=hashlib.sha256()
            with path.open('rb') as src:
                for b in iter(lambda:src.read(4*1024*1024),b''):h.update(b)
            assert h.hexdigest()==v['sha256']
    for item in manifest['runtime_dependencies']:
        snapshot_download(item['repo_id'],revision=item['revision'],allow_patterns=item['allow_patterns'],cache_dir=root/'hf/hub',token=False,max_workers=2)
        repo=root/'hf/hub'/('models--'+item['repo_id'].replace('/','--'));(repo/'refs').mkdir(exist_ok=True)
        (repo/'refs/main').write_text(item['revision'])
    if a.include_no_reps:
        snapshot_download('ajaysri/barx-no-reps-xp900-pnp-pretrain',revision=NO_REPS_REV,local_dir=root/'runs/no_reps',allow_patterns=['config.json','dataset_statistics.json','checkpoints/step-050000-epoch-15-loss=0.8424.pt'],token=False,max_workers=2)
if __name__=='__main__':main()
