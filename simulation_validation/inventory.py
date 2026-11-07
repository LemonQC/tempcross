"""Inventory external artifacts without treating partial files as loaded models."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime,timezone


def inventory(source,root):
    source=Path(source).resolve();root=Path(root).resolve()
    manifest=json.loads((source/'configs/public_artifacts.json').read_text());items=[]
    for item in manifest['artifacts']:
        if item['destination'].startswith(('data/','base-')):continue
        dest=root/('runs/joint_reps' if 'joint-reps' in item['destination'] else item['destination'])
        for filename,expected in item.get('expected_files',{}).items():
            p=dest/filename;row=dict(repo=item['repo_id'],revision=item['revision'],path=str(p),expected=expected,present=p.is_file(),verified=False)
            if p.is_file():
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
                row.update(size_bytes=p.stat().st_size,sha256=h.hexdigest());row['verified']=row['size_bytes']==expected['size_bytes'] and row['sha256']==expected['sha256']
            folder=p.parent/(p.name+'.chunks')
            row['partial_chunk_bytes']=sum(x.stat().st_size for x in folder.glob('*') if x.is_file()) if folder.exists() else 0
            items.append(row)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True)
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),source=str(source),artifact_root=str(root),gpu=gpu.stdout.strip(),artifacts=items,
                paths=dict(locked_python=str(source/'.venv/bin/python'),bootstrap_python=str(root.parent/'bootstrap/bin/python'),
                           logs=str(root.parent),rollouts=str(root.parent/'rollouts')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    Path(a.out).write_text(json.dumps(inventory(a.source,a.root),indent=2))
