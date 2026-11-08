"""Inventory reproducible external rollout files without uploading raw data."""
import argparse,csv,hashlib
from pathlib import Path


def manifest(root,out):
    root=Path(root).resolve();out=Path(out).resolve();rows=[]
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.resolve()==out:continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        rows.append(dict(absolute_path=str(path),relative_path=str(path.relative_to(root)),size_bytes=path.stat().st_size,sha256=digest.hexdigest()))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['absolute_path','relative_path','size_bytes','sha256']);writer.writeheader();writer.writerows(rows)
    print('Indexed',len(rows),'files,',sum(r['size_bytes'] for r in rows),'bytes')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();manifest(a.root,a.out)
