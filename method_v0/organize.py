"""Partition existing offline records into reviewable small Git artifacts."""
from pathlib import Path
import shutil
from method_v0.common import OUT, ROOT, read, write, digest


def organize(name,size):
    source=OUT/(name+'.json')
    if not source.exists():
        assert (OUT/(name+'.index.json')).exists()
        return
    rows=read(source);folder=OUT/name;folder.mkdir(exist_ok=True);parts=[]
    for i,start in enumerate(range(0,len(rows),size)):
        path=folder/f'part-{i:03d}.json';write(path,rows[start:start+size])
        parts.append(dict(path=str(path.relative_to(OUT)),records=min(size,len(rows)-start),sha256=digest(path)))
    dest=ROOT/(name+'.json');shutil.copyfile(source,dest)
    write(OUT/(name+'.index.json'),dict(records=len(rows),parts=parts,full_file=str(dest),full_file_sha256=digest(dest)))
    source.unlink()


def load_rows(name):
    index=read(OUT/(name+'.index.json'));result=[]
    for p in index['parts']:
        path=OUT/p['path'];assert digest(path)==p['sha256'];result.extend(read(path))
    assert len(result)==index['records']
    return result


if __name__=='__main__':
    organize('factual_continuations',180);organize('input_index',400)
