"""Create a committed-content manifest for publishing through the GitHub connector.

No credentials or remote writes occur in this helper. The caller uploads the
listed blobs/trees, checks tree identity, advances main with an expected-parent
lease, then rereads remote files. Large server artifacts never enter this list.
"""
import hashlib
import json
import subprocess
from pathlib import Path
from baselines.strong.common import PROJECT,ROOT,write

def git(*args):return subprocess.check_output(['git',*args],cwd=PROJECT).decode().strip()

def main():
    commit=git('rev-parse','HEAD');base=git('rev-parse','HEAD^')
    assert not git('status','--porcelain')
    files=[];directories=set()
    for line in git('diff','--name-status',base,commit).splitlines():
        change,path=line.split('\t');assert change in ['A','M']
        assert path.startswith(('baselines/strong/','results/strong_baselines/')) or path in ['docs/STRONG_BASELINE_MASTER_MATRIX.md','tests/test_strong_baseline_data.py']
        content=subprocess.check_output(['git','show',commit+':'+path],cwd=PROJECT)
        assert len(content)<1_000_000,'Partition large evidence before publication'
        text=content.decode();files.append(dict(path=path,bytes=len(content),chars=len(text),sha=git('rev-parse',commit+':'+path),sha256=hashlib.sha256(content).hexdigest()))
        parent=Path(path).parent
        while str(parent)!='.':directories.add(str(parent));parent=parent.parent
    directories.add('');trees=[]
    for path in sorted(directories,key=lambda x:(-len(Path(x).parts),x)):
        ref=commit+(':'+path if path else '^{tree}')
        entries=[]
        for row in subprocess.check_output(['git','ls-tree','-z',ref],cwd=PROJECT).split(b'\0'):
            if not row:continue
            meta,name=row.decode().split('\t',1);mode,typ,sha=meta.split();entries.append(dict(path=name,mode=mode,type=typ,sha=sha))
        trees.append(dict(path=path,sha=git('rev-parse',ref),entries=entries))
    # Root is always last, including when one top-level path has depth zero.
    trees.sort(key=lambda r:(r['path']=='',-len(Path(r['path']).parts),r['path']))
    manifest=dict(repository='LemonQC/tempcross',branch='main',base=base,local_commit=commit,message=git('log','-1','--format=%s'),tree=git('rev-parse','HEAD^{tree}'),files=files,trees=trees)
    dest=ROOT/'publications'/commit/'manifest.json';write(dest,manifest)
    print(json.dumps(dict(path=str(dest),base=base,commit=commit,files=len(files),trees=len(trees))))

if __name__=='__main__':main()
