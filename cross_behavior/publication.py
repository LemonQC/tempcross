"""Prepare only committed small files; publish through connected GitHub API.

The paired JS checks the parent and exact tree identity, updates only the
authorized research branch, then rereads all files and protected main.
"""
import hashlib
import subprocess
from pathlib import Path
from cross_behavior.common import PROJECT, ROOT, BRANCH, BASE, write


def git(*args):
    return subprocess.check_output(['git',*args],cwd=PROJECT,text=True).strip()


def main():
    assert git('branch','--show-current') == BRANCH
    assert not git('status','--porcelain')
    commit, base = git('rev-parse','HEAD'), git('rev-parse','HEAD^')
    assert base == BASE, 'Publish the isolated evidence commit on its reviewed base'
    files, directories = [], {''}
    allowed = ('cross_behavior/', 'configs/tasks/', 'results/cross_behavior/',
               'docs/CROSS_BEHAVIOR', 'tests/test_cross_behavior',
               'scripts/run_cross_behavior_expansion.py','scripts/cross_behavior_publish.js')
    for row in git('diff','--name-status',base,commit).splitlines():
        change,name = row.split('\t')
        extra = ('.gitignore','.github/workflows/ci.yml','requirements-cross-behavior-test.txt')
        assert change in ('A','M') and (name in extra or name.startswith(allowed)), name
        content = subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=PROJECT)
        assert len(content) < 1_000_000
        text=content.decode()
        files.append(dict(path=name,bytes=len(content),chars=len(text),
                          sha=git('rev-parse',f'{commit}:{name}'),sha256=hashlib.sha256(content).hexdigest()))
        parent=Path(name).parent
        while str(parent)!='.':
            directories.add(str(parent));parent=parent.parent
    trees=[]
    for folder in directories:
        ref=commit+(':'+folder if folder else '^{tree}')
        entries=[]
        for row in subprocess.check_output(['git','ls-tree','-z',ref],cwd=PROJECT).split(b'\0'):
            if not row:continue
            meta,name=row.decode().split('\t',1)
            mode,kind,sha=meta.split()
            entries.append(dict(path=name,mode=mode,type=kind,sha=sha))
        trees.append(dict(path=folder,sha=git('rev-parse',ref),entries=entries))
    trees.sort(key=lambda r:(r['path']=='',-len(Path(r['path']).parts),r['path']))
    manifest=dict(repository='LemonQC/tempcross',branch=BRANCH,base=base,
        protected_main='45d82e1640ee9557c2298d11c62dea53cbb2f9b9',local_commit=commit,
        message=git('log','-1','--format=%s'),tree=git('rev-parse','HEAD^{tree}'),files=files,trees=trees)
    dest=ROOT/'publications'/commit/'manifest.json'
    write(dest,manifest)
    print(str(dest))


if __name__=='__main__':
    main()
