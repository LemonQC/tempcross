"""Install upstream-locked artifacts over a verified mirror transport.

Only registry artifact URLs change. Versions, upstream artifact SHA256 hashes,
Git commits, project sources and evaluation settings remain fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse

import tomli
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name,parse_wheel_filename


def run(args):
    source=Path(args.source).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    python=source/'.venv/bin/python';uv=Path(args.uv).resolve();root=Path(__file__).resolve().parents[1]
    source_sha=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if source_sha!='0f435089f48b1f16d225f868dbef855a5e75d1e0':
        raise RuntimeError('Unexpected upstream commit: '+source_sha)
    env=dict(os.environ,UV_CACHE_DIR=args.cache,UV_SYSTEM_CERTS='true',UV_HTTP_TIMEOUT='3600',UV_CONCURRENT_DOWNLOADS='8',
             NO_PROXY='mirrors.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org',no_proxy='mirrors.tuna.tsinghua.edu.cn,pypi.org,files.pythonhosted.org',
             MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONPATH=str(root),UV_INDEX_URL='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple')
    env['PATH']=str(uv.parent)+os.pathsep+env.get('PATH','')
    def command(values,name,cwd=None,check=True):
        with (out/name).open('w') as stream:
            r=subprocess.run([str(x) for x in values],cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if check and r.returncode:raise RuntimeError(f'{name}: exit {r.returncode}')
        return r.returncode
    build_index=env.pop('UV_INDEX_URL')
    command([uv,'export','--project',source,'--locked','--no-dev','--extra','train','--no-hashes','--format','requirements-txt','--output-file',out/'resolved.txt'],'export-mirror.log')
    env['UV_INDEX_URL']=build_index
    lock=tomli.loads((source/'uv.lock').read_text());tags={tag:i for i,tag in enumerate(sys_tags())};context=default_environment()
    entries=[];git_sources=[];requirements=[]
    for line in (out/'resolved.txt').read_text().splitlines():
        line=line.strip()
        if not line or line.startswith(('#','-e')):continue
        req=Requirement(line)
        if req.marker and not req.marker.evaluate(context):continue
        if req.url:
            if req.url.startswith('git+'):
                sha=req.url.rsplit('@',1)[1]
                candidates=list(Path(args.cache).glob('git-v1/checkouts/*/'+sha[:7]))
                path=next((p for p in candidates if subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()==sha),None)
                if path is None:raise RuntimeError('Pinned source checkout missing: '+req.url)
                git_sources.append({'name':req.name,'commit':sha,'path':str(path),'upstream':req.url})
                continue
            raise RuntimeError('Unexpected source: '+str(req))
        if canonicalize_name(req.name)=='flash-attn':continue
        package=next(p for p in lock['package'] if canonicalize_name(p['name'])==canonicalize_name(req.name) and p['version'] in req.specifier and 'registry' in p['source'])
        choices=[]
        for item in package.get('wheels',[]):
            filename=urllib.parse.unquote(urllib.parse.urlsplit(item['url']).path.split('/')[-1])
            _,_,_,wheel_tags=parse_wheel_filename(filename)
            rank=min((tags[tag] for tag in wheel_tags if tag in tags),default=None)
            if rank is not None:choices.append((rank,item,filename))
        if choices:_,artifact,filename=min(choices,key=lambda x:x[0])
        else:
            artifact=package['sdist'];filename=urllib.parse.unquote(urllib.parse.urlsplit(artifact['url']).path.split('/')[-1])
        url=artifact['url'];local=Path(args.wheels)/filename
        if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==artifact['hash'].split(':',1)[1]:
            transport=local.as_uri()
        elif url.startswith('https://files.pythonhosted.org/packages/'):
            transport='https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/'+url.split('/packages/',1)[1]
        else:transport=url
        requirements.append(f'{req.name} @ {transport} --hash={artifact["hash"]}')
        entries.append({'name':req.name,'version':package['version'],'upstream_url':url,'transport_url':transport,'upstream_sha256':artifact['hash']})
    (out/'mirror-requirements.txt').write_text('\n'.join(requirements)+'\n')
    (out/'mirror-manifest.json').write_text(json.dumps({'source_commit':source_sha,'registry_artifacts':entries,'git_sources':git_sources,'lock_sha256':hashlib.sha256((source/'uv.lock').read_bytes()).hexdigest()},indent=2))
    command([uv,'pip','install','--python',python,'--no-deps','--index-url','https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple','setuptools==75.8.0','wheel==0.45.1','packaging==24.2'],'install-build-tools.log')
    command([uv,'pip','install','--python',python,'--no-deps','--require-hashes','-r',out/'mirror-requirements.txt'],'install-registry.log')
    command([uv,'pip','install','--python',python,'--no-deps','--no-build-isolation',*[item['path'] for item in git_sources],'-e',source,'-e',source/'policy','-e',source/'robocasa_x'],'install-pinned-sources.log')
    flash=Path(args.wheels)/'flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
    expected_flash_sha='3fc5d8813904d32cfc77aa4b88d40169dbc12053edb6ee0f1d94159527d05ab0'
    if hashlib.sha256(flash.read_bytes()).hexdigest()!=expected_flash_sha:
        raise RuntimeError('Official release wheel differs from the recorded installation artifact')
    command([uv,'pip','install','--python',python,'--no-deps',flash,'ninja==1.11.1.1','--index-url','https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple'],'install-flash.log')
    command([uv,'pip','check','--python',python],'dependency-check.log',check=False)
    command([python,'-m','simulation_validation.verify_runtime','--out',out/'runtime_verification.json'],'runtime-verification.log',root)
    command([python,'-m','unittest','discover','-s','tests','-v'],'official-tests.log',source)
    command([uv,'pip','freeze','--python',python],'installed-freeze.txt')
    (out/'status.json').write_text(json.dumps({'status':'RUNTIME_INSTALLED_VERIFIED','physical_task_episodes':0,'mirror_only_changes_transport':True},indent=2))
    print('RUNTIME_INSTALLED_VERIFIED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--uv',required=True);p.add_argument('--cache',required=True);p.add_argument('--wheels',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    try:run(a)
    except Exception as e:
        Path(a.out).mkdir(parents=True,exist_ok=True);(Path(a.out)/'status.json').write_text(json.dumps({'status':'INSTALLATION_FAILED','error':repr(e)},indent=2));raise
