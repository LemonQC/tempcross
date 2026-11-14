"""Durable offline handoff and isolated native-Git publication (standard library).

No Codex API, simulation, training, credential export or force-push dependency.
"""
import argparse
import datetime
import difflib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.mp4','.npz','.npy','.pkl','.h5','.hdf5','.pt','.pth','.ckpt','.safetensors'}
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|(?i:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)[\"\']?\s*[:=]\s*[\"\'][^\"\'\s]{4,}[\"\']')


class HandoffError(RuntimeError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value):
    atomic(path,(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())


def safe_text(text):
    text=SECRET.sub('[REDACTED]',text)
    return re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1[REDACTED]@',text)


def contains_json_credential(value):
    if isinstance(value,dict):
        for key,child in value.items():
            if re.fullmatch(r'(?i:password|passwd|token|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|github[_-]?token|gh[_-]?token)',str(key)) and child not in (None,False,''):
                return True
            if contains_json_credential(child):return True
    elif isinstance(value,list):return any(contains_json_credential(v) for v in value)
    return False


def process(args, repo=PROJECT, check=True, env=None, input_bytes=None, timeout=40, pass_fds=()):
    runtime=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    if env:runtime.update(env)
    try:
        p=subprocess.run(args,cwd=repo,input=input_bytes,capture_output=True,env=runtime,timeout=timeout,pass_fds=pass_fds)
    except subprocess.TimeoutExpired:
        raise HandoffError('COMMAND_TIMEOUT: '+args[0])
    if check and p.returncode:
        raise HandoffError(safe_text(p.stderr.decode(errors='replace')[-2000:]))
    return p


def git(repo,*args,check=True,env=None,input_bytes=None):
    return process(['git',*args],repo,check,env,input_bytes)


def textgit(repo,*args):
    return git(repo,*args).stdout.decode().strip()


def relative(repo,name):
    path=Path(name)
    if path.is_absolute() or '..' in path.parts or name.startswith('-'):
        raise HandoffError('Unsafe repository-relative path')
    resolved=(repo/path).resolve()
    if not resolved.is_relative_to(repo.resolve()) or path.name.startswith('.env'):
        raise HandoffError('Path escapes repository or is an environment file')
    return resolved


def settings(repo, config_path, args):
    cfg=read(config_path)
    cfg['data_root']=str(Path(args.data_root or cfg['data_root']).expanduser().resolve())
    cfg['branch']=args.branch or cfg['branch']
    if cfg['branch'] in ('main','master') or not cfg['branch']:
        raise HandoffError('MAIN_BRANCH_FORBIDDEN: select the research branch')
    git(repo,'check-ref-format','--branch',cfg['branch'])
    current=textgit(repo,'branch','--show-current')
    if current!=cfg['branch']:
        raise HandoffError('WRONG_BRANCH: checkout '+cfg['branch']+'; no automatic checkout')
    cfg['config_path']=str(Path(config_path).resolve())
    cfg['repo']=str(repo.resolve())
    cfg['output_root']=str(relative(repo,cfg['output_root']))
    cfg['state_root']=str(Path(textgit(repo,'rev-parse','--absolute-git-dir'))/'cross_behavior_handoff')
    return cfg


def remote_head(cfg,branch=None):
    branch=branch or cfg['branch']
    out=textgit(Path(cfg['repo']),'ls-remote','--heads',cfg['remote'],'refs/heads/'+branch)
    if not out:raise HandoffError('REMOTE_BRANCH_MISSING: '+branch)
    return out.split()[0]


def verify_remote(cfg):
    remote=textgit(Path(cfg['repo']),'remote','get-url','--push',cfg['remote'])
    if cfg.get('allow_local_test_remote') and Path(remote).is_absolute():
        return remote
    normalized=re.sub(r'^git@github.com:', 'https://github.com/', remote)
    normalized=re.sub(r'^ssh://git@github.com/', 'https://github.com/', normalized).removesuffix('.git')
    if normalized!='https://github.com/'+cfg['repository']:
        raise HandoffError('REMOTE_REPOSITORY_MISMATCH: expected configured GitHub repository without embedded credentials')
    return remote


def index_location(cfg, explicit=None):
    if explicit:return Path(explicit).resolve()
    root=Path(cfg['data_root'])
    return root/'artifact_index.json' if (root/'artifact_index.json').exists() else relative(Path(cfg['repo']),cfg['artifact_index'])


def discover(cfg,explicit=None):
    path=index_location(cfg,explicit)
    if not path.exists():raise HandoffError('ARTIFACT_INDEX_MISSING: use --artifact-index or place artifact_index.json under --data-root')
    index=read(path)
    if not isinstance(index.get('episodes'),list) or not index['episodes']:
        raise HandoffError('UNSUPPORTED_ARTIFACT_INDEX: expected nonempty episodes list')
    anchor=Path(index.get('server_root',cfg['data_root'])).resolve()
    root=Path(cfg['data_root']).resolve()
    entries=[];seen=set()
    for episode in index['episodes']:
        identity=f"{episode['task']}-{episode['robot']}-{int(episode['condition']):03d}"
        if identity in seen:raise HandoffError('DUPLICATE_EPISODE: '+identity)
        seen.add(identity);files={}
        for f in episode['files']:
            original=Path(f['path'])
            try:p=(root/original.relative_to(anchor)).resolve()
            except ValueError:raise HandoffError('ARTIFACT_PATH_OUTSIDE_DECLARED_ROOT')
            if not p.is_relative_to(root) or p.name.startswith('.env'):
                raise HandoffError('ARTIFACT_PATH_ESCAPE')
            files[p.name]=dict(path=str(p),expected_sha256=f.get('sha256'),expected_bytes=f.get('bytes'))
        for required in ('episode.json','condition.json','telemetry.jsonl'):
            if required not in files:raise HandoffError('MISSING_REQUIRED_INDEX_ENTRY: '+identity+'/'+required)
        full=episode.get('full_artifact_index')
        if full:
            p=(root/Path(full['path']).relative_to(anchor)).resolve()
            if not p.is_relative_to(root):raise HandoffError('ARTIFACT_PATH_ESCAPE')
            files['artifact_index.json']=dict(path=str(p),expected_sha256=full['sha256'])
        entries.append(dict(id=identity,task=episode['task'],robot=episode['robot'],condition=episode['condition'],files=files))
    return path,entries


def inspect_file(path,cache,episode_id=None,expected=None):
    path=Path(path)
    record=dict(path=str(path.resolve()),file_type=path.suffix.lower() or 'text',episode_id=episode_id,
                exists=path.exists(),read_success=False,bytes=None,sha256=None,expected_sha256=expected)
    if not record['exists']:
        record['error']='MISSING';return record
    try:
        before=path.stat()
        key=[before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns]
        hit=cache.get(str(path.resolve()))
        # Some server filesystems have coarse mtime/ctime resolution. A file
        # hashed in its metadata-change window is never eligible for cache reuse.
        mature=hit and max(before.st_mtime_ns,before.st_ctime_ns)+2_000_000_000 < hit.get('verified_at_ns',0)
        if hit and hit['stat']==key and mature:
            actual=hit['sha256'];record['hash_source']='VERIFIED_STAT_CACHE'
        else:
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
            after=path.stat()
            if (after.st_size,after.st_mtime_ns,after.st_ctime_ns)!=(before.st_size,before.st_mtime_ns,before.st_ctime_ns):
                raise HandoffError('INPUT_CHANGED_WHILE_HASHING')
            actual=h.hexdigest();cache[str(path.resolve())]=dict(stat=key,sha256=actual,verified_at_ns=time.time_ns())
            record['hash_source']='SHA256_READ'
        record.update(bytes=before.st_size,sha256=actual,read_success=True)
        record['integrity_ok']=expected is None or actual==expected
        if not record['integrity_ok']:record['error']='SHA256_MISMATCH'
    except (OSError,HandoffError) as e:
        record['error']=type(e).__name__+': '+safe_text(str(e))
    return record


def source_context(cfg):
    repo=Path(cfg['repo']);head=textgit(repo,'rev-parse','HEAD')
    programs=list(dict.fromkeys(cfg['publish_programs']))
    sources=list(dict.fromkeys(cfg['source_context']+programs))
    records=[];patch=[]
    for name in sources:
        path=relative(repo,name)
        if not path.is_file():raise HandoffError('CONTEXT_SOURCE_MISSING: '+name)
        data=path.read_bytes()
        if SECRET.search(data.decode(errors='replace')):raise HandoffError('SECRET_PATTERN_IN_CONTEXT: '+name)
        if path.suffix=='.json' and contains_json_credential(json.loads(data)):
            raise HandoffError('CREDENTIAL_KEY_IN_CONTEXT_JSON: '+name)
        tracked=git(repo,'cat-file','-e',head+':'+name,check=False).returncode==0
        records.append(dict(path=name,absolute_path=str(path),sha256=sha(data),bytes=len(data),
            exists_at_head=tracked,github_file_link=f"https://github.com/{cfg['repository']}/blob/{head if tracked else cfg['branch']}/{name}",
            snapshot_path='code_snapshot/'+name,publish_program=name in programs))
        if tracked:
            patch.append(git(repo,'diff','--no-ext-diff','--binary','HEAD','--',name).stdout.decode())
        else:
            patch.extend(difflib.unified_diff([],data.decode().splitlines(True),fromfile='/dev/null',tofile='b/'+name))
    dirty=bool(textgit(repo,'status','--porcelain'))
    return dict(head=head,branch=cfg['branch'],dirty=dirty,git_status=safe_text(textgit(repo,'status','--short')),
                files=records,controller_versions={r['path']:r['sha256'] for r in records if r['path'] in cfg['source_context']},
                source_patch_scope='Configured relevant sources only; unrelated edits are not staged or copied'),''.join(patch)


def state_paths(cfg):
    root=Path(cfg['state_root']);root.mkdir(parents=True,exist_ok=True)
    latest=root/('latest-'+sha(cfg['branch'].encode())[:12]+'.json')
    legacy=root/'latest.json'
    if not latest.exists() and legacy.exists():
        pointer=read(legacy);old=Path(pointer['state_path'])
        if old.exists() and read(old)['branch']==cfg['branch']:write(latest,pointer)
    return root,latest


def load_latest(cfg,run_id=None):
    root,latest=state_paths(cfg)
    if run_id:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id):raise HandoffError('INVALID_RUN_ID')
        p=root/'runs'/run_id/'state.json'
    elif latest.exists():p=Path(read(latest)['state_path'])
    else:return None,None
    if not p.exists():raise HandoffError('HANDOFF_STATE_MISSING')
    return p,read(p)


def doctor(cfg,explicit=None,auth_probe=True):
    repo=Path(cfg['repo']);result=dict(utc=now(),python=sys.executable,analysis_python=cfg['python'],
        branch=cfg['branch'],head=textgit(repo,'rev-parse','HEAD'),dirty=bool(textgit(repo,'status','--porcelain')),
        data_root=cfg['data_root'],data_exists=Path(cfg['data_root']).exists(),credential_values_logged=False,
        git_identity_ready=git(repo,'var','GIT_AUTHOR_IDENT',check=False).returncode==0)
    try:
        verify_remote(cfg);result['remote_repository']='MATCH'
        result['remote_head']=remote_head(cfg)
        path,entries=discover(cfg,explicit);result.update(artifact_index=str(path),episodes=len(entries))
        missing=[dict(episode=e['id'],file=name,path=f['path']) for e in entries for name,f in e['files'].items()
                 if not Path(f['path']).is_file()]
        result.update(missing_data_files=missing,data_ready=result['data_exists'] and not missing)
    except HandoffError as e:result['data_or_remote_error']=safe_text(str(e))
    p=process([cfg['python'],'-c','import json,numpy,cv2; print(json.dumps(dict(numpy=numpy.__version__,opencv=cv2.__version__)))'],repo,check=False)
    result['analysis_dependencies']=json.loads(p.stdout) if p.returncode==0 else 'MISSING'
    if auth_probe:
        try:
            ledger=Path(cfg['state_root'])/'last_publication.json'
            previous=read(ledger) if ledger.exists() else {}
            probe_ref=previous['commit'] if previous.get('branch',cfg['branch'])==cfg['branch'] and previous.get('commit') else 'HEAD'
            p=git(repo,'push','--dry-run',cfg['remote'],probe_ref+':refs/heads/'+cfg['branch'],check=False)
            # Non-fast-forward is a conflict, never classified as successful auth.
            result['push_auth_status']='READY' if p.returncode==0 else ('CONFLICT' if b'non-fast-forward' in p.stderr or b'fetch first' in p.stderr else 'UNAVAILABLE')
        except HandoffError:result['push_auth_status']='NETWORK_OR_AUTH_UNAVAILABLE'
    else:result['push_auth_status']='NOT_PROBED'
    result['one_time_auth_commands']=[
        'gh auth login --hostname github.com --git-protocol https', 'gh auth setup-git',
        'python scripts/cross_behavior_handoff.py doctor']
    result['auth_alternative']='If gh is absent, configure a standard credential helper or SSH outside this program. Never put a token in source/config or chat.'
    result['ready_for_run']=bool(result.get('remote_repository')=='MATCH' and result.get('data_ready') and result.get('episodes') and result['git_identity_ready'] and
        isinstance(result['analysis_dependencies'],dict) and result['push_auth_status']=='READY')
    return result


def analyze(cfg,args):
    repo=Path(cfg['repo']);state_root,latest=state_paths(cfg)
    index,episodes=discover(cfg,args.artifact_index)
    context,patch=source_context(cfg)
    cache_path=state_root/'hash_cache.json';cache=read(cache_path) if cache_path.exists() else {}
    records=[inspect_file(index,cache)]
    for episode in episodes:
        for entry in episode['files'].values():
            records.append(inspect_file(entry['path'],cache,episode['id'],entry.get('expected_sha256')))
    for record in context['files']:
        records.append(inspect_file(record['absolute_path'],cache,expected=record['sha256']))
    write(cache_path,cache)
    fingerprint=sha(json.dumps(dict(version=1,records=[{k:r.get(k) for k in ('path','bytes','sha256','expected_sha256','exists','read_success')} for r in records],
        controls={k:cfg[k] for k in ('branch','control_hz','maximum_overlay_count','maximum_overlay_width')}),sort_keys=True).encode())
    catalog_path=state_root/'catalog.json';catalog=read(catalog_path) if catalog_path.exists() else {}
    if fingerprint in catalog:
        state_path=Path(catalog[fingerprint]);state=read(state_path)
        if state['analysis_status']=='COMPLETE':
            finish_package(cfg,state_path,state)
            write(latest,dict(state_path=str(state_path)))
            return state_path,state,True
        if not args.resume:raise HandoffError('INCOMPLETE_RUN_EXISTS: rerun with --resume')
    else:
        run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+fingerprint[:10]
        output=Path(cfg['output_root'])/run_id
        output.mkdir(parents=True,exist_ok=False)
        state_path=state_root/'runs'/run_id/'state.json'
        # A previously published isolated commit is an allowed parent even if
        # the user's live HEAD/index were deliberately left untouched.
        known=read(state_root/'last_publication.json') if (state_root/'last_publication.json').exists() else {}
        base=known.get('commit',context['head']) if known.get('branch',cfg['branch'])==cfg['branch'] else context['head']
        state=dict(run_id=run_id,fingerprint=fingerprint,created_at=now(),output=str(output),
                   branch=cfg['branch'],data_root=cfg['data_root'],base_commit=base,
                   analysis_status='PENDING',package_status='PENDING',publish_status='NOT_RUN',episodes={},
                   config={k:v for k,v in cfg.items() if k not in ('config_path','_lock_fd')},frozen_context=context,frozen_patch=patch)
        write(state_path,state);catalog[fingerprint]=str(state_path);write(catalog_path,catalog)
    if not state.get('snapshot_initialized'):
        output=Path(state['output'])
        context=state['frozen_context'];patch=state['frozen_patch']
        def freeze(path,data):
            if path.exists():
                if path.read_bytes()!=data:raise HandoffError('INCOMPLETE_SNAPSHOT_CONTENT_CHANGED')
            else:atomic(path,data)
            temporary=path.with_name(path.name+'.tmp')
            if temporary.exists():temporary.unlink()
        if not (output/'input_manifest.json').exists():
            write(output/'input_manifest.json',dict(schema_version=1,fingerprint=fingerprint,artifact_index=str(index),
                  files=records,episodes=episodes,scope='Files listed in the top-level evidence index plus per-episode full indexes and source context. Full-index child files not consumed are referenced, not silently claimed rehashed.'))
        freeze(output/'code_context.json',(json.dumps(context,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
        freeze(output/'source_changes.patch',patch.encode())
        for record in context['files']:
            data=Path(record['absolute_path']).read_bytes()
            if sha(data)!=record['sha256']:raise HandoffError('CONTEXT_CHANGED_DURING_SNAPSHOT')
            freeze(output/record['snapshot_path'],data)
        state['snapshot_initialized']=True;write(state_path,state)
    write(latest,dict(state_path=str(state_path)))
    if any(not r['read_success'] or r.get('integrity_ok') is False for r in records):
        state.update(analysis_status='FAILED_INPUT_INTEGRITY',error='Inspect input_manifest.json; existing evidence is never repaired/overwritten')
        write(state_path,state);return state_path,state,False
    state.update(analysis_status='RUNNING',error=None);write(state_path,state)
    command=[cfg['python'],str(repo/'scripts/analyze_stage_a_offline.py'),'--manifest',str(Path(state['output'])/'input_manifest.json'),
             '--state',str(state_path),'--config',cfg['config_path']]
    if args.resume:command.append('--resume')
    if args.stop_after_episodes is not None:command+=['--stop-after-episodes',str(args.stop_after_episodes)]
    try:result=process(command,repo,check=False,timeout=1800,pass_fds=(cfg['_lock_fd'],) if '_lock_fd' in cfg else ())
    except HandoffError as error:
        state=read(state_path);state.update(analysis_status='INTERRUPTED',error=safe_text(str(error)))
        write(state_path,state);return state_path,state,False
    state=read(state_path)
    if result.returncode:
        state['analysis_status']='INTERRUPTED' if result.returncode==75 else 'FAILED_ANALYSIS'
        state['error']=safe_text(result.stderr.decode(errors='replace')[-2000:])
        write(state_path,state);return state_path,state,False
    if state['analysis_status']!='COMPLETE':raise HandoffError('ANALYZER_DID_NOT_COMPLETE')
    finish_package(cfg,state_path,state)
    return state_path,state,False


def finish_package(cfg,state_path,state):
    try:
        if not (Path(state['output'])/'snapshot_manifest.json').exists():seal_package(state)
        sizes=validate_package(cfg,state)
    except HandoffError as error:
        state.update(package_status='FAILED_VALIDATION',error=safe_text(str(error)))
        write(state_path,state);raise
    state.update(package_status='READY',package_size=sizes,error=None)
    write(state_path,state)


def seal_package(state):
    output=Path(state['output'])
    for required in ('input_manifest.json','code_context.json','source_changes.patch',
                     'CROSS_BEHAVIOR_STAGE_A_OFFLINE_DIAGNOSTIC.md','episode_metrics.csv',
                     'phase_timeline.csv','perception_audit.json','diagnostic_summary.json'):
        if not (output/required).is_file():raise HandoffError('REQUIRED_OUTPUT_MISSING: '+required)
    files=[]
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name!='snapshot_manifest.json':
            if p.name.endswith('.tmp'):raise HandoffError('UNCOMMITTED_TEMPORARY_OUTPUT')
            files.append(dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=sha(p.read_bytes())))
    write(output/'snapshot_manifest.json',dict(fingerprint=state['fingerprint'],files=files))


def validate_package(cfg,state):
    output=Path(state['output']);manifest=output/'snapshot_manifest.json'
    if not manifest.exists():raise HandoffError('PACKAGE_MANIFEST_MISSING')
    total=0;expected=set()
    for f in read(manifest)['files']:
        path=relative(output,f['path']);expected.add(path)
        if not path.is_file():raise HandoffError('PACKAGE_FILE_MISSING: '+f['path'])
        data=path.read_bytes()
        if sha(data)!=f['sha256'] or len(data)!=f['bytes']:raise HandoffError('IMMUTABLE_PACKAGE_CHANGED: '+f['path'])
    actual={p for p in output.rglob('*') if p.is_file() and p!=manifest}
    if actual!=expected:raise HandoffError('UNEXPECTED_PACKAGE_FILES')
    for path in sorted(actual|{manifest}):
        data=path.read_bytes();total+=len(data)
        if path.suffix.lower() in FORBIDDEN or path.name.startswith('.env'):
            raise HandoffError('FORBIDDEN_RAW_UPLOAD: '+path.name)
        if path.suffix.lower() in ('.png','.jpg','.jpeg'):
            if path.parent!=output/'overlays':raise HandoffError('IMAGE_OUTSIDE_OVERLAY_DIRECTORY')
        else:
            try:decoded=data.decode()
            except UnicodeDecodeError:raise HandoffError('UNAPPROVED_BINARY_UPLOAD')
            if SECRET.search(decoded):raise HandoffError('SECRET_PATTERN_IN_PACKAGE: '+str(path.relative_to(output)))
            if path.suffix=='.json' and contains_json_credential(json.loads(decoded)):
                raise HandoffError('CREDENTIAL_KEY_IN_PACKAGE_JSON: '+str(path.relative_to(output)))
        if len(data)>cfg['maximum_file_bytes']:raise HandoffError('OUTPUT_FILE_TOO_LARGE: '+path.name)
    if total>cfg['maximum_package_bytes']:raise HandoffError('PACKAGE_TOO_LARGE')
    return dict(files=len(actual)+1,bytes=total)


def publish(cfg,state_path,state,dry_run=False):
    repo=Path(cfg['repo']);verify_remote(cfg)
    if state['analysis_status']!='COMPLETE':raise HandoffError('ANALYSIS_NOT_COMPLETE')
    if state['branch']!=cfg['branch']:raise HandoffError('PACKAGE_BRANCH_MISMATCH')
    try:sizes=validate_package(cfg,state)
    except HandoffError as error:
        state.update(publish_status='FAILED_VALIDATION',error=safe_text(str(error)))
        write(state_path,state);raise
    if state.get('publish_status')=='COMPLETE':
        return dict(state,publication_already_complete=True)
    base=state['base_commit'];current=remote_head(cfg)
    candidate=state.get('candidate_commit')
    if current!=base and current!=candidate:
        state.update(publish_status='CONFLICT',error='REMOTE_CHANGED; no rebase, merge or force-push performed',
            recovery_commands=[f"git fetch {cfg['remote']} {cfg['branch']}",
                               f"git log --oneline --graph --all -12",
                               'Preserve the package. Review the remote changes before creating a new handoff; never force push.'])
        write(state_path,state);raise HandoffError('REMOTE_CHANGED: package retained; see status recovery_commands')
    if dry_run:return dict(state,dry_run=True,would_publish=sizes,expected_remote=base)
    if not candidate:
        # Fetch objects only; leave the user's branch, live index and worktree alone.
        git(repo,'fetch',cfg['remote'],cfg['branch'])
        output=Path(state['output'])
        files={str(p.relative_to(repo)):p for p in output.rglob('*') if p.is_file()}
        context=read(output/'code_context.json')
        for f in context['files']:
            if f['publish_program']:files[f['path']]=output/f['snapshot_path']
        with tempfile.TemporaryDirectory(prefix='handoff-index-',dir=cfg['state_root']) as tmp:
            index=str(Path(tmp)/'index');env={'GIT_INDEX_FILE':index}
            git(repo,'read-tree',base,env=env)
            for name,path in sorted(files.items()):
                relative(repo,name)
                blob=git(repo,'hash-object','-w','--stdin',input_bytes=path.read_bytes()).stdout.decode().strip()
                mode='100755' if name.endswith('.sh') else '100644'
                git(repo,'update-index','--add','--cacheinfo',f'{mode},{blob},{name}',env=env)
            tree=git(repo,'write-tree',env=env).stdout.decode().strip()
            message=f"Offline cross-behavior handoff {state['run_id']}\n"
            candidate=git(repo,'commit-tree',tree,'-p',base,input_bytes=message.encode()).stdout.decode().strip()
        state.update(candidate_commit=candidate,candidate_tree=tree,publish_paths=sorted(files),publish_status='COMMIT_CREATED')
        write(state_path,state)
    if current!=candidate:
        p=git(repo,'push',cfg['remote'],candidate+':refs/heads/'+cfg['branch'],check=False)
        if p.returncode:
            # A network loss after a successful receive is resolved by rereading
            # the branch; a retry never creates another commit for the package.
            try:observed=remote_head(cfg);received=observed==candidate
            except HandoffError:observed=None;received=False
            if not received:
                conflict=observed is not None and observed!=base
                state.update(publish_status='CONFLICT' if conflict else 'FAILED_AUTH_OR_NETWORK',error=safe_text(p.stderr.decode(errors='replace')[-2000:]))
                if conflict:state['recovery_commands']=[f"git fetch {cfg['remote']} {cfg['branch']}",'git log --oneline --graph --all -12','Preserve this package and review the remote changes; do not force push.']
                write(state_path,state);raise HandoffError('GIT_PUSH_FAILED: package and candidate retained; authenticate, then publish --resume')
    if remote_head(cfg)!=candidate:raise HandoffError('REMOTE_HEAD_VERIFICATION_FAILED')
    git(repo,'fetch',cfg['remote'],cfg['branch'])
    # Read the fetched remote object, not merely the local candidate object.
    fetched=textgit(repo,'rev-parse','FETCH_HEAD')
    if fetched!=candidate:raise HandoffError('FETCHED_REMOTE_COMMIT_MISMATCH')
    context=read(Path(state['output'])/'code_context.json')
    lookup={str(p.relative_to(repo)):p for p in Path(state['output']).rglob('*') if p.is_file()}
    lookup.update({f['path']:Path(state['output'])/f['snapshot_path'] for f in context['files'] if f['publish_program']})
    for name,path in lookup.items():
        if git(repo,'show',f'{fetched}:{name}').stdout!=path.read_bytes():
            raise HandoffError('REMOTE_FILE_READBACK_MISMATCH: '+name)
    state.update(publish_status='COMPLETE',github_commit=candidate,published_at=now(),error=None,
        github_url=f"https://github.com/{cfg['repository']}/commit/{candidate}",
        report_url=f"https://github.com/{cfg['repository']}/blob/{candidate}/{Path(state['output']).relative_to(repo)}/CROSS_BEHAVIOR_STAGE_A_OFFLINE_DIAGNOSTIC.md",
        remote_files_verified=len(lookup),publication_transport='NATIVE_GIT',live_index_and_worktree_preserved=True)
    write(state_path,state)
    write(Path(cfg['state_root'])/'last_publication.json',dict(commit=candidate,run_id=state['run_id'],branch=cfg['branch']))
    return state


def display_state(state):
    return {k:v for k,v in state.items() if k not in ('frozen_context','frozen_patch','config','publish_paths')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','analyze','publish','doctor','status'])
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--data-root');parser.add_argument('--branch');parser.add_argument('--artifact-index')
    parser.add_argument('--config',default=str(PROJECT/'configs/diagnostics/handoff_config.json'))
    parser.add_argument('--run-id');parser.add_argument('--stop-after-episodes',type=int,help='Clean analysis checkpoint for recovery testing; no physics')
    args=parser.parse_args(argv)
    try:
        cfg=settings(PROJECT,Path(args.config),args)
        root,_=state_paths(cfg)
        if args.command=='status':
            path,state=load_latest(cfg,args.run_id)
            print(json.dumps(display_state(state) if state else dict(status='NOT_STARTED'),indent=2,ensure_ascii=False));return 0
        if args.command=='doctor':
            result=doctor(cfg,args.artifact_index);print(json.dumps(result,indent=2,ensure_ascii=False))
            return 0 if result['ready_for_run'] else 2
        lock=(root/'handoff.lock').open('a+')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise HandoffError('HANDOFF_ALREADY_RUNNING')
        cfg['_lock_fd']=lock.fileno()
        if args.dry_run and args.command!='publish':
            result=doctor(cfg,args.artifact_index,auth_probe=False)
            result.update(dry_run=True,analysis_started=False,git_objects_created=False,push_performed=False)
            print(json.dumps(result,indent=2,ensure_ascii=False))
            return 0 if result.get('data_ready') and result.get('remote_repository')=='MATCH' and isinstance(result.get('analysis_dependencies'),dict) else 2
        if args.command=='publish':
            path,state=load_latest(cfg,args.run_id)
            if not state:raise HandoffError('NO_ANALYZED_PACKAGE: execute analyze first')
        else:
            path,state,reused=analyze(cfg,args)
            state['reused_existing_analysis']=reused
            if state['analysis_status']!='COMPLETE':
                print(json.dumps(display_state(state),indent=2,ensure_ascii=False));return 75 if state['analysis_status']=='INTERRUPTED' else 1
        if args.command in ('run','publish'):
            state=publish(cfg,path,state,args.dry_run)
        print(json.dumps(display_state(state),indent=2,ensure_ascii=False));return 0
    except (HandoffError,OSError,ValueError,KeyError) as error:
        print(json.dumps(dict(status='FAILED',error=safe_text(str(error))),ensure_ascii=False),file=sys.stderr)
        return 1
