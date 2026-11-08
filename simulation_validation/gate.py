"""Fixed P0 -> P1 gate. A technical error never becomes a failed task episode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

CHECKPOINTS={'joint_reps':'step-050000-epoch-15-loss=0.2577.pt','no_reps':'step-050000-epoch-15-loss=0.8424.pt'}

def complete_rollout(out,expected):
    files=list((Path(out)/'official').rglob('summary.json'))
    if len(files)!=1:return False
    summary=json.loads(files[0].read_text())
    episodes=files[0].parent/'episodes.jsonl'
    records=[json.loads(x) for x in episodes.read_text().splitlines()] if episodes.exists() else []
    videos=list(files[0].parent.rglob('*.mp4'))
    video_valid=False
    if len(videos)>=expected:
        try:
            import cv2
            video_valid=True
            for path in videos:
                cap=cv2.VideoCapture(str(path));ok,frame=cap.read();cap.release()
                video_valid=video_valid and ok and frame is not None
        except ImportError:
            video_valid=False
    return (summary.get('status')=='complete' and summary.get('episodes')==expected
            and len(records)==expected and video_valid
            and all(type(r.get('success')) is bool and 0<r.get('steps',0)<=600 for r in records))

def prepare_run_directory(out):
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite an existing run: {out}; choose a new BARX_RUN_ROOT')
    out.mkdir(parents=True,exist_ok=True)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--out',required=True);p.add_argument('--p0-only',action='store_true');a=p.parse_args()
    source=Path(a.source).resolve();artifacts=Path(a.artifacts).resolve();out=prepare_run_directory(Path(a.out).resolve())
    python=source/'.venv/bin/python';ledger=out/'gate.json'
    state=dict(status='P0_RUNNING',started_at=datetime.now(timezone.utc).isoformat(),commands=[],p1_started=False)
    def save():ledger.write_text(json.dumps(state,indent=2))
    def run(cmd,log,cwd=None,env=None):
        state['commands'].append(dict(command=[str(x) for x in cmd],log=str(log)));save()
        with log.open('w') as stream:
            result=subprocess.run([str(x) for x in cmd],stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,env=env)
        state['commands'][-1]['exit_code']=result.returncode;save()
        return result.returncode==0
    save()
    if not run([python,'-m','unittest','discover','-s','tests','-v'],out/'official_tests.log',source):
        state.update(status='P0_BLOCKED',blocker='OFFICIAL_UNIT_TESTS_FAILED');save();return 2
    def evaluate(method,arm,n,stage):
        checkpoint=artifacts/'runs'/method/'checkpoints'/CHECKPOINTS[method]
        directory=out/stage/method/arm
        if not checkpoint.is_file():
            state.update(status='P0_BLOCKED' if stage=='p0' else 'P1_INCOMPLETE',blocker=f'CHECKPOINT_MISSING: {checkpoint}');save();return False
        directory.mkdir(parents=True,exist_ok=False)
        cmd=[python,'-m','simulation_validation.run_official','--source',source,'--checkpoint',checkpoint,'--embodiment',arm,'--episodes',n,'--out',directory]
        if not run(cmd,directory/'console.log') or not complete_rollout(directory,n):
            state.update(status='P0_BLOCKED' if stage=='p0' else 'P1_INCOMPLETE',blocker=f'INVALID_OR_INCOMPLETE_ROLLOUT: {stage}/{method}/{arm}');save();return False
        return True
    for arm in ('ur5e','panda'):
        if not evaluate('joint_reps',arm,1,'p0'):return 2
    state['status']='P0_PASSED';save()
    if a.p0_only:return 0
    no_reps=artifacts/'runs/no_reps/checkpoints'/CHECKPOINTS['no_reps']
    if not no_reps.exists():
        state['status']='P1_ARTIFACT_PREPARATION';save()
        download_env=dict(os.environ,HF_HUB_OFFLINE='0')
        if not run([python,'-m','simulation_validation.download_inference','--source',source,'--root',artifacts,'--skip-joint','--include-no-reps'],out/'no_reps_download.log',env=download_env):
            state.update(status='P1_INCOMPLETE',blocker='NO_REPS_ARTIFACT_DOWNLOAD_FAILED');save();return 3
    state.update(status='P1_RUNNING',p1_started=True);save()
    for method in ('no_reps','joint_reps'):
        for arm in ('ur5e','panda'):
            if not evaluate(method,arm,20,'p1'):return 3
    state['status']='P1_COMPLETE';save();return 0
if __name__=='__main__':sys.exit(main())
