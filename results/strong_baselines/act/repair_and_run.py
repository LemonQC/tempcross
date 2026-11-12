"""One declared decoder repair, preserving the original attempt in full.

Wait for the already-running original child to close its frozen training stage.
Its stopped parent cannot start a reference evaluation. Only after that child
has exited do corrected probes, refreeze, training and evaluation run, in order.
No subsequent algorithm is launched by this script.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from baselines.strong.common import ROOT,PROJECT,PYTHON,out,read,write,digest,resource,now

def call(module,log):
    with (ROOT/log).open('a') as stream:
        subprocess.run([str(PYTHON),'-m',module],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    repair=read(out('act')/'technical_repair.json');assert str(ROOT)==repair['corrected_server_root']
    original=Path(repair['old_server_root']);archive=Path(repair['old_attempt_archive'])
    start=time.monotonic()
    try:
        while True:
            state=read(original/'act/training/target_only/status.json')
            if state['status']=='COMPLETE':break
            if state['status']=='TECHNICAL_FAILURE':raise RuntimeError('Original attempt failed; preserve before proceeding')
            if time.monotonic()-start>3600:raise TimeoutError('Waiting for original training stage exceeded one hour')
            time.sleep(15)
        assert state['actual_optimizer_updates']==24000
        # The parent was deliberately stopped before the original child ended.
        # Terminate only that owned, stopped orchestration process after its
        # training child has closed the complete checkpoint and consumption log.
        pid=read(original/'act-run-process.json')['pid']
        status=Path(f'/proc/{pid}/status');cmd=Path(f'/proc/{pid}/cmdline')
        if status.exists():
            assert b'results.strong_baselines.act.run' in cmd.read_bytes()
            assert '\nState:\tT' in status.read_text()
            children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
            for child in children:
                child_status=Path(f'/proc/{child}/status');deadline=time.monotonic()+120
                while child_status.exists() and '\nState:\tZ' not in child_status.read_text():
                    if time.monotonic()>deadline:raise TimeoutError('Original child did not close after COMPLETE checkpoint')
                    time.sleep(1)
            os.kill(pid,signal.SIGKILL)
        old_checkpoint=original/'act/training/target_only/latest.pt'
        original_summary=dict(status='PARTIAL',role='PRESERVED_UPSTREAM_FIRST_DECODER_EXECUTION',
            actual_optimizer_updates=24000,target_anchor_exposures=384000,probe_optimizer_updates=2,
            physical_smoke_episodes=4,formal_reference_episodes=0,formal_success_rate=None,
            checkpoint=str(old_checkpoint),checkpoint_sha256=digest(old_checkpoint),state=state,
            old_frozen_github_commit='1a3808e8c9707d35d09af23775774f196f714934',
            code_archive=str(archive),reason='Does not train the intended seven-layer decoder. Preserved as an upstream implementation limitation, not the repaired baseline result.')
        dest=out('act')/'attempts/upstream_first_decoder';dest.mkdir(parents=True,exist_ok=True)
        write(dest/'summary.json',original_summary)
        for name in ['training_target_only.json','target_only_selection.json']:
            shutil.copyfile(out('act')/name,dest/name)
        write(archive/'completed_training_summary.json',original_summary)
        original_files=[p for p in (original/'act').rglob('*') if p.is_file()]
        original_files += [p for p in archive.rglob('*') if p.is_file()]
        original_files += list(original.glob('act-*.log'))
        write(dest/'sha_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(original_files)])
        # Original child is complete; no old reference job was launched.
        assert not list((original/'act/physical').glob('formal/*/*/job.json'))
        write(ROOT/'resource_before_corrected_probe.json',resource())
        call('results.strong_baselines.act.finish_probes','act-probes-all.log')
        # A source-level repair is accepted only when gradients demonstrably
        # reach all seven configured decoder layers on real saved samples.
        import re
        for group in ['target_only','source_target']:
            changes=read(ROOT/'act/probe'/group/'parameter_changes.json')['changed_tensors']
            layers={int(match.group(1)) for name in changes if (match:=re.search(r'model\.transformer\.decoder\.layers\.(\d+)\.',name))}
            assert layers==set(range(7)),layers
        write(out('act')/'decoder_repair_acceptance.json',dict(status='PASS',all_seven_decoder_layers_updated_in_both_real_probes=True,
            repaired_probe_updates=2,repaired_full_physical_smoke_episodes=4,reference_evaluations_used_for_repair=0,technical_repairs_used=1))
        call('results.strong_baselines.act.freeze','act-freeze.log')
        call('results.strong_baselines.act.run','act-run.log')
    except BaseException as error:
        s=read(out('act')/'status.json');s.update(status='BLOCKED',phase='AFTER_SINGLE_DECODER_REPAIR',error=repr(error),traceback=traceback.format_exc(),technical_repairs=1,remaining_technical_repairs=0)
        write(out('act')/'status.json',s);write(out('act')/'blocked_after_repair.json',s);raise

if __name__=='__main__':main()
