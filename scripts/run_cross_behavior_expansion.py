#!/usr/bin/env python3
"""Single durable entry point. --status is CPU-only; --resume never repeats work."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_behavior.common import (ROOT, PROJECT, PYTHON, BRANCH, BASE, TASKS, ROBOTS,
                                  now, digest, read, write, environment, episode_path)


def status():
    p = ROOT/'status.json'
    return read(p) if p.exists() else dict(status='NOT_STARTED', root=str(ROOT), next='inventory_then_stage_A')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inventory','A','auto'], default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--background', action='store_true', help='Detach persistent runner; inspect with --status')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, str(Path(__file__).resolve())] + [x for x in sys.argv[1:] if x != '--background']
        with (ROOT/'runner.log').open('a') as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(status='BACKGROUND_LAUNCH', pid=process.pid, root=str(ROOT))))
        return
    lock = (ROOT/'runner.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='ALREADY_RUNNING', current=status())))
        return
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=PROJECT, text=True).strip()
    assert branch == BRANCH, 'Refuse execution on main or another research branch'
    if (ROOT/'status.json').exists() and not args.resume:
        parser.error('Run exists; explicitly use --resume. No automatic restart.')
    if args.resume and status()['status'] in ('BLOCKED_PHYSICAL_GATE','COMPLETE'):
        print(json.dumps(status(), ensure_ascii=False, indent=2))
        return
    if not (ROOT/'inventory.json').exists():
        resources = {}
        for name, command in [('gpu',['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv']),
                              ('disk',['df','-h',str(ROOT),'/data']),
                              ('processes',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'])]:
            probe = subprocess.run(command, capture_output=True, text=True)
            resources[name] = dict(returncode=probe.returncode, output=probe.stdout+probe.stderr)
        write(ROOT/'inventory.json', dict(utc=now(), branch=branch, base=BASE,
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd='/home/users/lemon/coft-p2/barx',text=True).strip(),
              project=str(PROJECT), root=str(ROOT), python=str(PYTHON), resources=resources,
              data_directory_writable=os.access('/data', os.W_OK),
              task_order=list(TASKS), all_three_tasks_before_training=True))
    from cross_behavior.preservation import audit
    audit()
    state = status()
    for stale in ('returncode','worker_pid','log','failure','next'):
        state.pop(stale, None)
    state.update(status='RUNNING_STAGE_A', pid=os.getpid(), branch=branch, base=BASE, root=str(ROOT), utc=now())
    write(ROOT/'status.json', state)
    if args.stage == 'inventory':
        state.update(status='INVENTORY_COMPLETE', next='Stage A Push', utc=now())
        write(ROOT/'status.json', state)
        print(json.dumps(state))
        return
    for task in TASKS:
        if args.task and task != args.task:
            continue
        for robot in ROBOTS:
            for i in range(8):
                directory = episode_path(task, robot, i)
                if (directory/'episode.json').exists():
                    continue
                completed = [read(episode_path(task,robot,j)/'episode.json') for j in range(8)
                             if (episode_path(task,robot,j)/'episode.json').exists()]
                failures = sum(not e['success'] for e in completed)
                if failures > 2 and not (directory/'launch.json').exists():
                    # With three retained failures 6/8 is mathematically impossible.
                    # Unrun episodes remain unrun, rather than being assigned failures.
                    break
                state.update(current=dict(stage='A', task=task, robot=robot, condition=i), utc=now())
                write(ROOT/'status.json', state)
                log = directory/'worker.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open('a') as stream:
                    child = subprocess.Popen([str(PYTHON),'-m','cross_behavior.worker','--task',task,'--robot',robot,'--condition',str(i)],
                                             cwd=PROJECT, env=environment(), stdout=stream, stderr=subprocess.STDOUT)
                    state.update(worker_pid=child.pid, log=str(log))
                    write(ROOT/'status.json', state)
                    def pause(*_):
                        child.send_signal(signal.SIGTERM)
                    signal.signal(signal.SIGTERM, pause)
                    signal.signal(signal.SIGINT, pause)
                    rc = child.wait()
                if rc or not (directory/'episode.json').exists():
                    state.update(status='PAUSED_OR_TECHNICAL_ERROR', returncode=rc, next='Inspect existing episode; resume without new condition', utc=now())
                    write(ROOT/'status.json', state)
                    return
                result = read(directory/'episode.json')
                # A physical failure is inspected before spending the rest of a task's budget.
                repair = ROOT/'repairs'/f'{task}.json'
                if not result['success'] and not repair.exists():
                    state.update(status='PHYSICAL_DIAGNOSIS_REQUIRED', failure=str(directory/'episode.json'),
                                 next=f'Diagnose {task}; at most one evidence-based repair before remaining trials', utc=now())
                    write(ROOT/'status.json', state)
                    return
        gates = {}
        for robot in ROBOTS:
            entries = [read(episode_path(task,robot,i)/'episode.json') for i in range(8)
                       if (episode_path(task,robot,i)/'episode.json').exists()]
            successes = sum(bool(e['success']) for e in entries)
            gates[robot] = dict(successes=successes, episodes=len(entries), not_run=8-len(entries),
                               threshold='6/8', passed=len(entries)==8 and successes>=6,
                               early_stop_reason='Three failures rule out 6/8' if len(entries)<8 else None)
        write(ROOT/'stage_a'/task/'gate.json', dict(task=task, status='PASS' if all(g['passed'] for g in gates.values()) else 'BLOCKED', robots=gates, utc=now()))
    gates = {task:read(ROOT/'stage_a'/task/'gate.json') for task in TASKS if (ROOT/'stage_a'/task/'gate.json').exists()}
    passed = len(gates)==3 and all(g['status']=='PASS' for g in gates.values())
    state.update(status='STAGE_A_PASSED' if passed else ('BLOCKED_PHYSICAL_GATE' if len(gates)==3 else 'STAGE_A_PARTIAL'),
                 gates=gates, training_updates=0, formal_policy_evaluations=0,
                 next='Implement and validate Stage B before any training' if passed else 'Preserve physical evidence; no training while any task gate is unpassed', utc=now())
    state['last_runner_pid'] = state.pop('pid',None)
    state['last_worker_pid'] = state.pop('worker_pid',None)
    state['last_condition'] = state.pop('current',None)
    write(ROOT/'status.json', state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
