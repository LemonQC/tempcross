"""One physical episode, with per-control-step transactional recovery.

An uncertain in-flight step is never silently replayed. Completed episodes are
immutable. A normal pause resumes integration/controller/expert/evaluator state.
"""
import argparse
import copy
import fcntl
import json
import os
import pickle
import signal
import time
import traceback
from pathlib import Path
import numpy as np
from cross_behavior.common import (ROOT, PROJECT, now, write, read, atomic, dump, digest,
                                   episode_path, condition_seed, recover_step)

STOP = False


def pause_request(*_):
    global STOP
    STOP = True


def numerical_attributes(obj):
    return {k: copy.deepcopy(v) for k,v in vars(obj).items()
            if v is None or isinstance(v, (bool, int, float, str, np.ndarray, np.number))}


def checkpoint(env, expert, evaluator, step, record=None, action=None, decision=None):
    import mujoco
    mode = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.sim.model._model, mode))
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, state, mode)
    robot = env.robots[0]
    return dict(step=step, integration=state, expert=expert, evaluator=evaluator,
                controllers={k:numerical_attributes(v) for k,v in robot.part_controllers.items()},
                grippers={k:numerical_attributes(v) for k,v in robot.gripper.items()},
                robot=numerical_attributes(robot), env=numerical_attributes(env),
                rng=copy.deepcopy(env.rng.bit_generator.state), numpy_rng=np.random.get_state(),
                record=record, action=action, decision=decision)


def restore(env, saved):
    import mujoco
    mode = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(env.sim.model._model, env.sim.data._data, saved['integration'], mode)
    env.sim.forward()
    for key, attrs in saved['controllers'].items():
        for k,v in attrs.items():
            setattr(env.robots[0].part_controllers[key], k, v)
    for key, attrs in saved['grippers'].items():
        for k,v in attrs.items():
            setattr(env.robots[0].gripper[key], k, v)
    # Robot caches include control histories; environment clocks must not restart.
    for k,v in saved['robot'].items():
        setattr(env.robots[0], k, v)
    for k,v in saved['env'].items():
        setattr(env, k, v)
    env.rng.bit_generator.state = saved['rng']
    np.random.set_state(saved['numpy_rng'])
    # forward recomputes derived kinematics; restore integration incl. warmstart again.
    mujoco.mj_setState(env.sim.model._model, env.sim.data._data, saved['integration'], mode)
    check = np.empty_like(saved['integration'])
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, check, mode)
    assert np.array_equal(check, saved['integration']), 'Integration-state restore mismatch'
    return saved['expert'], saved['evaluator']


def code_hashes():
    hashes = {str(p.relative_to(PROJECT)):digest(p) for p in sorted((PROJECT/'cross_behavior').glob('*.py'))}
    for name, sha in hashes.items():
        dest = ROOT/'source_blobs'/sha
        if not dest.exists():
            atomic(dest, (PROJECT/name).read_bytes())
    return hashes


def run(task, robot, condition, pause_after=None, inspect_only=False):
    import cv2
    from cross_behavior.tasks import create
    from cross_behavior.sensing import observe
    from cross_behavior.expert import Expert
    from cross_behavior.evaluator import Evaluator
    directory = episode_path(task, robot, condition)
    result_path = directory/'episode.json'
    if result_path.exists():
        return read(result_path)
    directory.mkdir(parents=True, exist_ok=True)
    episode_lock = (directory/'worker.lock').open('a+')
    try:
        fcntl.flock(episode_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return dict(status='EPISODE_ALREADY_RUNNING', directory=str(directory))
    signal.signal(signal.SIGTERM, pause_request)
    signal.signal(signal.SIGINT, pause_request)
    contract = read(PROJECT/'configs/tasks/cross_behavior_contracts.json')
    launch = directory/'launch.json'
    first = not launch.exists()
    if first:
        write(launch, dict(utc=now(), task=task, robot=robot, condition=condition,
                          seed=condition_seed(task, condition), role='DEVELOPMENT_FEASIBILITY',
                          contract_sha256=digest(PROJECT/'configs/tasks/cross_behavior_contracts.json'),
                          code_sha256=code_hashes(), physics_steps=0))
    else:
        original = read(launch)
        assert original['contract_sha256'] == digest(PROJECT/'configs/tasks/cross_behavior_contracts.json')
    write(directory/f'execution-{time.time_ns()}.json', dict(utc=now(), code_sha256=code_hashes()))
    committed_step = recover_step(directory)
    env = None
    started = time.monotonic()
    write(directory/'status.json', dict(status='INITIALIZING', pid=os.getpid(), step=committed_step, utc=now()))
    try:
        env, camera = create(task, robot, condition_seed(task, condition))
        initial = directory/'initial.pkl'
        if not initial.exists():
            if not first:
                # No physical policy step existed; ordinary constructor errors can be repaired.
                assert not (directory/'inflight.json').exists()
            expert, evaluator = Expert(task), Evaluator(task, contract)
            baseline = evaluator.observe(env, np.array([0.,0.,0.,0.,0.,0.,-1.]), 0)
            xml = env.sim.model.get_xml()
            atomic(directory/'model.xml', xml.encode())
            sensors = observe(env, camera)
            cv2.imwrite(str(directory/'initial.png'), cv2.cvtColor(sensors.rgb, cv2.COLOR_RGB2BGR))
            np.savez_compressed(directory/'initial_rgbd.npz', rgb=sensors.rgb, depth=sensors.depth, xyz=sensors.xyz)
            dump(initial, checkpoint(env, expert, evaluator, 0, record=baseline))
            write(directory/'condition.json', dict(task=task, robot=robot, condition=condition,
                  condition_id=f'cb-v1-{task}-development-{condition:03d}', seed=condition_seed(task, condition),
                  xml_sha256=digest(directory/'model.xml'), initial_state_sha256=digest(initial),
                  metadata=env.get_ep_meta(), camera=camera, native_initialization_control_steps=10,
                  role='DEVELOPMENT_FEASIBILITY', contract_sha256=read(launch)['contract_sha256']))
            write(directory/'cursor.json', dict(step=0))
        else:
            saved_xml = (directory/'model.xml').read_text()
            if digest(directory/'model.xml') != __import__('hashlib').sha256(env.sim.model.get_xml().encode()).hexdigest():
                env.reset_from_xml_string(saved_xml)
            path = directory/'steps'/f'{committed_step:06d}.pkl' if committed_step else initial
            with path.open('rb') as f:
                saved = pickle.load(f)
            expert, evaluator = restore(env, saved)
            write(directory/f'resume-{time.time_ns()}.json', dict(utc=now(), step=committed_step,
                  integration_exact=True, code_sha256=code_hashes(), physics_replayed=False))
        if inspect_only or pause_after == 0:
            write(directory/'status.json', dict(status='PAUSED', step=committed_step, reason='explicit_initial_inspection', utc=now()))
            return read(directory/'status.json')
        records = []
        final = None
        for step in range(committed_step+1, contract['max_steps']+1):
            if STOP or (pause_after is not None and step-committed_step > pause_after):
                write(directory/'status.json', dict(status='PAUSED', step=step-1, reason='boundary_pause', utc=now()))
                return read(directory/'status.json')
            sensors = observe(env, camera)
            action, decision = expert.predict(sensors)
            assert action.shape == (7,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.
            actual = np.r_[action[:6], action[6], np.zeros(4), -1.]
            frame = directory/'frames'/f'{step-1:06d}.png'
            frame.parent.mkdir(exist_ok=True)
            assert cv2.imwrite(str(frame), cv2.cvtColor(sensors.rgb, cv2.COLOR_RGB2BGR))
            # Durable intent must exist before any new physics is executed.
            write(directory/'inflight.json', dict(next_step=step, action=action.tolist(), utc=now()))
            env.step(actual)
            final = evaluator.observe(env, action, step)
            snap = checkpoint(env, expert, evaluator, step, final, action, decision)
            dump(directory/'steps'/f'{step:06d}.pkl', snap)
            write(directory/'cursor.json', dict(step=step))
            (directory/'inflight.json').unlink()
            write(directory/'status.json', dict(status='RUNNING', pid=os.getpid(), step=step,
                  expert_phase=expert.phase, diagnostic_error=expert.error, utc=now()))
            if step % 40 == 0:
                print(json.dumps(dict(task=task, robot=robot, condition=condition, step=step,
                      phase=expert.phase, decision=decision, metrics=final)), flush=True)
            if final['success']:
                break
            # A controller that cannot perceive the target cannot furnish a demonstration.
            if expert.error and expert.elapsed >= 40:
                break
        if final is None:
            path = directory/'steps'/f'{committed_step:06d}.pkl'
            with path.open('rb') as f:
                final = pickle.load(f)['record']
        video = directory/'rollout.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20., (320, 180))
        assert writer.isOpened()
        try:
            for path in sorted((directory/'frames').glob('*.png')):
                writer.write(cv2.imread(str(path)))
        finally:
            writer.release()
        # Human-readable telemetry is built from already committed steps, not new physics.
        with (directory/'telemetry.jsonl').open('w') as stream:
            for path in sorted((directory/'steps').glob('*.pkl')):
                with path.open('rb') as f:
                    item = pickle.load(f)
                stream.write(json.dumps(dict(step=item['step'], action=item['action'].tolist(),
                             decision=item['decision'], metrics=item['record']))+'\n')
        final_sensors = observe(env, camera)
        cv2.imwrite(str(directory/'final.png'), cv2.cvtColor(final_sensors.rgb, cv2.COLOR_RGB2BGR))
        outcome = dict(status='COMPLETE', task=task, robot=robot, condition=condition,
                       condition_id=f'cb-v1-{task}-development-{condition:03d}',
                       role='DEVELOPMENT_FEASIBILITY', success=final['success'], steps=final['step'],
                       final_metrics=final, expert_phase=expert.phase, expert_error=expert.error,
                       wall_seconds=time.monotonic()-started, video=str(video),
                       telemetry=str(directory/'telemetry.jsonl'), condition_path=str(directory/'condition.json'),
                       contract_sha256=read(launch)['contract_sha256'], code_sha256=read(launch)['code_sha256'],
                       video_sha256=digest(video), telemetry_sha256=digest(directory/'telemetry.jsonl'))
        write(result_path, outcome)
        write(directory/'status.json', dict(status='COMPLETE', success=outcome['success'], step=outcome['steps'], utc=now()))
        print(json.dumps(outcome), flush=True)
        return outcome
    except BaseException as error:
        write(directory/f'error-{time.time_ns()}.json', dict(utc=now(), error=repr(error), traceback=traceback.format_exc(),
              committed_step=read(directory/'cursor.json')['step'] if (directory/'cursor.json').exists() else 0))
        write(directory/'status.json', dict(status='TECHNICAL_ERROR', error=repr(error), utc=now()))
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['push','stack','drawer'])
    p.add_argument('--robot', required=True, choices=['ur5e','panda'])
    p.add_argument('--condition', required=True, type=int, choices=range(8))
    p.add_argument('--pause-after', type=int)
    p.add_argument('--inspect-only', action='store_true')
    args = p.parse_args()
    run(args.task, args.robot, args.condition, args.pause_after, args.inspect_only)
