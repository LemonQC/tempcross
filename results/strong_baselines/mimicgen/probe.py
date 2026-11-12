"""Finite official MimicGen generation + action-replay gate; never trains."""
import datetime
import json
import os
from pathlib import Path
import traceback

from common import (OUT, ROOT, SOURCE, TASKS, PYTHON, digest, install_paths,
                    install_annotation_frame_repair, read, write)


def completed(name):
    state = read(OUT / 'status.json')
    if name not in state['completed_stages']:
        state['completed_stages'].append(name)
    write(OUT / 'status.json', state)


def prepare():
    import h5py
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset
    from mimicgen.utils.file_utils import parse_source_dataset
    from mimicgen.configs import MG_TaskSpec, config_factory
    from barx.mimicgen_release import registry
    inputs = read(OUT / 'source_seed_manifest.json')
    for task in TASKS:
        prepared = ROOT / 'prepared' / (task + '.hdf5')
        mark = prepared.with_suffix('.json')
        if mark.exists():
            assert digest(prepared) == read(mark)['sha256']
            continue
        if prepared.exists():
            raise FileExistsError('Unfinished preparation requires diagnosis, no overwrite: ' + str(prepared))
        subset = prepared.with_name(task + '-seed-subset.hdf5')
        subset.parent.mkdir(parents=True, exist_ok=True)
        if subset.exists():
            raise FileExistsError(str(subset))
        rows = [r for r in inputs if r['task'] == task]
        with h5py.File(rows[0]['original_file'], 'r') as src, h5py.File(subset, 'x') as dst:
            data = dst.create_group('data')
            for k,v in src['data'].attrs.items():
                data.attrs[k] = v
            for row in rows:
                src.copy(src['data'][row['original_demo']], data, name=row['original_demo'])
        spec = registry()['tasks'][task]
        prepare_src_dataset(str(subset), spec['interface'], 'robosuite', n=5, output_path=str(prepared))
        cfg = config_factory(name=spec['config_name'], config_type='robosuite')
        task_spec = MG_TaskSpec.from_json(json_dict=json.loads(cfg.dump())['task']['task_spec'])
        _, indices, _, _, valid = parse_source_dataset(str(prepared),
            demo_keys=[r['original_demo'] for r in rows], task_spec=task_spec)
        if len(valid) != 5:
            raise RuntimeError('Not all five frozen source demos have valid official subtask annotations')
        write(mark, dict(status='COMPLETE', path=str(prepared), sha256=digest(prepared),
                         demos=valid, source_episode_ids=[r['episode_id'] for r in rows],
                         subtask_indices=indices.tolist()))
    completed('source_preparation')


class EpisodeFinished(Exception):
    pass


def run_episode(env, task, condition, role, key, generator=None, interface=None, replay_actions=None):
    import cv2
    import h5py
    import numpy as np
    from barx.benchmark import EMBODIMENTS, TASKS as specs
    from barx.evaluation_conditions import restore_frozen_condition, load_bundle
    from simulation_validation.observe import Recorder
    from baselines.common.metrics import observe_subpredicates
    directory = ROOT / 'probe' / key
    if directory.exists():
        marker = directory / 'episode.json'
        if marker.exists() and read(marker)['status'] == 'COMPLETE':
            return read(marker)
        raise FileExistsError('Preserve incomplete physical trajectory: ' + str(directory))
    directory.mkdir(parents=True)
    payload, states, models = load_bundle(SOURCE / 'evaluation/conditions', task=task, embodiment='panda')
    entry = payload['episodes'][condition]
    rec = Recorder(directory / 'telemetry')
    original_reset, original_step = env.reset, env.step
    frames, actions, flattened = [], [], []
    official_success = False
    last_metadata = None
    writer = None
    camera = EMBODIMENTS['panda'].camera + '_image'
    def reset(*args, **kwargs):
        nonlocal writer, last_metadata
        env.reset = original_reset
        try:
            obs, last_metadata = restore_frozen_condition(env, entry, states[condition], models[condition])
        finally:
            env.reset = reset
        assert env.env.control_freq == 20
        rec.start(entry, env, obs, last_metadata)
        flattened.append(env.get_state()['states'].copy())
        writer = cv2.VideoWriter(str(directory / 'video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (320,180))
        assert writer.isOpened()
        return obs
    def step(action):
        nonlocal official_success
        assert len(actions) < specs[task].max_steps
        before = env.get_observation()
        rgb = (255*np.transpose(before[camera],(1,2,0))).astype(np.uint8)
        frames.append(rgb);writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        actions.append(np.asarray(action).copy())
        result = original_step(action)
        obs, _, _, info = result
        rec.snapshot(env, obs, action, info)
        sub = observe_subpredicates(env.env, task)
        for k,v in sub.items():
            rec.append('mimicgen/post_action_'+k,v)
        official_success = bool(info['is_success']['task'])
        assert official_success == (sub['placement'] and sub['retreat'])
        flattened.append(env.get_state()['states'].copy())
        if official_success or len(actions) == specs[task].max_steps:
            raise EpisodeFinished()
        return result
    env.reset, env.step = reset, step
    ledger = ROOT / 'physical_launches.jsonl'
    with ledger.open('a') as f:
        f.write(json.dumps(dict(id=key,role=role,task=task,condition=condition,
            condition_id=entry['condition_id'],started=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
    state = read(OUT/'status.json')
    state['actual_new_physical_rollouts'] += 1
    if role == 'MIMICGEN_GENERATION': state['actual_generation_attempts'] += 1
    state['phase'] = key
    write(OUT/'status.json',state)
    upstream_result = None
    try:
        if generator is not None:
            upstream_result = generator.generate(env, interface, select_src_per_subtask=True,
                transform_first_robot_pose=True, interpolate_from_last_target_pose=True)
        else:
            env.reset()
            for action in replay_actions:
                env.step(action)
    except EpisodeFinished:
        pass
    finally:
        env.reset, env.step = original_reset, original_step
        if writer is not None: writer.release()
        rec.close()
    a = np.asarray(actions)
    assert len(a) > 0 and a.shape[1] == 12 and np.isfinite(a).all()
    assert np.all(a[:,7:11] == 0) and np.all(a[:,11] == -1)
    with h5py.File(directory/'trajectory.hdf5','x') as f:
        f.create_dataset('actions',data=a)
        f.create_dataset('states',data=np.asarray(flattened),compression='gzip')
        f.create_dataset('rgb',data=np.asarray(frames),compression='gzip')
        f.attrs['ep_meta']=json.dumps(last_metadata)
        f.attrs['condition_id']=entry['condition_id']
        f.attrs['role']=role
        if upstream_result is not None:
            f.attrs['upstream_success']=upstream_result['success']
            f.attrs['source_demo_indices']=json.dumps([int(i) for i in upstream_result['src_demo_inds']])
    record=dict(status='COMPLETE',id=key,role=role,task=task,condition=condition,
                condition_id=entry['condition_id'],success=official_success,steps=len(a),
                trajectory=str(directory/'trajectory.hdf5'),video=str(directory/'video.mp4'),
                telemetry=str(directory/'telemetry/episode-000.h5'),
                contact_log=str(directory/'telemetry/episode-000.jsonl'))
    record['artifacts']=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p))
                        for p in directory.rglob('*') if p.is_file()]
    write(directory/'episode.json',record)
    state=read(OUT/'status.json')
    field='generated_episode_ids' if generator is not None else 'replayed_episode_ids'
    state.setdefault(field,[]).append(key)
    if official_success:
        state['successful_generation_episode_ids' if generator is not None else 'replay_success_episode_ids'].append(key)
    write(OUT/'status.json',state)
    return record


def main():
    state=read(OUT/'status.json')
    if state['status'] in ['BLOCKED','COMPLETE']:
        print('Terminal status; no new physical trajectories launched.')
        return
    if 'probe_complete' in state['completed_stages']:
        print('Probe complete; no repeat.')
        return
    marker=read(ROOT/'freeze.json')
    for path,sha in marker['sha256'].items():
        assert digest(path)==sha, 'Frozen input changed: '+path
    install_paths();install_annotation_frame_repair()
    import h5py
    import numpy as np
    np.random.seed(7)
    import robocasa
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.configs import MG_TaskSpec, config_factory
    from mimicgen.env_interfaces.base import make_interface
    from barx.mimicgen_release import registry
    from barx.simulation import build_environment_config, initialize_observation_utils
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    prepare()
    records=[]
    for task,n in zip(TASKS,[3,2]):
        spec=registry()['tasks'][task]
        cfg=config_factory(name=spec['config_name'],config_type='robosuite')
        ts=MG_TaskSpec.from_json(json_dict=json.loads(cfg.dump())['task']['task_spec'])
        prepared=ROOT/'prepared'/(task+'.hdf5')
        generator=DataGenerator(ts,str(prepared),read(prepared.with_suffix('.json'))['demos'])
        config=build_environment_config(task,'panda');initialize_observation_utils(config)
        config['control_freq']=20
        env=create_env(**config,env_type=1,render=False,render_offscreen=True,
                       use_image_obs=True,use_camera_obs=False,rng=np.random.default_rng(7))
        interface=make_interface(name=spec['interface'],interface_type='robosuite',env=env.env)
        try:
            for condition in range(n):
                key=f'{task}-gen-{condition:03d}'
                generated=run_episode(env,task,condition,'MIMICGEN_GENERATION',key,generator,interface)
                records.append(generated)
                with h5py.File(generated['trajectory']) as f: actions=f['actions'][:]
                records.append(run_episode(env,task,condition,'ACTION_REPLAY',
                    f'{task}-replay-{condition:03d}',replay_actions=actions))
        finally:
            env.env.close()
    assert len(records)==10
    passed=sum(r['success'] for r in records)>=6
    passed=passed and all(any(r['success'] and r['task']==t and r['role']==role for r in records)
                         for t in TASKS for role in ['MIMICGEN_GENERATION','ACTION_REPLAY'])
    write(OUT/'probe_results.json',dict(status='PASS' if passed else 'BLOCKED',
          physical_rollouts=len(records),successes=sum(r['success'] for r in records),
          records=records,new_training_updates=0))
    completed('probe_complete')
    state=read(OUT/'status.json')
    state['phase']='POST_PROBE_DOWNSTREAM_FREEZE' if passed else 'PROBE_GATE_FAILED'
    if not passed:
        state['status']='BLOCKED';state['reason']='Frozen generation/replay gate not passed; no downstream training/evaluation.'
    write(OUT/'status.json',state)


if __name__=='__main__':
    import fcntl
    lock=(ROOT/'execution.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(ROOT/'runner_pid.json',dict(pid=os.getpid(),script=str(Path(__file__).resolve())))
    try:
        main()
    except BaseException as exc:
        state=read(OUT/'status.json')
        state.update(status='BLOCKED',reason=repr(exc),traceback=traceback.format_exc(),
                     reproduction_command=str(PYTHON)+' '+str(OUT/'probe.py'),
                     automatic_retry_permitted=False)
        write(OUT/'status.json',state)
        raise
