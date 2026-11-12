"""Bounded official MimicGen augmentation with exact, recorded source lineage."""
import json
import random
import time
from common import OUT, ROOT, SOURCE, TASKS, digest, install_paths, install_annotation_frame_repair, read, write


def main():
    import numpy as np
    cfg=read(OUT/'downstream_protocol.json')
    dest=ROOT/'augmentation';dest.mkdir(exist_ok=True)
    done=dest/'complete.json'
    if done.exists():
        assert read(done)['status']=='COMPLETE'
        return
    install_paths();install_annotation_frame_repair()
    import probe as shared
    shared.ROOT=dest
    import robocasa
    from mimicgen.configs import config_factory, MG_TaskSpec
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.env_interfaces.base import make_interface
    from barx.mimicgen_release import registry
    from barx.simulation import build_environment_config, initialize_observation_utils
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    seed_rows=read(OUT/'source_seed_manifest.json')
    all_rows=[];start=time.monotonic()
    for ti,task in enumerate(TASKS):
        spec=registry()['tasks'][task]
        official=config_factory(name=spec['config_name'],config_type='robosuite')
        ts=MG_TaskSpec.from_json(json_dict=json.loads(official.dump())['task']['task_spec'])
        prepared=ROOT/'prepared'/(task+'.hdf5');meta=read(prepared.with_suffix('.json'))
        assert digest(prepared)==meta['sha256']
        generator=DataGenerator(ts,str(prepared),meta['demos'])
        parents={r['original_demo']:r['episode_id'] for r in seed_rows if r['task']==task}
        config=build_environment_config(task,'panda');initialize_observation_utils(config)
        config['control_freq']=20
        env=create_env(**config,env_type=1,render=False,render_offscreen=True,use_image_obs=True,
                       use_camera_obs=False,rng=np.random.default_rng(7))
        interface=make_interface(name=spec['interface'],interface_type='robosuite',env=env.env)
        original_selection=generator.select_source_demo
        current=None
        def select(**kwargs):
            index=original_selection(**kwargs)
            name=generator.demo_keys[int(index)]
            current['selected_sources'].append(dict(subtask=int(kwargs['subtask_ind']),
                source_demo=name,episode_id=parents[name],source_index=int(index)))
            write(current['path'],current)
            return index
        generator.select_source_demo=select
        successes=0;task_rows=[]
        try:
            for attempt in range(cfg['augmentation']['max_attempts_per_task']):
                if time.monotonic()-start>cfg['augmentation']['max_wall_hours']*3600:
                    raise TimeoutError('Frozen augmentation wall cap')
                if __import__('shutil').disk_usage(ROOT).free<25*1024**3:
                    raise OSError('Frozen disk floor reached before generation')
                key=f'aug-{task}-{attempt:03d}'
                seed=700000+1000*ti+attempt;condition=attempt%20
                path=dest/'lineage'/(key+'.json')
                marker=dest/'probe'/key/'episode.json'
                if marker.exists():
                    assert path.exists()
                    row=read(marker);lineage=read(path)
                    assert lineage['seed']==seed and lineage['selected_sources']
                else:
                    if path.exists():raise FileExistsError('Incomplete generation; no implicit retry: '+str(path))
                    current=dict(id=key,task=task,seed=seed,condition=condition,path=str(path),
                                 selected_sources=[],stage='MIMICGEN_AUGMENTATION')
                    write(path,current)
                    np.random.seed(seed);random.seed(seed)
                    row=shared.run_episode(env,task,condition,'MIMICGEN_GENERATION',key,generator,interface)
                row=dict(row,generation_seed=seed,lineage=str(path),stage='AUGMENTATION')
                task_rows.append(row);all_rows.append(row)
                successes+=int(row['success'])
                write(dest/(task+'-progress.json'),dict(attempts=len(task_rows),successes=successes,rows=task_rows))
                state=read(OUT/'status.json')
                state['augmentation_progress']={t:read(dest/(t+'-progress.json'))['successes']
                    for t in TASKS if (dest/(t+'-progress.json')).exists()}
                write(OUT/'status.json',state)
                print(json.dumps(dict(stage='augmentation',task=task,attempt=attempt,success=row['success'],successes=successes)),flush=True)
                if successes==cfg['augmentation']['successful_demos_per_task']:break
            if successes!=cfg['augmentation']['successful_demos_per_task']:
                raise RuntimeError(f'Frozen generation cap exhausted for {task}: {successes}/{len(task_rows)}; no enlarged budget')
        finally:env.env.close()
    write(done,dict(status='COMPLETE',attempts=len(all_rows),successful_demos=sum(r['success'] for r in all_rows),
        rows=all_rows,elapsed_seconds=time.monotonic()-start,actual_physics_steps=sum(r['steps'] for r in all_rows)))


if __name__=='__main__':main()
