"""Extract factual task effects and suffix supervision; zero model/physics calls."""
from collections import Counter
import csv
import json
from pathlib import Path

from method_v0.common import OUT, ROOT, PROJECT, TASKS, guard, read, write, digest, ndarray_digest


def bilateral(records,target_geoms):
    result=[]
    for row in records:
        sides=set()
        for c in row['contacts']:
            if c['distance']>0:continue
            for a,b in [(c['geom1'],c['geom2']),(c['geom2'],c['geom1'])]:
                if b not in target_geoms:continue
                for side in ['left','right']:
                    if a in {f'gripper0_right_{side}_fingerpad_collision',f'gripper0_right_{side}_fingertip_collision'}:
                        sides.add(side)
        result.append(len(sides)==2)
    import numpy as np
    return np.asarray(result,dtype=bool)


def effect_word(placement,retreat):
    word=[]
    for p,r in zip(placement,retreat):
        event=f'{int(p)}{int(r)}'
        if not word or word[-1]!=event:word.append(event)
    return word


def inventory():
    import h5py
    import numpy as np
    source=read(PROJECT/'configs/datasets/baseline_source_manifest.json')
    target=read(PROJECT/'configs/datasets/baseline_target_manifest.json')
    counts=Counter((r['domain'],r['embodiment'],r['task'],r['split']) for r in source+target)
    metadata=Path('/home/users/lemon/coft-p2/p2/data/mg_pnp_lite/1.0.0/features.json')
    features=read(metadata)['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']
    state_meta=features['observation']['featuresDict']['features']['state']
    source_pack=Path('/home/users/lemon/baseline-benchmark/run-20261109/data/source.h5')
    packed_keys=set()
    with h5py.File(source_pack,'r') as f:
        assert len(f)==1800
        for g in f.values():packed_keys.update(g.keys())
    human=[]
    for path in sorted({r['original_file'] for r in target}):
        with h5py.File(path,'r') as f:
            rr=[]
            for name,g in f['data'].items():
                if not isinstance(g,h5py.Group):continue
                reward=g['rewards'][:];done=g['dones'][:]
                rr.append(dict(demo=name,frames=len(g['actions']),reward_unique=np.unique(reward).tolist(),
                    done_positive_states=int((done!=0).sum()),states_available='states' in g,
                    object_pose_dataset_available='object_poses' in g,
                    directly_recorded_contact_normals_available='contacts' in g))
            human.append(dict(path=path,sha256=digest(path),episodes=rr))
    result=dict(source_demonstrations=1800,panda_demonstrations=100,
        split_counts=[dict(domain=k[0],robot=k[1],task=k[2],split=k[3],episodes=v) for k,v in sorted(counts.items())],
        source_pack_keys=sorted(packed_keys),source_rlds_step_keys=sorted(features),
        source_rlds_observation_keys=sorted(features['observation']['featuresDict']['features']),
        source_state_description=state_meta.get('description'),source_state_dimensions=state_meta['tensor']['shape'],
        source_reward_description=features['reward'].get('description'),human_hdf5=human,
        exact_source_contact_labels_in_rlds=False,certified_alternative_continuation_outcomes_in_demonstrations=False,
        source_effect_labels='2D object bboxes / language motion labels; not a contact wrench or full physical state',
        deployment_joint_state_assumed=False,new_model_or_physics_calls=0)
    write(OUT/'supervision_inventory.json',result)


def entries():
    old=read(PROJECT/'results/baselines/analysis.json')['episodes']
    assert len(old)==1280
    for r in old:
        yield dict(r,corpus='baseline_1280',uid=f"baseline/{r['label']}/{r['task']}/{r['robot']}/{r['official_condition_index']}",
                   predicate_prefix='baseline',behavior=r['label'],declared_horizon=600 if r['task']==TASKS[0] else 650)
    with (PROJECT/'results/strong_baselines/barx_adaptation/episodes.csv').open() as f:
        new=[r for r in csv.DictReader(f) if r['new_physical_run']=='True']
    assert len(new)==160
    for r in new:
        yield dict(r,success=r['success']=='True',steps=int(r['steps']),official_condition_index=int(r['official_condition_index']),
                   corpus='barx_adaptation_160',uid=f"adapted/{r['task']}/{r['robot']}/{r['official_condition_index']}",
                   predicate_prefix='baseline',behavior='barx_adaptation_selected',declared_horizon=600 if r['task']==TASKS[0] else 650)
    paths=sorted(Path('/home/users/lemon/mimicgen-baseline-20261112/augmentation/probe').glob('*/episode.json'))
    assert len(paths)==177
    for path in paths:
        r=read(path)
        yield dict(r,corpus='mimicgen_augmentation_177',uid=r['id'],robot='panda',telemetry_path=r['telemetry'],
                   predicate_prefix='mimicgen',behavior='mimicgen_fixed',official_condition_index=r['condition'],
                   declared_horizon=600 if r['task']==TASKS[0] else 650)


def extract_entry(r):
    import h5py
    import numpy as np
    path=Path(r['telemetry_path']);logs_path=path.with_suffix('.jsonl')
    logs=[json.loads(x) for x in logs_path.read_text().splitlines()]
    with h5py.File(path,'r') as f:
        n=r['steps'];a=f['actions/environment'][:]
        assert len(a)==n and len(logs)==n+1
        body_names=json.loads(f.attrs['body_names']);geom_names=json.loads(f.attrs['geom_names'])
        geoms={g for g,b in zip(geom_names,f['model/geom_bodyid'][:]) if body_names[b].startswith('obj_')}
        contact=bilateral(logs,geoms);hits=np.flatnonzero(contact);anchor=int(hits[0]) if len(hits) else None
        prefix=r['predicate_prefix']
        placement=f[prefix+'/post_action_placement'][:].astype(bool)
        retreat=f[prefix+'/post_action_retreat'][:].astype(bool)
        goal=placement & retreat;goal_hits=np.flatnonzero(goal)
        assert len(goal)==n and bool(goal.any())==r['success']
        position=f['target/position'][:];assert len(position)==n+1 and np.isfinite(position).all()
        meta=json.loads(f.attrs['ep_meta']);objects={o['name']:o for o in meta['object_cfgs']}
        obj=objects['obj']['info'];asset=obj.get('mjcf_path');category=obj.get('cat')
        start_state=np.r_[f['sim/qpos'][0],f['sim/qvel'][0]]
        index=anchor if anchor is not None else 0
        initial_predicate=None if index==0 else f'{int(placement[index-1])}{int(retreat[index-1])}'
        word=effect_word(placement[max(index-1,0):],retreat[max(index-1,0):])
        first_goal=int(goal_hits[0]+1) if len(goal_hits) else None
        result=dict(uid=r['uid'],corpus=r['corpus'],task=r['task'],robot=r['robot'],behavior=r['behavior'],
            condition=r['official_condition_index'],condition_id=r.get('condition_id'),steps=n,
            benchmark_horizon=r['declared_horizon'],first_bilateral_contact_state=anchor,
            contact_is_not_load_bearing_label=True,official_success=bool(r['success']),
            first_official_goal_state=first_goal,initial_anchor_predicate=initial_predicate,
            continuation_effect_word=word,object_displacement_after_anchor=(position[-1]-position[index]).tolist(),
            object_initial_position=position[0].tolist(),object_initial_quaternion=f['target/quaternion'][0].tolist(),
            object_asset=asset,object_category=category,
            recorded_suffix_steps=n-index,remaining_benchmark_budget=r['declared_horizon']-index,
            truncated_before_benchmark_horizon=not r['success'] and n<r['declared_horizon'],
            factual_failure_is_physical_infeasibility=False,certified_physical_negative=False,
            initial_state_digest=ndarray_digest(start_state),
            telemetry_path=str(path),contact_log=str(logs_path))
        ROOT.mkdir(exist_ok=True)
        safe=__import__('hashlib').sha256(r['uid'].encode()).hexdigest()[:24]
        np.savez_compressed(ROOT/(safe+'.npz'),object_position=position,
            placement=placement,retreat=retreat,bilateral_contact=contact)
        result['extracted_arrays']=str(ROOT/(safe+'.npz'))
    return result,[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in [path,logs_path]]


def main():
    guard();OUT.mkdir(parents=True,exist_ok=True);ROOT.mkdir(exist_ok=True)
    inventory()
    rows=[];inputs=[]
    for i,r in enumerate(entries()):
        row,files=extract_entry(r);rows.append(row);inputs.extend(files)
        if (i+1)%200==0:print(json.dumps(dict(extracted=i+1)),flush=True)
    assert len(rows)==1617 and len({r['uid'] for r in rows})==1617
    write(OUT/'factual_continuations.json',rows)
    write(OUT/'input_index.json',inputs)
    print(json.dumps(dict(episodes=len(rows),physics_steps=0,training_updates=0)))


if __name__=='__main__':main()
