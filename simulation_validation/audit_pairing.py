"""Audit restored official scenes; never impose common robot joint states."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np

SCENE_KEYS=('object_cfgs','fixtures','fixture_refs','gen_textures','lang','layout_id','style_id')


def read(path):
    with h5py.File(path) as f:
        entry=json.loads(f.attrs['condition']);names=json.loads(f.attrs['body_names'])
        positions=f['sim/body_xpos'][0];quaternions=f['sim/body_xquat'][0]
        bodies={name:(positions[i],quaternions[i]) for i,name in enumerate(names)
                if name and not any(s in name.lower() for s in ('robot','gripper','mobilebase'))}
        return dict(entry=entry,bodies=bodies,camera=f['camera/T_world_optical'][0],
                    qpos=f['sim/qpos'][0],target=f['target/position'][0])


def semantic_metadata(value,key=None):
    if isinstance(value,dict):return {k:semantic_metadata(v,k) for k,v in value.items()}
    if isinstance(value,list):
        items=[semantic_metadata(v) for v in value]
        return sorted(items) if key in ('groups','groups_containing_sampled_obj') else items
    return value


def compare(first,second):
    a=read(first);b=read(second);common=a['bodies'].keys()&b['bodies'].keys()
    scene_equal=all(a['entry']['ep_meta'][k]==b['entry']['ep_meta'][k] for k in SCENE_KEYS)
    max_position=max(float(np.linalg.norm(a['bodies'][key][0]-b['bodies'][key][0])) for key in common)
    max_rotation=max(float(2*np.arccos(np.clip(abs(np.dot(a['bodies'][key][1],b['bodies'][key][1])),0,1))) for key in common)
    return dict(seed_a=a['entry']['seed'],seed_b=b['entry']['seed'],same_scene_metadata=scene_equal,
                different_scene_metadata_keys=[k for k in SCENE_KEYS if a['entry']['ep_meta'][k]!=b['entry']['ep_meta'][k]],
                same_semantic_scene_metadata=all(semantic_metadata(a['entry']['ep_meta'][k])==semantic_metadata(b['entry']['ep_meta'][k]) for k in SCENE_KEYS),
                same_nonrobot_body_names=a['bodies'].keys()==b['bodies'].keys(),common_nonrobot_bodies=len(common),
                max_nonrobot_position_difference_m=max_position,max_nonrobot_rotation_difference_rad=max_rotation,
                initial_target_difference_m=float(np.linalg.norm(a['target']-b['target'])),
                initial_camera_transform_max_abs_difference=float(np.abs(a['camera']-b['camera']).max()),
                robot_state_size_a=len(a['qpos']),robot_state_size_b=len(b['qpos']))


def audit(root,out):
    root=Path(root);rows=[]
    for method in ('no_reps','joint_reps'):
        for i in range(20):
            paths=[root/'p1'/method/arm/'telemetry'/f'episode-{i:03d}.h5' for arm in ('ur5e','panda')]
            summaries=[list((p.parents[1]/'official').rglob('summary.json')) for p in paths]
            if not all(s and json.loads(s[0].read_text())['status']=='complete' for s in summaries):continue
            rows.append(dict(method=method,episode=i,**compare(*paths)))
    methods=[]
    for arm in ('ur5e','panda'):
        for i in range(20):
            paths=[root/'p1'/method/arm/'telemetry'/f'episode-{i:03d}.h5' for method in ('no_reps','joint_reps')]
            if all(p.is_file() for p in paths) and all(list((p.parents[1]/'official').rglob('summary.json')) for p in paths):
                methods.append(dict(robot=arm,episode=i,**compare(*paths)))
    result=dict(cross_robot=rows,within_robot_between_methods=methods,
                strict_pairing_claimed=False,interpretation='Measured restored-scene agreement is reported numerically. Robot states remain official and robot-specific. We retain non-strictly-paired statistical reporting; pose agreement does not identify the causal role of dynamics, control, robot appearance or visibility.')
    Path(out).write_text(json.dumps(result,indent=2));print('Audited',len(rows),'cross-robot and',len(methods),'within-robot pairs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.root,a.out)
