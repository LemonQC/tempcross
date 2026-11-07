"""Pre-registered analytical pilot: counterfactual grasp orientation and wrist visibility.

This program deliberately does NOT use success labels from a neural model or physics
simulation. It can only establish (or fail to establish) geometric conflict. Results
from example calibration cannot validate real D435i placement or research novelty.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .geometry import SerialArm, inverse, line_of_sight, segment_box_intersects, target_tcp_pose

HERE=Path(__file__).resolve().parents[1]


def load_calibration(path, real=False):
    data=json.loads(Path(path).read_text())
    if real and data.get('status')!='MEASURED':
        raise ValueError('Real mode requires calibrated transforms with status=MEASURED; example is only a proxy.')
    for name in ('T_flange_tcp','T_flange_camera_optical'):
        t=np.asarray(data[name],dtype=float)
        if t.shape!=(4,4) or not np.all(np.isfinite(t)) or not np.allclose(t[3],[0,0,0,1]):
            raise ValueError('Invalid transform '+name)
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-3) or np.linalg.det(t[:3,:3])<0.99:
            raise ValueError('Camera/Tool transform not a valid rigid rotation: '+name)
    for attr in ('width','height','fx','fy','cx','cy','near_m','far_m'):
        if attr not in data['camera']: raise ValueError('Missing camera parameter '+attr)
    return data


def make_scene(idx, rng):
    """Pairs: the same target with/without a nearby box; direction is fixed per pair."""
    target=np.array([0.475+rng.uniform(-0.045,0.045), rng.uniform(-0.06,0.06),0.045])
    az=float(rng.uniform(0,2*np.pi))
    direction=np.array([np.cos(az),np.sin(az),0.])
    blocker={'center':(target+0.062*direction+np.array([0,0,0.09])).tolist(),
             'half_extent':[0.025,0.025,0.09]}
    return dict(scene_id=idx,target=target.tolist(),blocker=blocker,azimuth=az)


def evaluate_candidate(arm, cal, scene, yaw, with_blocker):
    target=np.array(scene['target'],dtype=float)
    tcp=np.asarray(cal['T_flange_tcp'])
    cam=np.asarray(cal['T_flange_camera_optical'])
    boxes=[scene['blocker']] if with_blocker else []
    # Vision required during approach, not at final contact, which may be inside D435i near limit.
    heights=[0.34,0.24,0.10]
    iks=[]; visibility=[]; prev=None
    for h in heights:
        world_tcp=target_tcp_pose(target,yaw,approach_z=h)
        world_flange=world_tcp@inverse(tcp)
        ik=arm.ik(world_flange,seed=prev)
        iks.append(ik)
        if ik['success']:
            prev=ik['q']
            world_camera=arm.fk(prev)@cam
            visibility.append(line_of_sight(world_camera,target,boxes,cal['camera']))
        else:
            # Explicitly unknown, not a zero-visibility measurement when IK fails.
            visibility.append(None)
    reached=all(x['success'] for x in iks)
    qstep=max([float(np.linalg.norm(iks[i]['q']-iks[i-1]['q'])) for i in (1,2)],default=float('inf'))
    # Coarse proxy: TCP straight approach segment should not intersect the box.
    tcp_intersection=any(segment_box_intersects(target+np.array([0,0,heights[i]]),
                                                target+np.array([0,0,heights[i+1]]),
                                                b['center'],b['half_extent'])
                         for b in boxes for i in (0,1))
    feasible=bool(reached and not tcp_intersection and qstep<2.75)
    # Evaluation uses only two approach observations to avoid confounding the known near-depth blind zone.
    observable=bool(all(v is not None and v['visible'] for v in visibility[:2]))
    vis_fraction=float(sum(v is not None and v['visible'] for v in visibility[:2]))/2
    return dict(yaw_rad=float(yaw),goal_same=True,ik_reached=bool(reached),
                max_ik_pos_error_m=max(x['pos_error'] for x in iks),
                max_ik_angle_error_rad=max(x['angle_error'] for x in iks),
                joint_path_jump_rad=qstep,tool_axis_box_intersection=bool(tcp_intersection),
                feasible=feasible,observable=observable,vis_fraction=vis_fraction,
                both=bool(feasible and observable),
                cam_depths_m=[None if v is None else v['depth'] for v in visibility],
                occluded=[None if v is None else v['occluded'] for v in visibility],
                notes='No physical execution; IK joint interpolation/tool collision self-occlusion not verified.')


def choose_baselines(candidates):
    """Baselines operate on the IDENTICAL candidate pool, costs, visibility and feasibility."""
    ids=list(range(len(candidates)))
    motion=max(ids,key=lambda i:(int(candidates[i]['feasible']),-candidates[i]['joint_path_jump_rad'],-i))
    vision=max(ids,key=lambda i:(candidates[i]['vis_fraction'],-i))
    weighted=max(ids,key=lambda i:(int(candidates[i]['feasible']),candidates[i]['vis_fraction'],
                                   -candidates[i]['joint_path_jump_rad'],-i))
    conjunction=next((i for i in ids if candidates[i]['both']),None)
    return dict(motion_only=motion,vision_only=vision,weighted=weighted,
                analytic_and=conjunction,oracle_any=conjunction)


def run(n_scenes=12,n_yaws=8,seed=20261107,calibration=None,robot=None,out='results/proxy',real=False,scene_file=None):
    root=HERE
    cal_path=Path(calibration or root/'configs/demo_proxy_calibration.json')
    rob_path=Path(robot or root/'configs/sr5_official_kinematics.json')
    cal=load_calibration(cal_path,real=real)
    arm=SerialArm(rob_path)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    if real and not scene_file:
        raise ValueError('Measured mode requires --scene-file in the robot base frame; do not validate on synthetic scenes.')
    if scene_file:
        data=json.loads(Path(scene_file).read_text())
        inputs=data['scenes']
        if not inputs: raise ValueError('scene-file has no scenes')
        for sc in inputs:
            if len(sc['target'])!=3 or any(not np.isfinite(float(x)) for x in sc['target']):
                raise ValueError('Invalid 3D target')
            b=sc['blocker']
            if len(b['center'])!=3 or len(b['half_extent'])!=3 or any(float(x)<=0 for x in b['half_extent']):
                raise ValueError('Invalid blocker AABB')
        n_scenes=len(inputs)
    else:
        inputs=[make_scene(i,rng) for i in range(n_scenes)]
    scenes=[]; all_rows=[]
    for si,sc in enumerate(inputs):
        for blocked in (False,True):
            records=[evaluate_candidate(arm,cal,sc,float(2*np.pi*j/n_yaws),blocked)
                     for j in range(n_yaws)]
            choices=choose_baselines(records)
            feasible_idxs=[i for i,r in enumerate(records) if r['feasible']]
            invisible_but_feasible=sum(r['feasible'] and not r['observable'] for r in records)
            some_both=any(r['both'] for r in records)
            good_alternative=bool(invisible_but_feasible>0 and some_both)
            row=dict(scene_id=si,occluder=blocked,n_candidates=n_yaws,
                     feasible_count=len(feasible_idxs),invisible_feasible_count=int(invisible_but_feasible),
                     visible_feasible_count=int(sum(r['both'] for r in records)),
                     conflict_with_alternative=good_alternative,
                     motion_only_both=records[choices['motion_only']]['both'],
                     vision_only_both=records[choices['vision_only']]['both'],
                     weighted_both=records[choices['weighted']]['both'],
                     analytic_and_both=bool(some_both))
            all_rows.append(row)
            scenes.append(dict(scene=sc,blocked=blocked,candidates=records,baselines=choices,summary=row))
        print(f'scene {si+1:02d}/{n_scenes}: open and occluder evaluated',flush=True)
    blocked=[r for r in all_rows if r['occluder']]
    open_rows=[r for r in all_rows if not r['occluder']]
    conflict_rate=sum(r['conflict_with_alternative'] for r in blocked)/len(blocked)
    simple_conjunction_solve=sum(r['analytic_and_both'] for r in blocked)/len(blocked)
    summary=dict(mode='measured' if real else 'PROXY_NOT_CALIBRATED',
                 hypothesis='Feasible target-arm actions may create different wrist visibility under the same task effect',
                 seed=seed,n_scene_pairs=n_scenes,n_candidates_per_scene=n_yaws,
                 scene_source=str(scene_file) if scene_file else 'SYNTHETIC',
                 blocked_conflict_with_alternative_rate=conflict_rate,
                 blocked_analytic_and_success_rate=simple_conjunction_solve,
                 blocked_motion_only_success_rate=sum(r['motion_only_both'] for r in blocked)/len(blocked),
                 blocked_vision_only_success_rate=sum(r['vision_only_both'] for r in blocked)/len(blocked),
                 blocked_weighted_success_rate=sum(r['weighted_both'] for r in blocked)/len(blocked),
                 open_analytic_and_success_rate=sum(r['analytic_and_both'] for r in open_rows)/len(open_rows),
                 geometry_evidence='COUNTERFACTUAL_DIFFERENCE_OBSERVED' if conflict_rate>0 else 'NO_CONFLICT_IN_TESTED_CASES',
                 algorithm_novelty='NOT_ESTABLISHED',
                 scientific_gate='ONLY_GEOMETRY_PROBE_NOT_A_TRO_GO',
                 limitations=[
                     'This is NOT real policy success or robot execution; no motion physics, contact dynamics, friction or force.',
                     'Only TCP-axis obstacle proxy; no finger/robot meshes or robot self-collision; IK solves each waypoint only.',
                     'Visibility checks one object center with 2 approach frames, axis-aligned boxes, no sensor noise/self occlusion.',
                     'Sample mount/camera parameters are NOT measured; any result in proxy mode is software-only.',
                     'Analytic conjunction is an equal-information strong baseline: if it solves the proxy, no learned algorithm novelty is established.',
                     'No source embodiment is modeled in v0; NOT a validated cross-embodiment transfer experiment.',
                     'Candidate gripper yaw invariance assumes a rotationally symmetric cylinder and equal task outcome.'
                 ])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (output/'trials.json').write_text(json.dumps(scenes,indent=2,ensure_ascii=False))
    with (output/'pairs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader();writer.writerows(all_rows)
    message=(f"# COFT analytical pilot ({summary['mode']})\n\n"
             f"Same-goal counterfactual, scene pairs: {n_scenes}; yaw candidates: {n_yaws}; seed: {seed}.\n\n"
             f"- Geometric conflict **with** feasible observable alternative (blocked scenes): **{conflict_rate:.1%}**\n"
             f"- Motion-only chosen candidate both-feasible-and-visible: **{summary['blocked_motion_only_success_rate']:.1%}**\n"
             f"- Visibility-only chosen candidate both: **{summary['blocked_vision_only_success_rate']:.1%}**\n"
             f"- Weighted/simple selector both: **{summary['blocked_weighted_success_rate']:.1%}**\n"
             f"- Analytic conjunction (same candidate pool) solves: **{simple_conjunction_solve:.1%}**\n\n"
             f"**Interpretation:** {summary['geometry_evidence']}. Algorithm novelty: NOT ESTABLISHED. "
             "A geometry conflict alone cannot justify T-RO or a learned COFT architecture. "
             "Review the analytic baseline before planning learning experiments.\n\n"
             "**Limitations:**\n"+''.join(f'- {s}\n' for s in summary['limitations']))
    (output/'REPORT.md').write_text(message)
    print(message)
    print('Results:',output.resolve())
    return summary


def main():
    p=argparse.ArgumentParser(description='SR5 analytical counterfactual probe; NEVER controls a real robot.')
    p.add_argument('--mode',choices=['demo','measured'],default='demo')
    p.add_argument('--robot')
    p.add_argument('--calibration')
    p.add_argument('--scenes',type=int,default=12)
    p.add_argument('--yaws',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--out',default='results/proxy')
    p.add_argument('--scene-file',help='JSON containing measured obstacle/target locations in SR5 base coordinates')
    args=p.parse_args()
    if args.scenes<=0 or args.yaws<2: p.error('--scenes > 0 and --yaws >= 2 required')
    if args.mode=='measured' and not args.calibration:p.error('measured mode requires --calibration')
    run(args.scenes,args.yaws,args.seed,args.calibration,args.robot,args.out,args.mode=='measured',args.scene_file)


if __name__=='__main__':main()
