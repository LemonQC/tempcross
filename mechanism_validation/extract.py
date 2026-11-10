"""Extract saved source successes and compare actual scenes. Zero physics steps."""
import json
import sys
import numpy as np
import h5py
from p3_validation.geometry import transform, inverse, site_definition, error
from p3_validation.scene_audit import canonical_metadata
from mechanism_validation.common import *
from mechanism_validation.contact import bilateral_pairs


def poses(f, xml):
    names = json.loads(f.attrs['body_names'])
    body, local = site_definition(xml)
    bid = names.index(body)
    tcp = np.stack([transform(p,q) @ local for p,q in zip(f['sim/body_xpos'][:,bid], f['sim/body_xquat'][:,bid])])
    obj = np.stack([transform(p,q) for p,q in zip(f['target/position'],f['target/quaternion'])])
    meta = json.loads(f.attrs['ep_meta'])
    sid = names.index(meta['fixture_refs']['sink'] + '_main')
    sink = transform(f['sim/body_xpos'][0,sid], f['sim/body_xquat'][0,sid])
    return tcp, obj, sink, meta


def first_run(mask, length):
    for i in range(len(mask)-length+1):
        if np.all(mask[i:i+length]):
            return i
    return None


def main():
    if (ROOT/'physical_launches.jsonl').exists():
        raise RuntimeError('Extraction/protocol cannot be regenerated after physics starts')
    sys.path[:0] = [str(SOURCE),str(SOURCE/'policy')]
    from barx.evaluation_conditions import load_bundle, validate_condition_bundle_for_evaluation
    from barx.benchmark import EMBODIMENTS
    bundles = {}
    for robot in ['ur5e','panda']:
        payload,states,models = load_bundle(SOURCE/'evaluation/conditions',task=TASK,embodiment=robot)
        validate_condition_bundle_for_evaluation(payload,states,models,task=TASK,camera_name=EMBODIMENTS[robot].camera,image_width=320,image_height=180,count=100)
        bundles[robot] = (payload,states,models)
    analysis = json.loads((PROJECT/'results/baselines/analysis.json').read_text())
    source = [r for r in analysis['episodes'] if r['label']=='A_barx_joint' and r['task']==TASK and r['robot']=='ur5e']
    target = [r for r in analysis['episodes'] if r['task']==TASK and r['robot']=='panda']
    labels = sorted({r['label'] for r in target}); assert len(labels)==12
    hard = sorted(r['official_condition_index'] for r in source if r['success'] and all(not t['success'] for t in target if t['official_condition_index']==r['official_condition_index']))
    assert hard == [64,68,71,72,76,88,93,94,97,99]
    assert hard[:8]==CASES
    bank=[]; scenes=[]
    for i in CASES:
        source_record=next(r for r in source if r['official_condition_index']==i)
        target_record=next(r for r in target if r['label']=='A_barx_joint' and r['official_condition_index']==i)
        sp=Path(source_record['telemetry_path']);tp=Path(target_record['telemetry_path'])
        with h5py.File(sp) as f, h5py.File(tp) as g:
            tcp,obj,sink,meta=poses(f,bundles['ur5e'][2][i]);ptcp,pobj,psink,pmeta=poses(g,bundles['panda'][2][i])
            actions=f['actions/environment'][:]
            records=[json.loads(x) for x in sp.with_suffix('.jsonl').read_text().splitlines()]
            geom_names=json.loads(f.attrs['geom_names']); body_names=json.loads(f.attrs['body_names'])
            target_geoms=[n for n,b in zip(geom_names,f['model/geom_bodyid'][:]) if body_names[b].startswith('obj_')]
            pad_contact=np.array([x['grasp_contact'] is True for x in records])
            contact=np.array([bilateral_pairs([(c['geom1'],c['geom2']) for c in x['contacts'] if c['distance']<=0],target_geoms) for x in records])
            rise=obj[:,2,3]-obj[0,2,3]
            lift=first_run(contact & (rise>=.03),10); assert lift is not None
            valid=[t for t in range(lift+1) if contact[t:t+10].all()]
            assert valid
            grasp=valid[0]
            closure=[int(k) for k in np.flatnonzero(actions[:,6]>0) if k<=grasp]
            close=closure[0] if closure else int(np.flatnonzero(actions[:,6]>0)[0])
            opened=np.flatnonzero((actions[:,6]<0)&(np.arange(len(actions))>=lift+9))
            release=int(opened[0]) if len(opened) else min(int(np.flatnonzero(contact)[-1])+1,len(actions))
            relative=inverse(obj[grasp])@tcp[grasp]
            observed=np.stack([transform(p,q,'xyzw') for p,q in zip(f['obs/robot0_eef_pos'],f['obs/robot0_eef_quat_site'])])
            pe=float(np.linalg.norm(tcp[:,:3,3]-observed[:,:3,3],axis=1).max())
            re=max(error(a,b)[1] for a,b in zip(tcp,observed));assert pe<.005 and re<np.deg2rad(2)
            source_asset=next(o for o in meta['object_cfgs'] if o['name']=='obj')['info']['mjcf_path']
            target_asset=next(o for o in pmeta['object_cfgs'] if o['name']=='obj')['info']['mjcf_path']
            assert source_asset==target_asset
            scene=dict(official_index=i,task=TASK,source_robot='ur5e',target_robot='panda',source_condition=bundles['ur5e'][0]['episodes'][i],target_condition=bundles['panda'][0]['episodes'][i],
                object_asset=source_asset,object_asset_identical=True,canonical_metadata_equal=canonical_metadata(meta)==canonical_metadata(pmeta),
                initial_object_position_difference_m=float(np.linalg.norm(obj[0,:3,3]-pobj[0,:3,3])),initial_object_rotation_difference_rad=error(obj[0],pobj[0])[1],
                sink_position_difference_m=float(np.linalg.norm(sink[:3,3]-psink[:3,3])),sink_rotation_difference_rad=error(sink,psink)[1],
                T_source_world_object_initial=obj[0].tolist(),T_target_world_object_initial=pobj[0].tolist(),T_source_world_sink=sink.tolist(),T_target_world_sink=psink.tolist())
            scenes.append(scene)
            # Preserve every recorded state of all four phases, not just endpoints.
            trace=dict(index_convention='state s precedes action s',state_indices=list(range(len(obj))),tcp_world=tcp.tolist(),object_world=obj.tolist(),
                object_in_sink=[(inverse(sink)@o).tolist() for o in obj],tcp_in_object=[(inverse(o)@t).tolist() for o,t in zip(obj,tcp)],
                bilateral_contact=contact.tolist(),legacy_pad_only_contact=pad_contact.tolist(),environment_actions=actions.tolist(),gripper_qpos=f['obs/robot0_gripper_qpos'][:].tolist())
            trace_path=ROOT/f'source_traces/condition-{i:03d}.json';write_json(trace_path,trace)
            # Two transport references plus release, directly from the real success.
            # No search or target-robot outcomes are used to select these states.
            middle=(lift+9+release)//2
            bank.append(dict(official_index=i,source_condition_id=source_record['condition_id'],target_condition_id=target_record['condition_id'],
                source_telemetry=str(sp),source_telemetry_sha256=digest(sp),source_contacts_sha256=digest(sp.with_suffix('.jsonl')),
                source_video=source_record['video_path'],target_legacy_telemetry=str(tp),target_legacy_telemetry_sha256=digest(tp),
                source_trace=str(trace_path),source_trace_bytes=trace_path.stat().st_size,source_trace_sha256=digest(trace_path),source_official_success=True,
                first_close_action_index=close,first_bilateral_contact_state=int(np.flatnonzero(contact)[0]),stable_contact_start_state=grasp,
                legacy_pad_only_contact_states=int(pad_contact.sum()),physical_two_sided_contact_states=int(contact.sum()),
                stable_lift_start_state=lift,stable_lift_confirmed_state=lift+9,release_action_index=release,retreat_final_state=len(obj)-1,
                release_event_kind='FIRST_OPEN_AFTER_LIFT' if len(opened) else 'FINAL_TWO_SIDED_CONTACT_LOSS_NO_OPEN_COMMAND',
                T_object_tcp=relative.tolist(),T_source_sink=sink.tolist(),T_target_sink=psink.tolist(),
                postgrasp_source_state_indices=[lift+9,middle,release],
                postgrasp_object_in_sink=[(inverse(sink)@obj[k]).tolist() for k in [lift+9,middle,release]],
                grasp_complete=dict(tcp_world=tcp[lift+9].tolist(),object_world=obj[lift+9].tolist(),gripper_qpos=f['obs/robot0_gripper_qpos'][lift+9].tolist()),
                synchronous_tcp_audit=dict(max_observation_position_error_m=pe,max_observation_rotation_error_rad=re)))
    write_json(OUT/'source_contact_bank.json',bank)
    write_json(OUT/'scene_audit.json',dict(new_physics_steps=0,hard_cases_verified=hard,target_settings=labels,records=scenes))
    write_json(OUT/'bundle_integrity.json',[dict(robot=r,files=[dict(path=str(SOURCE/f'evaluation/conditions/{TASK}/{r}{s}'),sha256=digest(SOURCE/f'evaluation/conditions/{TASK}/{r}{s}')) for s in ['.json','.npz']]) for r in bundles])
    print(json.dumps([dict(index=r['official_index'],grasp=r['stable_contact_start_state'],lift=r['stable_lift_start_state'],release=r['release_action_index']) for r in bank]))


if __name__=='__main__':main()
