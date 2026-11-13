"""Saved-state/RGB/instance-mask correspondence gate; no policy or learned mask.

Four training demonstrations, fixed frames0/middle/last. Saved simulator state
must round-trip exactly and rendered RGB must match within3% normalized MAE.
Initialization physics is counted; stepping after restoration is prohibited.
"""
import json
import traceback
from pathlib import Path
from baselines.strong.common import ROOT,SOURCE,PROJECT,TASKS,read,write,digest,setup,now


def materialize(value):
    import robocasa,robosuite
    roots={'ROBOCASA':Path(robocasa.__file__).resolve().parent,'ROBOSUITE':Path(robosuite.__file__).resolve().parent}
    if isinstance(value,dict):return {k:materialize(v) for k,v in value.items()}
    if isinstance(value,list):return [materialize(v) for v in value]
    if isinstance(value,str):
        for name,root in roots.items():
            prefix='<'+name+'>/'
            if value.startswith(prefix):return str(root/value[len(prefix):])
    return value


def main():
    setup()
    from baselines.common.language import install_core
    install_core()
    import cv2
    import h5py
    import numpy as np
    import mujoco
    from barx.simulation import initialize_observation_utils,build_environment_config
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    sources=read(ROOT/'shared_geometry/ur5e_raw_identity.json')
    targets=read(PROJECT/'configs/datasets/baseline_target_manifest.json')
    rows=[]
    for task in TASKS:
        r=next(r for r in sources if r['task']==task and r['split']=='train')
        rows.append(dict(domain='source',robot='ur5e',task=task,episode_id=r['episode_id'],raw_file=r['raw_file'],raw_demo=r['raw_demo']))
        r=next(r for r in targets if r['task']==task and r['split']=='train')
        rows.append(dict(domain='target',robot='panda',task=task,episode_id=r['episode_id'],raw_file=r['original_file'],raw_demo=r['original_demo']))
    completed=[]
    original=mujoco.mj_step
    for row in rows:
        folder=ROOT/'object_flow/repaired_01/label_probe'/(row['domain']+'-'+row['task'])
        if (folder/'acceptance.json').exists():
            old=read(folder/'acceptance.json');assert old['status']=='PASS';completed.append(old);continue
        if folder.exists():raise FileExistsError('Preserve unfinished probe: '+str(folder))
        folder.mkdir(parents=True);env=None;counters=dict(initialization_physics_steps=0,extraction_physics_steps=0,policy_calls=0)
        def initialization(*args,**kwargs):
            counters['initialization_physics_steps']+=1;return original(*args,**kwargs)
        def forbidden(*args,**kwargs):
            counters['extraction_physics_steps']+=1;raise RuntimeError('No physics stepping during saved-state label extraction')
        try:
            with h5py.File(row['raw_file'],'r') as f:
                g=f['data'][row['raw_demo']];meta=json.loads(g.attrs['ep_meta']);env_meta=json.loads(f['data'].attrs['env_args'])
                state=g['states'][:];initialize_observation_utils(build_environment_config(row['task'],row['robot']))
                kwargs=materialize(env_meta['env_kwargs'])
                kwargs.update(env_name=env_meta['env_name'],env_type=1,render=False,render_offscreen=True,use_image_obs=False)
                mujoco.mj_step=initialization;env=create_env(**kwargs)
                if hasattr(env.base_env,'set_attrs_from_ep_meta'):env.base_env.set_attrs_from_ep_meta(materialize(meta))
                else:env.base_env.set_ep_meta(materialize(meta))
                env.reset(unset_ep_meta=False);env.reset_to({'states':state[0]})
                raw=env.env;sim=raw.sim;model=sim.model
                camera=env_meta['env_kwargs']['camera_names'][0]
                cameras=[model.camera_id2name(i) for i in range(model.ncam)]
                selected_roles=[role for role in ['obj','container'] if role in raw.objects]
                assert 'obj' in selected_roles
                roots=[int(raw.obj_body_id[role]) for role in selected_roles]
                def relevant(body):
                    while body:
                        if body in roots:return True
                        body=int(model.body_parentid[body])
                    return False
                ids=[i for i in range(model.ngeom) if relevant(int(model.geom_bodyid[i]))]
                assert ids
                mujoco.mj_step=forbidden;env.step=forbidden;measurements=[]
                for t in [0,len(state)//2,len(state)-1]:
                    env.reset_to({'states':state[t]})
                    actual=sim.get_state().flatten();assert np.array_equal(actual,state[t]), 'Saved-state mismatch'
                    image=sim.render(width=320,height=180,camera_name=camera)
                    segments=sim.render(width=320,height=180,camera_name=camera,segmentation=True)
                    expected=g['obs/agentview_rgb'][t]
                    errors=[float(np.abs(image.astype(float)-expected).mean()/255),float(np.abs(image[::-1].astype(float)-expected).mean()/255)]
                    flip=int(np.argmin(errors));rgb_error=errors[flip]
                    # Gate is declared before any rendered comparison or model fitting.
                    assert rgb_error<=.03, f'RGB/state alignment gate failed:{rgb_error}'
                    if flip:image=image[::-1];segments=segments[::-1]
                    mask=(segments[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(segments[...,1],ids)
                    cv2.imwrite(str(folder/f'frame-{t:04d}-original.png'),cv2.cvtColor(expected,cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(folder/f'frame-{t:04d}-restored.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(folder/f'frame-{t:04d}-visible-objects.png'),mask.astype(np.uint8)*255)
                    measurements.append(dict(frame=t,rgb_normalized_mae=errors,vertical_flip=bool(flip),foreground_pixels=int(mask.sum())))
                assert counters['extraction_physics_steps']==0
                result=dict(status='PASS',**row,source_split='train',camera=camera,available_cameras=cameras,
                    labelled_object_roles=selected_roles,label_semantics='visible simulator-instance mask; privileged training annotation',
                    static_fixtures_not_labelled=True,frames=measurements,counters=counters,
                    new_policy_rollouts=0,new_training_updates=0,
                    files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(folder.iterdir()) if p.is_file()])
                write(folder/'acceptance.json',result);completed.append(result)
        except BaseException as e:
            write(folder/'error.json',dict(error=repr(e),traceback=traceback.format_exc(),counters=counters,row=row));raise
        finally:
            mujoco.mj_step=original
            if env is not None:env.env.close()
    write(ROOT/'object_flow/repaired_01/label_probe/summary.json',dict(status='PASS',utc=now(),cells=completed,
        method_scope='METHOD_PORT_SAME_BEHAVIOR_FEASIBILITY_ONLY',object_flow_predictor_trained=False,action_policy_trained=False))


if __name__=='__main__':main()
