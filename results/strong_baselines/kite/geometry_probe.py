"""Validate native geometric observations on existing UR5e training states.

No policy runs and no action trajectory is replayed. Initialization steps are
counted separately; physics stepping is forbidden during state extraction.
The upstream mesh/surface sampler functions are executed from their pinned
source AST, avoiding import of the new Torch runtime into the old simulator.
"""
import ast
import traceback
from pathlib import Path
from baselines.strong.common import ROOT, PROJECT, TASKS, read, write, digest, now, setup

UPSTREAM = Path('/home/users/lemon/baseline-benchmark/vendor/KITE')


def geometry_functions():
    import numpy as np
    path=UPSTREAM/'experiment/encode_contacts.py'
    names=['_geom_local_mesh','_allocate_counts','_sample_mesh_surface']
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    module=ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[])
    namespace={'np':np}
    exec(compile(module,str(path),'exec'),namespace)
    return namespace,path


def main():
    setup()
    from baselines.common.language import install_core
    install_core()
    import numpy as np
    import h5py
    import cv2
    import mujoco
    from barx.simulation import initialize_observation_utils, build_environment_config
    from robocasa.utils.robomimic.robomimic_env_utils import create_env
    from mimicgen.scripts.prepare_src_dataset import _materialize_portable_paths
    from robosuite.utils.camera_utils import get_real_depth_map
    funcs,source=geometry_functions()
    raw_rows=read(ROOT/'shared_geometry/ur5e_raw_identity.json')
    counts={'bootstrap_mujoco_steps':0,'extraction_mujoco_steps':0}
    original_step=mujoco.mj_step
    def counted_step(*args,**kwargs):
        counts['bootstrap_mujoco_steps']+=1
        return original_step(*args,**kwargs)
    def forbidden_step(*args,**kwargs):
        counts['extraction_mujoco_steps']+=1
        raise RuntimeError('Physics stepping forbidden while reading saved states')
    results=[]
    for task in TASKS:
        row=next(r for r in raw_rows if r['task']==task and r['split']=='train')
        folder=ROOT/'kite/geometry_probe'/task
        if (folder/'acceptance.json').exists():
            result=read(folder/'acceptance.json');assert result['status']=='PASS';results.append(result);continue
        if folder.exists():raise FileExistsError('Preserve unfinished geometric probe: '+str(folder))
        folder.mkdir(parents=True)
        env=None
        try:
            with h5py.File(row['raw_file'],'r') as f:
                g=f['data'][row['raw_demo']]
                states=g['states'][:];meta=json.loads(g.attrs['ep_meta'])
                env_meta=json.loads(f['data'].attrs['env_args'])
                initialize_observation_utils(build_environment_config(task,'ur5e'))
                kwargs=_materialize_portable_paths(env_meta['env_kwargs'])
                kwargs.update(env_name=env_meta['env_name'],env_type=1,render=False,
                    render_offscreen=True,use_image_obs=False,postprocess_visual_obs=False)
                mujoco.mj_step=counted_step
                env=create_env(**kwargs)
                env.base_env.set_attrs_from_ep_meta(_materialize_portable_paths(meta))
                env.reset(unset_ep_meta=False)
                env.reset_to({'states':states[0]})
                mujoco.mj_step=forbidden_step
                env.step=forbidden_step
                raw=env.env;sim=raw.sim;model=sim.model
                camera=env_meta['env_kwargs']['camera_names'][0]
                camera_id=model.camera_name2id(camera)
                robot=raw.robots[0];gripper=robot.gripper['right']
                pad_ids=[model.geom_name2id(gripper.important_geoms[k][0]) for k in ['left_fingerpad','right_fingerpad']]
                assert len(set(pad_ids))==2
                local=[];normals=[];owners=[]
                for i,geom in enumerate(pad_ids):
                    mesh=funcs['_geom_local_mesh'](model,geom)
                    direction=sim.data.geom_xpos[pad_ids[1-i]]-sim.data.geom_xpos[geom]
                    direction=direction/np.linalg.norm(direction)
                    local_direction=sim.data.geom_xmat[geom].reshape(3,3).T@direction
                    alignment=mesh.face_normals@local_direction
                    assert alignment.max()>.95, 'No unambiguous inward pad face'
                    part=mesh.submesh([np.flatnonzero(alignment>.95)],append=True,repair=False)
                    p,n=funcs['_sample_mesh_surface'](part,2048,seed=7+geom)
                    local.append(p);normals.append(n);owners.extend([geom]*len(p))
                local=np.concatenate(local);normal=np.concatenate(normals);owners=np.asarray(owners)
                object_root=int(raw.obj_body_id['obj'])
                def below(body):
                    while body:
                        if body==object_root:return True
                        body=int(model.body_parentid[body])
                    return False
                object_geoms=[i for i in range(model.ngeom) if below(int(model.geom_bodyid[i]))]
                colliders=[i for i in object_geoms if model.geom_contype[i] or model.geom_conaffinity[i]]
                meshes=[(i,funcs['_geom_local_mesh'](model,i)) for i in colliders]
                meshes=[(i,m.convex_hull) for i,m in meshes if m is not None and m.area>0]
                assert meshes
                allocation=funcs['_allocate_counts'](np.array([m.area for _,m in meshes]),4096)
                obj_local=[];obj_normals=[];obj_owners=[]
                for (geom,mesh),count in zip(meshes,allocation):
                    p,n=funcs['_sample_mesh_surface'](mesh,int(count),seed=42+geom)
                    obj_local.append(p);obj_normals.append(n);obj_owners.extend([geom]*len(p))
                obj_local=np.concatenate(obj_local);obj_normal=np.concatenate(obj_normals);obj_owners=np.asarray(obj_owners)
                frames=[0,len(states)//2,len(states)-1];frame_results=[];arrays={}
                for t in frames:
                    env.reset_to({'states':states[t]})
                    assert np.array_equal(sim.get_state().flatten(),states[t]), 'Saved simulator state did not round-trip'
                    image,depth=sim.render(width=320,height=180,camera_name=camera,depth=True)
                    segmentation=sim.render(width=320,height=180,camera_name=camera,segmentation=True)
                    expected=g['obs/agentview_rgb'][t]
                    errors=[float(np.abs(image.astype(float)-expected).mean()/255),float(np.abs(image[::-1].astype(float)-expected).mean()/255)]
                    flip=int(np.argmin(errors))
                    assert errors[flip]<=.03, 'Raw camera RGB reconstruction failed predefined 3-percent MAE gate'
                    if flip:image=image[::-1];depth=depth[::-1];segmentation=segmentation[::-1]
                    depth=get_real_depth_map(sim,depth)
                    mask=(segmentation[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(segmentation[...,1],object_geoms)
                    v,u=np.nonzero(mask);z=depth[v,u]
                    assert np.isfinite(z).all() and np.all(z>0)
                    focal=90/np.tan(np.deg2rad(model.cam_fovy[camera_id])/2)
                    optical=np.stack([(u-160)*z/focal,(v-90)*z/focal,z],axis=-1)
                    # OpenCV pixel coordinates use +Y down, +Z forward.
                    rotation=sim.data.cam_xmat[camera_id].reshape(3,3)@np.diag([1.,-1.,-1.])
                    cloud=optical@rotation.T+sim.data.cam_xpos[camera_id]
                    def world(points,nrm,ids):
                        rot=sim.data.geom_xmat[ids].reshape(-1,3,3)
                        return np.einsum('nij,nj->ni',rot,points)+sim.data.geom_xpos[ids],np.einsum('nij,nj->ni',rot,nrm)
                    pad_world,pad_norm=world(local,normal,owners)
                    object_world,object_norm=world(obj_local,obj_normal,obj_owners)
                    assert np.allclose(np.linalg.norm(pad_norm,axis=-1),1,atol=1e-5)
                    assert np.allclose(np.linalg.norm(object_norm,axis=-1),1,atol=1e-5)
                    arrays.update({f'frame{t}_{k}':a for k,a in dict(pad_points=pad_world,pad_normals=pad_norm,
                        object_points=object_world,object_normals=object_norm,visible_object_cloud=cloud,qpos=sim.data.qpos.copy(),
                        camera_world_optical=rotation, camera_position=sim.data.cam_xpos[camera_id].copy()).items()})
                    cv2.imwrite(str(folder/f'frame{t}_original.png'),cv2.cvtColor(expected,cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(folder/f'frame{t}_restored.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    frame_results.append(dict(frame=t,normalized_rgb_mae=errors,selected_vertical_flip=bool(flip),
                        object_visible_pixels=len(u),pad_surface_points=len(pad_world),object_surface_points=len(object_world)))
                np.savez_compressed(folder/'geometry.npz',**arrays)
                (folder/'model.xml').write_text(sim.model.get_xml())
                assert counts['extraction_mujoco_steps']==0
                result=dict(status='PASS',task=task,episode_id=row['episode_id'],split='train',raw_file=row['raw_file'],raw_demo=row['raw_demo'],
                    camera=camera,frames=frame_results,initialization_and_extraction_counters=dict(counts),
                    policy_calls=0,new_policy_rollouts=0,geometry_only_not_complete_contact_supervision=True,
                    upstream_sampler_source_sha256=digest(source),files=[dict(path=str(p),sha256=digest(p)) for p in sorted(folder.iterdir()) if p.is_file()])
                write(folder/'acceptance.json',result);results.append(result)
        except BaseException as e:
            write(folder/'error.json',dict(error=repr(e),traceback=traceback.format_exc(),counters=counts));raise
        finally:
            mujoco.mj_step=original_step
            if env is not None:env.env.close()
    write(ROOT/'kite/geometry_probe/summary.json',dict(status='PASS',cells=results,utc=now(),new_training_updates=0,new_policy_rollouts=0))


if __name__=='__main__':
    import json
    main()
