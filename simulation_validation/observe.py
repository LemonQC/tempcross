"""Read-only simulator telemetry. Does not change policy, action, observation or success.

Installed only in the official evaluation process. Missing fields are recorded as
unavailable, never inferred from an unsuccessful episode.
"""
import atexit
import json
from pathlib import Path
import numpy as np

class Recorder:
    def __init__(self,directory):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.episode=-1;self.file=None;self.stream=None;self.step=0
        atexit.register(self.close)

    def close(self):
        if self.file is not None:self.file.close();self.file=None
        if self.stream is not None:self.stream.close();self.stream=None

    def start(self,entry,env,obs,metadata):
        import h5py
        self.close();self.episode+=1;self.step=0
        self.file=h5py.File(self.root/f'episode-{self.episode:03d}.h5','w')
        self.stream=(self.root/f'episode-{self.episode:03d}.jsonl').open('w')
        self.file.attrs['condition']=json.dumps(entry,default=str)
        self.file.attrs['ep_meta']=json.dumps(metadata,default=str)
        self.snapshot(env,obs,None,{})

    def append(self,key,value):
        if value is None:return
        a=np.asarray(value)
        if a.dtype.kind not in 'biuf' or a.size==0:return
        if key not in self.file:
            self.file.create_dataset(key,shape=(0,)+a.shape,maxshape=(None,)+a.shape,dtype=a.dtype,compression='gzip',chunks=True)
        d=self.file[key]
        if d.shape[1:]!=a.shape:return
        d.resize(d.shape[0]+1,axis=0);d[-1]=a

    def snapshot(self,env,obs,action,info):
        if self.file is None:return
        raw=env.env;sim=raw.sim;errors=[]
        for key,value in obs.items():
            if 'image' not in key:self.append('obs/'+key,value)
        # Policy images are already saved unmodified by the official evaluator.
        self.append('sim/qpos',sim.data.qpos.copy());self.append('sim/qvel',sim.data.qvel.copy())
        self.append('sim/body_xpos',sim.data.body_xpos.copy())
        self.append('sim/body_xquat',sim.data.body_xquat.copy())
        if action is not None:self.append('actions/environment',action)
        contacts=[]
        for i in range(sim.data.ncon):
            c=sim.data.contact[i]
            contacts.append(dict(geom1=sim.model.geom_id2name(c.geom1),geom2=sim.model.geom_id2name(c.geom2),distance=float(c.dist)))
        names=[sim.model.body_id2name(i) for i in range(sim.model.nbody)]
        if self.step==0:self.file.attrs['body_names']=json.dumps(names)
        grasp=None
        try:
            obj=raw.objects.get('obj')
            if obj is not None:
                self.append('target/position',sim.data.body_xpos[raw.obj_body_id['obj']].copy())
                self.append('target/quaternion',sim.data.body_xquat[raw.obj_body_id['obj']].copy())
                gripper=raw.robots[0].gripper
                if isinstance(gripper,dict):gripper=gripper['right']
                grasp=bool(raw._check_grasp(gripper=gripper,object_geoms=obj.contact_geoms))
        except (KeyError,AttributeError,TypeError) as e:errors.append(type(e).__name__+': '+str(e))
        static_projection=None
        try:
            from coftprobe.geometry import camera_projection
            camera=next(name for name in raw.camera_names if name.startswith('barx_'))
            camera_id=sim.model.camera_name2id(camera)
            world_optical=np.eye(4)
            # MuJoCo looks along -Z with +Y up; COFT optical convention is +Z/+Y down.
            world_optical[:3,:3]=sim.data.cam_xmat[camera_id].reshape(3,3)@np.diag([1.,-1.,-1.])
            world_optical[:3,3]=sim.data.cam_xpos[camera_id]
            fy=90/np.tan(np.deg2rad(sim.model.cam_fovy[camera_id])/2)
            intrinsics=dict(width=320,height=180,fx=fy,fy=fy,cx=160,cy=90,
                            near_m=sim.model.vis.map.znear*sim.model.stat.extent,
                            far_m=sim.model.vis.map.zfar*sim.model.stat.extent)
            target=sim.data.body_xpos[raw.obj_body_id['obj']]
            static_projection=camera_projection(world_optical,target,intrinsics)
            static_projection['role']='POSTHOC_CENTER_FRUSTUM_ONLY_NOT_SUCCESS_OR_OCCLUSION'
            self.append('camera/T_world_optical',world_optical)
        except (KeyError,AttributeError,TypeError,StopIteration) as e:
            errors.append('static_projection_unavailable: '+str(e))
        self.stream.write(json.dumps(dict(step=self.step,success=info.get('is_success'),grasp_contact=grasp,contacts=contacts,
                                         coft_static_projection=static_projection,telemetry_errors=errors),default=str)+'\n')
        self.stream.flush();self.file.flush();self.step+=1

    def install(self):
        import barx.evaluation_conditions as conditions
        from robocasa.utils.robomimic import robomimic_env_utils as utils
        create=utils.create_env;restore=conditions.restore_frozen_condition
        def wrapped_create(*args,**kwargs):
            env=create(*args,**kwargs);step=env.step
            def wrapped_step(action,*a,**kw):
                result=step(action,*a,**kw)
                self.snapshot(env,result[0],np.asarray(action).copy(),result[3])
                return result
            env.step=wrapped_step
            return env
        def wrapped_restore(env,entry,*args,**kwargs):
            obs,meta=restore(env,entry,*args,**kwargs)
            self.start(entry,env,obs,meta)
            return obs,meta
        import experiments.robot.robot_utils as robot_utils
        predict=robot_utils.get_action
        def wrapped_predict(*args,**kwargs):
            result=predict(*args,**kwargs)
            if self.file is not None:
                self.append('actions/policy_chunks',np.asarray(result['action']).copy())
                self.append('actions/policy_chunk_step',self.step)
            return result
        robot_utils.get_action=wrapped_predict
        utils.create_env=wrapped_create;conditions.restore_frozen_condition=wrapped_restore
