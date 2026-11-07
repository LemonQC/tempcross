"""FK/IK and visibility primitives; all distances in metres, angles in radians.

The supplied SR5 chain was transcribed from the official Rokae xMateSR5 file.
This proxy intentionally excludes robot/hand mesh collision and sensor noise.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(translation=None, rotation=None):
    out = np.eye(4)
    if translation is not None: out[:3, 3] = translation
    if rotation is not None: out[:3, :3] = rotation
    return out


def inverse(t):
    out=np.eye(4); out[:3,:3]=t[:3,:3].T; out[:3,3]=-out[:3,:3]@t[:3,3]
    return out


def apply(t, p):
    return t[:3,:3]@np.asarray(p,dtype=float)+t[:3,3]


class SerialArm:
    def __init__(self, json_file):
        self.spec=json.loads(Path(json_file).read_text())
        self.joints=self.spec['joints']
        self.lower=np.array([j['lower'] for j in self.joints],dtype=float)
        self.upper=np.array([j['upper'] for j in self.joints],dtype=float)
        self.home=np.array([0,-0.7,1.1,0,0.7,0],dtype=float)
        self.home=np.clip(self.home,self.lower,self.upper)

    def fk(self,q,link_points=False):
        q=np.asarray(q,dtype=float)
        if len(q)!=len(self.joints):raise ValueError('joint vector length mismatch')
        t=np.eye(4)
        pts=[t[:3,3].copy()]
        for angle,j in zip(q,self.joints):
            t=t@transform(j['xyz'], Rotation.from_euler('xyz',j['rpy']).as_matrix())
            t=t@transform(rotation=Rotation.from_rotvec(np.asarray(j['axis'])*angle).as_matrix())
            pts.append(t[:3,3].copy())
        return (t,np.asarray(pts)) if link_points else t

    def ik(self,target,seed=None,position_tolerance=0.012,angular_tolerance=0.18,max_nfev=85):
        """Multi-seed bounded 6D IK; no continuity or collision guarantee."""
        seeds=[self.home]
        if seed is not None:seeds.insert(0,np.clip(seed,self.lower,self.upper))
        seeds.extend([np.clip(np.array([0,-0.9,1.8,0,-0.2,0]),self.lower,self.upper),
                      np.clip(np.array([0,0.5,-0.6,0,0.6,0]),self.lower,self.upper)])
        best=None
        for s in seeds:
            def residual(q):
                fk=self.fk(q)
                e_p=(fk[:3,3]-target[:3,3])
                e_r=Rotation.from_matrix(target[:3,:3]@fk[:3,:3].T).as_rotvec()
                return np.r_[e_p,0.22*e_r]
            result=least_squares(residual,s,bounds=(self.lower+1e-7,self.upper-1e-7),
                                 max_nfev=max_nfev,ftol=3e-5,xtol=3e-5,gtol=3e-5)
            f=self.fk(result.x)
            pos=float(np.linalg.norm(f[:3,3]-target[:3,3]))
            ang=float(np.linalg.norm(Rotation.from_matrix(target[:3,:3]@f[:3,:3].T).as_rotvec()))
            quality=pos+0.22*ang
            if best is None or quality<best['quality']:
                best=dict(q=result.x,pos_error=pos,angle_error=ang,quality=quality)
            if pos<=position_tolerance and ang<=angular_tolerance:break
        best['success']=bool(best['pos_error']<=position_tolerance and best['angle_error']<=angular_tolerance)
        return best


def camera_projection(t_world_camera,p_world,camera):
    p=apply(inverse(t_world_camera),p_world)
    z=float(p[2])
    if z<=0:return dict(in_frustum=False,depth=z,uv=None)
    u=float(camera['fx']*p[0]/z+camera['cx']);v=float(camera['fy']*p[1]/z+camera['cy'])
    ok=(camera['near_m']<=z<=camera['far_m'] and 0<=u<camera['width'] and 0<=v<camera['height'])
    return dict(in_frustum=bool(ok),depth=z,uv=[u,v])


def segment_box_intersects(a,b,center,half_extent,eps=1e-9):
    """Robust 3D slab test for *finite* segment vs axis-aligned occluder AABB."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    lo=np.asarray(center)-np.asarray(half_extent); hi=np.asarray(center)+np.asarray(half_extent)
    d=b-a;t0=0.;t1=1.
    for axis in range(3):
        if abs(d[axis])<eps:
            if a[axis]<lo[axis] or a[axis]>hi[axis]:return False
        else:
            l=(lo[axis]-a[axis])/d[axis];u=(hi[axis]-a[axis])/d[axis]
            t0=max(t0,min(l,u));t1=min(t1,max(l,u))
            if t0>t1:return False
    return True


def line_of_sight(t_world_camera,point,blocks,camera):
    proj=camera_projection(t_world_camera,point,camera)
    origin=t_world_camera[:3,3]
    occluded=any(segment_box_intersects(origin,point,b['center'],b['half_extent']) for b in blocks)
    return dict(visible=bool(proj['in_frustum'] and not occluded),
                occluded=bool(occluded), **proj)


def target_tcp_pose(point,yaw,approach_z=0.25):
    """Top-down parallel grasp; only valid for symmetric cylinder proxy scenes."""
    rot=Rotation.from_euler('z',yaw).as_matrix()@Rotation.from_euler('x',math.pi).as_matrix()
    return transform(np.asarray(point)+np.array([0,0,approach_z]),rot)
