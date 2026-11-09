"""Explicit gripper-site transforms and ordinary bounded OSC commands."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(position,quaternion,order='wxyz'):
    q=np.asarray(quaternion,dtype=float)
    if order=='wxyz':q=q[[1,2,3,0]]
    elif order!='xyzw':raise ValueError(order)
    result=np.eye(4);result[:3,:3]=Rotation.from_quat(q).as_matrix();result[:3,3]=position
    return result


def inverse(pose):
    result=np.eye(4);result[:3,:3]=pose[:3,:3].T;result[:3,3]=-result[:3,:3]@pose[:3,3]
    return result


def error(current,goal):
    return float(np.linalg.norm(current[:3,3]-goal[:3,3])),float(Rotation.from_matrix(goal[:3,:3]@current[:3,:3].T).magnitude())


def interpolate(first,last,fraction):
    fraction=float(np.clip(fraction,0.,1.));result=np.eye(4)
    result[:3,3]=(1-fraction)*first[:3,3]+fraction*last[:3,3]
    result[:3,:3]=Slerp([0.,1.],Rotation.from_matrix(np.stack([first[:3,:3],last[:3,:3]])))(fraction).as_matrix()
    return result


def site_definition(xml,name='gripper0_right_grip_site'):
    for marker in ['ROBOCASA','ROBOSUITE']:xml=xml.replace('<'+marker+'>','__'+marker+'__')
    root=ET.fromstring(xml);found=[]
    for body in root.iter('body'):
        for site in body.findall('site'):
            if site.get('name')==name:
                found.append((body.attrib['name'],transform(np.fromstring(site.get('pos','0 0 0'),sep=' '),
                    np.fromstring(site.get('quat','1 0 0 0'),sep=' '))))
    if len(found)!=1:raise ValueError('Ambiguous TCP site')
    return found[0]


def bounded_delta(current,goal,origin_rotation,position_gain=.4,rotation_gain=.4,
                  position_step=.012,rotation_step=.12):
    """Return metric deltas in the unchanged OSC controller reference frame."""
    dp=position_gain*(goal[:3,3]-current[:3,3])
    dr=rotation_gain*Rotation.from_matrix(goal[:3,:3]@current[:3,:3].T).as_rotvec()
    for v,limit in [(dp,position_step),(dr,rotation_step)]:
        n=np.linalg.norm(v)
        if n>limit:v*=limit/n
    return np.r_[origin_rotation.T@dp,origin_rotation.T@dr]
