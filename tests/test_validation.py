import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
from coftvalidation.model import Arm, ROOT
from coftvalidation.scene import scenes, candidates, dense_path
from coftvalidation.geometry import TCP, MOUNTS, visibility, collision
from coftvalidation.experiment import selectors
from coftprobe.geometry import SerialArm, inverse


def independent_fk(spec,q):
    # Matrix exponential product independently implements URDF revolute transforms.
    t=np.eye(4)
    for j,v in zip(spec['joints'],q):
        rx,ry,rz=j['rpy']; x,y,z=j['axis']
        origin=np.eye(4); origin[:3,3]=j['xyz']
        def spin(a,k):
            axis=np.zeros(3); axis[k]=a
            return Rotation.from_rotvec(axis).as_matrix()
        origin[:3,:3]=spin(rz,2)@spin(ry,1)@spin(rx,0)
        twist=np.zeros((4,4)); twist[:3,:3]=[[0,-z,y],[z,0,-x],[-y,x,0]]
        t=t@origin@expm(twist*v)
    fixed=np.eye(4); fixed[:3,3]=spec['flange_fixed_xyz']
    return t@fixed


def test_official_sources_and_v0_transcription():
    xml=ET.parse(ROOT/'vendor/rokae/xMateSR5.urdf.xacro')
    nodes=[j for j in xml.iter('joint') if j.get('type')=='revolute']
    old=SerialArm(ROOT/'configs/sr5_official_kinematics.json')
    new=Arm('sr5')
    for node,j in zip(nodes,new.joints):
        for key in ('xyz','rpy'):
            assert np.allclose(j[key],[float(v) for v in node.find('origin').get(key).split()])
        assert np.allclose(j['axis'],[float(v) for v in node.find('axis').get('xyz').split()])
        assert j['lower']==float(node.find('limit').get('lower'))
        assert j['upper']==float(node.find('limit').get('upper'))
    rng=np.random.default_rng(982)
    for _ in range(10):
        q=rng.uniform(new.lo,new.hi)
        assert np.allclose(old.fk(q),new.fk(q),atol=1e-12)


def test_independent_fk_and_tight_ik_roundtrip():
    rng=np.random.default_rng(8)
    for name in ('panda','sr5'):
        arm=Arm(name)
        for _ in range(8):
            q=rng.uniform(arm.lo,arm.hi)
            assert np.allclose(arm.fk(q),independent_fk(arm.spec,q),atol=1e-11)
        result=arm.ik(arm.fk(arm.home),arm.home)
        assert result['found']
        assert result['position_error_m']<1e-8
        # TCP-to-flange inversion and optical transform composition.
        tcp=arm.fk(arm.home)@TCP
        assert np.allclose(tcp@inverse(TCP),arm.fk(arm.home))
        for mount in MOUNTS.values():
            assert np.allclose(tcp@inverse(TCP)@mount,arm.fk(arm.home)@mount)
    p=Arm('panda')
    assert np.isclose(np.linalg.norm(p.frames(p.home)[-1][:3,3]-p.frames(p.home)[-2][:3,3]),.107)


def test_shared_task_and_fixed_sampling():
    for sc in scenes(314159,3):
        pool=candidates(sc); assert len(pool)==8
        for c in pool:
            assert np.allclose(c['poses'][3][:3,3],sc['target'])
            if sc['family']=='oriented_place':
                assert np.allclose(c['poses'][-2][:3,3],sc['place'])
                # Parallel jaw +/- pi offsets cancel at release, giving the same object orientation.
                obj_r=c['poses'][-2][:3,:3]@c['poses'][3][:3,:3].T@Rotation.from_euler('z',sc['yaw']).as_matrix()
                assert np.allclose(obj_r,Rotation.from_euler('z',sc['place_yaw']).as_matrix(),atol=1e-12)
            path=dense_path(c)
            assert max(np.linalg.norm(b[0][:3,3]-a[0][:3,3]) for a,b in zip(path,path[1:]))<=.025001
    assert scenes(271828)==scenes(271828)
    assert not {s['id'] for s in scenes(314159)} & {s['id'] for s in scenes(271828)}


def test_hand_visibility_and_nonlocal_collision():
    sc=scenes(314159,1)[0]; f=candidates(sc)[0]['poses'][0]@inverse(TCP)
    clear=visibility(f,MOUNTS['mount_a'],sc,[])
    box=dict(center=(np.array(sc['target'])+[0,0,.15]).tolist(),half_extent=[.3,.3,.025])
    hidden=visibility(f,MOUNTS['mount_a'],sc,[box])
    assert 0<=clear['fraction']<=1
    assert hidden['fraction']<=clear['fraction']
    assert hidden['fraction']==0
    arm=Arm('panda'); frames=arm.frames(arm.home)
    big=dict(center=[0,0,.5],half_extent=[2,2,2])
    assert 'arm_environment_proxy' in collision(frames,frames[-1],[big],False,sc)


def test_baseline_budget_and_conjunction_tautology():
    rows=[dict(ik_found=True,feasible_proxy=True,desired_visibility={'mount_a':.2},joint_length_rad=1),
          dict(ik_found=True,feasible_proxy=True,desired_visibility={'mount_a':.8},joint_length_rad=2),
          dict(ik_found=False,feasible_proxy=False,desired_visibility={'mount_a':1.},joint_length_rad=None)]
    r=selectors(rows,'mount_a')
    assert r['ik_only']==0 and r['ik_collision']==0
    assert r['visibility_only']==2
    assert r['weighted_feasibility']==1
    assert r['analytic_and']==r['geometry_planning']==r['same_information_mpc']==1
    rows[1]['feasible_proxy']=False
    r=selectors(rows,'mount_a')
    assert r['analytic_and'] is None and r['same_information_mpc'] is None
