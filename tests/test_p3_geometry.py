import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from p3_validation.geometry import transform,inverse,error,interpolate,bounded_delta
from p3_validation.plan import WaypointPlan


def test_frame_conventions_and_object_relative_pose_roundtrip():
    object_pose=transform([1,2,3],[.5,.5,.5,.5])
    tcp=transform([1.1,2.2,3.3],Rotation.from_euler('xyz',[.3,.4,-.5]).as_quat(),'xyzw')
    relative=inverse(object_pose)@tcp
    assert np.allclose(object_pose@relative,tcp)
    assert np.allclose(inverse(tcp)@tcp,np.eye(4))


def test_shortest_rotation_interpolation_and_translation():
    a=transform([0,0,0],Rotation.from_euler('z',170,degrees=True).as_quat(),'xyzw')
    b=transform([1,2,3],Rotation.from_euler('z',-170,degrees=True).as_quat(),'xyzw')
    middle=interpolate(a,b,.5)
    assert np.allclose(middle[:3,3],[.5,1,1.5])
    assert abs(error(a,middle)[1]-np.deg2rad(10))<1e-10


def test_osc_reference_frame_and_vector_norm_caps():
    current=np.eye(4);goal=transform([1,1,0],Rotation.from_euler('xyz',[1,1,1]).as_quat(),'xyzw')
    origin=Rotation.from_euler('z',90,degrees=True).as_matrix()
    action=bounded_delta(current,goal,origin)
    assert np.isclose(np.linalg.norm(action[:3]),.012)
    assert np.isclose(np.linalg.norm(action[3:]),.12)
    assert action[0]>0 and action[1]<0
    assert np.allclose(bounded_delta(current,current,origin),0)


def test_frozen_budget_and_test_window():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/p3_protocol.json').read_text())
    assert c['development_indices']==[22,24,26,29,34,35,37,39]
    assert c['independent_indices']==list(range(40,60))
    assert set(c['development_indices']).isdisjoint(c['independent_indices'])
    assert sum(n for _,n in c['controls']['phases'])==326
    for batch,updates in c['training']['updates_by_batch'].items():assert int(batch)*updates==96000


def test_matched_release_and_identical_pose_only_combined_waypoints():
    root=Path(__file__).resolve().parents[1];c=json.loads((root/'configs/p3_protocol.json').read_text())
    release_offset=0
    for name,n in c['controls']['phases']:
        if name=='release':break
        release_offset+=n
    assert release_offset==c['controls']['A1']['release_offset']==275
    branch=np.eye(4);grasp=transform([.2,.1,.3],[1,0,0,0]);place=transform([.5,-.2,.3],[1,0,0,0])
    first=WaypointPlan(branch,grasp,place,c);second=WaypointPlan(branch,grasp,place,c)
    for step in range(600):
        a=first.goal(step);b=second.goal(step)
        assert a[0]==b[0] and np.array_equal(a[1],b[1])
    assert first.goal(275)[0]=='release'
    assert first.goal(325)[0]=='retreat' and first.goal(326)[0]=='post_retreat'
