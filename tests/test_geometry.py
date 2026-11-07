from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from coftprobe.geometry import SerialArm, inverse, transform, camera_projection, segment_box_intersects, line_of_sight
from coftprobe.experiment import choose_baselines, load_calibration, run

ROOT=Path(__file__).resolve().parents[1]


def test_inverse_pose_and_fk():
    t=transform([.1,-.3,.4],Rotation.from_euler('xyz',[.3,-.5,.7]).as_matrix())
    assert np.allclose(t@inverse(t),np.eye(4),atol=1e-10)
    arm=SerialArm(ROOT/'configs/sr5_official_kinematics.json')
    assert len(arm.joints)==6
    q=np.zeros(6)
    f=arm.fk(q)
    assert f.shape==(4,4)
    assert np.isfinite(f).all()
    s=arm.ik(f)
    assert s['success']


def test_ray_and_depth():
    assert segment_box_intersects([0,0,1],[0,0,-1],[0,0,0],[.1,.1,.1])
    assert not segment_box_intersects([1,0,1],[1,0,-1],[0,0,0],[.1,.1,.1])
    c=dict(fx=100,fy=100,cx=100,cy=100,width=200,height=200,near_m=.2,far_m=2)
    assert camera_projection(np.eye(4),[0,0,1],c)['in_frustum']
    assert not camera_projection(np.eye(4),[0,0,.1],c)['in_frustum']
    assert not line_of_sight(np.eye(4),[0,0,1],[{'center':[0,0,.5], 'half_extent':[.1,.1,.1]}],c)['visible']
    assert line_of_sight(np.eye(4),[0,0,1],[],c)['visible']


def test_equal_info_baselines():
    cases=[dict(feasible=True,observable=False,both=False,vis_fraction=0,joint_path_jump_rad=.1),
           dict(feasible=True,observable=True,both=True,vis_fraction=1,joint_path_jump_rad=.3),
           dict(feasible=False,observable=True,both=False,vis_fraction=1,joint_path_jump_rad=.01)]
    results=choose_baselines(cases)
    assert results['motion_only']==0
    assert results['analytic_and']==1
    assert results['weighted']==1
    assert results['vision_only']==1


def test_one_scene_reproducibility(tmp_path):
    kwargs=dict(n_scenes=1,n_yaws=3,seed=8,calibration=str(ROOT/'configs/demo_proxy_calibration.json'),
                robot=str(ROOT/'configs/sr5_official_kinematics.json'),real=False)
    a=run(**kwargs,out=str(tmp_path/'a'))
    b=run(**kwargs,out=str(tmp_path/'b'))
    assert a==b
    assert a['algorithm_novelty']=='NOT_ESTABLISHED'
    assert (tmp_path/'a'/'trials.json').exists()


def test_proxy_cannot_be_claimed_measured():
    import pytest
    with pytest.raises(ValueError,match='MEASURED'):
        load_calibration(ROOT/'configs/demo_proxy_calibration.json',real=True)
