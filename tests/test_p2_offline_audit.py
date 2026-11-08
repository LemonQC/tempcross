import numpy as np
import pytest

from p2_validation.offline_audit import (
    center_prism, expand_chunks, fingerpad_specs, relative_pose, world_pad,
)


def test_fingerpad_pose_uses_body_rotation_and_local_offset():
    q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    spec = dict(position=np.array([1.,0,0]), quaternion=np.array([1.,0,0,0]))
    position, orientation = world_pad(np.array([2.,3,4]), q, spec)
    np.testing.assert_allclose(position, [2,4,4], atol=1e-12)
    offset, relative_q = relative_pose(np.array([2.,3,4]), q, position, orientation)
    np.testing.assert_allclose(offset, [1,0,0], atol=1e-12)
    np.testing.assert_allclose(relative_q, [1,0,0,0], atol=1e-12)


def test_center_prism_distinguishes_pad_depth_from_center_distance():
    identity = np.array([[1.,0,0,0], [1.,0,0,0]])
    left = (np.array([[0.,-.04,0], [0.,-.04,0]]), identity)
    right = (np.array([[0.,.04,0], [0.,.04,0]]), identity)
    specs = {side: {'half_size': np.array([.01,.001,.01675])} for side in ['left','right']}
    _, gap, _, inside = center_prism(left, right, np.array([[0.,0,0], [0.,0,.04]]), specs)
    assert inside.tolist() == [True, False]
    np.testing.assert_allclose(gap, [.078,.078])


def test_chunk_expansion_trims_unexecuted_prediction_and_rejects_gaps():
    chunks = np.arange(2*8*7).reshape(2,8,7)
    expanded = expand_chunks(chunks, [1,9], 10)
    np.testing.assert_equal(expanded, np.concatenate([chunks[0], chunks[1,:2]]))
    with pytest.raises(ValueError, match='Incomplete'):
        expand_chunks(chunks[:1], [1], 10)
    with pytest.raises(ValueError, match='overlapping'):
        expand_chunks(chunks, [1,8], 10)
    with pytest.raises(ValueError, match='counts differ'):
        expand_chunks(chunks, [1], 10)


def test_xml_uses_the_named_contact_pads_not_visual_meshes():
    xml = '<mujoco><asset><mesh file="<ROBOCASA>/unused"/></asset><worldbody>'
    for side in ['left','right']:
        xml += (f'<body name="{side}"><geom name="gripper0_right_{side}_fingerpad_collision" '
                'type="box" pos="0 0 .03" size=".01 .001 .01675"/></body>')
    specs = fingerpad_specs(xml+'</worldbody></mujoco>')
    assert specs['left']['body'] == 'left'
    np.testing.assert_allclose(specs['right']['position'], [0,0,.03])
