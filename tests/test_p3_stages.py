import numpy as np
from p3_validation.stages import measure_stages


def test_confirmed_lift_release_and_retreat_use_state_action_alignment():
    obj=np.zeros((21,3));obj[2:,2]=.04;obj[5:,0]=.2
    eef=obj.copy();eef[16:,0]+=.15
    contact=np.zeros(21,dtype=bool);contact[2:16]=True
    gripper=np.ones(20);gripper[15:]=-1
    r=measure_stages(obj,eef,contact,gripper)
    assert r['first_stable_lift_start_state_index']==2
    assert r['first_stable_lift_confirmation_state_index']==11
    assert r['first_release_after_confirmed_lift_action_index']==15
    assert np.isclose(r['maximum_horizontal_displacement_from_lift_start_while_in_contact_m'],.2)
    assert np.isclose(r['maximum_eef_object_distance_after_release_m'],.15)


def test_rise_without_contact_and_interrupted_contact_are_not_stable_grasp():
    obj=np.zeros((21,3));obj[:,2]=.04;obj[0,2]=0
    contact=np.ones(21,dtype=bool);contact[[0,9,18]]=False
    r=measure_stages(obj,obj,contact,-np.ones(20))
    assert not r['stable_lift_confirmed']
    assert r['maximum_horizontal_displacement_from_lift_start_while_in_contact_m'] is None
    assert r['first_release_after_confirmed_lift_action_index'] is None
