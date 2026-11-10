"""Nonphysical checks protecting the one-shot intervention comparisons."""
import unittest
import numpy as np
from mechanism_validation.contact import bilateral_pairs
from mechanism_validation.control import Prefix,Postgrasp

C=dict(position_tolerance_m=.015,rotation_tolerance_degrees=10,pose_dwell_states=5,contact_dwell_states=5,
       stable_rise_m=.03,stable_dwell_states=10,clearance_up_m=.08,pregrasp_up_m=.08,lift_m=.08,
       close_min_steps=16,preplace_up_m=.12,release_steps=20,retreat_up_m=.15,
       phase_caps=dict(clearance=60,pregrasp=160,descend=100,close=60,lift=100,transport=130,preplace=130,place=140,retreat=100))


class Controls(unittest.TestCase):
    def test_fingertips_are_two_sided_but_self_or_distractor_contact_is_not(self):
        pairs=[('gripper0_right_left_fingertip_collision','obj_g1'),('obj_g2','gripper0_right_right_fingertip_collision')]
        self.assertTrue(bilateral_pairs(pairs,['obj_g1','obj_g2']))
        self.assertFalse(bilateral_pairs(pairs,['obj_g1']))
        self.assertFalse(bilateral_pairs([('gripper0_right_left_fingertip_collision','gripper0_right_right_fingertip_collision')],['obj_g1']))

    def test_elapsed_time_cannot_advance_an_unreached_phase(self):
        p=Prefix(np.eye(4),np.eye(4),np.eye(4),C)
        for _ in range(60):self.assertEqual(p.command(np.eye(4),np.eye(4),False)[0],'clearance')
        self.assertIsNone(p.command(np.eye(4),np.eye(4),False))
        self.assertEqual(p.failed,'clearance_TIMEOUT');self.assertFalse(p.done)

    def test_pose_arrival_without_contact_does_not_authorize_handoff(self):
        p=Prefix(np.eye(4),np.eye(4),np.eye(4),C)
        p.stage=3
        for _ in range(C['phase_caps']['close']):p.command(p.grasp,np.eye(4),False)
        self.assertIsNone(p.command(p.grasp,np.eye(4),False))
        self.assertEqual(p.failed,'close_TIMEOUT');self.assertFalse(p.done)

    def test_fixed_and_feedback_goals_agree_without_slip_and_diverge_under_slip(self):
        release=np.eye(4);release[0,3]=.3
        template=dict(T_target_sink=np.eye(4).tolist(),postgrasp_object_in_sink=[np.eye(4).tolist(),release.tolist(),release.tolist()])
        tcp=np.eye(4);tcp[2,3]=.05;obj=np.eye(4)
        a=Postgrasp(tcp,obj,template,C,'M0');b=Postgrasp(tcp,obj,template,C,'M2')
        np.testing.assert_array_equal(a.object_goals,b.object_goals)
        np.testing.assert_allclose(a.command(tcp,obj)[1],b.command(tcp,obj)[1])
        slip=obj.copy();slip[0,3]=.03
        fixed=a.command(tcp,slip)[1];feedback=b.command(tcp,slip)[1]
        self.assertAlmostEqual(fixed[0,3]-feedback[0,3],.03)


if __name__=='__main__':unittest.main()
