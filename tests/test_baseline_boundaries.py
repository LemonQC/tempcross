import numpy as np
import pytest
from baselines.common.action_adapter import canonical_action
from baselines.common.dataset_adapter import window_indices


def test_gripper_sign_and_inactive_mobile_dimensions():
 for g in [-1.,0.,1.]:
  a,_=canonical_action([.1,.2,-.3,.1,0.,-.2,g]);assert a[6]==g and np.array_equal(a[7:11],np.zeros(4)) and a[11]==-1
 for native,expected in [(0.,1.),(1.,-1.),(.5,0.)]:
  a,_=canonical_action([0,0,0,0,0,0,native],'barx_unnormalized');assert a[6]==expected


def test_bad_actions_are_technical_errors_not_task_failures():
 for a in [np.zeros(6),[0,0,0,0,0,0,float('nan')]]:
  with pytest.raises(ValueError):canonical_action(a)


def test_diffusion_time_alignment_and_history_padding():
 obs,actions=window_indices('diffusion_policy',7,30)
 assert obs.tolist()==[6,7] and actions[1]==7 and actions[-1]==21
 obs,actions=window_indices('diffusion_policy',0,10)
 assert obs.tolist()==[0,0] and actions[:2].tolist()==[0,0] and actions[-1]==9


def test_transformer_only_sees_history_while_rnn_sequence_is_causal():
 obs,actions=window_indices('bc_transformer',3,30);assert len(obs)==10 and obs[-1]==3 and obs.max()==3 and actions[-1]==3
 obs,actions=window_indices('bc_rnn',3,30);assert obs.tolist()==list(range(3,13)) and np.array_equal(obs,actions)


def test_retreat_requires_placement_not_merely_initial_distance():
 from baselines.common.metrics import summarize_states
 obj=np.zeros((4,3));tcp=obj.copy();tcp[:,0]=.2
 actions=np.zeros((3,12));actions[:,6]=-1
 r=summarize_states(tcp,obj,[False]*4,actions,[False]*3,[True]*3,False)
 assert r['ever_gripper_object_far'] and not r['retreat_after_placement']
 assert not r['stable_contact_lift'] and not r['official_task_success']


def test_official_success_does_not_require_a_grasp_proxy():
 from baselines.common.metrics import summarize_states
 obj=np.zeros((4,3));actions=np.zeros((3,12));actions[:,6]=-1
 r=summarize_states(obj,obj,[False]*4,actions,[False,False,True],[True]*3,True)
 assert r['official_task_success'] and r['retreat_after_placement']
 assert not r['stable_contact_lift']


def test_contact_lift_is_consecutive_and_transport_is_separate():
 from baselines.common.metrics import summarize_states
 obj=np.zeros((15,3));obj[2:12,2]=.04;obj[5:,0]=.15
 contact=np.zeros(15,bool);contact[2:12]=True
 actions=np.zeros((14,12));actions[:,6]=1
 r=summarize_states(obj,obj,contact,actions,[False]*14,[False]*14,False)
 assert r['stable_contact_lift'] and r['longest_contact_lift_states']==10
 assert r['transport_10cm_contact_proxy'] and not r['placement_official_subpredicate']
 assert not r['official_task_success']
