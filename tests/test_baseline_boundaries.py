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
