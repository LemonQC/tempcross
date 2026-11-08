import pytest
from p2_validation.controls import GripperCorrection, select_conditions


def test_hysteresis_and_hold_only_depend_on_signal_history():
    controller=GripperCorrection(.9,hysteresis=.03,minimum_hold=3)
    assert [controller.command(g) for g in [.95,.89,.99,.99,.91,.94]]==[-1,1,1,1,1,-1]
    controller.reset();assert controller.command(.95)==-1


def test_window_preserves_original_ids_indices_and_alignment():
    payload={'episodes':[{'episode':i,'condition_id':str(i),'seed':1000+2*i} for i in range(100)],'episode_count':100}
    selected,states,xmls=select_conditions(payload,list(range(100)),['xml'+str(i) for i in range(100)],list(range(20,40)))
    assert selected['episodes'][0]=={'episode':20,'condition_id':'20','seed':1040}
    assert states==list(range(20,40)) and xmls[0]=='xml20'
    assert payload['episode_count']==100 and selected['episode_count']==20
    with pytest.raises(ValueError):select_conditions(payload,list(range(100)),['x']*100,[20,20])
