import json
import pytest
from simulation_validation.analyze_rollouts import analyze,wilson


def test_wilson_small_sample_does_not_imply_certainty():
    lo,hi=wilson(0,20)
    assert lo==pytest.approx(0.,abs=1e-12)
    assert hi==pytest.approx(.161125158,abs=1e-8)
    assert wilson(20,20)[0]==pytest.approx(1-hi)
    assert wilson(0,0) is None


def test_unexecuted_cells_have_no_success_rate_or_failure_count(tmp_path):
    analyze(tmp_path/'missing-run',tmp_path/'out')
    payload=json.loads((tmp_path/'out/analysis.json').read_text())
    assert payload['episodes']==[]
    assert len(payload['cells'])==4
    assert all(c['status']=='INCOMPLETE' and c['success_rate'] is None
               and c['successes'] is None and c['wilson95'] is None
               for c in payload['cells'])


def test_running_cell_retains_completed_episode_count_without_final_rate(tmp_path):
    official=tmp_path/'run/p1/no_reps/panda/official';official.mkdir(parents=True)
    (official/'episodes.jsonl').write_text(json.dumps({'episode':0,'success':True})+'\n')
    analyze(tmp_path/'run',tmp_path/'out')
    cell=json.loads((tmp_path/'out/analysis.json').read_text())['cells'][1]
    assert cell['completed']==1 and cell['successes']==1
    assert cell['status']=='INCOMPLETE' and cell['success_rate'] is None


def test_string_false_is_rejected_instead_of_counted_as_success(tmp_path):
    official=tmp_path/'run/p1/no_reps/panda/official';official.mkdir(parents=True)
    (official/'episodes.jsonl').write_text(json.dumps({'episode':0,'success':'False'})+'\n')
    with pytest.raises(ValueError,match='JSON booleans'):
        analyze(tmp_path/'run',tmp_path/'out')
