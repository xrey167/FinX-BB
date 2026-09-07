import copy
import pytest
from so.strict_evidence import assess_records


def record(seed=0, rate=.99):
    return {"experiment":"E-000070", "rows":[{
        "seed":seed,"screening_pass":True,"fresh_alias_read_rate":rate,
        "alias_relink":{"old_truth":7,"stale_snapshot_pred":7,"commit_pred":7,"pod_only_pred":7,
                        "cavi_equals_explicit_neural_rejection_maxabs":0.0},"bypass_maxabs":0.0}]}


def test_legacy_green_does_not_satisfy_stronger_bar():
    result = assess_records([record(0,.875),record(1,46/48),record(2,41/48)])
    assert result['seed_set_complete']
    assert not result['capability_gate_pass']
    assert [r['fresh_alias_capability']['status'] for r in result['rows']] == ['FAIL','PASS','FAIL']


@pytest.mark.parametrize('v', [None,float('nan'),float('inf'),True,'0.99', -0.1, 1.1])
def test_absent_or_invalid_metric_never_passes(v):
    report = assess_records([record(0,v),record(1),record(2)])
    assert report['rows'][0]['fresh_alias_capability']['status']=='NOT_MEASURED'
    assert not report['capability_gate_pass']


def test_missing_seeds_cannot_pass():
    assert not assess_records([record(0)])['capability_gate_pass']


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError,match='unique integer seed'):
        assess_records([record(0),record(0)])


def test_capability_success_cannot_stand_in_for_full_evidence():
    report = assess_records([record(i) for i in range(3)])
    assert report['capability_gate_pass']
    assert not report['breakthrough_established']
    assert report['rows'][0]['learned_scope_bypass']['status']=='NOT_MEASURED'
    assert report['rows'][0]['retained_pod_output_locality']['status']=='NOT_MEASURED'


def test_selected_case_must_have_answered_correctly():
    r=record();r['rows'][0]['alias_relink']['stale_snapshot_pred']=256
    report=assess_records([r],expected_seeds=(0,))
    assert report['rows'][0]['selected_old_answer_case']['status']=='FAIL'


def test_historical_records_are_not_mutated():
    data=[record(i) for i in range(3)]; original=copy.deepcopy(data)
    assess_records(data)
    assert data==original
