import pytest

from so.screen import COORDINATES, CostVector, Measurement, VacuousComparison, screen


def _m(label, **kw):
    return Measurement(label, CostVector(**kw))


# ------------------------------------------------------------------ defect 4: the vacuity guard


def test_same_provenance_is_refused():
    """e000104:224-225 and e000105:176-177 in the form the guard makes unrepresentable."""
    a = _m("mutation_multiplies(net)", multiplies=9)
    b = _m("mutation_multiplies(net)", multiplies=9)
    with pytest.raises(VacuousComparison):
        screen(a, b, states_equal=True)


def test_independent_provenance_is_allowed_even_when_the_numbers_tie():
    """A real tie is a real result; only a shared source is refused."""
    v = screen(
        _m("candidate_local_update", multiplies=12),
        _m("generic_cached_environment_update", multiplies=12),
        states_equal=True,
    )
    assert v.verdict == "KILL"
    assert v.improved == {} and v.regressed == {}


def test_a_measurement_must_name_its_provenance():
    with pytest.raises(ValueError):
        Measurement("   ", CostVector(multiplies=1))


# --------------------------------------------------------- defect 1: improvement, not equality


def test_a_strict_regression_is_killed_not_promoted():
    """NOV-001's M5: identical result, 32 multiplies against 16, better at nothing."""
    v = screen(_m("counted_wasteful", multiplies=32), _m("counted_plain", multiplies=16), True)
    assert v.verdict == "KILL"
    assert v.regressed == {"multiplies": [32, 16]}
    assert v.improved == {}


def test_a_strict_improvement_is_promoted():
    v = screen(_m("strassen_counter", multiplies=7), _m("schoolbook_counter", multiplies=8), True)
    assert v.verdict == "PROMOTE"
    assert v.improved == {"multiplies": [7, 8]}


# ------------------------------------------------- defect 2: every coordinate the mechanism moves


def test_a_win_on_memory_traffic_is_read():
    """NOV-001's M1: ties on multiplies at 72, moves 36 words against 60."""
    v = screen(
        _m("tiled_counter", multiplies=72, slow_memory_words=36),
        _m("materialising_counter", multiplies=72, slow_memory_words=60),
        True,
    )
    assert v.verdict == "PROMOTE"
    assert v.improved == {"slow_memory_words": [36, 60]}


def test_a_trade_promotes_and_records_both_sides():
    """NOV-001's M2: fewer rounds bought with more arithmetic."""
    v = screen(
        _m("draft_verify_counter", multiplies=9564, sequential_rounds=1000),
        _m("sequential_counter", multiplies=4536, sequential_rounds=1134),
        True,
    )
    assert v.verdict == "PROMOTE"
    assert v.improved == {"sequential_rounds": [1000, 1134]}
    assert v.regressed == {"multiplies": [9564, 4536]}


def test_undeclared_coordinates_are_reported_not_silently_scored():
    cand = Measurement("a", CostVector(multiplies=5, slow_memory_words=1), ("multiplies",))
    gen = Measurement("b", CostVector(multiplies=5, slow_memory_words=99), ("multiplies",))
    v = screen(cand, gen, True)
    assert v.verdict == "KILL"
    assert "slow_memory_words" in v.unread_coordinates
    assert v.improved == {}


def test_unknown_coordinate_is_rejected():
    with pytest.raises(ValueError):
        Measurement("a", CostVector(), ("flops_per_watt",))


# ----------------------------------------------------------- defect 3: the representation charged


def test_representation_is_charged_to_whoever_builds_it():
    """NOV-001's M4: subsidised the baseline ties; charged, the candidate wins."""
    cand = _m("prefix_index_counter", multiplies=0, representation_multiplies=16)
    subsidised = _m("baseline_given_the_table", multiplies=0, representation_multiplies=0)
    unsubsidised = _m("baseline_own_scan", multiplies=121)

    assert screen(cand, subsidised, True).verdict == "KILL"
    assert screen(cand, unsubsidised, True).verdict == "PROMOTE"


def test_e000105_shape_is_killed_for_the_right_reason():
    """NOV-002: charging exposes a regression, it does not rescue the candidate."""
    v = screen(
        _m("counted_candidate_sensitivity", multiplies=31104, representation_multiplies=906624),
        _m("counted_generic_sensitivity", multiplies=31104, representation_multiplies=298368),
        True,
    )
    assert v.verdict == "KILL"
    assert v.regressed == {"multiplies": [937728, 329472]}


def test_charged_folds_representation_into_arithmetic():
    c = CostVector(multiplies=10, representation_multiplies=5).charged()
    assert c.multiplies == 15 and c.representation_multiplies == 0


# --------------------------------------------------------------------------------- exactness first


def test_unequal_states_short_circuit_before_cost():
    v = screen(_m("a", multiplies=1), _m("b", multiplies=1000), states_equal=False)
    assert v.verdict == "NOT_EXACT"
    assert v.improved == {} and v.regressed == {}


def test_verdict_serialises_with_its_provenance():
    d = screen(_m("cand", multiplies=1), _m("gen", multiplies=2), True).as_dict()
    assert d["verdict"] == "PROMOTE"
    assert d["candidate_label"] == "cand" and d["generic_label"] == "gen"
    assert set(COORDINATES) >= set(d["improved_coordinates"])
