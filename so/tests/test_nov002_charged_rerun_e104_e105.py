from fractions import Fraction as F

from so.experiments import e000104_tensor_train_lifecycle_quotient_reduction as e104
from so.experiments import e000105_stable_gate_cohort_transport_reduction as e105
from so.experiments.nov002_charged_rerun_e104_e105 import (
    counted_candidate_sensitivity,
    counted_environments,
    counted_generic_sensitivity,
    run,
    work_term_audit,
)

_RESULT = run()


def _by_experiment():
    return {r["experiment"]: r for r in _RESULT["results"]}


def test_counters_reproduce_the_originals():
    """The counters must describe the code that produced the kills, not a model of it."""
    assert _RESULT["counters_reproduce_originals"] is True
    for r in _RESULT["results"]:
        assert r["counter_disagreements_with_original"] == 0, r["experiment"]


def test_arms_still_reach_identical_exact_states():
    """Charging changes the accounting, not the arithmetic — both originals' state claims hold."""
    assert _RESULT["arms_still_reach_identical_exact_states"] is True
    for r in _RESULT["results"]:
        assert r["exact_state_mismatches"] == 0, r["experiment"]


def test_both_kills_survive_charging():
    assert _RESULT["decision"] == "BOTH_KILLS_SURVIVE_CHARGING"
    assert set(_RESULT["kills_surviving"]) == {"E-000104", "E-000105"}
    assert _RESULT["kills_reversed"] == []


def test_e000104_charges_both_arms_identically():
    """Same cached environments, same arithmetic shape: the tie is real."""
    r = _by_experiment()["E-000104"]
    assert r["representation_build_ratio_candidate_over_generic"] == 1.0
    assert r["candidate_total_charged"] == r["generic_total_charged"]
    assert r["charged_screen_detail"]["improved_coordinates"] == {}
    assert r["charged_screen_detail"]["regressed_coordinates"] == {}
    assert r["verdict_charged"] == "KILL"


def test_e000105_candidate_is_strictly_worse_once_charged():
    """Matrix propagation costs width^3 per layer; column propagation costs width^2 per axis."""
    r = _by_experiment()["E-000105"]
    assert r["representation_build_ratio_candidate_over_generic"] > 3.0
    assert r["candidate_total_charged"] > r["generic_total_charged"]
    assert r["charged_screen_detail"]["improved_coordinates"] == {}
    assert "multiplies" in r["charged_screen_detail"]["regressed_coordinates"]
    assert r["verdict_charged"] == "KILL"


def test_apply_costs_were_equal_which_is_what_the_originals_compared():
    """Both originals compared only this, and on this the arms do tie."""
    for r in _RESULT["results"]:
        assert r["candidate_cost"]["multiplies"] == r["generic_cost"]["multiplies"], r["experiment"]
        assert r["verdict_as_originally_screened"] == "KILL", r["experiment"]


def test_work_terms_in_both_originals_cannot_fail():
    a = work_term_audit()
    assert a["cases"] == 64
    assert a["e000104_work_mismatches"] == 0
    assert a["e000105_work_mismatches"] == 0
    assert a["either_reachable"] is False


def test_counted_sensitivity_matches_original_pointwise():
    net = e105.generate(3, 6, 4, codes=4)
    for code in range(4):
        c_j, c_m = counted_candidate_sensitivity(net, code)
        g_j, g_m = counted_generic_sensitivity(net, code)
        assert c_j == e105.candidate_sensitivity(net, code)
        assert g_j == e105.generic_region_sensitivity(net, code)
        assert c_j == g_j
        assert c_m > g_m > 0


def test_counted_environments_match_original_pointwise():
    states, sessions = e104.generate_cell(1, 8, 3, 4)
    cores = e104.materialize(states)
    target = 8 // 2
    for left, right in sessions:
        (l, r), mults = counted_environments(cores, target, left, right)
        assert (l, r) == e104.candidate_environments(cores, target, left, right)
        assert mults > 0


def test_full_replay_is_still_the_expensive_reference():
    """Neither kill says the mechanism is useless against replay — only against the generic arm."""
    for r in _RESULT["results"]:
        assert r["full_replay_multiplies"] > r["generic_total_charged"], r["experiment"]


def test_e000105_build_gap_is_the_width_cubed_term():
    """A direct check of the stated cause, on one net rather than through the aggregate."""
    w, depth, pd, od, codes = 8, 6, 3, 3, 4
    net = e105.generate(0, w, depth, codes=codes)
    _, c_m = counted_candidate_sensitivity(net, 0)
    _, g_m = counted_generic_sensitivity(net, 0)
    assert c_m == depth * (w ** 3 + w * w * pd) + od * w * pd
    assert g_m == pd * (depth * w * w + od * w)
    assert c_m > g_m
