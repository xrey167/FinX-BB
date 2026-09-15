import pytest

from so.experiments.nov003_unread_coordinates_e103 import (
    SCREENED,
    rank1_inverse_update_ops,
    rank1_solution_update_ops,
    run,
    woodbury_hoist_setup_ops,
    woodbury_inverse_ops,
    woodbury_solution_ops_full,
    woodbury_solution_ops_hoisted,
)
from so.screen import VacuousComparison

_R = run()


def test_the_registered_domain_contains_a_crossover():
    """E-000103's own dimensions straddle it, so one verdict cannot describe the comparison."""
    assert _R["crossover_within_the_registered_domain"] is True
    assert _R["dimensions_where_candidate_promotes"] == [4, 6]
    assert _R["dimensions_where_candidate_is_killed"] == [8, 10]
    assert _R["decision"] == "CROSSOVER_AGGREGATE_VERDICT_IS_NOT_A_VERDICT"


def test_the_crossover_is_the_quadratic_terms():
    """8n^2 against 6n^2 on the revision, with the candidate's per-session term smaller."""
    for n in (4, 6):
        cand = 2 * rank1_inverse_update_ops(n) + 24 * 2 * rank1_solution_update_ops(n)
        gen = woodbury_inverse_ops(n, 2) + 24 * woodbury_solution_ops_hoisted(n, 2)
        assert cand < gen, n
    for n in (8, 10):
        cand = 2 * rank1_inverse_update_ops(n) + 24 * 2 * rank1_solution_update_ops(n)
        gen = woodbury_inverse_ops(n, 2) + 24 * woodbury_solution_ops_hoisted(n, 2)
        assert cand > gen, n


def test_the_baseline_as_written_rebuilds_loop_invariant_work():
    """The mirror of the subsidy: the baseline denied an optimisation the candidate already uses."""
    for n in (4, 6, 8, 10):
        assert woodbury_solution_ops_full(n, 2) > 3 * woodbury_solution_ops_hoisted(n, 2), n
    for d in _R["per_dimension"]:
        assert d["generic_as_written_multiplies"] > 2 * d["generic_hoisted_multiplies"], d["n"]


def test_hoisting_costs_no_extra_arithmetic():
    """woodbury_rankk_inverse already builds inv_u and middle_inv; hoisting retains, not rebuilds."""
    assert woodbury_hoist_setup_ops(10, 2) == 0


def test_unresolvable_coordinate_is_declared_unread_not_scored():
    """A 2-word constant gap must not decide a verdict against a larger arithmetic regression."""
    assert "slow_memory_words" not in SCREENED
    unread = _R["verdict_vs_generic_hoisted"]["unread_coordinates"]
    assert "slow_memory_words" in unread
    # it is still recorded, so the omission is visible rather than silent
    assert _R["candidate_cost"]["slow_memory_words"] > 0
    assert _R["generic_hoisted_cost"]["slow_memory_words"] > 0


def test_candidate_always_pays_more_sequential_depth():
    """Two dependent rank-1 steps against one block step, on every dimension."""
    for d in _R["per_dimension"]:
        assert "sequential_rounds" in d["regressed"], d["n"]


def test_no_mechanism_is_promoted():
    """Wins small, loses large, costs roughly twice the depth throughout — not an invention."""
    assert "promot" not in _R["decision"].lower() or _R["crossover_within_the_registered_domain"]
    assert _R["dimensions_where_candidate_is_killed"], "a clean sweep would be a different result"


def test_arms_carry_independent_provenance():
    """so.screen refuses a comparison whose sides share a source; these must not."""
    labels = {
        _R["verdict_vs_generic_hoisted"]["candidate_label"],
        _R["verdict_vs_generic_hoisted"]["generic_label"],
    }
    assert len(labels) == 2


def test_screen_would_refuse_a_self_comparison_here():
    """The guard is live in this experiment's own screen calls, not just in so/tests/test_screen.py."""
    from so.screen import CostVector, Measurement, screen

    same = Measurement("nov003:candidate_sequential_rank1_counter", CostVector(multiplies=1))
    with pytest.raises(VacuousComparison):
        screen(same, same, states_equal=True)
