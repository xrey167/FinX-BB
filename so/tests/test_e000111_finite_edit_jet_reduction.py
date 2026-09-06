from fractions import Fraction

from so.experiments.e000111_finite_edit_jet_reduction import (
    candidate_neural_edit_jet,
    evaluate_jet,
    generic_taylor_mode_ad,
    make_net,
    rational_nonpolynomial_fresh,
    rational_nonpolynomial_jet,
    run_registered,
    true_degree,
)


def test_full_edit_jet_matches_fresh_and_generic() -> None:
    net = make_net(seed=111, depth=4)
    x = Fraction(3, 19)
    p0 = Fraction(0)
    order = 16
    candidate, candidate_work = candidate_neural_edit_jet(net, x, p0, order)
    generic, generic_work = generic_taylor_mode_ad(net, x, p0, order)

    assert candidate == generic
    assert candidate_work == generic_work
    assert true_degree(candidate) == order

    for delta in (Fraction(1, 5), Fraction(-1, 7), Fraction(2, 11)):
        fresh = net.fresh(x, p0 + delta)
        assert evaluate_jet(candidate, delta) == fresh
        assert evaluate_jet(generic, delta) == fresh


def test_finite_jet_fails_exact_rational_nonpolynomial_transport() -> None:
    a = Fraction(2, 3)
    delta = Fraction(1, 5)
    fresh = rational_nonpolynomial_fresh(a, delta)
    for order in (1, 2, 4, 8, 16, 32):
        assert evaluate_jet(rational_nonpolynomial_jet(a, order), delta) != fresh


def test_registered_kill_screen() -> None:
    result = run_registered()
    totals = result["polynomial_edit_jet"]["totals"]
    witness = result["nonpolynomial_exactness_witness"]

    assert result["kill_screen_pass"]
    assert result["decision"] == (
        "KILL_FINITE_EDIT_JETS_AND_CO_COMPUTED_HIGHER_ORDER_SENSITIVITY_AS_STANDALONE_EXACT_TRANSPORT_NOVELTY"
    )
    assert totals["registered_cells"] == 96
    assert totals["exact_transport_cases"] == 3456
    assert totals["candidate_fresh_mismatches"] == 0
    assert totals["generic_fresh_mismatches"] == 0
    assert totals["candidate_generic_jet_mismatches"] == 0
    assert totals["candidate_generic_work_mismatches"] == 0
    assert totals["degree_mismatches"] == 0
    assert totals["max_required_exact_jet_order"] == 64
    assert totals["truncated_transport_failure_fraction"] == 1.0
    assert totals["candidate_to_generic_slot_ratio"] == 1.0
    assert totals["candidate_to_generic_multiply_ratio"] == 1.0
    assert witness["cases"] == 36
    assert witness["failures"] == 36
    assert witness["exact_matches"] == 0
