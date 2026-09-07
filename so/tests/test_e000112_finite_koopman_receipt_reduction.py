from fractions import Fraction

from so.experiments.e000112_finite_koopman_receipt_reduction import (
    apply_receipt,
    candidate_compile_receipt,
    candidate_propagate,
    generic_compile_receipt,
    generic_propagate,
    lift_original,
    make_net,
    original_fresh_lift,
    run_registered,
)


def test_exact_finite_lift_matches_original_and_generic() -> None:
    net = make_net(112)
    context = Fraction(7, 67)
    pod = Fraction(1, 5)
    y0 = Fraction(9, 71)
    depth = 8
    initial = lift_original(context, pod, y0)
    candidate = candidate_propagate(net, initial, depth)
    generic = generic_propagate(net, initial, depth)
    fresh = original_fresh_lift(net, context, pod, y0, depth)
    assert candidate == generic == fresh


def test_one_edit_receipt_is_exact_state_dependent_and_generic() -> None:
    net = make_net(3)
    old_pod = Fraction(1, 5)
    new_pod = Fraction(3, 7)
    depth = 16
    cand_a, cand_b, cand_work = candidate_compile_receipt(
        net, old_pod, new_pod, depth
    )
    gen_a, gen_b, gen_work = generic_compile_receipt(
        net, old_pod, new_pod, depth
    )
    assert (cand_a, cand_b, cand_work) == (gen_a, gen_b, gen_work)
    assert cand_b[5] != 0

    hidden_deltas = set()
    for session_id in range(8):
        context = Fraction(session_id + 1, 67)
        y0 = Fraction(session_id + 2, 71)
        stale = original_fresh_lift(net, context, old_pod, y0, depth)
        transported = apply_receipt(stale, cand_a, cand_b)
        fresh = original_fresh_lift(net, context, new_pod, y0, depth)
        assert transported == fresh
        hidden_deltas.add(transported[5] - stale[5])
    assert len(hidden_deltas) == 8


def test_registered_kill_screen() -> None:
    result = run_registered()
    totals = result["registered"]["totals"]
    assert result["kill_screen_pass"]
    assert result["decision"] == (
        "KILL_FINITE_EXACT_KOOPMAN_CARLEMAN_LIFT_AS_STANDALONE_LIFECYCLE_TRANSPORT_NOVELTY"
    )
    assert totals["registered_cells"] == 96
    assert totals["lifecycle_cases"] == 24576
    assert totals["candidate_fresh_mismatches"] == 0
    assert totals["generic_fresh_mismatches"] == 0
    assert totals["candidate_generic_receipt_mismatches"] == 0
    assert totals["candidate_generic_state_mismatches"] == 0
    assert totals["candidate_generic_work_mismatches"] == 0
    assert totals["material_fraction"] == 1.0
    assert totals["state_dependent_edit_cells"] == totals["expected_state_dependent_edit_cells"]
    assert totals["candidate_to_generic_work_ratio"] == 1.0
    assert totals["candidate_to_generic_slot_ratio"] == 1.0
    assert totals["replay_to_candidate_advantage"] > 8.0
