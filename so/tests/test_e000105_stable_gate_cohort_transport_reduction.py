from fractions import Fraction as F

from so.experiments.e000105_stable_gate_cohort_transport_reduction import (
    apply,
    candidate_sensitivity,
    fresh_forward,
    generate,
    generic_region_sensitivity,
    run,
    vsub,
)


def test_candidate_and_generic_sensitivity_match_exactly():
    net = generate(seed=3, width=6, depth=4)
    for code in range(4):
        assert candidate_sensitivity(net, code) == generic_region_sensitivity(net, code)


def test_exact_lifecycle_transport_matches_fresh():
    net = generate(seed=7, width=4, depth=2)
    context = [F(2), F(5), F(9)]
    old = [F(1), F(2), F(3)]
    new = [F(2), F(4), F(6)]
    for code in range(4):
        j = candidate_sensitivity(net, code)
        old_y = fresh_forward(net, context, old, code)
        assert apply(old_y, j, vsub(new, old)) == fresh_forward(net, context, new, code)


def test_context_cohorts_have_materially_different_pod_corrections():
    net = generate(seed=11, width=8, depth=6)
    delta = [F(1), F(2), F(3)]
    corrections = []
    for code in range(4):
        j = candidate_sensitivity(net, code)
        corrections.append(tuple(sum((row[k] * delta[k] for k in range(3)), F(0)) for row in j))
    assert len(set(corrections)) == 4


def test_registered_small_screen_kills_only_after_valid_controls():
    result = run(seed_count=2, widths=(4, 6), depths=(2, 4), session_count=8, codes=4)
    assert result["validity_pass"] is True
    assert result["candidate_fresh_mismatch"] == 0
    assert result["generic_fresh_mismatch"] == 0
    assert result["candidate_generic_mismatch"] == 0
    assert result["work_mismatch"] == 0
    assert result["kill_screen_pass"] is True
