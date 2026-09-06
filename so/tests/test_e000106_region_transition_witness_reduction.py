from fractions import Fraction

from so.experiments.e000106_region_transition_witness_reduction import (
    GenericPiecewiseAffine,
    _serialize,
    make_session,
    run,
    tent_tape,
)


def test_candidate_matches_generic_exactly():
    w = make_session(7, dims=4, region_count=9)
    g = GenericPiecewiseAffine(_serialize(w.segments))
    for p in [Fraction(0), Fraction(1, 8), Fraction(1, 3), Fraction(7, 8), Fraction(1)]:
        assert w.evaluate(p) == g.evaluate(p)


def test_lifecycle_screen_small():
    r = run(seed_count=2, sessions_per_seed=5, dims=4, region_count=7)
    assert r["kill_screen_pass"] is True
    assert r["candidate_generic_mismatches"] == 0
    assert r["endpoint_materialization_mismatches"] == 0
    assert r["material_updates"] == r["total_sessions"]


def test_tent_region_count_is_exponential():
    for depth in range(1, 9):
        assert len(tent_tape(depth).segments) == 2**depth


def test_tape_is_continuous_at_breakpoints():
    w = make_session(11, dims=3, region_count=8)
    for left, right in zip(w.segments, w.segments[1:]):
        t = left.right
        a = tuple(m * t + b for m, b in zip(left.slopes, left.intercepts))
        b = tuple(m * t + c for m, c in zip(right.slopes, right.intercepts))
        assert a == b
