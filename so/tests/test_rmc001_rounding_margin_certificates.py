"""RMC001 exact operator regressions; not trained-model tests."""
from copy import deepcopy
import numpy as np
import pytest

from so.experiments.rmc001_rounding_margin_certificates import (
    A, Q, BLOCK, boundary_controls, build_model, candidate_edit, edit_path,
    exact_equal, full_rebuild, init_candidate, init_sparse, make_certs,
    qround_scalar, safe_radius, sparse_edit, verify_certs,
)


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (Q // 2, 0),                 # tie: 0 is even
        (Q // 2 + 1, 1),
        (Q + Q // 2, 2),             # tie: 2 is even
        (-(Q // 2), 0),
        (-(Q // 2 + 1), -1),
        (-(Q + Q // 2), -2),
    ],
)
def test_round_to_nearest_ties_to_even(n, expected):
    assert qround_scalar(n) == expected


def test_safe_radius_respects_midpoint_parity():
    assert safe_radius(Q, 1) == Q // 2 - 1
    assert safe_radius(2 * Q, 2) == Q // 2
    row = boundary_controls()
    assert row["naive_non_strict_midpoint_is_unsound"]


def test_certificate_tampering_is_detected():
    model = build_model(0)
    certs = make_certs(model)
    assert verify_certs(model, certs)
    bad = deepcopy(certs)
    bad[0][0].max_abs_weight -= 1
    assert not verify_certs(model, bad)
    bad = deepcopy(certs)
    bad[0][0].safe_radius += 1
    assert not verify_certs(model, bad)


def test_one_update_matches_full_rebuild_for_both_exact_methods():
    model = build_model(1)
    target = np.zeros(A, dtype=np.int64)
    target[0] = 1
    fresh, _ = full_rebuild(model, target)

    sparse = init_sparse(model)
    candidate = init_candidate(model)
    sparse_edit(model, sparse, target)
    candidate_edit(model, candidate, target)

    assert all(exact_equal(a, b) for a, b in zip(sparse.h, fresh))
    assert all(exact_equal(a, b) for a, b in zip(candidate.h, fresh))


def test_accumulated_margin_eventually_forces_refresh_without_stale_reuse():
    model = build_model(2)
    state = init_candidate(model)
    refreshed = 0
    for target in edit_path(2):
        fresh, _ = full_rebuild(model, target)
        _, _, count, _ = candidate_edit(model, state, target)
        refreshed += count
        assert all(exact_equal(a, b) for a, b in zip(state.h, fresh))
    assert refreshed > 0


def test_never_and_restoration_are_exact():
    model = build_model(3)
    state = init_candidate(model)
    nonzero = np.zeros(A, dtype=np.int64); nonzero[1] = -2; nonzero[5] = 1
    zero = np.zeros(A, dtype=np.int64)
    for target in (nonzero, zero, nonzero):
        fresh, _ = full_rebuild(model, target)
        candidate_edit(model, state, target)
        assert all(exact_equal(a, b) for a, b in zip(state.h, fresh))


def test_noop_edit_preserves_state_exactly():
    model = build_model(4)
    state = init_candidate(model)
    before = [x.copy() for x in state.h]
    ops, refresh_ops, refreshed, changed = candidate_edit(model, state, state.source.copy())
    assert all(exact_equal(a, b) for a, b in zip(before, state.h))
    assert refresh_ops == 0 and refreshed == 0
    assert sum(changed) == 0
    assert ops > 0  # certificate checks are intentionally counted


def test_leaky_high_gain_case_falls_back_but_remains_exact():
    model = build_model(0, leaky=True)
    state = init_candidate(model)
    target = np.ones(A, dtype=np.int64)
    fresh, _ = full_rebuild(model, target)
    _, refresh_ops, refreshed, _ = candidate_edit(model, state, target)
    assert refreshed > 0 and refresh_ops > 0
    assert all(exact_equal(a, b) for a, b in zip(state.h, fresh))
