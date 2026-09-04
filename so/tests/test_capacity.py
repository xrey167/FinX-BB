"""Pressure and efficiency, and the three verdicts they are allowed to reach."""

from __future__ import annotations

import torch

from so.capacity import allocation, capacity_bound, orthonormalise, subspace_overlap


def _basis(d, idx):
    a = torch.zeros(len(idx), d)
    for r, i in enumerate(idx):
        a[r, i] = 1.0
    return a


def test_capacity_bound_is_linear_in_the_dimension():
    assert capacity_bound(768, 3.33) == 768 / 3.33
    assert capacity_bound(768, 1) == 768.0


def test_orthogonal_subspaces_have_zero_overlap():
    d = 32
    assert subspace_overlap(_basis(d, [0, 1]), _basis(d, [2, 3])) < 1e-5


def test_a_shared_direction_shows_as_overlap_one():
    d = 32
    assert subspace_overlap(_basis(d, [0, 1]), _basis(d, [1, 2])) > 1.0 - 1e-5


def test_a_rank_deficient_input_is_not_counted_as_two_dimensions():
    """Two nearly collinear rows are one direction, not two with a tiny singular value."""
    a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]])
    assert orthonormalise(a).shape[1] == 1


def test_disjoint_allocation_is_efficient_and_low_pressure():
    d = 64
    subs = [_basis(d, [2 * i, 2 * i + 1]) for i in range(4)]
    a = allocation(subs, d)
    assert a.efficiency > 0.99
    assert a.pressure == 8 / 64
    assert a.max_overlap < 1e-5
    assert "ALLOCATED" in a.verdict()


def test_overlapping_subspaces_with_budget_to_spare_read_as_an_allocation_failure():
    d = 64
    subs = [_basis(d, [0, 1]), _basis(d, [1, 2]), _basis(d, [2, 3])]
    a = allocation(subs, d)
    assert a.efficiency < 0.95
    assert a.headroom > 0.8
    assert "ALLOCATION, NOT CAPACITY" in a.verdict()


def test_filling_the_budget_reads_as_the_capacity_limit_instead():
    d = 8
    subs = [_basis(d, [i, (i + 1) % d]) for i in range(4)]
    a = allocation(subs, d)
    assert a.pressure == 1.0
    assert "AGAINST THE BOUND" in a.verdict()


def test_efficiency_is_the_rank_of_the_union_over_the_sum_of_dimensions():
    d = 16
    subs = [_basis(d, [0, 1]), _basis(d, [0, 1])]          # identical: rank 2, sum 4
    a = allocation(subs, d)
    assert a.union_rank == 2 and sum(a.dims) == 4
    assert abs(a.efficiency - 0.5) < 1e-6


def test_empty_input_is_not_a_verdict():
    a = allocation([], 64)
    assert "no deletion subspaces" in a.verdict()
