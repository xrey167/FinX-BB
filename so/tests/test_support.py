"""The must-hit certificate, and the ways it is meant to fail."""

from __future__ import annotations

import pytest
import torch

from so.support import (MustHitCertificate, certified_closure, certify_must_hit,
                        disjoint_lower_bound, nonneg_pursuit)


def _dictionary(n: int = 12, d: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, d, generator=g)
    return a / a.norm(dim=1, keepdim=True)


# ------------------------------------------------------------------------- the decomposition
def test_pursuit_recovers_a_planted_nonnegative_support():
    a = _dictionary()
    h = 2.0 * a[0] + 1.5 * a[3] + 0.7 * a[7]
    s = nonneg_pursuit(h, a, n_atoms=6, tol=0.02)
    assert set(s.directions) == {0, 3, 7}
    assert s.residual_fraction < 0.02


def test_pursuit_reports_what_it_could_not_explain():
    """A support that stopped on budget is not a trace, and the number saying so is on the result."""
    a = _dictionary()
    g = torch.Generator().manual_seed(1)
    h = torch.randn(64, generator=g)
    s = nonneg_pursuit(h, a, n_atoms=2, tol=1e-6)
    assert s.size <= 2
    assert s.residual_fraction > 0.5


def test_pursuit_uses_the_callers_own_numbering():
    a = _dictionary()
    ids = [100 + i for i in range(a.shape[0])]
    h = 1.0 * a[2] + 1.0 * a[5]
    s = nonneg_pursuit(h, a, ids=ids, n_atoms=4, tol=0.02)
    assert set(s.directions) == {102, 105}


def test_pursuit_coefficients_are_nonnegative():
    a = _dictionary()
    h = 3.0 * a[1] - 2.0 * a[4]
    s = nonneg_pursuit(h, a, n_atoms=8, tol=0.01)
    assert all(c >= 0.0 for c in s.coefficients)


# ------------------------------------------------------------------------- the certificate
def test_certificate_holds_when_the_support_really_carries_the_answer():
    pool = list(range(8))
    support = (0, 1, 2)

    def silences(dirs):
        return bool(set(dirs) & set(support))

    c = certify_must_hit(silences, support, pool)
    assert c.holds and c.exhaustive and c.counterexample is None
    assert c.subsets_tested == 2 ** 5 - 1


def test_certificate_fails_and_names_the_counterexample():
    """A direction outside the support silences the query: the support is not a must-hit set."""
    pool = list(range(8))
    support = (0, 1, 2)

    def silences(dirs):
        return bool(set(dirs) & {0, 1, 2, 6})

    c = certify_must_hit(silences, support, pool)
    assert not c.holds
    assert c.counterexample == (6,)


def test_certificate_refuses_a_query_that_was_never_answered():
    """The instrument that cannot fail: silence everywhere means every subset 'silences'."""
    with pytest.raises(ValueError, match="does not answer with nothing removed"):
        certify_must_hit(lambda dirs: True, (0, 1), list(range(5)))


def test_certificate_is_honest_about_a_budget_it_ran_out_of():
    pool = list(range(20))
    support = (0,)

    def silences(dirs):
        return 0 in set(dirs)

    c = certify_must_hit(silences, support, pool, budget=200)
    assert c.holds
    assert not c.exhaustive
    assert 0 < c.max_size_exhausted < len(c.complement)


# ------------------------------------------------------------------------- the bound
def _cert(support, holds=True, exhaustive=True):
    return MustHitCertificate(tuple(support), (), 1, 1, exhaustive, holds, None)


def test_disjoint_supports_bound_the_closure_from_below():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3)), _cert((4, 5))])
    assert b.lower_bound == 3 and b.certified
    assert b.shared_atoms == ()


def test_one_shared_atom_collapses_the_bound_to_one_and_that_is_correct():
    """Every access path running through one direction IS a pod, and a pod's closure is one."""
    b = disjoint_lower_bound([_cert((0, 1)), _cert((0, 2)), _cert((0, 3))])
    assert b.lower_bound == 1
    assert b.shared_atoms == (0,)


def test_an_untested_support_makes_the_bound_a_conjecture():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3), holds=False)])
    assert b.lower_bound == 2 and not b.certified
    assert "NOT CERTIFIED" in b.summary()


def test_a_partially_tested_support_also_makes_it_a_conjecture():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3), exhaustive=False)])
    assert not b.certified


def test_empty_input_is_not_a_bound_of_zero_dressed_as_a_certificate():
    b = disjoint_lower_bound([])
    assert b.lower_bound == 0 and not b.certified


# ------------------------------------------------------------------------- the interval
def test_greedy_meeting_a_certified_bound_is_exact():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3))])
    c = certified_closure(7, 2, b, n_queries=2, workload="two phrasings")
    assert c.optimal and c.lower == 2 and c.upper == 2
    assert "EXACT" in c.summary()


def test_greedy_meeting_an_uncertified_bound_proves_nothing():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3), holds=False)])
    c = certified_closure(7, 2, b, n_queries=2)
    assert not c.optimal
    assert "lower side not certified" in c.summary()


def test_a_gap_between_the_two_sides_is_reported_as_a_gap():
    b = disjoint_lower_bound([_cert((0, 1)), _cert((2, 3))])
    c = certified_closure(7, 5, b, n_queries=2)
    assert not c.optimal and c.lower == 2 and c.upper == 5
    assert "[2, 5]" in c.summary()


# ------------------------------------------------------------------------- vacuity
def test_a_support_that_fills_the_pool_is_flagged_vacuous():
    """It passes by having nothing to test against, which is not the same as passing."""
    pool = [0, 1, 2]
    c = certify_must_hit(lambda d: bool(d), (0, 1, 2), pool)
    assert c.vacuous and c.subsets_tested == 0
    assert "VACUOUS" in c.summary()


def test_a_vacuous_support_cannot_certify_a_bound():
    b = disjoint_lower_bound([_cert((0, 1)), MustHitCertificate((2, 3), (), 0, 0, True, True, None,
                                                                vacuous=True)])
    assert b.lower_bound == 2 and not b.certified
