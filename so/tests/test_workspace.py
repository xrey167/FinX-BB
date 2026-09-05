"""Workspace closure: the same algorithm as so/closure.py, over directions instead of records.

These tests do not need a language model. They build a toy readout whose carriers are known by
construction, so what is asserted is that the instrument reports the number that is actually there --
including the case where it must report that it could not find one.
"""
import numpy as np
import pytest
import torch

from so.workspace import (WorkspaceClosure, carrier_candidates, fit_tuned_lens, lens_logits,
                          project_out, workspace_closure)

D, V = 16, 12


def _basis(seed=0):
    torch.manual_seed(seed)
    w = torch.randn(V, D)
    return w / w.norm(dim=-1, keepdim=True)


def test_projecting_out_a_direction_removes_it_and_leaves_the_rest():
    w = _basis()
    h = 3.0 * w[4] + 2.0 * w[7]
    out = project_out(h.reshape(1, -1), w[4].reshape(1, -1)).reshape(-1)
    assert abs(float(out @ w[4])) < 1e-5
    assert abs(float(out @ w[7])) > 0.5          # the other component survives


def test_projecting_out_a_span_removes_the_whole_span_not_one_vector_at_a_time():
    """Non-orthogonal directions: projecting one at a time leaves part of the span standing, which
    would understate the closure. The QR in project_out is what stops that."""
    torch.manual_seed(0)
    a = torch.randn(1, D)
    b = a + 0.15 * torch.randn(1, D)             # nearly parallel to a
    h = (2.0 * a + 1.0 * b)
    both = project_out(h, torch.cat([a, b]))
    assert float(both.norm()) < 1e-4             # h is IN the span, so nothing is left
    one_then_one = project_out(project_out(h, a), b)
    assert float(one_then_one.norm()) > 0.5      # and one at a time leaves most of it standing
    # which is the whole reason for the QR: the second projection puts back part of the first
    h2 = h + 5.0 * torch.randn(1, D)
    assert float((project_out(h2, torch.cat([a, b])) @ a.t()).abs().max()) < 1e-4
    assert float((project_out(project_out(h2, a), b) @ a.t()).abs().max()) > 1e-3


def test_the_object_direction_is_offered_first_so_the_search_measures_closure_not_ranking():
    w = _basis()
    h = 0.1 * w[3] + 9.0 * w[9]                  # the object (3) is NOT the largest component
    cands = carrier_candidates(h, w, obj_id=3, n=5)
    assert cands[0] == 3
    assert 9 in cands


def test_a_single_shared_carrier_gives_a_closure_of_one_proved_optimal():
    """The pod case, in activation space: every query reads the same direction, so one removal ends
    them all and the certified lower bound meets it."""
    calls = {"n": 0}

    def answers(dirs):
        calls["n"] += 1
        return [7 if 7 not in dirs else -1] * 5   # five queries, one carrier

    wc = workspace_closure(answers, candidates=[7, 1, 2, 3], obj_id=7, n_queries=5)
    assert wc.size == 1 and wc.directions == (7,)
    assert wc.lower_bound == 1 and wc.optimal and not wc.exhausted
    assert "meets the hardest-query lower bound" in wc.summary()


def test_one_carrier_per_query_gives_a_closure_of_k_proved_optimal():
    """The duplicated case: each query has its own carrier, the supports are disjoint, so the bound
    equals the answer and greedy is exact."""
    carriers = {0: 10, 1: 11, 2: 12}

    def answers(dirs):
        return [7 if carriers[i] not in dirs else -1 for i in range(3)]

    wc = workspace_closure(answers, candidates=[7, 10, 11, 12], obj_id=7, n_queries=3, max_dirs=8)
    assert wc.size == 4                        # 7 is tried first and does nothing, then the three
    assert set(wc.directions) == {7, 10, 11, 12}
    # the hardest single query needs [7, 10, 11, 12] in candidate order to reach ITS carrier, so the
    # sound bound is 4 and greedy meets it. It is a weaker argument than so.closure's disjointness --
    # see the class docstring -- and it is the one that actually holds for directions.
    assert wc.lower_bound == 4 and wc.optimal


def test_a_query_that_never_answered_is_not_counted():
    """The attack-validity floor, one level down: a phrasing the model never got right cannot be
    evidence that a deletion worked."""
    def answers(dirs):
        return [7, -1, 7]                       # query 1 never says it, with or without ablation

    wc = workspace_closure(answers, candidates=[9], obj_id=7, n_queries=3, max_dirs=2)
    assert wc.exhausted                         # 9 does nothing, so the search runs out
    assert wc.size == 1
    assert wc.n_queries == 3                    # but only two of them were ever evidence


def test_a_fact_the_model_does_not_produce_at_all_has_an_empty_closure():
    wc = workspace_closure(lambda dirs: [-1, -1], candidates=[1, 2], obj_id=7, n_queries=2)
    assert wc.size == 0 and wc.optimal and wc.lower_bound == 0


def test_the_search_reports_exhaustion_rather_than_a_number_it_did_not_reach():
    def answers(dirs):
        return [7]                              # nothing ever removes it

    wc = workspace_closure(answers, candidates=[1, 2, 3, 4], obj_id=7, n_queries=1, max_dirs=3)
    assert wc.exhausted and wc.size == 3 and not wc.optimal
    assert "SEARCH EXHAUSTED" in wc.summary()


def test_the_collateral_is_carried_beside_the_closure_and_printed_with_it():
    """A closure of one is worthless if the direction removed was carrying everything."""
    def answers(dirs):
        return [7 if 7 not in dirs else -1]

    wc = workspace_closure(answers, candidates=[7], obj_id=7, n_queries=1,
                           collateral_with=lambda dirs: 0.30 if dirs else 0.95)
    assert wc.collateral == pytest.approx(0.30) and wc.collateral_before == pytest.approx(0.95)
    assert "collateral 0.3000 from 0.9500" in wc.summary()


def test_the_lens_is_named_in_the_result_because_two_of_them_mean_different_things():
    wc = workspace_closure(lambda d: [-1], candidates=[1], obj_id=7, n_queries=1,
                           lens="tuned (fitted A)")
    assert "tuned (fitted A)" in wc.summary()


def test_the_tuned_lens_fit_moves_the_readout_towards_the_target():
    """It has to do something, or reporting it as a different lens would be decoration."""
    torch.manual_seed(0)
    w = _basis()
    h = torch.randn(64, D)
    a_true = torch.randn(D, D) * 0.3
    target = lens_logits(h, w, None, a_true)
    before = torch.nn.functional.kl_div(torch.log_softmax(lens_logits(h, w), -1),
                                        torch.log_softmax(target, -1), log_target=True,
                                        reduction="batchmean")
    a = fit_tuned_lens(h, target, w, steps=250, lr=0.05)
    after = torch.nn.functional.kl_div(torch.log_softmax(lens_logits(h, w, None, a), -1),
                                       torch.log_softmax(target, -1), log_target=True,
                                       reduction="batchmean")
    assert float(after) < float(before) * 0.5
