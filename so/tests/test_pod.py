"""The pod objective: what each half rewards, and what it refuses to ask for."""

from __future__ import annotations

import numpy as np
import torch

from so.pod import fact_directions, pod_loss, pod_private, private_loss


def _states(dirs, per_fact, jitter=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows, ids = [], []
    for f, v in enumerate(dirs):
        for _ in range(per_fact):
            rows.append(v + jitter * torch.randn(v.shape[0], generator=g))
            ids.append(f)
    return torch.stack(rows), torch.as_tensor(ids)


def test_a_perfect_pod_has_no_spread_loss():
    d = 32
    dirs = [torch.eye(d)[0] * 3, torch.eye(d)[1] * 3, torch.eye(d)[2] * 3]
    x, ids = _states(dirs, 4)
    lp, _, stats = pod_private(x, ids)
    assert float(lp) < 1e-4
    assert stats["pod/spread"] < 1e-4


def test_phrasings_scattered_around_their_fact_cost_the_pod_term():
    d = 32
    dirs = [torch.eye(d)[0] * 3, torch.eye(d)[1] * 3, torch.eye(d)[2] * 3]
    tight, ids = _states(dirs, 4, jitter=0.0)
    loose, _ = _states(dirs, 4, jitter=2.0, seed=1)
    assert float(pod_private(loose, ids)[0]) > float(pod_private(tight, ids)[0])


def test_orthogonal_facts_pay_no_privacy_penalty():
    d = 32
    dirs = torch.eye(d)[:4]
    loss, bound = private_loss(dirs, centred=False)
    assert float(loss) < 1e-8
    assert bound == 0.0                     # n <= d: the Welch bound permits exact orthogonality


def test_collinear_facts_do_pay():
    d = 32
    dirs = torch.stack([torch.eye(d)[0], torch.eye(d)[0], torch.eye(d)[1]])
    assert float(private_loss(dirs, centred=False)[0]) > 0.1


def test_the_penalty_is_hinged_so_it_never_asks_for_the_impossible():
    """More directions than dimensions: exact orthogonality does not exist and is not demanded."""
    g = torch.Generator().manual_seed(0)
    d, n = 8, 40
    x = torch.randn(n, d, generator=g)
    dirs = x / x.norm(dim=1, keepdim=True)
    loss, bound = private_loss(dirs, centred=False)
    assert bound > 0.0
    scaled = dirs * 1.0
    at_bound = torch.zeros(n, d)
    at_bound[:, 0] = 1.0                    # fully collinear, far above the bound
    assert float(private_loss(at_bound, centred=False)[0]) > float(loss)


def test_directions_are_computed_on_centred_states():
    """A shared offset is the common mode, not the fact, and must not drive the cosine.

    Three facts, not two: two CENTRED directions are always exactly antipodal, which is arithmetic
    and not a property of the states.
    """
    d = 16
    base = torch.ones(d) * 10.0
    dirs = [base + torch.eye(d)[i] for i in range(3)]
    x, ids = _states(dirs, 3)
    _, v = fact_directions(x, ids)
    off = [float((v[i] * v[j]).sum().abs()) for i in range(3) for j in range(i + 1, 3)]
    assert max(off) < 0.9                     # not driven to 1 by the shared offset


def test_two_centred_directions_are_antipodal_and_the_hinge_knows_it():
    """The floor no objective can beat: n centred vectors sum to zero, so at n = 2 the cosine is 1."""
    d = 16
    base = torch.ones(d) * 10.0
    x, ids = _states([base + torch.eye(d)[0], base + torch.eye(d)[1]], 3)
    _, v = fact_directions(x, ids)
    assert float((v[0] * v[1]).sum()) < -0.99
    loss, bound = private_loss(v)
    assert bound == 1.0                        # 1/(n-1) at n = 2
    assert float(loss) < 1e-8                  # and the hinge therefore asks for nothing


def test_the_centring_floor_shrinks_as_facts_are_added():
    g = torch.Generator().manual_seed(0)
    for n, expect in ((2, 1.0), (5, 0.25), (17, 0.0625)):
        v = torch.randn(n, 64, generator=g)
        v = v / v.norm(dim=1, keepdim=True)
        assert abs(private_loss(v)[1] - expect) < 1e-9


def test_stats_report_the_bound_and_the_excess_over_it():
    d = 16
    dirs = torch.stack([torch.eye(d)[0], torch.eye(d)[1]])
    x, ids = _states([dirs[0] * 3, dirs[1] * 3], 3)
    _, _, stats = pod_private(x, ids)
    assert "pod/welch_bound" in stats and "pod/excess_coherence" in stats
    assert stats["pod/excess_coherence"] <= stats["pod/coherence_mean"] + 1e-9
