"""Shaping a carrier: the arithmetic that bounds it, and the two losses that reach for it.

The privacy loss makes a claim a theorem can check -- n unit vectors in d dimensions cannot beat the
Welch bound -- so most of what is asserted here is that the code agrees with that theorem rather than
with an intuition.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from so.carrier import ablate_carrier, ablation_loss, carriers, privacy_loss, welch_bound
from so.model import ModelConfig, MutableKnowledgeTransformer


def _model(n_entities=64, d_model=64):
    cfg = ModelConfig(n_entities=n_entities, n_relations=4, d_model=d_model, marker_dim=16,
                      n_core_layers=1, n_heads=2)
    torch.manual_seed(0)
    return MutableKnowledgeTransformer(cfg)


def test_the_welch_bound_is_zero_exactly_when_orthogonality_is_possible():
    assert welch_bound(64, 128) == 0.0          # fewer vectors than dimensions: exact, no floor
    assert welch_bound(128, 128) == 0.0
    assert welch_bound(256, 768) == 0.0         # GPT-2's space holds 256 objects orthogonally
    assert welch_bound(256, 128) == pytest.approx(math.sqrt(128 / (128 * 255)), rel=1e-9)
    assert welch_bound(256, 128) == pytest.approx(0.0626, abs=5e-4)


def test_the_bound_rises_as_the_space_gets_more_crowded():
    assert welch_bound(1000, 128) > welch_bound(256, 128) > 0.0


def test_an_orthonormal_set_costs_the_privacy_loss_nothing():
    """The loss must be zero where the theorem says nothing more is available."""
    m = _model(n_entities=16, d_model=32)
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(32, 16))
        m.v_fwd.weight.copy_(torch.eye(32))
        m.ent_emb.weight.copy_(q.t())
    loss, st = privacy_loss(m)
    assert float(loss) < 1e-10
    assert st["welch_bound"] == 0.0 and st["coherence_max"] < 1e-5


def test_a_collapsed_set_costs_it_a_lot_and_the_loss_reduces_the_coherence():
    m = _model(n_entities=16, d_model=32)
    with torch.no_grad():                      # every object on one direction: maximal coherence
        m.ent_emb.weight.copy_(torch.ones(16, 32) + 0.01 * torch.randn(16, 32))
    before, st0 = privacy_loss(m)
    assert float(before) > 0.1 and st0["coherence_max"] > 0.9
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss, _ = privacy_loss(m)
        loss.backward()
        opt.step()
    after, st1 = privacy_loss(m)
    assert float(after) < float(before) * 0.2
    assert st1["coherence_max"] < st0["coherence_max"] - 0.3


def test_the_loss_is_hinged_so_it_never_asks_for_more_than_the_theorem_allows():
    """256 objects in 128 dimensions cannot go below 0.0626, and the loss must not push there."""
    m = _model(n_entities=256, d_model=128)
    loss, st = privacy_loss(m)
    assert st["welch_bound"] == pytest.approx(0.0626, abs=5e-4)
    # a set exactly AT the bound contributes nothing, whatever its size
    with torch.no_grad():
        v = carriers(m)
        u = F.normalize(v, dim=-1)
        g = u @ u.t()
        off = (g - torch.diag(torch.diagonal(g))).abs()
    hinged = ((off - st["welch_bound"]).clamp(min=0.0) ** 2).sum() / (256 * 255)
    assert float(loss) == pytest.approx(float(hinged), rel=1e-5)


def test_ablating_a_carrier_removes_that_direction_and_leaves_the_orthogonal_part():
    torch.manual_seed(0)
    c = torch.randn(1, 16)
    h = 3.0 * F.normalize(c, dim=-1) + torch.randn(1, 16) * 0.1
    out = ablate_carrier(h, c)
    assert float((out * F.normalize(c, dim=-1)).sum().abs()) < 1e-5
    assert float(out.norm()) > 0.0             # the rest of the state survives


def test_the_ablation_loss_reports_whether_the_answer_survived_the_removal():
    """The number that says whether the carrier was where the fact was."""
    m = _model()
    h = torch.randn(8, m.cfg.d_model)
    target = torch.randint(0, m.cfg.n_entities, (8,))
    loss, st = ablation_loss(m, h, target, m.cfg.n_entities)
    assert 0.0 <= st["answer_survives_ablation"] <= 1.0
    assert float(loss) > 0.0


def test_the_ablation_loss_pushes_the_answer_to_unknown():
    """Training against it must move the survival rate down, or it is decoration."""
    m = _model()
    torch.manual_seed(1)
    h = torch.randn(32, m.cfg.d_model)
    target = torch.randint(0, m.cfg.n_entities, (32,))
    _, st0 = ablation_loss(m, h, target, m.cfg.n_entities)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(150):
        opt.zero_grad()
        loss, _ = ablation_loss(m, h, target, m.cfg.n_entities)
        loss.backward()
        opt.step()
    _, st1 = ablation_loss(m, h, target, m.cfg.n_entities)
    assert st1["answer_survives_ablation"] <= st0["answer_survives_ablation"]
    loss1, _ = ablation_loss(m, h, target, m.cfg.n_entities)
    assert float(loss1) < 1.0


def test_carriers_come_from_the_projection_and_not_from_the_embedding():
    """Shaping the embedding would leave v_fwd free to undo it, so the loss acts on what is read."""
    m = _model()
    v = carriers(m)
    assert v.shape == (m.cfg.n_entities, m.cfg.d_model)
    assert torch.allclose(v, m.v_fwd(m.ent_emb.weight))
    assert not torch.allclose(v, m.ent_emb.weight)
