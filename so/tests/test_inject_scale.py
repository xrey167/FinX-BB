"""The injected read at a fraction of its size: the dose axis, which must not move a record.

``KnowledgeAdapterLM.set_inject_scale`` exists because an audit that reports "no residual trace after
deletion" is asserting something it has never measured: that had a residue been present, it would have
been seen. To measure that, a residue must be supplied in a KNOWN AMOUNT, and until this instrument
there were exactly two amounts available -- the trained write, and ``set_inject_projection(None,
"zero")``, which is none of it.

The properties the ladder has to have, or the rungs on it mean nothing:

  1. OFF BY DEFAULT AND EXACT. With no scale set the forward is bit-identical to the trained one, so
     every recorded number stands.
  2. THE ENDPOINTS ARE THE ARMS ALREADY IN USE. ``alpha=1`` is the trained write and ``alpha=0`` is the
     'zero' mode of ``set_inject_projection`` -- the no-memory floor every capability row is read
     against. A ladder whose ends are not the two arms the records already use is a different
     experiment wearing their labels.
  3. THE RUNG IS THE WRITE, EXACTLY. At the first read site the injected vector is alpha times the
     trained one, to floating point. At LATER sites it is not, and the test says so rather than
     asserting a linearity the architecture does not have: the block-8 write is in the residual the
     block-10 read builds its routing query from (ledger 31.56), so scaling the earlier write changes
     which cell the later read routes to. The ladder is per-site for the same reason the projection
     arms are.
  4. IT COMPOSES WITH THE PROJECTION ARMS IN ONE ORDER ONLY. The scale is applied last, so a rung of a
     projected write is alpha times that projected write and not a projection of a scaled one.
  5. BAD INPUT IS REFUSED, not silently coerced: a negative alpha, a NaN, or a site that is not a read
     layer.
"""

import numpy as np
import pytest
import torch

from so.data import bank_from_world
from so.jlens import random_basis
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.train import make_centre
from so.world import World

transformers = pytest.importorskip("transformers")

N_ENT, UNK, D = 20, 5, 32


def _adapter(**kw):
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=D, n_layer=2, n_head=2)
    torch.manual_seed(0)
    lm = transformers.GPT2LMHeadModel(cfg).eval()
    torch.manual_seed(1)
    acfg = AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16, **kw)
    return KnowledgeAdapterLM(lm, acfg, list(range(10, 10 + N_ENT)), UNK).eval()


def _bank(seed=0):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, N_ENT, 4, 30, 2)
    return bank_from_world(rng, world, make_centre(seed, 16), 0.3, 0.2, 0.1).tensors()


def _prompt(B=4, T=6):
    g = torch.Generator().manual_seed(2)
    return (torch.randint(0, 64, (B, T), generator=g), torch.ones(B, T, dtype=torch.long),
            torch.full((B,), T - 1, dtype=torch.long))


def _run(m, b, alpha=None, layers=None):
    ids, am, last = _prompt()
    m.set_inject_scale(alpha, layers=layers)
    with torch.no_grad():
        cand, full, _, _ = m(b, ids, am, last)
    inj = m.last_injected.clone()
    m.set_inject_scale(None)
    return cand, full, inj


def test_no_scale_is_bit_identical_to_the_trained_forward():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    ids, am, last = _prompt()
    with torch.no_grad():
        base_cand, base_full, _, _ = m(b, ids, am, last)
    cand, full, _ = _run(m, b)
    assert torch.equal(cand, base_cand) and torch.equal(full, base_full)


def test_alpha_one_is_the_trained_forward_and_alpha_zero_is_the_no_memory_floor():
    """The ends of the ladder are the two arms the records already use, not new ones."""
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    base_cand, base_full, base_inj = _run(m, b)

    one_cand, one_full, one_inj = _run(m, b, alpha=1.0)
    assert torch.equal(one_cand, base_cand) and torch.equal(one_full, base_full)
    assert torch.equal(one_inj, base_inj)

    zero_cand, zero_full, zero_inj = _run(m, b, alpha=0.0)
    assert torch.allclose(zero_inj, torch.zeros_like(zero_inj), atol=0)

    ids, am, last = _prompt()
    m.set_inject_projection(None, "zero")
    with torch.no_grad():
        proj_cand, proj_full, _, _ = m(b, ids, am, last)
    m.set_inject_projection(None)
    assert torch.equal(zero_cand, proj_cand) and torch.equal(zero_full, proj_full)


def test_the_rung_is_exactly_alpha_times_the_write_at_the_first_read_site_only():
    """alpha scales the FIRST site's write exactly; the later site is not linear in alpha, and why.

    This is the ladder's half of what ``test_inject_projection`` found: with two read layers the
    block-0 write sits in the residual when the block-1 read computes its routing query, so scaling the
    first write moves the second read's CELL CHOICE and not merely its magnitude. The test asserts the
    exact relation where it holds and asserts that it FAILS at the later site, so a future refactor
    that made the sites independent would be caught here rather than silently changing what a rung is.
    """
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    _, _, full_inj = _run(m, b)
    moved_later = False
    for a in (0.25, 0.5, 0.75):
        _, _, inj = _run(m, b, alpha=a)
        assert torch.allclose(inj[:, 0], full_inj[:, 0] * a, atol=1e-6), f"first site not exact at alpha={a}"
        if not torch.allclose(inj[:, 1], full_inj[:, 1] * a, atol=1e-4):
            moved_later = True
    assert moved_later, ("the second site scaled linearly with alpha at every rung: the read layers "
                         "have become independent, and the per-site rule this ladder is written under "
                         "no longer applies")


def test_a_single_site_ladder_leaves_the_other_site_alone():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    _, _, full_inj = _run(m, b)
    _, _, inj = _run(m, b, alpha=0.5, layers=(0,))
    assert torch.allclose(inj[:, 0], full_inj[:, 0] * 0.5, atol=1e-6)
    _, _, inj1 = _run(m, b, alpha=0.5, layers=(1,))
    assert torch.equal(inj1[:, 0], full_inj[:, 0])


def test_the_scale_is_applied_after_the_projection_not_before():
    """A rung of a projected write, in that order, so 'keep' arms stay inside the span at every rung."""
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    basis = random_basis(D, 8, seed=3)
    ids, am, last = _prompt()

    m.set_inject_projection(basis, "keep")
    with torch.no_grad():
        m(b, ids, am, last)
    kept = m.last_injected.clone()

    m.set_inject_scale(0.4)
    with torch.no_grad():
        m(b, ids, am, last)
    scaled = m.last_injected.clone()
    m.set_inject_scale(None)
    m.set_inject_projection(None)
    assert torch.allclose(scaled[:, 0], kept[:, 0] * 0.4, atol=1e-6)


def test_bad_input_is_refused():
    m = _adapter(status_gated=True, use_links=True, n_deref=1)
    with pytest.raises(ValueError):
        m.set_inject_scale(-0.1)
    with pytest.raises(ValueError):
        m.set_inject_scale(float("nan"))
    with pytest.raises(ValueError):
        m.set_inject_scale(0.5, layers=(7,))
    assert getattr(m, "_inject_scale", None) is None, "a refused call must leave the instrument off"
