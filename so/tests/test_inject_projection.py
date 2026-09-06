"""The injected read, restricted to a subspace: an evaluation-only instrument that must not move a record.

``KnowledgeAdapterLM.set_inject_projection`` exists so an experiment can ask what a linear
interpretability basis (the J-lens family, the logit lens, an SAE dictionary) accounts for in a memory
write that is RMS-matched to the residual stream. The three properties an instrument like this has to
have, or the arms it produces mean nothing:

  1. OFF BY DEFAULT AND EXACT. With no projection set, the forward is bit-identical to the trained one,
     so every recorded number stands.
  2. THE ARMS PARTITION THE WRITE, PER READ SITE. keep + drop is the write exactly at the first read
     site; at later sites the arms have already diverged, because the earlier write is in the residual
     the later read builds its routing query from. If the basis were not orthonormal the two arms would
     overlap even at one site, which is why ``set_inject_projection`` refuses a basis whose Gram matrix
     is not the identity.
  3. ZERO IS THE NO-MEMORY FLOOR. The 'zero' mode must reproduce the model with no memory at all, which
     is the floor every capability row is read against.
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


def _run(m, b, arm=None, layers=None):
    ids, am, last = _prompt()
    if arm is None:
        m.set_inject_projection(None)
    else:
        m.set_inject_projection(arm[0], arm[1], layers=layers)
    with torch.no_grad():
        cand, full, _, _ = m(b, ids, am, last)
    inj = m.last_injected.clone()
    m.set_inject_projection(None)
    return cand, full, inj


def test_no_projection_is_bit_identical_to_the_trained_forward():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    ids, am, last = _prompt()
    with torch.no_grad():
        base_cand, base_full, _, _ = m(b, ids, am, last)
    cand, full, _ = _run(m, b)
    assert torch.equal(cand, base_cand) and torch.equal(full, base_full)


def test_keep_and_drop_partition_the_write_at_the_first_read_site():
    """keep + drop is the write EXACTLY at the first read site, and only there.

    Found by this test rather than assumed: with two read layers the arms stop being a decomposition
    after the first injection, because the block-8 write is in the residual when the block-10 read
    computes its own routing query (so/llm_adapter.py, the hook reads ``h[ar, last_idx]`` of the
    block it is attached to). Restricting the first write therefore changes WHICH CELL the second read
    routes to, not merely how that read is projected. An experiment that reports a keep/drop
    decomposition of "the write" over a multi-site reader is reporting two different trajectories; the
    decomposition is per-site, and this is the reason.
    """
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    basis = random_basis(D, 8, seed=3)
    _, _, full_inj = _run(m, b)
    _, _, keep = _run(m, b, (basis, "keep"))
    _, _, drop = _run(m, b, (basis, "drop"))
    assert keep.shape == full_inj.shape
    assert torch.allclose(keep[:, 0] + drop[:, 0], full_inj[:, 0], atol=1e-5)
    assert float(keep[:, 0].norm()) > 1e-3 and float(drop[:, 0].norm()) > 1e-3
    # the second site is a different trajectory, not a projection of the same one
    assert not torch.allclose(keep[:, 1] + drop[:, 1], full_inj[:, 1], atol=1e-4)


def test_with_one_read_site_the_partition_is_exact_everywhere():
    """The control for the test above: with a single read site there is no downstream site to disturb."""
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=D, n_layer=2, n_head=2)
    torch.manual_seed(0)
    lm = transformers.GPT2LMHeadModel(cfg).eval()
    torch.manual_seed(1)
    single = KnowledgeAdapterLM(lm, AdapterConfig(read_layers=(1,), d_key=16, marker_dim=16,
                                                  status_gated=True, use_links=True, n_deref=1),
                                list(range(10, 10 + N_ENT)), UNK).eval()
    b = _bank()
    basis = random_basis(D, 8, seed=3)
    _, _, full_inj = _run(single, b)
    _, _, keep = _run(single, b, (basis, "keep"))
    _, _, drop = _run(single, b, (basis, "drop"))
    assert torch.allclose(keep + drop, full_inj, atol=1e-5)


def test_renorm_preserves_the_write_magnitude_and_only_changes_direction():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    basis = random_basis(D, 8, seed=4)
    _, _, full_inj = _run(m, b)
    _, _, keep_r = _run(m, b, (basis, "keep_renorm"))
    # at the first site, where the two arms see the same upstream state, the norm is preserved exactly
    assert torch.allclose(keep_r[:, 0].norm(dim=-1), full_inj[:, 0].norm(dim=-1), atol=1e-4)
    cos = torch.nn.functional.cosine_similarity(keep_r[:, 0], full_inj[:, 0], dim=-1)
    assert float(cos.max()) < 0.999   # the direction did move


def test_zero_mode_is_the_no_memory_forward():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    ids, am, last = _prompt()
    with torch.no_grad():
        none_cand, none_full, _, _ = m(None, ids, am, last)
    cand, full, inj = _run(m, b, (None, "zero"))
    assert float(inj.abs().max()) == 0.0
    assert torch.allclose(cand, none_cand, atol=1e-6) and torch.allclose(full, none_full, atol=1e-6)


def test_a_single_site_arm_leaves_the_other_site_untouched():
    """Per-site arms: restricting the write at the LAST read layer cannot change an earlier one."""
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    basis = random_basis(D, 8, seed=5)
    _, _, full_inj = _run(m, b)
    _, _, late = _run(m, b, (basis, "keep"), layers=(1,))
    assert torch.equal(late[:, 0], full_inj[:, 0])          # site 0 is the trained write, bit for bit
    assert not torch.allclose(late[:, 1], full_inj[:, 1])   # site 1 was restricted
    with pytest.raises(ValueError, match="not read layers"):
        m.set_inject_projection(basis, "keep", layers=(7,))


def test_a_per_layer_basis_mapping_is_accepted():
    m, b = _adapter(status_gated=True, use_links=True, n_deref=1), _bank()
    per = {0: random_basis(D, 8, seed=6), 1: random_basis(D, 12, seed=7)}
    m.set_inject_projection(per, "keep")
    ids, am, last = _prompt()
    with torch.no_grad():
        m(b, ids, am, last)
    inj = m.last_injected.clone()
    m.set_inject_projection(None)
    assert torch.isfinite(inj).all()


def test_a_non_orthonormal_basis_is_refused():
    m = _adapter()
    bad = torch.randn(D, 6)          # columns neither unit nor orthogonal
    with pytest.raises(ValueError, match="orthonormal"):
        m.set_inject_projection(bad, "keep")
    with pytest.raises(ValueError, match="mode must be"):
        m.set_inject_projection(random_basis(D, 4, 0), "sideways")


def test_random_basis_keeps_the_expected_share_of_a_random_vector():
    """The null's calibration: an r-dimensional subspace holds r/d of a random vector's mass."""
    d, r = 128, 32
    b = random_basis(d, r, seed=7)
    g = torch.randn(2000, d, generator=torch.Generator().manual_seed(8))
    g = g / g.norm(dim=1, keepdim=True)
    share = float((((g @ b) ** 2).sum(1)).mean())
    assert abs(share - r / d) < 0.02
