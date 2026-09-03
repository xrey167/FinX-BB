"""Adapter semantics on a tiny randomly initialised GPT-2 (no download, no pretrained weights)."""
import numpy as np
import pytest
import torch

from so.data import bank_from_world
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.train import make_centre
from so.world import World

transformers = pytest.importorskip("transformers")

N_ENT, UNK = 20, 5


def _lm():
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def _adapter(**kw):
    cfg = AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16, **kw)
    torch.manual_seed(1)
    return KnowledgeAdapterLM(_lm(), cfg, list(range(10, 10 + N_ENT)), UNK).eval()


def _bank(seed=0, p_revoked=0.3, p_shred=0.2):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, N_ENT, 4, 30, 2)
    return bank_from_world(rng, world, make_centre(seed, 16), p_revoked, p_shred, 0.1).tensors()


def _prompt(B=4, T=6):
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 64, (B, T), generator=g)
    am = torch.ones(B, T, dtype=torch.long)
    return ids, am, torch.full((B,), T - 1, dtype=torch.long)


def test_unknown_mode_default_allowed_set_is_active():
    m = _adapter(); b = _bank()
    enc = m.encode_bank(b)
    assert torch.equal(enc["active"], b["active"])


def test_status_gated_unknown_mode_revoked_cell_reads_as_unknown_and_stays_routable():
    m = _adapter(status_gated=True); b = _bank()
    enc = m.encode_bank(b)
    assert torch.equal(enc["active"], b["routable"])
    inactive = ~b["active"]
    assert inactive.any()
    unk = m.v_proj(m.wte[m.candidate_ids[-1]][None])
    assert torch.allclose(enc["values"][inactive], unk.expand(int(inactive.sum()), -1), atol=1e-6)
    assert torch.all(enc["gate"][inactive] == 0)


def test_prior_mode_masked_bank_equals_base_model():
    m = _adapter(fallback="prior"); b = _bank()
    ids, am, last = _prompt()
    with torch.no_grad():
        base = m(None, ids, am, last)[1]
        masked = m(b, ids, am, last, cell_mask=torch.zeros(b["subject"].shape[0], dtype=torch.bool))[1]
    assert torch.allclose(base, masked, atol=1e-5)


def test_prior_mode_all_revoked_status_gated_equals_base_model():
    m = _adapter(fallback="prior", status_gated=True); b = _bank(p_revoked=1.0, p_shred=0.0)
    assert not b["active"].any()
    ids, am, last = _prompt()
    with torch.no_grad():
        base = m(None, ids, am, last)[1]
        _, out, routing, _ = m(b, ids, am, last)
    assert torch.allclose(base, out, atol=1e-5)
    assert routing is not None and routing.shape[-1] == b["subject"].shape[0] + 1


def test_prior_mode_active_bank_changes_logits_and_null_value_is_frozen():
    m = _adapter(fallback="prior"); b = _bank(p_revoked=0.0, p_shred=0.0)
    ids, am, last = _prompt()
    with torch.no_grad():
        base = m(None, ids, am, last)[1]
        out = m(b, ids, am, last)[1]
    assert not torch.allclose(base, out, atol=1e-3)
    assert not m.null_value.requires_grad and torch.all(m.null_value == 0)
    assert all(p.requires_grad for p in m.adapter_parameters())
    assert not any(p is m.null_value for p in m.adapter_parameters())


def test_unknown_mode_null_value_is_learnable_and_nonzero():
    m = _adapter()
    assert m.null_value.requires_grad and torch.any(m.null_value != 0)
    assert any(p is m.null_value for p in m.adapter_parameters())
