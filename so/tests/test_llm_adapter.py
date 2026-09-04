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


def _link_bank(seed=0, n_links=8):
    """A bank whose last ``n_links`` rows are aliases pointing at earlier fact rows."""
    b = _bank(seed, p_revoked=0.0, p_shred=0.0)
    n = b["subject"].shape[0]
    is_link = torch.zeros(n, dtype=torch.bool)
    is_link[-n_links:] = True
    b["is_link"] = is_link
    b["link_subject"] = b["subject"].clone()
    b["link_relation"] = b["relation"].clone()
    b["link_subject"][-n_links:] = b["subject"][:n_links]      # point at the first rows
    b["link_relation"][-n_links:] = b["relation"][:n_links]
    b["obj"] = torch.where(is_link, torch.zeros_like(b["obj"]), b["obj"])
    return b


@pytest.mark.parametrize("n_deref", [1, 2])
def test_dereference_slot_is_an_identity_at_initialisation(n_deref):
    """The whole point of the zero-initialised query and the size-aware passthrough bias.

    A dereference slot that is not an identity at the start injects the average of every cell into
    the frozen model from the first step; that is what collapsed the first two attempts at E-000020,
    where a checkpoint reading at 98% dropped to 0% (flat bias) and 14% (size-aware bias alone).
    """
    ids, am, last = _prompt()
    bank = _link_bank()
    linked = _adapter(use_links=True, n_deref=n_deref)
    with torch.no_grad():
        _, _, routing, _ = linked(bank, ids, am, last)
    # the dereference slots are the odd ones out of each (resolve, deref...) group; their last column
    # is the passthrough, and it must hold essentially all of the mass before training
    per_read = 1 + n_deref
    deref_slots = [r * per_read + 1 + d for r in range(2) for d in range(n_deref)]
    passthrough = routing[:, deref_slots, -1]
    assert float(passthrough.min()) > 0.99, f"passthrough only {float(passthrough.min()):.4f} at init"


def test_dereference_routing_has_one_slot_per_read_plus_one():
    ids, am, last = _prompt()
    _, _, routing, _ = _adapter(use_links=True, n_deref=1)(_link_bank(), ids, am, last)
    assert routing.shape[1] == 2 * (1 + 1)      # two read layers, one dereference each


def test_alias_row_value_differs_from_a_fact_row_value():
    bank = _link_bank()
    enc = _adapter(use_links=True, n_deref=1).encode_bank(bank)
    v = enc["values"]
    assert not torch.allclose(v[bank["is_link"]].mean(0), v[~bank["is_link"]].mean(0))
