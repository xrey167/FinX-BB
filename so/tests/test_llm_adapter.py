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


def _injection_norm(adapter, bank, ids, am, last):
    """How far the adapter moves the frozen model's last-token logits."""
    with torch.no_grad():
        base = adapter(None, ids, am, last)[1]
        out = adapter(bank, ids, am, last)[1]
    return float((out - base).abs().mean())


def test_match_gate_actually_reduces_the_injection():
    """It did not, before the injection path was restructured.

    The gate multiplied the read and the RMS match then divided by the RMS of that same gated read,
    so the factor cancelled exactly and E-000018's gate arm measured no effect at all. The
    normaliser is now taken from the ungated read.
    """
    ids, am, last = _prompt()
    bank = _bank()
    a = _adapter(match_gate=True)
    with torch.no_grad():
        a.match_tau.fill_(0.99)          # nothing can match: the gate should close
        a.match_temp.fill_(50.0)
    closed = _injection_norm(a, bank, ids, am, last)
    with torch.no_grad():
        a.match_tau.fill_(-0.99)         # everything matches: the gate should be open
    open_ = _injection_norm(a, bank, ids, am, last)
    assert closed < 0.2 * open_, f"closed gate injects {closed:.4f} against an open {open_:.4f}"


def test_two_channel_null_can_silence_the_refusal_channel():
    ids, am, last = _prompt()
    bank = _bank()
    a = _adapter(two_channel_null=True)
    with torch.no_grad():                # force the relevance score to zero and to one
        for m in a.query_relevance.values():
            m[-1].bias.fill_(-20.0)
    silent = _injection_norm(a, bank, ids, am, last)
    with torch.no_grad():
        for m in a.query_relevance.values():
            m[-1].bias.fill_(20.0)
    loud = _injection_norm(a, bank, ids, am, last)
    assert silent != pytest.approx(loud, rel=1e-3), "the relevance channel changes nothing"


def test_ungated_adapter_is_unchanged_by_the_restructuring():
    """With both channels off the read is cell + null again, so nothing recorded earlier moves."""
    ids, am, last = _prompt()
    bank = _bank()
    a = _adapter()
    with torch.no_grad():
        cand, full, _, _ = a(bank, ids, am, last)
    assert torch.isfinite(cand).all() and torch.isfinite(full).all()
    assert _injection_norm(a, bank, ids, am, last) > 0


# --------------------------------------------------------------------- tied vs untied output embeddings

def _untied_lm():
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2,
                                  tie_word_embeddings=False)
    torch.manual_seed(0)
    lm = transformers.GPT2LMHeadModel(cfg).eval()
    with torch.no_grad():                                  # make the two matrices genuinely unrelated
        lm.get_output_embeddings().weight.copy_(torch.randn_like(lm.get_output_embeddings().weight) * 0.05)
    return lm


def test_gpt2_ties_its_embeddings_so_the_two_matrices_are_one():
    m = _adapter()
    assert m.ties_embeddings
    assert torch.equal(m.w_in, m.w_out)
    assert torch.equal(m.wte, m.w_in)          # the retained name still means the input side


def test_the_payload_comes_from_the_output_embedding_when_they_differ():
    """The value must raise the object's logit at the LM head, so it is built from the head's rows.

    GPT-2 ties the two matrices, which let the layer be written against `wte` alone and still work.
    Llama, Qwen, OLMo and Pythia do not tie them; there a payload built from the INPUT embedding
    raises nothing at the head, and the mechanism would read near chance for a reason that has
    nothing to do with the architecture being tested.
    """
    cfg = AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16)
    torch.manual_seed(1)
    m = KnowledgeAdapterLM(_untied_lm(), cfg, list(range(10, 10 + N_ENT)), UNK).eval()
    assert not m.ties_embeddings
    assert not torch.equal(m.w_in, m.w_out)
    b = _bank()
    with torch.no_grad():
        payload = m.v_proj(m.w_out[m.entity_token_ids[b["obj"]]])
        enc = m.encode_bank(b)
    # values are payload*g + unk*(1-g); where the gate is fully open the value IS the payload
    open_gate = enc["gate"] > 0.999
    if open_gate.any():
        assert torch.allclose(enc["values"][open_gate], payload[open_gate], atol=1e-5)
    wrong = m.v_proj(m.w_in[m.entity_token_ids[b["obj"]]])
    assert not torch.allclose(payload, wrong, atol=1e-3)    # the two really are different directions


def test_the_null_value_is_an_output_direction_on_an_untied_model():
    cfg = AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16)
    torch.manual_seed(1)
    lm = _untied_lm()
    m = KnowledgeAdapterLM(lm, cfg, list(range(10, 10 + N_ENT)), UNK).eval()
    assert torch.allclose(m.null_value[0], lm.get_output_embeddings().weight[UNK], atol=1e-6)


# ------------------------------------------- the key channel, in the frozen-LM adapter (E-000028 scope)

def test_the_adapter_key_carries_no_object_so_shredding_hides_it_from_routing():
    """E-000028's leak cannot occur here, and this is why rather than an assertion from reading.

    In the synthetic model the reverse key is k_rev(LN(object + relation)) and is never gated, so a
    shredded cell's key still names the object and a candidate sweep recovers it. The adapter's key is
    k_proj(ln_key(subject + relation)) and contains no object at all: two banks that differ only in the
    objects of gate-closed cells must be indistinguishable to every routing distribution the layer
    produces.
    """
    m = _adapter(status_gated=True)
    with torch.no_grad():                              # an untrained gate sits near 0.5 for any marker
        m.marker_gate[-1].weight.zero_(); m.marker_gate[-1].bias.fill_(-20.0)
    b = _bank(p_revoked=0.0, p_shred=1.0)              # every cell shredded: the gate is shut everywhere
    assert float(m.gate(b["marker"]).max()) < 1e-6
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["obj"] = (b2["obj"] + 1) % N_ENT
    ids, am, last = _prompt()
    with torch.no_grad():
        enc, enc2 = m.encode_bank(b), m.encode_bank(b2)
        assert torch.equal(enc["keys"], enc2["keys"])          # the keys do not move with the object
        _, _, r1, _ = m(b, ids, am, last)
        _, _, r2, _ = m(b2, ids, am, last)
    assert torch.allclose(r1, r2, atol=1e-6)


def test_an_active_cell_object_does_move_the_adapter_values_but_not_its_keys():
    """The complement, so the test above cannot pass by the bank being ignored altogether."""
    m = _adapter(status_gated=True)
    with torch.no_grad():
        m.marker_gate[-1].weight.zero_(); m.marker_gate[-1].bias.fill_(20.0)
    b = _bank(p_revoked=0.0, p_shred=0.0)              # every marker valid: the gate is open
    assert float(m.gate(b["marker"]).min()) > 1 - 1e-6
    b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
    b2["obj"] = (b2["obj"] + 1) % N_ENT
    with torch.no_grad():
        enc, enc2 = m.encode_bank(b), m.encode_bank(b2)
    assert torch.equal(enc["keys"], enc2["keys"])
    assert not torch.allclose(enc["values"], enc2["values"], atol=1e-5)
