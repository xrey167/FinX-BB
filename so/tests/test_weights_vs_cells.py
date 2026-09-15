"""Mechanics of E-000024's weights arm, on a tiny randomly initialised GPT-2 (no download).

The head-to-head is only fair if three things hold, so they are pinned here rather than assumed:
the LoRA is a no-op at initialisation, its state round-trips exactly (both unlearning modes start
from the same trained weights), and the knowledge adapter is genuinely bypassed when the weights
arm is scored — otherwise the "weights" arm would be quietly reading cells.
"""
import numpy as np
import pytest
import torch

from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM

transformers = pytest.importorskip("transformers")

from so.experiments.e000024_weights_vs_cells import (  # noqa: E402
    LoRAConv1D, attach_lora, load_lora_state, lora_delta_norm, lora_state,
)

N_ENT, UNK = 20, 5


def _lm():
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def _model():
    torch.manual_seed(1)
    cfg = AdapterConfig(read_layers=(0, 1), d_key=16, marker_dim=16)
    return KnowledgeAdapterLM(_lm(), cfg, list(range(10, 10 + N_ENT)), UNK).eval()


def _prompt(B=3, T=6):
    g = torch.Generator().manual_seed(2)
    return torch.randint(0, 64, (B, T), generator=g), torch.ones(B, T, dtype=torch.long)


def test_lora_is_a_no_op_at_initialisation():
    m = _model()
    ids, am = _prompt()
    with torch.no_grad():
        before = m.lm(input_ids=ids, attention_mask=am).logits.clone()
    attach_lora(m.lm, rank=4)
    with torch.no_grad():
        after = m.lm(input_ids=ids, attention_mask=am).logits
    assert torch.allclose(before, after, atol=1e-6)
    assert lora_delta_norm(m.lm) == 0.0


def test_lora_state_round_trips_exactly():
    m = _model()
    params = attach_lora(m.lm, rank=4)
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.01)
    trained = lora_state(m.lm)
    norm = lora_delta_norm(m.lm)
    assert norm > 0.0
    with torch.no_grad():
        for p in params:
            p.add_(torch.randn_like(p) * 0.5)          # a second "unlearning run" moves them
    assert lora_delta_norm(m.lm) != pytest.approx(norm)
    load_lora_state(m.lm, trained)                      # ... and the arm is reset before the next one
    assert lora_delta_norm(m.lm) == pytest.approx(norm, rel=1e-6)
    for n, p in m.lm.named_parameters():
        if n in trained:
            assert torch.equal(p.detach(), trained[n])


def test_pretrained_weights_stay_frozen_under_a_lora_step():
    m = _model()
    params = attach_lora(m.lm, rank=4)
    base = {n: p.detach().clone() for n, p in m.lm.named_parameters() if not n.endswith((".a", ".b"))}
    assert all(not p.requires_grad for n, p in m.lm.named_parameters() if not n.endswith((".a", ".b")))
    ids, am = _prompt()
    opt = torch.optim.AdamW(params, lr=1e-2)
    out = m.lm(input_ids=ids, attention_mask=am, labels=ids)
    out.loss.backward()
    opt.step()
    for n, p in m.lm.named_parameters():
        if n in base:
            assert torch.equal(p.detach(), base[n]), n
    assert lora_delta_norm(m.lm) > 0.0


def test_scoring_the_weights_arm_bypasses_the_knowledge_adapter():
    """With no bank in context the adapter hook must return None, so the LM is the bare model."""
    m = _model()
    ids, am = _prompt()
    bare = _lm()
    bare.load_state_dict({k: v for k, v in m.lm.state_dict().items()})
    with torch.no_grad():
        a = m.lm(input_ids=ids, attention_mask=am).logits
        b = bare(input_ids=ids, attention_mask=am).logits
    assert torch.allclose(a, b, atol=1e-6)
    assert m._ctx is None


def test_wrapped_layers_are_lora_layers():
    m = _model()
    attach_lora(m.lm, rank=4)
    for block in m.lm.transformer.h:
        for mod in (block.attn.c_attn, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
            assert isinstance(mod, LoRAConv1D)
