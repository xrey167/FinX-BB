"""The J-lens must give the same vectors on a frozen core as on a trainable one.

E-000063's CI run (33955376015, both seeds) trained its adapter for 2000 steps and then died in
``jlens_vectors`` with "element 0 of tensors does not require grad": ``KnowledgeAdapterLM`` freezes
every core parameter, a forward from ``input_ids`` then builds no graph, and the VJP has nothing to
differentiate. The lens is a derivative with respect to the hidden state, so the fix is a leaf at the
embeddings; this test pins that the frozen and trainable paths agree exactly.
"""

import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore")


def _gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained("gpt2")
    lm.eval()
    return tok, lm


def test_jlens_frozen_equals_trainable():
    from so.jlens import jlens_vectors
    tok, lm = _gpt2()
    enc = tok(["The capital of France is Paris.", "A train crossed the bridge."], return_tensors="pt", padding=True)
    w = lm.get_output_embeddings().weight
    ids = [tok.encode(" Paris")[0], tok.encode(" Tokyo")[0]]
    trainable = jlens_vectors(lm, 9, ids, enc["input_ids"], enc["attention_mask"], w)
    for p in lm.parameters():
        p.requires_grad_(False)
    frozen = jlens_vectors(lm, 9, ids, enc["input_ids"], enc["attention_mask"], w)
    assert frozen.n_prompts == 2 and frozen.n_positions == int(enc["attention_mask"].sum())
    assert torch.equal(trainable.vectors, frozen.vectors)
    assert torch.equal(trainable.raw_norms, frozen.raw_norms)
    assert float(frozen.vectors.norm(dim=1).sub(1).abs().max()) < 1e-5
