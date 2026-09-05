"""The J-lens: the directions at a layer that a token's final logit is actually a function of.

Anthropic's workspace paper (Gurnee, Sofroniew, ... Lindsey, "Verbalizable Representations Form a
Global Workspace in Language Models", Transformer Circuits, 6 July 2026) defines

    J_l = E_{t, t' >= t, prompt} [ d h_final,t' / d h_l,t ]

and calls **the rows of W_U J_l** the J-lens vectors: "each J-lens vector is a direction in
residual-stream space associated with a single token". The readout is
``lens(h_l) = softmax(W_U norm(J_l h_l))``.

WHY THIS MODULE EXISTS RATHER THAN A LOGIT LENS. E-000042's first run used rows of ``W_U`` -- the
logit lens, which is the J = I special case -- and went VOID: removing every one of eight top lens
directions did not stop GPT-2 answering a single fact. That is the paper's own point about why the
J-lens beats the logit lens, arriving as a failed experiment. A direction that merely CORRELATES with
a token at the output is not a direction the computation passes through; ablating it removes nothing.

WHY IT IS AFFORDABLE, which is the part worth stating. J_l is d x d and estimating all of it costs d
backward passes. But nothing here needs all of it. The J-lens vector for token u is

    v_u = J_l^T W_U[u],   and   (J_l^T w)_b = d (w . h_final) / d h_l,b

so v_u is ONE vector-Jacobian product: push ``w = W_U[u]`` back from the target layer to layer l. A
pool of twenty-four candidate tokens costs one forward and twenty-four backward passes, not 768. The
causal mask supplies ``t' >= t`` for free, since ``d h_final,t' / d h_l,t`` is identically zero below
the diagonal.

TWO DETAILS TAKEN FROM THE PAPER RATHER THAN GUESSED. The expectation is over source position t,
every later position t', and a corpus of prompts -- so the sum over t' is taken before the backward
pass and the sum over t is what the backward pass returns. And the target is the PENULTIMATE residual,
not the final one: "The default lens on Sonnet 4.5, used throughout the paper, computes d z_t' /
d h_l,t with z taken at the penultimate layer... including the last layer can sometimes increase the
number of noisy artifacts."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = ["JLens", "jlens_vectors", "jlens_logits"]


@dataclass
class JLens:
    """J-lens vectors for a fixed set of tokens at one layer, with the corpus they were estimated on.

    ``vectors`` is (n_tokens, d), unit-normalised. ``raw_norms`` keeps what the normalisation removed,
    because a token whose J-lens vector is tiny before normalising is one the layer barely reaches and
    a caller ranking directions should be able to see that.
    """

    token_ids: Tuple[int, ...] = ()
    vectors: Optional[torch.Tensor] = None
    raw_norms: Optional[torch.Tensor] = None
    layer: int = 0
    target_layer: int = -2
    n_prompts: int = 0
    n_positions: int = 0

    def summary(self) -> str:
        return (f"{len(self.token_ids)} J-lens vector(s) at layer {self.layer} -> target "
                f"{self.target_layer}, estimated over {self.n_prompts} prompt(s) and "
                f"{self.n_positions} position pair(s)")


@torch.enable_grad()
def jlens_vectors(lm, layer: int, token_ids: Sequence[int], input_ids: torch.Tensor,
                  attention_mask: torch.Tensor, w_out: torch.Tensor, target_layer: int = -2,
                  batch: int = 8) -> JLens:
    """``v_u = J_l^T W_U[u]`` for each token in ``token_ids``, averaged over the given prompts.

    One forward per batch and one backward per token per batch. ``input_ids`` and ``attention_mask``
    are the corpus the expectation is taken over -- the paper uses a thousand pretraining-like
    sequences and reports that "J-lens beats the logit lens and tuned lens baselines with as few as 10
    prompts", so a caller on a CPU can afford a real estimate.
    """
    ids = [int(t) for t in token_ids]
    dev = next(lm.parameters()).device
    acc = torch.zeros(len(ids), lm.config.n_embd if hasattr(lm.config, "n_embd") else w_out.shape[1],
                      dtype=torch.float32, device=dev)
    n_prompts, n_pos = 0, 0

    for start in range(0, input_ids.shape[0], batch):
        chunk = input_ids[start: start + batch].to(dev)
        mask = attention_mask[start: start + batch].to(dev)
        out = lm(input_ids=chunk, attention_mask=mask, output_hidden_states=True)
        h = out.hidden_states[layer]                       # (B, T, d), the state being differentiated
        z = out.hidden_states[target_layer]                # (B, T, d), the target the paper uses
        keep = mask.unsqueeze(-1).to(z.dtype)
        n_prompts += chunk.shape[0]
        n_pos += int(mask.sum())
        for j, tid in enumerate(ids):
            w = w_out[tid].detach().to(z.dtype)
            s = ((z * keep) @ w).sum()                     # sum over t' of w . z_t', valid positions
            g, = torch.autograd.grad(s, h, retain_graph=(j < len(ids) - 1))
            acc[j] += (g * keep).sum(dim=(0, 1)).to(torch.float32)   # sum over t, accumulate prompts

    norms = acc.norm(dim=1)
    vec = acc / norms.clamp(min=1e-12).unsqueeze(1)
    return JLens(tuple(ids), vec.detach(), norms.detach(), layer, target_layer, n_prompts, n_pos)


def jlens_logits(h: torch.Tensor, jl: JLens) -> torch.Tensor:
    """``h`` projected onto each J-lens vector: the lens readout, up to the model's own norm.

    The paper's readout applies the model's normalisation before the unembedding. This is the inner
    product only, which is what a SUPPORT search needs -- the pursuit is over directions, and a
    monotone per-row rescaling does not change which directions a state is made of.
    """
    return h.reshape(-1, h.shape[-1]) @ jl.vectors.t()
