"""Frozen-backbone-safe Jacobian-lens estimator, kept separate from historical jlens.py.

This is an audit instrument, NOT a deletion certificate or a new J-lens method.
Source indices denote INPUTS to decoder blocks: source=8 is before block 8,
source=9 is after block 8. Targets denote hidden_states entries. Only targets
strictly downstream of source and before the final normalization are accepted.

We estimate mean_{prompt, valid t <= t'} W_out[u] d h_target[t']/d h_source[t].
The causal decoder supplies the triangular zeros. Normalization is by the number
of valid causal position pairs, not by the number of batches or source tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FrozenJLens:
    token_ids: tuple[int, ...]
    mean_vectors: Tensor
    directions: Tensor
    raw_norms: Tensor
    source: int
    target: int
    n_prompts: int
    n_source_positions: int
    n_position_pairs: int

    def project(self, states: Tensor, *, unit_directions: bool = False) -> Tensor:
        """Raw linearized effect by default; direction-only scores are not logit ranks."""
        basis = self.directions if unit_directions else self.mean_vectors
        if states.shape[-1] != basis.shape[-1]:
            raise ValueError("state and lens hidden dimensions differ")
        return states @ basis.to(device=states.device, dtype=states.dtype).T


def decoder_blocks(lm: nn.Module):
    """Narrow decoder-family adapter; no downloads or remote code execution."""
    for path in ("transformer.h", "gpt_neox.layers", "model.layers", "transformer.blocks"):
        value = lm
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, (nn.ModuleList, list, tuple)) and len(value):
            return value
    raise TypeError("unsupported decoder: provide a model with an explicit block list")


def estimate_frozen_jlens(
    lm: nn.Module,
    source: int,
    token_ids: Sequence[int],
    input_ids: Tensor,
    attention_mask: Tensor,
    w_out: Tensor,
    *,
    target: int = -2,
    batch_size: int = 8,
) -> FrozenJLens:
    """Compute VJPs without unfreezing weights or populating parameter gradients.

    Run on an eval-mode, non-shared model instance with no active injection hooks.
    Each source input is replaced by a detached gradient-enabled leaf; only the
    downstream graph is needed. The temporary hook is removed even on failure.
    The model's standard hidden_states convention is explicitly checked.

    This diagnostic must run outside inference_mode (no_grad is supported).
    Accumulation uses float64, but downstream derivatives use the model dtype.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("J-lens audit requires inference_mode(False)")
    if any(module.training for module in lm.modules()):
        raise ValueError("call lm.eval() before estimating a deterministic J-lens")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or not input_ids.numel():
        raise ValueError("input_ids and attention_mask must be nonempty equal-shape matrices")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be integer token IDs")
    if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("attention_mask must be binary")
    lengths = attention_mask.sum(dim=1).to(torch.int64)
    if bool((lengths < 1).any()):
        raise ValueError("every prompt must have at least one valid token")
    ids = tuple(int(t) for t in token_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("token_ids must be nonempty and unique")
    if w_out.ndim != 2 or min(ids) < 0 or max(ids) >= w_out.shape[0]:
        raise ValueError("output embedding matrix or selected token IDs are invalid")
    if not bool(torch.isfinite(w_out[list(ids)]).all()):
        raise ValueError("selected output rows must be finite")
    blocks = decoder_blocks(lm)
    n_states = len(blocks) + 1
    target_index = target if target >= 0 else n_states + target
    if not (0 <= source < target_index < len(blocks)):
        raise ValueError("require 0 <= source < target < n_blocks; final-normalized state is excluded")
    device = next(lm.parameters()).device
    total = torch.zeros((len(ids), w_out.shape[1]), device=device, dtype=torch.float64)
    captured: list[Tensor] = []

    def mark_source(_module, args, kwargs):
        if captured:
            raise RuntimeError("source block executed more than once in one audit forward")
        positional = bool(args)
        h = args[0] if positional else kwargs.get("hidden_states")
        if not torch.is_tensor(h) or not h.is_floating_point():
            raise TypeError("source block must receive a floating-point hidden_states tensor")
        leaf = h.detach().requires_grad_(True)
        captured.append(leaf)
        if positional:
            return (leaf, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": leaf}

    handle = blocks[source].register_forward_pre_hook(mark_source, with_kwargs=True)
    try:
        with torch.enable_grad():
            for start in range(0, input_ids.shape[0], batch_size):
                captured.clear()
                chunk = input_ids[start:start + batch_size].to(device)
                mask = attention_mask[start:start + batch_size].to(device)
                out = lm(input_ids=chunk, attention_mask=mask, output_hidden_states=True,
                         use_cache=False, return_dict=True)
                if len(captured) != 1:
                    raise RuntimeError("source block was not observed exactly once")
                hidden_states = getattr(out, "hidden_states", None)
                if hidden_states is None or len(hidden_states) != n_states:
                    raise RuntimeError("unsupported hidden_states indexing convention")
                h = captured[0]
                z = hidden_states[target_index]
                if z.shape != h.shape or z.shape[-1] != w_out.shape[-1]:
                    raise ValueError("source/target/output-embedding dimensions do not match")
                if not z.requires_grad:
                    raise RuntimeError("target is detached from the gradient-enabled source")
                keep = mask.unsqueeze(-1).to(z.dtype)
                for j, token in enumerate(ids):
                    row = w_out[token].detach().to(device=device, dtype=z.dtype)
                    scalar = ((z * keep) @ row).sum()
                    grad, = torch.autograd.grad(scalar, h, retain_graph=j + 1 < len(ids))
                    contribution = (grad * keep).sum(dim=(0, 1)).to(torch.float64)
                    if not bool(torch.isfinite(contribution).all()):
                        raise FloatingPointError("non-finite J-lens derivative")
                    total[j] += contribution
    finally:
        handle.remove()
        captured.clear()
    pair_count = int((lengths * (lengths + 1) // 2).sum())
    means = total / pair_count
    norms = means.norm(dim=-1)
    directions = means / norms.clamp_min(torch.finfo(means.dtype).tiny).unsqueeze(-1)
    return FrozenJLens(ids, means.detach(), directions.detach(), norms.detach(), source,
                       target_index, input_ids.shape[0], int(lengths.sum()), pair_count)
