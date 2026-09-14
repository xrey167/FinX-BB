from __future__ import annotations

from dataclasses import dataclass

import torch

from research import r350b_qwen_learned_world_port_exact_mask as exact_mask

base = exact_mask.base

# Preserve exact-mask split functions from R350b.
base.PositionalContext = exact_mask.PositionalContext
base.prepare = exact_mask.prepare
base.run_layers = exact_mask.run_layers

_original_cache_pns = base.cache_pns


def cache_pns_normal(model, tok, rows, barrier):
    """Cross the inference->training boundary explicitly.

    The frozen-Qwen cache builder correctly runs under inference_mode, but PyTorch
    marks those tensors as inference tensors. R350 trains only the small residual
    materializer and therefore needs the cached Qwen activations as ordinary
    constant tensors so autograd can save them while differentiating the injected
    delta through the final frozen decoder block.
    """
    cached = _original_cache_pns(model, tok, rows, barrier)
    for x in cached:
        # clone outside inference_mode converts the retained activation into a
        # normal non-requires-grad tensor; Qwen parameters remain frozen.
        x.hidden = x.hidden.clone().detach()
        x.semantic = x.semantic.clone().detach()
    return cached


base.cache_pns = cache_pns_normal
base.STAGE = "R350C-QWEN-LEARNED-INTERNAL-WORLD-PORT"

if __name__ == "__main__":
    raise SystemExit(base.main())
