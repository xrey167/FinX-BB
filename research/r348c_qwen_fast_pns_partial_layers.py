from __future__ import annotations

import torch.nn as nn

from research import r348b_qwen_fast_pns_bridge as fixed

base = fixed.base
_orig_encode_fast = base.encode_fast


def encode_fast_partial(model, tok, rows):
    # Keep a genuine pretrained Transformer representation but cap hosted-CPU
    # cost by evaluating only the first six frozen Qwen decoder layers.
    # This is explicitly a bridge gate, not a full-model quality claim.
    if len(model.model.layers) > 6:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:6])
    return _orig_encode_fast(model, tok, rows)


base.encode_fast = encode_fast_partial

if __name__ == "__main__":
    raise SystemExit(base.main())
