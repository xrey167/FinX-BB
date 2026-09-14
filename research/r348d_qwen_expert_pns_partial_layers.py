from __future__ import annotations

import torch.nn as nn

from research import r348c_qwen_expert_pns_bridge as expert

base = expert.base
_orig_encode_fast = base.encode_fast


def encode_fast_partial(model, tok, rows):
    """Fast real-pretrained bridge for hosted CPU CI.

    The representation is still produced by genuine pinned Qwen2.5 decoder
    layers, but only the first six frozen layers are evaluated. This is a
    mechanism bridge, not a full-model quality or language-capability claim.
    """
    if len(model.model.layers) > 6:
        model.model.layers = nn.ModuleList(list(model.model.layers)[:6])
    return _orig_encode_fast(model, tok, rows)


base.encode_fast = encode_fast_partial
base.STAGE = "R348D-QWEN6-PROSPECTIVE-NEURAL-STATE-BRIDGE"

if __name__ == "__main__":
    raise SystemExit(base.main())
