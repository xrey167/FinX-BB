from __future__ import annotations

import torch

from research import qwen25_05b_r296_multipod_jspace as r296


def extract_features_left_padded(model, tok, triples, batch_size: int = 16):
    """Correct feature extraction for decoder-only left-padded batches.

    `attention_mask.sum()-1` is an index into an unpadded sequence, not the physical
    index in a left-padded batch. The generation-prompt token is always at the final
    physical sequence position, so use that position for every item.
    """
    feats = {}
    for start in range(0, len(triples), batch_size):
        batch = triples[start:start + batch_size]
        texts = [r296.prompt(tok, op, a, b) for op, a, b in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            out = model(**enc, use_cache=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        idx = h.shape[1] - 1
        for i, key in enumerate(batch):
            feats[key] = h[i, idx].float().cpu()
    return feats


r296.extract_features = extract_features_left_padded

if __name__ == "__main__":
    raise SystemExit(r296.main())
