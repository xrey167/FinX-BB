from __future__ import annotations

from research import r350b_qwen_learned_world_port_exact_mask as exact_mask

base = exact_mask.base
base.PositionalContext = exact_mask.PositionalContext
base.prepare = exact_mask.prepare
base.run_layers = exact_mask.run_layers

_original_cache_pns = base.cache_pns


def _normal(x):
    return None if x is None else x.clone().detach()


def cache_pns_normal(model, tok, rows, barrier):
    cached = _original_cache_pns(model, tok, rows, barrier)
    for x in cached:
        # Every tensor produced under inference_mode must cross the boundary to an
        # ordinary constant tensor before autograd traverses the final frozen Qwen
        # block. This includes RoPE cos/sin and the causal mask, not only hidden state.
        x.hidden = _normal(x.hidden)
        x.semantic = _normal(x.semantic)
        p = x.pos
        x.pos = exact_mask.PositionalContext(
            position_ids=_normal(p.position_ids),
            cache_position=_normal(p.cache_position),
            position_embeddings=tuple(_normal(t) for t in p.position_embeddings),
            causal_mask=_normal(p.causal_mask),
        )
    return cached


base.cache_pns = cache_pns_normal
base.STAGE = "R350D-QWEN-LEARNED-INTERNAL-WORLD-PORT"

if __name__ == "__main__":
    raise SystemExit(base.main())
