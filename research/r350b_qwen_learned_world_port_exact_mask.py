from __future__ import annotations

from dataclasses import dataclass

import torch

from research import r350_qwen_learned_internal_world_port as base


@dataclass(frozen=True)
class PositionalContext:
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    causal_mask: torch.Tensor | None


def prepare(model, input_ids: torch.Tensor):
    core = model.model
    hidden = core.embed_tokens(input_ids)
    seq = hidden.shape[1]
    cache_position = torch.arange(seq, device=hidden.device)
    position_ids = cache_position.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    causal_mask = core._update_causal_mask(attention_mask, hidden, cache_position, None, False)
    position_embeddings = core.rotary_emb(hidden, position_ids)
    return hidden, PositionalContext(position_ids, cache_position, position_embeddings, causal_mask)


def run_layers(model, hidden: torch.Tensor, pos: PositionalContext, start: int, end: int) -> torch.Tensor:
    for layer in model.model.layers[start:end]:
        hidden = layer(
            hidden,
            attention_mask=pos.causal_mask,
            position_ids=pos.position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=pos.cache_position,
            position_embeddings=pos.position_embeddings,
        )[0]
    return hidden


base.PositionalContext = PositionalContext
base.prepare = prepare
base.run_layers = run_layers
base.STAGE = "R350B-QWEN-LEARNED-INTERNAL-WORLD-PORT-EXACT-MASK"


if __name__ == "__main__":
    raise SystemExit(base.main())
