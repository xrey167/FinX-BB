from __future__ import annotations

from dataclasses import dataclass

import torch

from research import r349_qwen_internal_prospective_state as base


@dataclass(frozen=True)
class PositionalContext:
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    causal_mask: torch.Tensor | None


def prepare(model, input_ids: torch.Tensor):
    """Reproduce Qwen2Model.forward pre-layer state for Transformers 4.45.2.

    The first R349 split passed `attention_mask=None` directly to decoder layers.
    In eager attention that omits the causal 4-D mask constructed by Qwen2Model,
    so the split path was internally self-consistent but was not the standard
    pretrained model path.  This repair deliberately calls the model's own
    `_update_causal_mask` helper and carries that exact mask across the split.
    """
    core = model.model
    hidden = core.embed_tokens(input_ids)
    seq = hidden.shape[1]
    cache_position = torch.arange(seq, device=hidden.device)
    position_ids = cache_position.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    causal_mask = core._update_causal_mask(
        attention_mask,
        hidden,
        cache_position,
        None,
        False,
    )
    position_embeddings = core.rotary_emb(hidden, position_ids)
    return hidden, PositionalContext(position_ids, cache_position, position_embeddings, causal_mask)


def run_layers(model, hidden: torch.Tensor, pos: PositionalContext, start: int, end: int) -> torch.Tensor:
    core = model.model
    for layer in core.layers[start:end]:
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
base.STAGE = "R349B-QWEN-INTERNAL-PROSPECTIVE-STATE-EXACT-MASK"


if __name__ == "__main__":
    raise SystemExit(base.main())
