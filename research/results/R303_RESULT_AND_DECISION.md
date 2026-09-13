# R303 Result — Universal Model-Native Value Codes

Status: **future-value emission mechanism gate passed; not DoD and not a novelty claim.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

## Result

32 held-out one-token symbolic values were tested, including capital names, colors, directions, seasons and shape labels. Their value rows were excluded from the shared-fit vocabulary set.

The simplest representation won:

`J_value(token_id) = frozen_Qwen_lm_head_weight[token_id]`

Without **any per-value training**, the unchanged full Qwen vocabulary head greedily decoded the intended token for **32/32 values (100%)**.

- raw output-row top-1: **100%**;
- raw output-row top-5: **100%**;
- raw output-row top-20: **100%**;
- minimum target margin over the best competing vocabulary token: ~**0.0291**;
- int8-quantized row top-1: **100%**;
- int8 storage: ~**50.22%** of BF16 row bytes.

A rank-64 mapping from Qwen input embeddings to output rows failed completely on these held-out values (0% top-20), so it should not be used as the current universal code path.

Artifact ZIP SHA256: `11a8ea9569d7e87cda483d9279d1d383de03579bf142f0804cb34fb7b3979e4f`.

## Architectural consequence

For token-representable leaf values, CKCA does **not need to persist a per-fact KV capsule or even a full latent vector**. The authoritative Pod page can store a compact model-native symbolic code such as token ID(s), and the serving runtime can materialize the corresponding immutable model output-code row from the frozen model revision.

This changes the preferred payload hierarchy:

1. **symbolic leaf value:** token ID / token-ID sequence + lifecycle metadata;
2. **structured relation/pointer:** typed Pod references + operators;
3. **only when necessary:** compiled latent/KV Port artifact for content that cannot be represented efficiently by model-native symbols.

That can reduce a simple single-token value payload from kilobytes to a few bytes plus lifecycle metadata, while keeping future values writeable without gradient optimization.

## Important boundary

Vocabulary/output embeddings and embedding lookup are established techniques. R303 is an engineering simplification, not a claimed invention. Multi-token value emission, compositional reasoning and free-form continuation still require explicit gates.
