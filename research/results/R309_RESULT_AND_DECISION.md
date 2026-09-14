# R309 Result — Late-Bound Decode Checkpoint

Status: **real-model late-binding gate passed; not DoD and not a standalone novelty claim.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`.

## Executed result

Ten prefix-stable multi-token future values were tested. The decoder was checkpointed immediately before mutable value binding; the verified current value was then spliced into the assistant stream and only the generation-dependent suffix was executed.

- full recompute vs late-bound splice top-match: **100%**;
- maximum full-vocabulary logit delta: **0.0**;
- median generation-dependent token fraction: **7.5758%**;
- median full recompute time on Actions CPU: **20.8417 s**;
- median late-splice time: **1.60609 s**;
- median wall-time speedup: **12.9766×**;
- arbitrary mutation of the previous physically resident generation-dependent suffix changed current full-vocabulary logits by **0.0**.

Artifact ZIP SHA256: `aad4dee15618c8b2c2b6d932da46e615df15fdba35a70d8ca840cd7ca2063d2b`.

## Architectural decision

Adopt the **Late-Bound Decode Checkpoint (LBDC)** as the preferred neural materialization boundary when exact token-sequence output is sufficient:

`immutable B-prefix -> verified Port read -> bind current value -> execute minimal dependent suffix`

An edit does not require re-prefilling the immutable query/skill prefix. Only the post-binding suffix has the mutable generation in its causal lifetime.

This is a concrete implementation of CKVM's rule: bind mutable world state at the latest semantic point that needs it.

## Claim boundary

Prefix/KV caching, forced-token/copy decoding and partial recomputation are established. R309 is evidence that they fit CKCA's generation-lifetime architecture with exact real-model equivalence; the splice itself is not the invention.
