# R346 Result — Trainable Reference-Valued Attention (TRVA)

Status: **trainable reference-attention mechanism gate passed; not DoD and not a novelty proof.**

## What changed

R340–R345 showed that a retained neural activation can be a live function/reference over future world state. R346 asks whether the *reference selection itself* can be learned by attention rather than supplied by a hand-coded resolver.

Two trainable attention heads map a query representation to canonical reference IDs. The attention output retained in the descriptor is the references themselves, not the current memory values. An identity-blind J continuation dereferences current world values only when serving the descriptor.

During training, the entire entity->value world was redrawn every batch, preventing a stable factual binding from being memorized by the reference router.

## Result

- canonical Pods: **256**
- retained descriptors: **6,000**
- full reference-descriptor accuracy: **100%**
- future-world accuracy across 64 fresh worlds: **100%**
- stale numeric value-cache accuracy on those worlds: **26.82%**
- gain from live reference descriptor over stale numeric cache: **+73.18 percentage points**
- descriptor bytes unchanged after 30,000 world updates: **true**
- write-time descriptor invalidations/patches: **0**
- numeric value-cache patches counterfactual: **1,404,290**

Race / generation safety:

- injected generation races: **1,070**
- detected: **1,070 / 1,070**
- escaped: **0**
- semantic mismatches after retry: **0**

Report SHA256: `6d5876591064b45b18e2c9f144343a56da9fa7015d2197453acf0e6cec0f30ba`.
Artifact ZIP SHA256: `ebdac25d67fcb673d7b88ff69e7577f7410456da84f153536344357b88faac84`.

## Decision

Promote **trainable reference-valued attention** into the Prospective Transformer track.

The important point is not that pointer networks exist—they do. The useful new evidence is that an attention module can be trained to output a persistent reference-valued activation which remains valid across arbitrary future payload rewrites because it never cached those payloads.

The next novelty-critical step is to make this a native mixed activation stream inside a Transformer layer: numeric language heads plus reference-valued world heads, with multiple subsequent layers operating on `(H,R)` before an explicit materialization barrier.
