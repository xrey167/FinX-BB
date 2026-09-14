# R322 Result — Cross-Model Causal Neural World ABI

Status: **two-model World ABI integration gate passed; not DoD and not a novelty proof.**

## Construction

One canonical Pod generation log is consumed by two different frozen pretrained transformer families:

- `Qwen/Qwen2.5-0.5B`, pinned revision `060db6499f32faf8b98477b0a26969ef7d8b9987`;
- `HuggingFaceTB/SmolLM2-360M-Instruct`, pinned revision `cbcad7f4d160a10174f725b968ab6faf2a76399e`.

The authority plane transitions the Pod once. Each model has its own immutable prefix/materializer, but neither owns the fact and neither receives gradient updates or model-specific reindex operations when the world changes.

The transition log includes a same-value generation rewrite so temporal identity changes even when semantic bytes do not.

## Result

Across both model families:

- frozen backbones: **true**;
- max full-recompute vs late-splice full-vocabulary logit delta: **0.0**;
- minimum top-token match rate: **100%**;
- model-specific gradient updates after world change: **0**;
- model-specific world reindex operations: **0**.

Per model:

- Qwen2.5-0.5B median full/splice speedup: **2.868×** on this short gate;
- SmolLM2-360M-Instruct median full/splice speedup: **2.845×**.

Report SHA256: `80bc77889afe317ab394e212d61d802dc62c10bc3dfd9a467978d20c6fe13556`.
Artifact ZIP SHA256: `20acfdcdf70e321403ae11630f2eeebf18bf83513eea4a50dcbb096931b0f829`.

## Architectural decision

Promote **model-independent mutable world state** as an explicit CKCA V3 principle:

> A model is a consumer/compiler of the World ABI; it is not the owner of the current fact.

The same canonical update/revocation lifecycle can therefore govern multiple neural runtimes without copying the mutable fact into each model or rebuilding a model-specific factual index.

This does not mean external shared state is new. The research target remains the stronger composition: canonical generations + binding-equivariant execution + temporal neural types + commit validation + transitive neural lifetime + generation-aware KV/rewind across multiple model consumers.
