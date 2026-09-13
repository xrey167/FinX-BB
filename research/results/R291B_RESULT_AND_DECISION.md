# R291b Result — Causal Page Table (CPT)

Status: **bounded real-model authority/materialization gate passed; not DoD.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

## Executed result

R291b executed 4,096 monotonic generation updates while keeping historical generation pages physically resident outside the neural attention sequence. Sixteen bounded real-model probes were interleaved across the lifecycle run.

- stale-handle rejection: **100%**;
- forged-future-handle rejection: **100%**;
- revoke returns no attention page: **pass**;
- active verified page top-match versus direct current-generation prefix: **100%**;
- maximum full-vocabulary logit delta, verified active page versus direct current generation: **0.0**;
- maximum full-vocabulary logit delta after arbitrary mutation of stale external pages: **0.0**;
- verified authority lookup median: **1,482 ns**;
- p99 verified authority lookup: **1,593 ns**.

Artifact ZIP SHA256: `249f71214307cbfe80cdbd44819e2f240f023b8c4f3a5628137376d528862b8c`.

## Architectural decision

Use **active-only materialization** as the normal CKCA generation mechanism. Historical pages can remain physically stored, but they are not concatenated into the transformer attention sequence. This gives revision-count-independent logical context length and makes historical-generation position-overlay machinery unnecessary for the normal serving path.

The Causal Page Table is an authority boundary, not merely a cache index:

`pod_id -> verified current generation capability -> exactly one materialized neural page`

Old page bytes have no causal path to logits unless the verifier admits them. The stale-page randomization experiment gives a direct causal test of that boundary for the executed real-model probes.

## What this does not prove

The expensive neural-equivalence checks were deliberately bounded to keep CI tractable; lifecycle/control checks ran for all 4,096 revisions. This is not a large-model or end-to-end serving benchmark, and the page-table idea by itself is not claimed as novel.
