# R312b Result — Integrated CKVM Real-Model Gate

Status: **reduced real-model integration gate passed; not DoD.**

The original R312 integrated job exceeded its wall-time budget. R312b deliberately removes the learned planner from the integrated timing path and isolates the runtime mechanisms that must compose correctly on a real frozen transformer.

## Target

Pinned frozen `Qwen/Qwen2.5-0.5B` with:

- canonical alias -> Pod identity;
- exact Pod generations and capabilities;
- CKVM execution;
- CLFD lifetime invalidation;
- late-bound decoder checkpoint;
- same-value generation rewrites;
- no gradient updates after world changes.

## Result

Across four online transitions:

- max full-recompute vs late-splice full-vocabulary logit delta: **0.0**;
- every old factor whose dependency changed became dead: **true**;
- same-value generation rewrite checks: **passed**;
- canonical aliases and late aliases resolved to the same Pod identity: **true**;
- median full recompute: **31.1067 s**;
- median late splice: **2.49565 s**;
- median full/splice speedup: **12.464×**;
- median authority update + CLFD invalidation: **18,538 ns**;
- update gradient steps: **0**.

Report SHA256: `6deacd76c8e7557d36a4011f1553dd736746216f8da363526a20429501ec650b`.
Artifact ZIP SHA256: `8a1f8f9bc7475805b4ae052bac4fbb773144268963673e787b7e79059fe9dd55`.

## Decision

The integrated runtime path is now supported on a real pretrained transformer for the reduced operation set:

`immutable prefix checkpoint -> verified current CKVM result -> late decode splice`

A generation transition invalidates the old derived lifetime even when the replacement value bytes are identical, while the immutable prefix remains reusable. This is exactly the architectural separation CKCA needs.

This gate does **not** establish free-form knowledge reasoning, cross-model replication, end-to-end RAG superiority, or novelty of the full system. Those remain DoD requirements.
