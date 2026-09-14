# R343 Result — Compact Referential Continuation (CRC)

Status: **practicality gate passed for the first compact future-bindable activation descriptor; not DoD and not a novelty proof.**

## Why R343

R341 proved exact reference-valued attention/residual semantics but its dense per-query compiled transform used 18.125x the memory of a BF16 numeric hidden cache. R342 proved nonlinear referential state but expanded polynomial storage was even worse.

R343 keeps the referential semantics but removes dense per-query continuation matrices. A cached descriptor contains only:

- sparse canonical PodRefs;
- stable attention coefficients;
- an 8-dimensional BF16/FP16 B-plane continuation code.

Global shared neural weights decode that code and execute the world-dependent J continuation after late dereference. The descriptor contains no current world values and no last-seen generations.

## Result

- Pods: **4,096**
- cached query descriptors: **3,000**
- heads: **4**
- top-k/head: **3**
- hidden dimension: **64**
- continuation-code dimension: **8**
- reference-preserving blocks executed after dereference: **8**
- world updates: **50,000**
- same-value updates: **17,526**
- transactional serves: **80,000**

Correctness:

- semantic mismatches vs full B compile + route + current-world J execution: **0**
- max absolute error: **0.0**
- injected generation races: **6,447**
- detected: **6,447 / 6,447**
- escaped: **0**
- retries: **6,447 / 6,447**

Write behavior:

- descriptor invalidations/patches on ordinary world writes: **0**
- conventional numeric-cache invalidations counterfactual: **438,202**
- write-fanout elimination: **100%**

Practicality:

- compact referential descriptor: **336,000 bytes**
- BF16 numeric hidden-cache counterfactual: **384,000 bytes**
- descriptor/numeric-cache ratio: **0.875**
- median cached continuation execution: **128.319 µs**
- median full B compile + route + J execution: **356.114 µs**
- full/cached speedup: **2.775x**
- median canonical world write: **1.814 µs**

Report SHA256: `2a6f9c4d4bf190b8d5a382f4ef74b6bcf0209478d2ae59b3a6da9807b63f2f72`.
Artifact ZIP SHA256: `911b497d8a9f3cdb2fe3395b6db8ee20e7e94f106130dfa916f105268a0aad9d`.

## Architectural decision

Promote **Compact Referential Continuation** as the practical carrier for the R340/R341 semantics.

The key result is now stronger than the dense R341 prototype:

> A cached neural activation descriptor can remain a function of future mutable world state, require no write-time maintenance, be exact against full current-world recomputation, and be smaller than the BF16 numeric hidden vector it replaces in this gate.

The descriptor is not merely metadata around an old numeric tensor. There is no old world-derived hidden vector to invalidate. It is a compact continuation waiting for the current world.

The next decisive gate must move this primitive onto a real pretrained model: frozen Qwen B features should compile a compact referential descriptor, and current external world values should be dereferenced only in a small J continuation. That is the R344 direction.
