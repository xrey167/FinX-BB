# E-000105 result — stable-gate cohort transport reduces to generic exact region specialization

Date: 2026-09-06
Status: **DECISIVE DIRECTION CHANGE / scoped family kill**

## Decision

E-000105 closes exact lifecycle transport based only on a small set of **stable piecewise-affine activation cohorts** as a standalone major-invention seam.

Within a fixed activation signature the registered network is exactly affine in the mutable Pod:

`y_s(p) = a_s + J_z p`.

The lifecycle-labelled candidate caches one exact Pod sensitivity `J_z` per cohort and transports a canonical mutation with `J_z (p_new-p_old)`. The strongest generic baseline receives the same network, same stable masks/signatures, same cohort partition and same cached old mixed outputs, and independently computes the identical fixed-region sensitivity using ordinary forward-mode propagation.

## Executed evidence

GitHub Actions run `34009753437`, head SHA `19cede2142ad0bcdfc04dcafb2a5f800d949c3e1`, completed successfully on Ubuntu 24.04 / Python 3.11.

Registered exact assay:

- 16 deterministic seeds;
- hidden widths 4, 6, 8;
- depths 2, 4, 6;
- 4 stable activation cohorts per network;
- 32 distinct context sessions per network cell;
- 144 network cells / 4,608 session cells;
- exact Python `Fraction` arithmetic;
- lifecycle checks UPDATE, DELETE, RESTORE, ROLLBACK and ABA;
- fresh full evaluation under the same registered stable activation signature as gold.

Observed:

- lifecycle cases: **23,040**;
- materially changed UPDATE sessions: **4,608 / 4,608**;
- cross-cohort interaction checks: **864 / 864 material**;
- candidate sensitivity vs independently derived generic sensitivity mismatches: **0**;
- candidate vs fresh mismatches: **0**;
- generic baseline vs fresh mismatches: **0**;
- candidate vs generic mismatches: **0**;
- ABA cases: **4,608**, mismatches **0**;
- candidate mutation multiplications: **31,104**;
- generic mutation multiplications: **31,104**;
- descriptive full-replay multiplications: **8,008,704**;
- per-cell full-replay / cohort-update multiplication ratio: **89.78x to 541.33x**;
- focused regressions: **4 passed**.

Artifact `9982093861` was uploaded by the successful run. GitHub reported artifact ZIP SHA-256 `f6e3c81007fef3dea5da46a2786bdfc91fea15ae08b87704fc3de87e67cc7d3c`.

## Interpretation

Stable activation cohorts can create the desired fleet topology: one exact correction per cohort can update many already-materialized Pod-dependent sessions, producing very large arithmetic savings relative to independent full replay in this structural assay.

But the complete exactness and the complete mutation-work advantage are reproduced by ordinary exact specialization / forward-mode sensitivity of the **same fixed piecewise-affine regions**. The speedup therefore belongs to generic piecewise-affine algebra, not to Symlink–Pod lifecycle semantics.

This is not E-000094 static-cache noninterference: the cached old session output genuinely depends on the old Pod. The reduction instead says that once the activation signature is fixed, exact Pod transport is ordinary affine sensitivity propagation.

## Fresh prior-art boundary

Current work makes the fixed-region baseline particularly strong:

- Wei et al., **AffineLens: Capturing the Continuous Piecewise Affine Functions of Neural Networks** (2026), computes exact affine regions and exploits the affine restriction under fixed activation patterns across residual/MLP/CNN components: https://arxiv.org/abs/2605.06218
- Barua, Ahmed & Begum, **Mechanistic Interpretability of ReLU Neural Networks Through Piecewise-Affine Mapping** (Machine Learning, 2026), treats ReLU networks through their piecewise-affine maps: https://doi.org/10.1007/s10994-025-06957-0
- Braniff & Tian, **YANNs: Y-wise Affine Neural Networks for Exact and Efficient Representations of Piecewise Linear Functions** (2025), gives exact efficient representations of piecewise-affine functions: https://arxiv.org/abs/2505.07054
- The 2026 Regime Change Hypothesis work explicitly studies stability of activation patterns, including Transformer ReLU-FFN patterns, and notes local affine behavior when patterns are stable: https://arxiv.org/abs/2602.08333

These references are baseline/context evidence, not claims that they implement the FinX lifecycle mechanism or anticipate every possible successor.

## What is killed

Do not spend major-invention budget on:

- a small library of exact Jacobians indexed by stable activation pattern;
- one correction per ReLU/gating cohort;
- activation-signature caching used solely to choose an affine Pod transport;
- exact fixed-region sensitivity/partial-evaluation transport;
- training whose only lifecycle benefit is to make the Pod remain inside one of a small set of already-known affine regions.

If the same stable signature and network are given to a generic exact evaluator, it obtains the same state and mutation work.

## What remains open

E-000105 deliberately does **not** cover Pod revisions that change activation regions/signatures. A successor could still be interesting if it supports exact cheap transport **across nonlinear region transitions** and beats the strongest generic exact branch evaluation / incremental recomputation given identical cached state.

Nor does E-000105 rule out a learned neural quotient whose representation or transition law is materially smaller than the exact piecewise-affine region structure itself.

Any such successor still requires exact fresh-state equality and ultimately the full real LINK->Pod reader, >=3 seeds, >=2 backbones, leakage/UNKNOWN, stale-state attack, lifecycle/race/rollback, independent audit, matched-memory, <=5% inference-overhead and material fleet-level mutation-to-ready gates.

No patentability/infringement conclusion is claimed.