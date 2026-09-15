# E-000098 — Cross-Context Exact Correction Span

Date: 2026-09-05
Status: **PREREGISTERED REPRESENTATION KILL SCREEN / not a novelty claim**

## Trigger

Current branch evidence closes the obvious transport shortcuts:

- E-000095: one translation vector and a per-dimension affine receipt fail exact held-out transport on DistilGPT-2 and Pythia-70M across three seeds.
- E-000096: a substantially more flexible state-dependent nonlinear RBF receipt can fit calibration corrections extremely well yet still fails exact held-out transport.
- E-000097: once exact composable transition summaries are assumed, associative scan / dynamic segment-tree recomposition is a generic baseline with the same exact guarantee and update algebra.

The remaining representation-level question is whether a single canonical Pod edit nevertheless induces an **exact low-dimensional shared correction quotient** across distinct cached neural sessions.

## Candidate class under test

For one fixed old->new Pod edit and many contexts `c`, let

`Delta(c) = H_new(c) - H_old(c)`

be the exact fresh-recompute correction at the final hidden state after a real nonlinear suffix.

A shared linear correction quotient assumes there exists a small context-independent basis `U_r` such that every relevant correction lies in `span(U_r)`. A deployable mechanism would still need to recover the context-specific coefficients without suffix replay, but E-000098 tests the easier representation-existence question first.

This screen is deliberately **more favorable than a deployable receipt**: on held-out contexts it uses the true fresh `Delta(c)` to compute the optimal orthogonal projection coefficients into the calibration-derived basis. Therefore a held-out failure cannot be blamed on a weak coefficient predictor.

## Frozen protocol

Backbones:

- `distilgpt2`;
- `EleutherAI/pythia-70m`.

Seeds: `0,1,2` independently for each backbone.

For each backbone/seed:

1. freeze the pretrained causal LM;
2. select `layer = n_blocks - 3`, leaving at least two actual nonlinear transformer blocks after the memory-read intervention;
3. construct one fixed old payload and one fixed new payload from different vocabulary embeddings, RMS normalized to `2.0`;
4. generate 128 deterministic random token contexts of length 16;
5. use the first 64 only as calibration contexts and the remaining 64 only as held-out contexts;
6. run old and fresh-new forwards and record final hidden state and logits;
7. form the calibration correction matrix `D_cal` with one correction row per context;
8. compute an SVD of `D_cal` in float64 and derive nested row-space bases for registered ranks `1,2,4,8,16,32,48,64` (clipped to available rank);
9. for every held-out exact correction `d`, compute the **oracle** orthogonal projection of `d` into each calibration basis;
10. reconstruct candidate final hidden state as `H_old + Proj_U(d)` and apply the model's real output head.

No rank, split, payload, layer, threshold or basis is selected from held-out results.

## Exact bars

A rank survives a backbone/seed cell only if, over the complete held-out set:

- final-hidden maximum absolute error `<= 1e-6`;
- reconstructed-logit maximum absolute error `<= 1e-5`;
- top-1 agreement `== 1.0`;
- the underlying edit is behaviorally material on `>= 95%` of all 128 contexts.

Approximate KL, mean error, explained variance, or top-1 agreement alone receives **zero lifecycle/deletion guarantee credit**.

## Calibration lower bound

For each registered rank `r`, also compute the Eckart-Young optimal calibration Frobenius tail

`sqrt(sum_{j>r} sigma_j^2)`.

If this exceeds

`1e-6 * sqrt(n_cal * hidden_dim)`,

then no rank-`r` approximation of the calibration correction matrix can have every entry within the hidden-state exactness tolerance, because entrywise max error `<= 1e-6` would imply Frobenius error at or below that bound.

This is a finite-matrix representation bound only. It is not a general lower bound on transformer editing.

## Kill rule

Kill **shared low-rank exact cross-context correction span** as the next major-invention seam if all six backbone/seed cells satisfy the material-edit validity condition and **no registered rank, including the full 64-context calibration span, reaches the exact held-out hidden/logit bars**.

This would be stronger than E-000095/E-000096 in one specific sense: even oracle held-out coefficients cannot rescue a correction representation restricted to the calibration span.

## Survival rule

Do not promote a novelty claim merely because a small span happens to reconstruct held-out corrections. A surviving rank only authorizes a successor test that must still show:

1. coefficients can be computed from already-materialized stale session state plus the Pod edit without executing the ordinary nonlinear suffix;
2. the mechanism beats exact last-read-site patch + minimal suffix recomputation / exact residual/KV reconstruction in fleet mutation-to-ready cost;
3. the basis/state footprint is counted against matched memory;
4. real LINK->Pod capability and the full lifecycle attack contract pass.

## Strong baselines / no-credit territory

No novelty credit is assigned to low-rank KV compression, SVD/PCA, low-rank adapters, Jacobian/JVP linearization, cache blending, generic learned cache repair, KVEraser-style steering, KV-Direct/residual reconstruction, selective recomputation, late binding, or associative trees/scans.

Recent 2025-2026 low-rank KV work (Palu, ReCalKV, PuzzleKV, eOptShrinkQ and related methods) further establishes low-rank cache representation as crowded prior-art territory. Intel `US20260080217A1` also covers gauge/rank-r KV-cache compression and references a provisional titled `COMPOSABLE EXACT KEY-VALUE CACHE COMPRESSION`. E-000098 therefore treats low rank only as a **falsification target**, never as the claimed invention.

## Major-break gates remain unchanged

No result from this representation screen is a major break by itself. Any successor still requires real LINK->Pod reader `>=0.95` on every held-out template in every interpreted job; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched suffix-recompute/KV-repair baseline.
