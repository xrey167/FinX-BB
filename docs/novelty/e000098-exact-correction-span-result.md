# E-000098 result — shared linear exact correction span is falsified

Date: 2026-09-05
Status: **DECISIVE FALSIFICATION / SCOPED FAMILY KILL**

## Decision

**Kill the shared linear / low-rank exact cross-context correction-span route as a major-invention seam for the tested transformer states.**

This result does not establish a universal impossibility theorem for nonlinear, program-derived lifecycle transport. It closes the strategy in which one canonical old->new Pod edit is represented by a fixed linear correction subspace learned from calibration sessions and then reused across arbitrary unseen cached sessions.

The key strengthening over E-000095/E-000096 is that E-000098 gives the candidate an oracle advantage on held-out sessions: the true fresh correction itself is used to compute the optimal projection coefficients into the calibration-derived subspace. Therefore held-out failure cannot be blamed on a weak coefficient predictor.

## Provenance

Preregistration: `docs/novelty/e000098-exact-correction-span-preregister.md`.

Executable: `so/experiments/e000098_exact_correction_span.py`.

Workflow: `.github/workflows/e000098-exact-correction-span.yml`.

Executed commit/run: `7dc031059cb4d09d4d6fd667431d7b0d9ef51abf` / GitHub Actions run `33994723116`.

Artifacts:

- DistilGPT-2 artifact `9977707607`, ZIP SHA-256 `93c3a5c9427a6177e15e485dc0190c5ba11d2e116433d059c7907088e6961baa`;
- Pythia-70M artifact `9977717635`, ZIP SHA-256 `adf8e25683acbdf5c78a8eb1935f1a95ce1700c1c01db0c7831fa75002adcabc`.

## Registered protocol

For each backbone and seed, one fixed old payload and one fixed new payload were injected at block `n_blocks-3`, leaving two real nonlinear transformer blocks downstream. There were 128 deterministic token contexts per cell: 64 calibration contexts and 64 disjoint held-out contexts.

For each context `c`:

`Delta(c) = H_new(c) - H_old(c)`.

An SVD of the 64 calibration correction rows defined nested calibration spans of registered ranks `1,2,4,8,16,32,48,64`. On held-out contexts, the true `Delta(c)` was projected orthogonally into each span. The reconstructed final state was then passed through the model's real output head.

Exact bars remained:

- final-hidden maxabs <= `1e-6`;
- reconstructed-logit maxabs <= `1e-5`;
- top-1 agreement = `1.0`;
- material edit rate >= `0.95`.

All six backbone x seed cells had material edit rate `1.0`.

## Exact held-out result at the strongest registered span

The table below reports the **full 64-context calibration row span**, not a small-rank approximation. Even this largest registered span fails every exact-state gate.

| Backbone | Seed | Hidden maxabs | Logit maxabs | Top-1 | KL mean | Oracle delta relative Frobenius residual |
|---|---:|---:|---:|---:|---:|---:|
| DistilGPT-2 | 0 | 1.429235 | 6.546806 | 1.00000 | 0.000184 | 0.214112 |
| DistilGPT-2 | 1 | 1.481250 | 5.686707 | 1.00000 | 0.000960 | 0.287828 |
| DistilGPT-2 | 2 | 1.397194 | 4.396484 | 1.00000 | 0.004609 | 0.210674 |
| Pythia-70M | 0 | 0.891733 | 2.000000 | 1.00000 | 0.009871 | 0.107711 |
| Pythia-70M | 1 | 0.966095 | 2.000000 | 1.00000 | 0.015333 | 0.085883 |
| Pythia-70M | 2 | 0.744343 | 2.000000 | 0.96875 | 0.027615 | 0.084353 |

For every cell and every registered rank:

- exact hidden fraction at `1e-6` = `0`;
- exact logit fraction at `1e-5` = `0`;
- no registered span is exact;
- the full calibration span is not exact;
- decision = `KILL_SHARED_LINEAR_EXACT_CORRECTION_SPAN`.

Thus unseen contexts contribute correction components outside the entire span of the 64 calibration corrections. The remaining held-out correction energy at full calibration rank is material: about 8.4% to 28.8% in relative Frobenius norm across the six cells.

## Calibration rank lower bound

The calibration matrices are themselves not approximately low-rank at the programme's exact hidden-state tolerance.

For every one of the six cells, the smallest rank not already ruled out by the Eckart-Young/Frobenius condition for all-entry error <= `1e-6` is **64**, the maximum available calibration row-space rank.

Examples:

- DistilGPT-2 seed 0: optimal rank-48 calibration Frobenius tail `12.848471` versus all-entry-1e-6 bound `0.000221703`; smallest calibration singular value `2.370604`.
- Pythia-70M seed 0: rank-48 tail `7.503715` versus bound `0.000181019`; smallest calibration singular value `1.339900`.

The conclusion is finite and scoped: within the sampled correction matrix, no rank below 64 can meet the registered entrywise exactness tolerance even on calibration data. Full calibration rank then still fails unseen contexts.

## Why approximate behavioral success receives zero guarantee credit

This screen again shows why top-1/KL are misleading for lifecycle integrity. DistilGPT-2 keeps top-1 agreement at 1.0 for all three seeds at full calibration rank and KL is only `1.84e-4` to `4.61e-3`, while hidden-state max error remains about `1.4` and logit max error `4.4–6.5`. Pythia shows similarly small KL/top-1 changes while remaining orders of magnitude outside the exact-state bars.

Therefore a low-rank correction basis may still be useful as an approximate serving technique, but it cannot support the registered deletion/freshness guarantee.

## Prior-art boundary

Broad low-rank / compressed KV representation is crowded and receives no novelty credit. Current boundary references include:

- `Residual Stream Is All You Need: Lossless KV-Cache Compression with KV-Direct` (arXiv:2603.19664), which exploits deterministic residual-to-K/V projections and exact reconstruction;
- 2025-2026 low-rank KV methods including Palu, ReCalKV, PuzzleKV, eOptShrinkQ and VarRate;
- Intel `US20260080217A1`, published 2026-03-19, covering gauge/rank-r KV-cache transformation/compression and tracing priority to a provisional titled `COMPOSABLE EXACT KEY-VALUE CACHE COMPRESSION`.

The targeted search this turn did not identify a source that supplies the narrow CAVI result "one canonical knowledge edit has an exact reusable cross-session correction span". That is a limited search observation, not evidence of legal novelty or patent clearance. E-000098 itself is a falsification result, not an invention claim.

## Programme consequence

Do not allocate additional major-invention search budget to:

- a larger PCA/SVD basis for one edit;
- a learned coefficient predictor over a fixed shared linear correction span;
- low-rank cross-session delta dictionaries;
- linear/Jacobian correction subspaces whose exactness is inferred from high explained variance or task accuracy.

E-000095 killed compact translation/diagonal-affine receipts. E-000096 killed calibration-fitted nonlinear interpolation as an exact held-out transport law. E-000097 killed associative recomposition as standalone novelty once exact summaries are assumed. E-000098 now kills the next escape: a shared exact linear correction quotient across sessions.

## Surviving frontier

The next admissible mechanism must get exactness from **session-specific model computation or a genuinely nonlinear exact sufficient state**, not from a static basis learned across neighboring contexts.

A successor should answer one of the following:

1. Can an already-materialized session retain a compact exact local computational witness that lets the Pod-induced delta be propagated through nonlinear transformer operators with substantially less work than ordinary suffix recomputation?
2. Can the model expose an exact causal quotient whose value is session-specific and whose update is cheaper than generic incremental/change-propagation execution using the same cached intermediates?
3. Can a lifecycle mutation transform stale mixed state through an exact nonlinear operator derived from the actual cached state while beating last-read-site patch + minimal suffix recomputation / KV-Direct-style exact reconstruction?

If the candidate merely replaces the static linear span with another approximate predictor, E-000095/E-000096 already kill it. If it merely organizes exact summaries in a scan/tree, E-000097 kills it.

## Major-break gates remain unchanged

No major useful invention is promoted. Any successor still requires real LINK->Pod reader >=0.95 on every held-out template in every interpreted job; >=3 genuine seeds; >=2 backbone families; <=2% old/deleted leakage; >=90% UNKNOWN in declared missing-key scope; exact bypass or <=0.05 nats generic divergence; stale Bank/router/resolved-payload/Hidden/KV attacks; UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU; key/reconstruction attacks; independent J-space/J-lens audit only; <=5% steady-state inference overhead; matched memory; and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched per-session suffix-recompute/KV-repair baseline.
