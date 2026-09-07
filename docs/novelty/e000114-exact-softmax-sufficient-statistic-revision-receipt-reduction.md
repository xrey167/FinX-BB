# E-000114 — Exact Softmax Sufficient-Statistic Revision Receipt Reduction

## Status

Preregistered structural kill screen. This experiment is not a major-invention claim and does not reopen E-000095 or any earlier closed lane.

## Question

Can a canonical `old -> new` Pod mutation update an **already-materialized post-softmax neural memory read** exactly across many heterogeneous cached sessions by carrying a compact task-essential sufficient state, and does that mechanism exceed the guarantee-matched generic dynamic-aggregation plus exact suffix-recompute baseline?

This directly attacks the post-nonlinear boundary left by E-000113. The candidate is intentionally stronger than a pre-activation sparse accumulator: the mutable record already participates in a normalized exponential attention read.

## Candidate

Each session has an integer query `q_s`, static key/value memory records, and one canonical mutable Pod record `(k_p, v_p)`. Attention is

`A_s = sum_i exp(log(2) * q_s k_i) v_i / sum_i exp(log(2) * q_s k_i)`.

Because all registered scores are integer multiples of `log(2)`, every exponential weight is exactly `2^(q_s k_i)` and the entire assay can be executed with `fractions.Fraction`, with no numerical tolerance.

The cached task state is:

- the session query,
- the scalar softmax normalizer `Z_s`, and
- the actual materialized attention output `A_s`.

The unnormalized numerator is recoverable exactly as `N_s = Z_s A_s`. A canonical Pod lifecycle mutation removes the old weighted Pod contribution and adds the new one, then re-normalizes. This is an active state-dependent correction: heterogeneous queries and contexts yield different exact output corrections from the same canonical Pod edit.

Two dense nonlinear task layers follow the attention read. Exact lifecycle readiness therefore requires the patched attention state to traverse this suffix.

## Independent generic baseline

A separately implemented generic dynamic normalized-weighted-average engine receives the identical `(query, normalizer, aggregate output)` state and the same removed/added records. It has no Symlink, Pod, J-space, neural-memory, or lifecycle semantics.

The strong exact-ready comparison additionally includes the user-required baseline boundary:

`exact last-memory-read patch -> minimal nonlinear suffix recomputation`.

An oracle version that is handed the exact read-site delta pays only the suffix work. A candidate that cannot beat this boundary cannot claim a fleet-level exact transport advantage merely because it updates the softmax read cheaply.

## Registered domain

- 16 deterministic seeds.
- Widths `4, 8, 16`.
- 32 heterogeneous cached sessions per cell.
- 64 static memory records per session.
- Eight query buckets.
- Sequential lifecycle: `UPDATE -> DELETE -> RESTORE -> ABA`.
- Exact rational arithmetic only.
- Session static values are independently perturbed while the canonical mutable Pod record is shared across the fleet.

Total registered lifecycle cases: `16 * 3 * 32 * 4 = 6,144`.

## Kill screen

Close this mechanism as a standalone lifecycle novelty seam if all of the following hold:

1. every lifecycle transition materially changes the final nonlinear task state;
2. candidate post-softmax state equals fresh attention exactly in every case;
3. generic baseline post-softmax state equals fresh attention exactly in every case;
4. candidate and generic states are identical in every case;
5. after required suffix recomputation, both candidate and generic final states equal fresh recomputation exactly;
6. candidate and generic local mutation work are identical;
7. candidate and generic exact-ready work are identical;
8. every registered lifecycle transition exhibits heterogeneous context-dependent attention-output corrections rather than one fleet-global output translation; and
9. candidate exact-ready work remains at least the cost of the oracle exact-read-site-patch plus suffix baseline.

If this screen passes, assign zero standalone invention credit to:

- caching a softmax normalizer plus materialized attention output,
- subtracting/adding one edited key/value contribution,
- online-softmax-style sufficient-statistic repair,
- dynamic normalized attention aggregation, or
- claiming exact neural lifecycle transport solely from an exact post-softmax read-site patch.

The scoped conclusion is **not** that exact mutable attention is useless. It is that local exact softmax repair is generic aggregation and does not remove the downstream nonlinear exact-state obligation.

## Promotion boundary unchanged

No E-000114 structural success can be promoted to major-invention status without later satisfying the full real-system gates: DistilGPT-2 and Pythia-70M, at least three seeds, real LINK->Pod reader >=0.95 on every held-out template, exact fresh-state equality, <=5% steady-state overhead, material fleet-level mutation-to-ready advantage over guarantee-matched suffix/KV repair at matched memory, lifecycle/race/stale-state/key/reconstruction attacks, <=2% old/deleted leakage, >=90% UNKNOWN in declared missing-key scope, generic-divergence bypass requirement, and independent J-space/J-lens content audit.
