# E-000106 — Canonical-Edit Region-Transition Witness Reduction

Date: 2026-09-06
Status: **preregistered scoped falsification; not a novelty claim**

## Trigger

E-000105 showed that exact transport inside a fixed activation signature collapses to ordinary fixed-region affine specialization. The surviving transport escape is a Pod revision that **crosses nonlinear regions** but is accompanied by a compact exact per-session transition witness.

This screen asks whether an exact precomputed region-transition witness for the programme's canonical old->new Pod edit creates a distinct lifecycle mechanism, or merely moves ordinary piecewise-affine evaluation / materialized-view work to session construction time.

## Registered candidate class

For each cached session `s`, let mutable scalar Pod state be `p` and the exact downstream state be a continuous piecewise-affine function `H_s(p)` on a bounded lifecycle domain.

A candidate stores a per-session exact witness `W_s` containing enough information to evaluate `H_s(p)` after a lifecycle transition without replaying the original nonlinear network. The primary instantiation stores ordered exact region boundaries and, for each region, the exact affine map.

The canonical programme edit is `p_old -> p_new`; DELETE, RESTORE and ABA endpoints are also evaluated.

This is an intentionally favorable structural candidate. It is not a trained transformer experiment.

## Strong baselines

A. **Generic piecewise-affine evaluator** given the exact same region boundaries and affine maps as the candidate. Representation location and Pod naming receive zero credit.

B. **Materialized endpoint state/delta** for the finite registered lifecycle endpoints. If the canonical edit is known before mutation and witness construction had access to the exact target state, a generic materialized-view baseline may store the target state or exact delta directly.

C. Last-read-site patch + minimal suffix replay / exact residual-KV reconstruction remains the production baseline for mechanisms that do not precompute the target.

## Exact finite assay

Use Python `Fraction` arithmetic.

Primary family:
- 32 deterministic seeds;
- 64 distinct cached sessions per seed;
- 8-dimensional downstream state;
- 12 exact piecewise-affine regions per session;
- distinct continuous rational maps per session;
- lifecycle endpoints `OLD=1/8`, `CURRENT=7/8`, `DELETE=0`, `RESTORE=7/8`, `ABA=1/8`.

The candidate and generic baseline are implemented independently from the same serialized exact representation. Every lifecycle output must agree exactly.

A materiality control requires `H_s(OLD) != H_s(CURRENT)` for every registered session.

## Folding control

A separate exact control uses the iterated tent map on `[0,1]`.

One tent-map application is ReLU-realizable:
`T(x)=2 ReLU(x) - 4 ReLU(x-1/2) + 2 ReLU(x-1)` on the registered domain.

After depth `d`, the exact one-dimensional piecewise-affine function has `2^d` regions. Evaluate depths 1..12 and require exactly `2^d` segments. This is a constructive representation-growth witness for this family only, not a universal lower bound for neural networks.

## Kill rule

Kill **precomputed exact region-transition witnesses / activation-transition tapes as a standalone major-invention seam** if all are true:

1. candidate and generic evaluator produce exactly the same state for every registered lifecycle case;
2. the generic evaluator uses the same exact representation and therefore has the same asymptotic mutation-time evaluation work;
3. finite canonical-edit endpoint materialization reproduces the same registered endpoint states;
4. the folding control confirms that an explicit exact region tape can grow exponentially in depth in at least one compact ReLU-realizable compositional family.

Passing this rule means any future survivor must earn its advantage from a representation that is both materially smaller/cheaper than generic exact piecewise-affine state **and** unavailable to a generic evaluator given the same sufficient state.

## What is not killed

This is not an impossibility theorem for:
- an exact nonlinear sufficient state computed at mutation time without fresh-target access;
- a neural-specific quotient whose compactness survives region changes and is not generic piecewise-affine representation;
- causal-lineage discovery/certification;
- an architecture with measured fleet-level systems advantage after matched memory and <=5% steady-state overhead.

## Programme gates unchanged

No major invention may be promoted without the real LINK->Pod reader >=0.95 on every held-out template, >=3 seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% UNKNOWN in declared missing-key scope, exact bypass or <=0.05 nats generic divergence, stale Bank/router/resolved-payload/Hidden/KV attacks, UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU, key/reconstruction attacks, independent J-space/J-lens audit, matched memory, <=5% steady-state inference overhead, and material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
