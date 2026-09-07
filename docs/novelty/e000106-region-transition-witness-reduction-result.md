# E-000106 result — precomputed exact region-transition witnesses collapse to generic piecewise-affine evaluation

Date: 2026-09-06
Status: **DECISIVE SCOPED FAMILY KILL / no invention promoted**

## Decision

E-000106 kills **precomputed exact region-transition witnesses / activation-transition tapes as a standalone major-invention seam** for the registered class.

The candidate stores, for each cached session, an exact piecewise-affine representation of downstream state as a function of mutable Pod state: ordered region boundaries plus the exact affine map in every region. This representation can cross nonlinear activation regions exactly, unlike E-000105's fixed-region cohort mechanism.

However, an independently implemented generic piecewise-affine evaluator given the identical serialized representation returns the identical lifecycle states with the same mutation-time evaluation structure. The Pod interpretation adds no stronger computational guarantee.

For a finite preregistered lifecycle endpoint set, an even stronger baseline can simply materialize the exact endpoint states/deltas if witness construction already had access to those fresh target states.

## CI evidence

Registered source commit: `4ee0a1b0d708e0fc3bd470da44453135d273c348`.
GitHub Actions run: `34011994108`.
Job conclusion: success.
Focused regressions: **4 passed**.

Exact `Fraction` assay:

- 32 deterministic seeds;
- 64 cached sessions per seed = **2,048 sessions**;
- 8-dimensional downstream state;
- 12 exact continuous rational regions per primary session;
- lifecycle endpoints OLD, UPDATE, DELETE, RESTORE and ABA;
- **10,240 candidate-vs-generic comparisons, 0 mismatches**;
- **10,240 endpoint-materialization comparisons, 0 mismatches**;
- **2,048 / 2,048 materially changed UPDATE sessions**;
- candidate and generic baseline use the same exact representation footprint: **203 rational slots/session**;
- the finite three-value lifecycle endpoint materialization baseline needs only **24 state slots/session** in this registered assay.

Artifact ID: `9982747699`.
Uploaded artifact ZIP SHA-256 reported by Actions: `85224cc024253ef3d188ef615eba20478d31c3b983b9455f3819a543c7e2ecd6`.
The workflow also records SHA-256 hashes for preregistration/source/tests and emitted JSON.

## Exact folding control

The separate iterated-tent-map control confirms that an explicit exact region tape can grow rapidly even for a compact ReLU-realizable composition.

`T(x) = 2 ReLU(x) - 4 ReLU(x-1/2) + 2 ReLU(x-1)` on `[0,1]`.

After depth `d`, the exact registered representation has `2^d` affine pieces. E-000106 verified depths 1 through 12 exactly:

- depth 8: 256 regions;
- depth 10: 1,024 regions;
- depth 11: 2,048 regions;
- depth 12: **4,096 regions / 12,287 rational representation slots**.

There were **0 folding-control failures**.

This is a constructive growth witness for this family, not a universal lower bound for neural networks.

## External boundary update

Fresh 2026 work further makes generic exact piecewise-affine representation a mandatory baseline. AffineLens (`arXiv:2605.06218`, 2026-05-07) explicitly enumerates exact continuous piecewise-affine regions and their affine restrictions over bounded domains. Region Seeding (`arXiv:2605.06300`) likewise studies and exactly enumerates realized affine regions.

Ramesh, `Subtract or Replay? Exact Deletion from Language-Model Memory` (`arXiv:2607.27539`, 2026-07-30), remains the strongest directly relevant systems boundary: addressable influence can be algebraically removed, whereas edit-dependent later writes in shared recurrent state require rebuilding from before the affected write under the tested interface; deterministic replay matches omitted state bit-for-bit in that scope.

Targeted 2026 patent search continues to crowd generic mutable/cache representations, including IBM `WO2026087278A1` / `US20260119893A1` for direct knowledge insertion/modification/deletion via a KV-cache network layer. No claim is made that it anticipates the narrow surviving FinX-BB target, and this is not patent-clearance advice.

## What is now closed

Do not allocate major-invention search budget to:

- exact activation-transition tapes;
- pre-enumerated region libraries indexed by Pod value;
- per-session piecewise-affine Pod functions;
- exact breakpoint + Jacobian/affine-map tables;
- canonical-edit witnesses whose construction already materialized the fresh endpoint;

when a generic exact evaluator/materialized-view baseline receives the identical sufficient representation.

## Surviving target

E-000106 does **not** kill a mutation-time exact nonlinear sufficient state that does not require fresh-target precomputation and is materially more compact/cheaper than generic piecewise-affine representation, suffix replay, exact residual/KV reconstruction, or generic incremental evaluation.

The remaining transport target is therefore narrower:

> derive, at mutation time from the old session state plus the Pod edit, a compact exact nonlinear quotient that survives edit-induced region changes without enumerating the region function or replaying the suffix, and whose advantage is not inherited by a generic evaluator given the same state.

The parallel surviving programme seam remains exact neural causal-lineage discovery/certification that identifies the exact source set required for lifecycle repair/revocation without being supplied ordinary provenance metadata.

## Major-break status

**No major invention is promoted.** All full programme gates remain required: real LINK->Pod reader >=0.95 on every held-out template, >=3 genuine seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% scoped UNKNOWN, exact bypass or <=0.05 nats generic divergence, complete stale-state/lifecycle/race/reconstruction battery, independent J-space/J-lens audit, matched memory, <=5% steady-state inference overhead, and material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
