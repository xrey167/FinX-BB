# E-000085 — revocation-cone boundary

**Status:** falsification/architecture boundary only. **No novelty claim.**

## Question

E-000084 showed in a numerical construction that a pod represented as an invertible group action can be removed *after* deep equivariant computation by applying the inverse action, without replaying the depth stack. That observation is only useful to FinX-BB if pod-dependent semantic computation can remain both expressive and exactly revocable.

E-000085 tests a necessary boundary: what happens when pod-dependent information leaves the representation on which the lifecycle action operates and is written into persistent state that the action does not transform?

## Representation argument

Let the neural state be `(b, z)` and let a lifecycle group `G` act as

`g · (b, z) = (b, rho(g) z)`.

Here `b` is a trivial representation (unchanged by lifecycle actions), while `z` is revocable/equivariant state. For a downstream map `F=(F_b,F_z)` to be `G`-equivariant,

`F(g · (b,z)) = g · F(b,z)`.

The `b` component therefore obeys

`F_b(b, rho(g)z) = F_b(b,z)`.

So a phase/group-dependent pod signal cannot be written into `b` while retaining exact equivariance. If a layer performs such a write anyway, applying `rho(g^{-1})` to `z` after the layer cannot undo the stale value already persisted in `b`.

This is standard representation theory, not a new theorem. The research consequence is specific: **the full persistence cone of pod-dependent semantics must remain inside state that is itself transformed/revoked, or every affected state must be recomputed/repaired.** Merely making the original memory carrier invertible is insufficient.

## Falsification screen

`so/experiments/e000085_revocation_cone_boundary.py` compares five numerical seeds under two stacks:

1. **equivariant stack:** the durable lane consumes only group-invariant quantities; the phase-carrying lane remains non-trivial and a late readout can still use it;
2. **leaky stack:** the same model additionally writes phase-sensitive features into the durable lane.

Pre-registered checks:

- inverse revocation in the equivariant stack matches clean recomputation to `1e-10`;
- the durable/trivial lane has no pod-dependent signal to `1e-10`;
- a late readout still receives a non-trivial pod-dependent signal, so the symmetry is not merely numerically decorative;
- the late readout after inverse repair matches the clean counterfactual to `1e-10`;
- any nonzero tested phase-sensitive durable write creates measurable durable pod dependence (`>=1e-4`) and makes inverse-only repair fail (`>=1e-4`).

Passing this screen does **not** establish real-language capability, deletion safety, novelty or a speedup.

## Why this changes the E-000084 design

The original structural screen can be misread as: “encode a pod as a reversible action and it can be removed later from arbitrary neural computation.” That is too strong.

The correct candidate is narrower:

> A live LLM would need an explicitly revocation-equivariant **persistence cone**: every persistent neural-derived state that can absorb pod-dependent semantics (hidden state, K/V, activation caches, routing-derived state and generated internal history) must either transform under the lifecycle action, remain discardable, or be recomputed from an authority-clean boundary.

This immediately gives strong baselines and failure tests. If the practical implementation degenerates into a late sidecar/readout, source-isolated cache, generic recomputation, or ordinary cache invalidation, it has no novelty claim.

## Direct nearby prior art checked on 2026-09-05

The following removes broad claims around the ingredients:

- **KVEraser**, arXiv:2606.17034 — learned localized KV-cache erasing without suffix recomputation; approximate rather than exact counterfactual cache repair.
- **Multiplicative Orthogonal Sequential Editing (MOSE)**, AAAI 2026 / arXiv:2601.07873 — factual knowledge edits represented by multiplicative orthogonal matrices; orthogonal/invertible knowledge updates are therefore not new.
- **Complex-Valued Phase-Coherent Transformer**, arXiv:2605.10123 — phase-preserving multi-layer transformer computation; phase-coherent neural computation is not new.
- **Phase-Associative Memory**, arXiv:2604.05030 — complex phase-based associative memory and superposition; phase-coded memory is not new.
- **LieTransformer**, arXiv:2012.10885, and the broader equivariant-network literature — group-equivariant attention/networks are established.
- **Intel US20260080217A1, “Key-value cache compression based on gauge transformation”** — exact/composable gauge transformations of transformer attention weights and KV representations are patented prior art for compression, so gauge/rotation of KV itself is not a claim.
- **IBM US20260119893A1** — mutable LLM KV-cache knowledge insertion/modification/deletion already collides with the broad editable-neural-memory target.
- **Knowledge Externalization, ICLR 2026** — reversible externalized memory and modular retrieval are already established.

No direct source was found in this pass for the much narrower end-to-end property “semantic lifecycle action remains attached to all persistent neural descendants and supports exact post-computation pod revocation while preserving a qualified live symlink reader.” Absence from this search is not evidence of patentability or novelty.

## Promotion conditions remain unchanged

No CAVI attack result is interpreted until fresh real-symlink correctness is `>=0.95` on at least three training seeds. A surviving implementation still needs `>=0.95` held-out paraphrase reading and REVOKE/SHRED, `<=0.02` deleted-object leakage, `>=0.90` UNKNOWN, exact/no-damage BYPASS or generic KL `<=0.05` nats, stale Bank/router/route/payload/hidden/KV replay attacks, independent J-space/J-lens audit, alias fan-out and lifecycle scaling, practical inference/memory/rollback costs, and preferably a second public backbone.

E-000085 is useful only as a **negative design constraint**: if pod semantics leak into state outside the revocation action's representation, inverse-only repair is not a valid lifecycle operation.
