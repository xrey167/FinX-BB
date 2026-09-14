# R340 Result — Causal Referential Tensor (CRT)

Status: **first-class referential neural-state gate passed; this is the first branch in the current sequence that changes the semantics of retained neural state itself rather than only adding lifecycle metadata around ordinary numeric tensors. Novelty is still a hypothesis, not a claim.**

## New primitive

A retained neural state is represented as a live world-parameterized closure rather than a numeric tensor that already copied mutable world values:

`X(W) = b + Σ_j A_j · deref(W, ref_j)`

The closure stores canonical live references and world-independent neural coefficients. It stores neither mutable value bytes nor a last-seen world generation. Ordinary world writes therefore do not invalidate or patch the retained closure. Current values and generations are dereferenced only when the closure is materialized for use; a commit barrier rejects a race before publication.

This differs from the preceding Delta World Attention / Transactional Lazy Delta Attention mechanisms: those retained a numeric attention output and repaired/caught it up. CRT retains a function over future world state.

## Result

- canonical Pods: **4,096**
- retained query closures: **8,000**
- live references/closure: **6**
- mutable value dimension: **16**
- world-independent affine/gated neural layers compiled into each closure: **6**
- scheduled world updates: **50,000**
- same-value scheduled updates: **17,426**
- transactional serves: **120,000**
- injected generation races: **9,502**
- same-value race conflicts: **4,208**

Correctness:

- semantic mismatches vs full current-world neural recomputation: **0**
- max absolute error: **3.997e-15**
- race conflicts detected: **9,502 / 9,502**
- race conflicts escaped: **0**
- retries: **9,502 / 9,502**

Write behavior:

- derived closure invalidations on ordinary world writes: **0**
- derived closure numeric patches on ordinary world writes: **0**
- eager numeric-cache patches that would otherwise have been required: **585,552**
- write-fanout elimination for the referential region: **100%**

Reference Python timing:

- median canonical write: **1.382 µs**
- median live closure materialization: **5.599 µs**
- median full six-layer affine-region recomputation: **13.350 µs**
- full/closure materialization ratio: **2.384x**

Report SHA256: `120add5442ee21244a39217c479ae7bec9b248393d1741bc049e0b4a6ba5d079`.
Artifact ZIP SHA256: `83391ca313ca91ea7d1d38b4ff67f1c3094f287a8bf6198e6919d9eae0912c5f`.

## Architectural decision

Promote **Causal Referential Tensor** to the next novelty track.

The project should no longer treat every mutable-world-derived hidden state as an ordinary numeric tensor plus freshness metadata. For reference-preserving neural regions, the hidden state can itself remain *referential* — a partially evaluated neural function of live world cells.

This creates a qualitatively different cache object:

`numeric cache = value from an old world`

versus

`referential cache = reusable neural computation awaiting the current world`.

The immediate next gate is R341: make **attention values themselves reference-valued**, propagate those references through multiple residual/gated neural blocks without dereference, and test whether a reference-valued residual stream can remain exact under arbitrary world rewrites with zero derived-cache maintenance.

Partial evaluation, closures, symbolic execution, pointers and affine algebra all have prior art. What remains to establish is whether **live generation-scoped references as first-class neural activation operands** and a reference-valued attention/residual stream have direct prior art or constitute a new neural-state semantics. No novelty claim is made yet.
