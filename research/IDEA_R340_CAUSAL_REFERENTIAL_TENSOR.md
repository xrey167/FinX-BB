# R340 hypothesis — Causal Referential Tensor (CRT)

A CKCA-derived neural state should not always cache a numeric value that has already absorbed mutable world bytes. For a large class of neural computations, it can instead cache a **function of live world references**.

A Causal Referential Tensor is an affine neural closure:

`X(W) = b + sum_j A_j * deref(W, ref_j)`

where `ref_j` is a canonical live world reference, `A_j` and `b` depend only on reusable language/skill/query state, and the mutable payload is dereferenced only when the result is actually materialized.

This changes the update problem. If a world value changes, the closure itself does not become stale because it never copied the value. Only the next materialization observes the new generation/value. A commit barrier detects races between dereference and publication.

The closure can be propagated exactly through any world-independent affine neural region. Stable-key attention is affine in `V`; base-conditioned linear/gated transforms are also affine in the live value stream. Thus multiple neural layers can be partially evaluated once into a live referential closure.

This is stronger than R334/R335 numeric cache repair:

- DWA stores numeric `O` and patches it on updates;
- TLDA stores numeric `O` plus last-seen values/generations and catches up lazily;
- CRT stores **no copied mutable numeric state** for the region. It stores a compiled neural function over canonical references.

World writes therefore have zero derived-cache fanout for CRT regions.

Hard limits: arbitrary value-dependent nonlinearities break affine closure and require either materialization or a richer symbolic operator node. R340 intentionally tests the exact affine/gated region first; R341 will test composition with typed nonlinear referential nodes.
