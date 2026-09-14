# R355 — World-Natural Neural Layers

Status: architecture formalization / falsification framework; not a novelty claim.

## Motivation

PNS needs a compositional answer to a concrete engineering question:

> Which neural operations may be executed and cached **before** mutable world values are materialized while preserving exact equivalence under every future world revision?

A layer-by-layer criterion is stronger than choosing one materialization layer by convention.

## Materialization law

Let `K` be a prospective state, `W` an authoritative world, and `M_W(K)` its numeric materialization.

A lifted neural layer `L^` is **world-natural** with numeric counterpart `L` when, for every legal world `W`:

`M_W(L^(K)) = L(M_W(K))`.

The square commutes:

```text
K  ----------------->  L^(K)
|                       |
M_W                     M_W
|                       |
v                       v
M_W(K)  ------------>   L(M_W(K))
```

If `L1^` and `L2^` are world-natural, their composition is world-natural by substitution. This gives a safe algebra of pre-materialization neural blocks.

## Two-plane state

The R355 executable model uses:

`K = (h, R)`

where:

- `h` is ordinary world-independent neural continuation state;
- `R` is a sparse reference-valued linear form over canonical world cells.

Materialization yields `(h, R(W))`.

World-natural operations include:

- arbitrary nonlinear transformation of `h` that does not read mutable values;
- stable reference permutation / merge / scaling;
- query-dependent reference coefficient transforms whose control depends only on `h`;
- residual composition between world-independent state and reference metadata without dereference.

The first operation that fuses `R(W)` into an ordinary numeric residual hidden vector is a **materialization barrier** unless a stronger symbolic/certified lift exists (R352).

## Consequence

Instead of asking “at which Transformer layer should the World Port live?”, a compiler can ask:

`does this block satisfy the world-naturality contract?`

and push the materialization barrier through the longest prefix that does.

This generalizes the B/J split:

- B is the maximal world-natural prefix/state;
- J begins where current mutable payload becomes decoder-observable numeric state.

## Negative control

An early-value-fusion cache computes a hidden vector from `R(W_old)` and retains that numeric result. Under `W_new`, the retained state cannot satisfy the commuting square unless the old-world contribution is decoder-null or the cache is repaired. This is the R351 quotient criterion in layer form.

## Prior-art boundary

Naturality/commuting diagrams, staged computation, partial evaluation, symbolic execution and lifted operators are established mathematical/PL ideas. R355 does not claim them as new. Its role is to make the PNS architecture testable at the neural-layer interface and to supply a compiler rule for legal pre-materialization blocks.
