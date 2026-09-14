# Prospective Transformer V1 — Reference-Valued Residual Architecture

Status: research architecture hypothesis. Not a novelty or breakthrough claim.

## Core change

A standard Transformer layer carries one dominant activation type:

`H_l ∈ R^{n×d}`

When mutable external knowledge is injected into `H_l`, every later cached hidden/KV state becomes a numeric function of that historical world snapshot.

A Prospective Transformer carries two activation kinds:

`State_l = (H_l, R_l)`

where:

- `H_l` is an ordinary numeric language/skill stream that is world-independent until an explicit materialization barrier;
- `R_l` is a **referential stream**: sparse canonical live references plus coefficients/operator state. It denotes a function of the current/future world rather than copied world values.

## Reference attention head

For governed external memory with stable address keys `K_addr`, a reference head computes:

`A_l = softmax(Q(H_l) K_addr^T)`

but does **not** immediately compute:

`A_l V_world`.

Instead it emits:

`R_l = RefAttention(A_l, PodRefs)`

possibly sparsified/top-k.

This is the architectural discontinuity: attention's value-side result may be a live reference object rather than a numeric vector.

## Reference-preserving blocks

Operations independent of mutable payload can transform the prospective state without dereferencing it:

- routing over stable keys;
- sparse reference merge/prune;
- static/B-conditioned linear transforms of coefficients;
- residual composition;
- query-conditioned continuation-code updates;
- relation traversal that returns new references, guarded by predicate generations.

The model therefore may execute multiple layers while the world is still unresolved.

## Materialization barrier

A layer explicitly requests current world values:

`(V, G) = deref(R_l, WorldAuthority)`

and computes a numeric J state:

`J_l = Materialize(H_l, R_l, V)`.

The exact `(PodRef, generation)` read-set becomes the temporal lifetime of `J_l` and its downstream numeric derivatives. Commit validation rejects a generation race.

The pre-materialization `(H,R)` state is prospective and does not inherit ordinary value-generation lifetimes because it has not copied those values.

## Hybrid heads

A layer can contain both conventional heads and reference heads:

- language/self-attention heads: numeric `V`, ordinary residual contribution;
- world-reference heads: `RefV`, contribute to `R` rather than the numeric residual;
- optional materializing heads: explicitly dereference selected references and start/update J.

This allows a pretrained language backbone to remain mostly conventional while adding prospective external-world channels.

## Cache semantics

A normal cached hidden tensor is retrospective:

`cache = H(q, W_t)`.

A prospective cache is:

`cache = κ_q`, with `κ_q(W_current) -> H_current`.

World writes therefore need not patch/invalidate `κ_q`. Numeric materializations remain generation-scoped.

## Relation to known ideas

This architecture must not claim novelty for:

- external memories / Neural Turing Machines;
- pointer networks;
- lazy tensors / partial evaluation;
- cached query plans / closures;
- sparse attention;
- symbolic computation.

The narrower novelty hypothesis to audit is:

> A Transformer residual/attention state in which mutable-memory `V` is represented by live generation-scoped references that remain undereferenced as first-class hidden operands across multiple neural layers, so the retained activation is intentionally a function of future world state.

R340–R346 test pieces of this semantics. R347 should compare a cached prospective residual stream against an otherwise matched early-materializing numeric cross-attention stream under arbitrary world rewrites.
