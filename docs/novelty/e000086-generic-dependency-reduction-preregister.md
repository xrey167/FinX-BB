# E-000086R — generic-dependency reduction test

Date: 2026-09-05
Status: **pre-registered before executable reduction test; candidate-kill screen**

## Question

E-000086 currently proposes one alias-qualified canonical Pod-generation witness

`(alias_id, alias_incarnation, pod_id, pod_incarnation)`

as a coherence domain for heterogeneous derived neural artifacts (Bank, routing, resolved payload, post-read hidden state, KV). The major-invention programme explicitly requires killing this seam if it reduces to ordinary cache coherence, self-adjusting computation, PAMSPEC-style authoritative-object/derived-state semantics, or a generic dependency graph.

This experiment asks the narrower systems question **before spending further compute on a heterogeneous neural implementation**:

> Under E-000086's explicit-tag contract, is full object coherence observationally different from ordinary versioned dependency validation over an alias node and a Pod node?

## Generic baseline fixed before execution

Represent the authoritative state as two ordinary dependency nodes:

- alias node value = `(alias_incarnation, current_pod_id, live)`;
- Pod node value = `(pod_incarnation, live)`.

When a derived artifact is produced, capture the values of the alias node and the resolved Pod node. At reuse, a generic memoization/dependency validator accepts the artifact iff both current node values equal their captured values. This is a type-agnostic dependency record; the validator does not know whether the derived value is a Bank row, router output, payload, hidden activation, or KV tensor.

Compare that decision with the current `CAVIAuthority.validate_witness` decision for the exact same authority state and lifecycle trace. The generic baseline is allowed to depend on the alias relationship itself; omitting that dependency would intentionally weaken it and would not be a strongest guarantee-matched baseline.

## Traces fixed before execution

Cover at least:

- no mutation;
- alias RELINK to a different live Pod;
- Pod UPDATE;
- alias REVOKE and REVOKE->RESTORE;
- Pod SHRED and SHRED->RESTORE;
- same-id delete/recreate ABA;
- repeated Pod updates;
- RELINK then update the new Pod;
- RELINK then mutate the old Pod;
- mutation of an unrelated Pod;
- mutation/relink of an unrelated alias.

For every trace, compare decisions for all five E-000086 artifact classes. Also compare a global-epoch baseline to preserve the expected distinction between selective dependency validation and indiscriminate invalidation.

## Kill criterion

The current **coherence-domain mechanism** is rejected as a major-invention seam if all of the following hold:

1. full object-coherence reuse decisions equal the generic dependency validator for every registered trace and artifact type;
2. unrelated mutations are preserved by both selective methods while a global epoch over-invalidates them;
3. the alias-qualified witness contains no runtime freshness information beyond the two captured dependency-node states needed by the generic validator;
4. independent J-space/J-lens attestation remains outside the runtime authorization path, so adding it does not distinguish the coherence algorithm itself.

A pass of this reduction test does **not** prove that every possible neural-memory architecture is old, nor does it settle legal novelty. It kills only the currently specified explicit-tag/versioned-generation coherence seam. Any surviving candidate must add a technical mechanism not obtainable by generic dependency capture/change propagation—for example a new way to discover/certify causal neural lineage, create materially smaller exact affected work inside the model, or otherwise beat a guarantee-matched dependency/change-propagation baseline under matched memory.

## Prior-art boundary fixed before execution

- Self-adjusting computation already records dynamic data/control dependences and change-propagates only affected computation while memoizing unaffected computation.
- PAMSPEC (July 2026 Internet-Draft, work in progress) already separates canonical versioned Memory Objects from non-authoritative derived state; its Update semantics states that Derived Indexes become stale until rebuilt, and its deletion/redaction guidance propagates to derived state.
- Invalidation Contracts (August 31, 2026) already uses version stamps for selective invalidation of cached agent memory.
- LineageKV already occupies lineage-guided stale-memory KV repair/selective reuse.

These references are prior-art/architecture boundaries, not assertions that any one document implements this repository's neural experiment verbatim.

## What cannot rescue the seam after a reduction pass

Merely changing names from dependency nodes to Symlink/Pod/generation, applying the same generic validator to more neural artifact types, adding version counters, moving the check closer to consumption, or reporting J-space as an independent audit does not reverse the reduction. A new candidate must change the mechanism or the guarantee/performance frontier.
