# E-000094 — Noninterference / Late-Binding Reduction

Date: 2026-09-05
Status: **PREREGISTERED ARCHITECTURE-LEVEL KILL SCREEN**

## Trigger

E-000093 closes passive in-band freshness when the reuse decision factors through current authority plus a statistic recoverable from cached neural state. A tempting successor is to modify the architecture so mutable Pod information never contaminates the large reusable neural state: a small mutable stream may read or carry Pod state, while a large static stream remains exactly reusable across Pod lifecycle transitions.

This experiment asks whether **that noninterference property by itself** creates a new lifecycle mechanism, or whether it is structurally just late-bound mutable memory/cross-attention with a static cached computation.

## Candidate family

Let:

- `x` be immutable/static request context;
- `p` be current mutable Pod/lifecycle state;
- `q` be the current query/continuation;
- `R(x,p)` be the state claimed reusable across arbitrary allowed changes of `p`;
- `Y(x,p,q)` be the final model output.

The family under test requires exact Pod noninterference of the reusable state:

`R(x,p1) == R(x,p2)` for every registered `p1,p2` in the lifecycle domain.

The architecture may still contain a mutable branch or read operator that consumes current `p` and the reusable state at inference time.

## Reduction

If the noninterference requirement holds, choose any reference Pod state `p0` and define:

`F(x) := R(x,p0)`.

Because `R` is invariant to `p`, for all `p`:

`R(x,p) = F(x)`.

Whatever computation remains can be written as a function `G` of the reusable state, current Pod state and query:

`Y(x,p,q) = G(F(x), p, q)`.

This is an exact static-computation + late-bound-mutable-argument factorization. A correctly implemented external/co-located memory branch can materialize `F(x)` once and provide current `p` to the same `G`. Thus the **freshness/reuse guarantee obtained solely from noninterference** is not stronger merely because the mutable branch lives inside transformer blocks.

If some cached state `M(x,p)` does vary with `p`, then that state is not in the exact reusable domain. On a lifecycle transition it must be recomputed, transformed, invalidated, or otherwise repaired. That returns the programme to active-repair / affected-work questions and is outside this reduction.

## Registered executable assay

The executable uses many deterministic finite neural-operator analogues with:

- static context states;
- multiple Pod states including active/update/revoke/restore/ABA/rollback transitions;
- multiple query states;
- a reusable state constructed to satisfy exact noninterference;
- an arbitrary nonlinear late-bound read/output operator;
- a sidecar/external-memory shadow that supplies the same current Pod state to the same operator.

It exhaustively checks exact output equality for all registered cases and lifecycle traces.

A negative control deliberately lets the supposedly reusable state depend on `p`; the assay must detect that the noninterference premise has been violated rather than falsely applying the factorization.

A second control materializes a mutable cached substate `M(x,p)` and verifies that lifecycle transitions change it in non-vacuous cases, demonstrating why such a substate cannot be counted as reusable merely by naming the overall architecture a firewall.

## KILL rule

Kill **causal firewall / mutable-stream noninterference by itself** as a major lifecycle novelty seam if:

1. every noninterfering registered candidate is exactly reproduced by the late-bound sidecar shadow; and
2. the negative control detects Pod-dependent contamination of the purported reusable state; and
3. mutable cached substates are shown to require lifecycle work when they actually depend on the Pod.

This kill is scoped to the guarantee obtained from exact noninterference and late binding. It is not a lower bound on latency and does not prove that every internal implementation has identical hardware cost.

## Escape condition

A successor remains interesting only if it adds a mechanism not captured by the factorization, for example:

- an **active** lifecycle-conditioned algebraic transform of already-materialized neural state that is materially cheaper than guarantee-matched replay/recompute;
- a model architecture that yields a measured >=10x mutation-to-ready advantage over the strongest late-bound sidecar/external-memory baseline under matched memory while retaining <=5% normal inference overhead;
- exact affected-work compression unavailable to ordinary dependency/change-propagation machinery at comparable cost;
- causal-source discovery/certification not supplied as normal metadata.

## Prior-art boundary

No novelty is claimed for static/dynamic prompt segregation, cross-attention, external memory, memory tokens, modular streams, prompt caching, selective recomputation, or the reduction itself. Contemporary systems already separate static cached computation from mutable late-bound content, and external/layer-local memory architectures already read mutable memory through cross-attention.

Relevant adjacent examples include:

- ELMUR (2025/ICLR 2026): layer-local external memory with cross-attention and update/rewrite;
- Memory Layers at Scale (ICML 2025): trainable key-value memory layers;
- production prefix/prompt caching: stable reusable prefixes plus late dynamic content.

These references are context for the baseline boundary, not a claim that they anticipate every possible lifecycle-aware neural architecture.

## Programme consequence

If E-000094 passes its kill screen, do not spend major-invention credit on a two-stream or causal-firewall architecture **unless it demonstrates an additional active computational mechanism or measured systems frontier beyond late binding**.

All real-reader and major-break gates remain unchanged.
