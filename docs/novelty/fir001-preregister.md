# FIR-001 — exact bounded influence horizon versus persistent recall

Registered 2026-09-05 BEFORE numerical execution. Parent BHC001: `2153cab1bf64ef8bb73b263b8b624354319ff378`. Status UNRUN. This is an architecture-boundary test, not a novelty claim.

## Question

Can a source-local neural memory guarantee that one pod's influence vanishes from ALL future-relevant state after an exact finite horizon K, yet still support source-dependent recall arbitrarily later without rereading/retaining that source anywhere else?

For a deterministic Markov state `s_{t+1}=F_t(s_t,u_t)` under source-independent future forcing, if two source worlds have exactly equal complete future-relevant state at t=K, their future trajectories are identical. Therefore any later source-dependent output must come from (a) source dependence that never actually left the retained state, (b) future source re-injection/lookup, or (c) another retained channel excluded from the equality test. This is elementary state sufficiency, not a new theorem.

## Fixed synthetic architecture

1. A four-stage source channel `z` implemented as a shift register with zero-preserving nonlinear transforms. Source payload enters only stage0. Structural shifting guarantees exact extinction after K=4 state transitions in integer arithmetic, regardless of payload value.
2. A separate long-term recurrent state `b` receives source-INDEPENDENT exogenous inputs only. The combined read is nonlinear in `(b,z)`. While z is alive, reads may depend on the source; after four source-free transitions, complete `(b,z)` is source-independent.
3. Persistent-recall arm: query again at step64. Compare:
   - no reread: exact source-dependent recall must be impossible after coalescence;
   - canonical reread: inject the current pod payload again at query time and measure exact update/revoke fan-out;
   - finite-window baseline: retain last4 source events or reread current canonical source at query, receiving the same information and operator budget.
4. Source revision: edit old payload A to B after the original write. If no reread occurs after the K-horizon, all states beyond K are already source-independent; mutation-to-ready is constant but there is no long-term source recall. If query-time reread is enabled, late recall follows the canonical source without repairing old states, but this is late binding/lookup and receives zero novelty credit.
5. A deliberately leaky control lets z write into b. Long-term source recall returns, but exact K-step influence extinction fails. This demonstrates the tradeoff within the tested Markov architecture.

## Exactness and utility controls

Use Python integer arithmetic plus a separate Fraction implementation for exact trajectories. Five seeded source-independent background recurrences, 128-dimensional b, four 32-dimensional source stages, 80 steps. Check every state write. No floating tolerances.

Measure operation counts and wall-time microbenchmarks for source edit readiness and late query across:
- finite-horizon candidate without reread,
- candidate with canonical reread,
- conventional finite-window/reread baseline,
- full replay of all80 steps.
The strongest baseline receives the same source access and finite-window information. If the candidate's benefit disappears, no utility novelty survives.

This is an operator microbenchmark, not end-to-end LLM throughput. Memory accounting includes b, z, source store, and finite-window events for the compared arms; Python object overhead is reported separately/not used as a system memory gate.

## Prior-art boundary fixed before execution

- Finite impulse response/finite-memory systems are classical; a 2025 IEEE SDS paper explicitly classifies attention as FIR-like and SSMs as IIR-like. A structurally finite horizon is not novel by itself.
- Forgetting Transformer (ICLR2025) adds explicit learned forgetting but does not claim exact finite-step lifecycle deletion.
- Recurrent Memory Transformer and ELMUR retain bounded external memory but are not exact pod-repair mechanisms.
- BHC001 already shows contraction is insufficient for finite native coalescence; FIR001 tests a structurally guaranteed horizon instead.
- Canonical reread/late binding, finite windows, event sourcing, replay and source generation checks receive zero novelty credit.

## Claim limits

No trained reader, semantic pod routing, J-space audit, UNKNOWN behavior, generic divergence, generation/publication race handling, or second-backbone qualification. If exact bounded horizon and persistent recall conflict without external source retention, that is a boundary of this tested deterministic state interface, not an impossibility result for all architectures. A system may retain a compact sufficient source statistic elsewhere; that extra channel must then be included in memory/repair accounting.
