# CKCA Architecture V2 — A Causal Knowledge Coherence Machine for Neural Systems

Date: 2026-09-14
Status: **research architecture candidate; no breakthrough/novelty claim yet.**

## 1. Thesis

Today's LLM stacks usually treat mutable knowledge as one of three things:

1. text to retrieve and inject;
2. parameters to edit;
3. external/persistent memory to read through a learned adapter.

All three can carry useful information, but none of those categories alone defines the lifecycle of every **neural execution artifact that has already absorbed mutable knowledge**.

CKCA proposes a different system contract:

> Mutable world knowledge is authoritative, generation-scoped state. Any neural artifact that causally depends on that state inherits its generation lifetime. Language and reusable reasoning skill remain in a separate immutable plane. Current knowledge is bound as late as possible and becomes externally visible only through a validated knowledge transaction.

The target is not a new database wrapper around an LLM. It is a coherent execution model for mutable knowledge inside a neural runtime.

---

## 2. Four planes and one transaction protocol

### B-plane — immutable language/skill plane

The pretrained transformer supplies language understanding, reusable reasoning operators and general skills. Mutable governed facts are not written into the reusable B-plane cache.

A fact update should not invalidate the B-plane unless the interpretation/schema itself changed.

### P-plane — canonical temporal knowledge plane

Natural aliases resolve to one stable `pod_id`. A Pod owns monotonic generations and typed data/relations.

`pod_id : g1 -> g2 -> g3 -> ...`

The same semantic value written twice is still two distinct generations.

### A-plane — authority plane

Physical pages are not authoritative by possession. A current page is admitted only through an independently verified generation capability bound to:

- canonical `pod_id`;
- exact generation;
- payload/page digest;
- model/schema revision where relevant;
- authenticated authority-log identity.

Historical pages may remain physically stored but are not members of neural execution.

### J-plane — ephemeral causal workspace

A J artifact is any derived hidden/KV/typed execution state that may later affect model output. It carries a lifetime factor over the exact mutable generations that causally created it.

`lifetime(J) = AND lifetime(pod_i, generation_i)`

Derived J states compose lifetimes transitively.

### CKT — Causal Knowledge Transaction

The execution protocol is:

1. reuse/compile a B-plane plan;
2. resolve canonical identities;
3. acquire verified current generation capabilities lazily;
4. execute typed CKVM operations and record the dynamic read-set;
5. validate the exact generation read-set at a **Neural Commit Barrier**;
6. abort/retry only mutable J execution if any generation changed;
7. on successful commit, attach a CLFD lifetime factor to every retained derivative;
8. materialize mutable value state only at the latest neural/decode boundary that needs it;
9. serve retained artifacts only while their factor remains live.

Transaction-time validation and post-transaction lifetime are complementary. A commit barrier prevents publishing a mixed world; lifetime factors prevent later serving of an artifact whose once-valid sources have since changed.

---

## 3. CKVM — Causal Knowledge Virtual Machine

CKVM is the typed late-binding execution layer between B and J.

The B-plane compiles natural language into a small K-SSA/dataflow program. The program names canonical operands, operators and control flow; it does not copy current fact values into its immutable plan state.

Conceptual program:

```text
%a       = RESOLVE(alias_a)
%b       = RESOLVE(alias_b)
%ca      = CAPABILITY(%a)
%flag    = READ_FIELD(%ca, flag)
%result  = SELECT_LAZY(
             %flag,
             { EMIT_REF(READ_VALUE(CAPABILITY(%a))) },
             { EMIT_REF(READ_VALUE(CAPABILITY(%b))) })
COMMIT(%result)
```

Every runtime register conceptually contains:

`(typed value/handle, lifetime_factor)`.

Only the actually executed path contributes generation dependencies. Static supersets are safe but create unnecessary invalidation fanout.

---

## 4. Typed lifetime domains

Not all state should expire with the same event.

### Plan lifetime

Depends on alias-binding generation, schema/compiler revision and immutable B-plan assumptions.

Ordinary fact-value updates do **not** invalidate the plan.

### J-data lifetime

Depends on the plan plus exact mutable generations actually read.

### Decode-suffix lifetime

Depends on J-data lifetime plus any generations materialized later during generation.

### Authority-page lifetime

Depends on the current Pod generation and authenticated authority state.

This phase separation prevents world writes from destroying reusable interpretation state.

---

## 5. CLFD — Causal Lifetime Factor DAG

Naively scanning every source generation on every cache/J read is too expensive. CKCA compiles the conjunction into a factor graph:

`generation leaf -> conjunction factor -> O(1) live bit`

When a source generation expires, reverse edges invalidate only causal descendants. Old leaves never revive. New writes create new leaves even when the value bytes are identical.

This provides the runtime anti-ABA rule:

`same_value(old_generation, new_generation) != same_lifetime`.

---

## 6. Late-bound neural materialization

Mutable knowledge should remain a compact handle/token sequence/typed value until the last semantic point requiring neural representation.

Preferred hierarchy:

1. compact symbolic/token ID value;
2. typed relation/pointer/value handle;
3. late-bound token sequence at decoder checkpoint;
4. compact latent/KV Port only when richer neural materialization is necessary.

R309 supports a concrete decoder boundary:

`reusable B-prefix -> verified current result -> small dependent suffix`.

This minimizes the amount of neural state that inherits mutable lifetime.

---

## 7. Governed namespace firewall

A lifecycle-governed relation should not have an uncontrolled second factual address through lexical parametric recall.

After canonical resolution, the governed factual execution path should receive:

- operator/query semantics;
- opaque canonical identity/capability;
- verified current value or explicit null.

It should not rely on an instruction such as “prefer external memory over what you already know.”

On revocation, the governed value channel returns an explicit null/unknown state instead of falling back to pretrained entity-specific factual memory.

This is a governance rule, not a claim that generic entity aliasing is novel.

---

## 8. Persistence and replicas

Authority and lifetime state are durable but separate.

Recovery rule:

```text
restore J/lifetime snapshot @ watermark S
recover authenticated authority ledger to current tip T
replay authority transitions S<T into lifetime invalidation
only then admit restored J/cache artifacts
```

Possession of old bytes, a valid historical signature or an old replica snapshot is never enough to recreate current authority.

---

## 9. Concurrency semantics

A multi-read neural execution can otherwise observe generations that were never jointly current at commit time.

CKCA therefore uses generation read-set validation before publication. On conflict:

- mutable J execution aborts/retries;
- immutable B-plan remains reusable;
- no mixed-generation answer becomes externally visible.

Post-commit lifetime invalidation then handles later writes.

---

## 10. What current experiments support

Current evidence is distributed across mechanism gates rather than one completed product:

- R287: future bindings/aliases/updates can be runtime data rather than learned pairs.
- R290/R293b: compressed fact capsules can preserve exact-Port decisions on bounded probes; naïve PCA failed.
- R291b: active-only authority materialization gives zero logit influence from mutated historical external pages on real-model probes.
- R292: source-only replacement leaves measurable downstream neural contamination; transitive lifetime masking removes it to zero full-vocabulary delta in tested cases.
- R295b: mutable Port/world updates leave the tested frozen B-plane query hidden/logit state unchanged.
- R297: factorized lifetime admission is exact in the audit and makes read-time validity O(1).
- R299/R306: authenticated authority plus replay prevents stale replica/restart state from regaining admission under the declared safety model.
- R301/R314: bounded state exploration finds no stale accept under exact generation rules while weaker semantic/identity-only policies admit many counterexamples.
- R303/R304: many token-representable leaf values can remain extremely compact and be emitted without per-record training; governed alias collapse/null semantics are implementable.
- R305: ordinary non-governed traffic can bypass the knowledge plane with zero logit change and tiny measured control overhead on the CPU gate.
- R309: late-bound decode is exactly equivalent to full recompute on the tested probes while re-executing only a small suffix.
- R310/R311: lazy dynamic read sets and typed lifetime domains materially reduce invalidation fanout and preserve reusable plans.
- R313: read-set commit validation rejects concurrent mixed-generation transactions and retries only J execution.

R312 is the first real-model gate intended to combine several of these pieces in one neural transaction.

---

## 11. What is explicitly NOT the contribution

The project must not claim novelty for any single one of these:

- RAG or better retrieval ranking;
- precomputed KV reuse;
- KV quantization;
- external memory;
- frozen-backbone memory sidecars/cross-attention;
- provenance tracking;
- memory revocation;
- dependency graphs/cascade repair;
- OCC/read-set validation;
- MESI/cache-coherence analogy;
- virtual-memory terminology;
- SSA/lazy execution;
- entity anonymization/aliasing;
- append-only authenticated ledgers;
- copy/pointer decoding.

Those are established techniques or neighboring prior art.

---

## 12. Surviving major-contribution hypothesis

The only claim worth pursuing is the **joint neural knowledge coherence contract**:

> A neural runtime can keep general language/skill state reusable while treating mutable knowledge as canonical monotonic generations; admit only independently verified current generations; record the exact dynamic generation read-set of neural execution; validate it atomically before output commit; attach its transitive lifetime to every retained neural derivative; and bind mutable values late enough that updates/revocations invalidate only the minimal dependent neural suffix — including across restart and replica recovery.

This is still a hypothesis, not a certified novelty claim.

---

## 13. Breakthrough DoD

No “new stone” / major architecture claim before all of the following are simultaneously shown:

1. integrated real-model implementation, not disconnected mechanism proxies;
2. multiple open-model families/scales;
3. unseen post-training entities, relations and multi-token values with no per-record optimizer loop;
4. multi-hop reasoning and free-form multi-token generation;
5. exact online update/revoke/ABA behavior while stale neural bytes remain resident;
6. transitive closure over hidden/J/KV, assistant/tool/cache and speculative artifacts;
7. concurrent update safety through commit validation;
8. restart, stale snapshot and replica replay with zero invalid admissions under a declared consistency model;
9. governed-relation parametric-bypass mitigation;
10. <5% production serving degradation for non-governed traffic and competitive governed-path latency;
11. quality/TTFT/context/write/update/delete frontier against idealized strong RAG, dedicated-memory readers and KV baselines;
12. systematic current literature, source-code and patent audit showing that the **joint protocol over live neural execution artifacts** is not already disclosed.

Until then, CKCA/CKVM is a serious architecture candidate under active falsification, not a proclaimed breakthrough.
