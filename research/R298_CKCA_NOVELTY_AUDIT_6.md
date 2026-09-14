# CKCA Novelty Audit 6 — Transactional Agent Memory and Continuity Kernels

Date: 2026-09-14. Status: adversarial claim narrowing, **not novelty certification**.

Recent work makes the transactional/control-plane neighborhood substantially more crowded than earlier audits assumed.

## MemTX occupies transactional belief commit + cascade repair

`MemTX: Transactional Belief Commit for Stateful Agent Memory` (arXiv:2607.23929) stages memory changes inside snapshot-isolated transactions, validates before commit, gates irreversible actions on in-flight belief state and triggers typed cascading repair when a belief is retracted. It reports bounded exhaustive/property evidence for action-safety and repair completeness.

Therefore CKCA cannot claim novelty for:

- treating agent-memory updates as transactions;
- snapshot-isolated memory writes;
- validate-and-commit before downstream action;
- retraction-triggered cascade repair;
- transactional memory as a general safety primitive.

## Continuity Kernel occupies authoritative-head activation contracts

`Beyond Memory: A Transactional Continuity Kernel for Long-Lived AI Agents` (arXiv:2608.11632) explicitly distinguishes retained state from **authoritative activated state**, uses exact predecessor heads, validates ownership/freshness before activation and advances an accepted branch head atomically. Its bounded executable model explores millions of states with zero modeled invariant violations.

Therefore CKCA cannot claim novelty for:

- saying retained bytes are not automatically authoritative;
- an explicit authority/control plane over persistent agent state;
- typed activation/commit contracts;
- monotonic branch heads / accepted state lineages;
- bounded formal verification of such control-plane transitions.

## Concurrency hierarchy work occupies stale-generation transaction safety

`Verified Detection and Prevention of Concurrency Anomalies in Multi-Agent Large Language Model Systems` (arXiv:2606.17182) formalizes stale-generation and related anomalies, implements locking/snapshot-isolation style runtimes and machine-checks parts of the consistency hierarchy.

Together with S-Bus, this kills any broad claim around generation-aware OCC, read-set commit barriers or snapshot consistency for LLM *agent state*.

## What still survives as the target

The research question is now narrower again:

> Can the same authority/transaction semantics be extended **inside live neural execution**, so that hidden/KV/J/decoder artifacts which have already absorbed mutable world knowledge inherit exact generation lifetimes and become causally inadmissible after source expiry — while immutable language/skill state remains reusable and old neural bytes may remain physically resident?

This is materially different from claiming a transactional memory store, because the object under coherence is the neural execution artifact itself.

The surviving joint target requires:

1. **neural materialization boundary** — current authority determines which mutable state may enter hidden/KV/J execution;
2. **exact neural read-set lifetime** — every retained neural derivative carries the exact generations that causally produced it;
3. **pre-publication barrier + post-publication lifetime** — transaction consistency and later invalidation are both required;
4. **causal-byte irrelevance** — once invalid, arbitrary mutation of physically resident stale neural bytes leaves current logits unchanged;
5. **late binding** — mutable values are materialized only in the minimal neural suffix that needs them;
6. **immutable-base reuse** — fact updates do not destroy unrelated language/skill/query-plan state;
7. **parametric-bypass governance** — governed relations cannot silently fall back to obsolete lexical parametric recall;
8. **restart/replica neural admission** — restored neural artifacts remain blocked until current authority replay revalidates their lifetime;
9. **quality/latency frontier** — the complete system must beat or match strong retrieval/dedicated-memory alternatives on a joint metric, not only safety.

## Implication for R313/R314 language

The Neural Commit Barrier and Causal Knowledge Transaction are **required architecture composition**, not standalone novelty. OCC and transactional belief commit are occupied.

R314's useful contribution to the research program is the counterexample structure: commit-time validation alone does not stop a once-valid neural artifact becoming stale later, while post-hoc lifetime alone does not stop an inconsistent in-flight read-set from being committed. The possible claim can only involve how those known semantics are mapped onto and enforced over live neural derivatives.

## Current claim discipline

Avoid novelty language around:

- transactional agent memory;
- snapshot isolation;
- stale-generation conflict detection;
- read-set validation;
- atomic authority-head activation;
- cascade repair;
- provenance/lineage ledgers.

The candidate “new stone” is now strictly **generation-coherent neural execution**, if and only if the integrated real-model, benchmark and broader literature/patent gates continue to survive.
