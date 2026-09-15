# CAVI prior-art boundary — 2026-09-05

This note is a novelty **falsification map**, not a novelty or patent claim.  The architecture must remain narrower than the union of established ingredients below.

## Decisive narrowing after fresh prior-art search

A July 19, 2026 IETF Internet-Draft, **PAMSPEC / Architecture and Data Model for Persistent Memory in Agentic Systems** (`draft-infantado-agent-memory-architecture-00`), materially removes a large part of the broad CAVI novelty story.  PAMSPEC already specifies stable canonical Memory Object identity, immutable logical versions, independently versioned Relationship Objects, explicit scope/lifecycle/availability/validation state, an authoritative Persistent State Plane, and non-authoritative derived state including embeddings, generated retrieval summaries, ranking caches and retrieval caches.  It further says relationship traversal re-applies scope/authorization, stale expected versions fail, and deletion/redaction should propagate to derived state.

Therefore **canonical objects + versioned relationships/pointers + authoritative-vs-derived separation + stale derived-cache handling cannot be claimed as CAVI novelty**, even as a composition.  They are now treated as required baselines/architecture hygiene.

The remaining candidate is deliberately narrower and neural-execution-specific.  Internally we call this **CAVI-N / neural-consumption continuity** until it survives falsification:

> A memory-derived neural effect is valid only while the exact authoritative reference binding and referent incarnation that causally licensed it remain live; that dependency is preserved across non-authoritative neural derivations (routing distributions, resolved payload vectors, serialized activations, cached intermediate states) and is revalidated at the *actual later neural injection/consumption boundary*.  Invalid state must collapse to exact no-memory BYPASS, while fresh current-generation state retains capability.  Independent J-space/J-lens measurement may attest causal accessibility but never participates in optimization, routing or authorization.

This is not merely "derived state is non-authoritative"; PAMSPEC already says that.  The testable question is whether **authority lineage survives conversion into otherwise opaque neural tensors and remains enforceable at subsequent model-consumption sites**.  If ordinary cache invalidation/version checking provides the same property, the candidate is dead.

## Current candidate property

**Causally Attested Versioned Indirection (CAVI)** is retained as the implementation umbrella, but no broad novelty is assigned to its individual state-management ingredients:

> A canonical knowledge object is reached through pointer-only aliases; BYPASS / RESOLVE / UNKNOWN are explicit; reference binding and referent incarnation are live-revalidated adjacent to actual neural memory consumption; any memory-derived state that survives beyond that boundary remains dependent on the same live referential witness and must be revalidated at later neural injection/consumption; no-memory scope is an exact bypass; store reachability and causal neural accessibility are audited independently.

The only live differentiator is therefore **end-to-end authority-lineage preservation into and through neural computation**, including post-authorization derived neural material.  Epochs, HMACs, capabilities, pointers, transactions, authorization, freshness, object/version models, derived-state classification, audit probes and cryptographic erasure are not individually claimed.

## Prior art that removes broad claims

| Prior art | What is already established | Consequence for CAVI |
|---|---|---|
| PAMSPEC / `draft-infantado-agent-memory-architecture-00` (Infantado & Leroux, 19 Jul 2026) | canonical stable Memory Objects, immutable versions, independently versioned Relationship Objects, authoritative Persistent State Plane, non-authoritative embeddings/summaries/ranking/retrieval caches, scope enforcement on relationship traversal, expected-version concurrency and derived-index deletion/redaction propagation | **broad canonical-object / versioned-pointer / authoritative-vs-derived CAVI novelty is falsified**; only a neural-consumption-specific lineage invariant remains testable |
| SERAC (Mitchell et al., 2022, `arXiv:2206.06520`) | explicit edit memory + learned scope/routing + counterfactual model | external memory and routing are not novel |
| WISE (Wang et al., 2024, `arXiv:2405.14768`) | side memory, router, lifelong edit sharding/merging | dual memory, routing and sharding are not novel |
| DKME (Zheng et al., ACL Findings 2026) | decoupled semantic addressing and partitioned knowledge storage | address/storage decoupling is not novel |
| Knowledge Externalization (Li et al., ICLR 2026) | modular external memory tokens, reversible forgetting/restoration and post-externalization edits | removable/modular external knowledge is not novel |
| STALE (Chao et al., 2026, `arXiv:2605.06527`) | stale/invalid memory is a first-class agent-memory failure mode | freshness as a problem statement is not novel |
| TEPA (Zhou et al., 2026, `arXiv:2608.07429`) | explicit revocation of superseded evidence memory under current keys, preserving revoked history for audit | revocable validity state and stale-memory exclusion are not novel |
| FreshCache (Mansoor et al., 2026, `arXiv:2607.04281`) | explicit freshness-risk gating for reuse of cached LLM retrieval results | freshness-aware cache reuse is not novel |
| SleepGate (Xie, 2026, `arXiv:2603.14517`) | temporal/conflict tagging and eviction/compression of stale KV-cache associations | stale neural/cache material and selective eviction are not new problem classes |
| MemTX (Li et al., 2026, `arXiv:2607.23929`) | snapshot-isolated transactional belief commit, provenance/validity, cascade repair | transactions, validity metadata and commit discipline are not novel |
| Commit-Time Authorization (Santos-Grueiro, 2026, `arXiv:2607.10487`) | earlier authority evidence cannot authorize a later durable effect after invalidation | revalidation at a later effect boundary is not novel in itself; CAVI-N must show a distinct neural-consumption boundary and derived-tensor attack |
| J-Access / *Measure, Don't Optimize* (Song et al., 2026, `arXiv:2608.11408`) | Jacobian-lens accessibility is useful as an independent recovery-risk audit; optimizing it causes audit evasion / worse recovery | J-space/J-lens stays audit-only; never a CAVI router or training objective |
| canonical pointers / indirection / MVCC / ABA prevention / linearizability | long-established systems semantics | pointer identity, generations/incarnations, ABA-safe handles and atomic read/update semantics are not novel |
| crypto-shredding | long-established cryptographic erasure by key destruction | keys/HMAC/key destruction cannot carry novelty |

## The current differentiating tests

E-000066 showed that an old exported memory Bank can replay deleted knowledge after SHRED/DELETE.  E-000068 showed that a live-incarnation one-use capability can close that replay class, but that mechanism is established systems/security prior art.  E-000070 and E-000071 therefore move validation to real symlink neural consumption and compare full alias+pod referential validation against pod-only version checks and cached commit-time authorization.

E-000073 attacks **post-read derived neural state**.  It captures the real post-symlink-read hidden state, serializes it, changes alias/pod authority, then injects the stale tensor with `bank=None`.  Baselines are no guard, cached commit-time authorization and pod-only version validation.  Full CAVI only passes if it rejects stale derived state at the actual replay/injection hook while the fresh current incarnation remains usable.

E-000074 attacks **cached routing distributions and resolved payload vectors**.  It serializes real router output and reconstructed post-dereference values, changes the alias binding while the old pod remains live/current, then injects the old resolved values directly at adapter read sites with `bank=None`.  This is intentionally downstream of the exported Bank and differentiates full alias+pod lineage from a simple referent-version check.

If E-000073/E-000074 expose replay and full referential revalidation closes it, that still does not establish novelty.  It only establishes that neural derived state creates a real attack surface in this adapter and that authority lineage must cross the tensor boundary.  Required next work includes old-alias/rollback/restore ABA, true concurrent mutation, cached route-index/current-payload cross-generation mixes, multiple seeds/backbones, and E-000063 independent J-lens audit.

## Claim-killing conditions

Do **not** claim CAVI-N novelty if any of the following is true:

1. A simple pod/version equality check performs identically on alias-relink and ABA attacks.
2. Commit-time authorization alone is sufficient when authority changes after resolution but before neural consumption.
3. Stale post-authorization neural state is not actually replayable in a way that resurrects stale knowledge.
4. Stale derived neural state can be made safe by ordinary cache invalidation/version tags with no need to carry authoritative reference lineage to neural consumption.
5. The boundary is semantically equivalent to PAMSPEC's non-authoritative Derived Index handling or ordinary relationship/version traversal, with no neural-consumption-specific correctness property.
6. Exact BYPASS requires hidden memory injection or materially perturbs the base model.
7. J-space/J-lens must be optimized or used as a router for the architecture to work.
8. The composed benefit disappears across seeds/backbones or under fresh-current controls.
9. The independent causal audit cannot distinguish current live-memory influence from query/base-model influence under NEVER-memory controls.

## Sources checked on 2026-09-05

- https://ftp.kaist.ac.kr/ietf/draft-infantado-agent-memory-architecture-00.html — PAMSPEC / Architecture and Data Model for Persistent Memory in Agentic Systems, published 19 Jul 2026 (Internet-Draft, work in progress)
- https://arxiv.org/abs/2206.06520 — SERAC / Memory-Based Model Editing at Scale
- https://arxiv.org/abs/2405.14768 — WISE
- https://aclanthology.org/2026.findings-acl.792/ — DKME
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html — Knowledge Externalization
- https://arxiv.org/abs/2605.06527 — STALE
- https://arxiv.org/abs/2608.07429 — TEPA
- https://arxiv.org/abs/2607.04281 — FreshCache
- https://arxiv.org/abs/2603.14517 — SleepGate
- https://arxiv.org/abs/2607.23929 — MemTX
- https://arxiv.org/abs/2607.10487 — Commit-Time Authorization
- https://arxiv.org/abs/2608.11408 — J-Access / Measure, Don't Optimize
