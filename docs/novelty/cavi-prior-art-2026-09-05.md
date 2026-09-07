# CAVI prior-art and falsification boundary — 2026-09-05

This note narrows the research claim. It is **not** a novelty or patentability opinion. Individual ingredients below are treated as prior art unless a narrower neural-execution property survives direct baselines.

## Decisive collision: PAMSPEC

The 19 July 2026 IETF Internet-Draft `draft-infantado-agent-memory-architecture-00` (**Architecture and Data Model for Persistent Memory in Agentic Systems / PAMSPEC**) materially falsifies the broad CAVI composition as a novelty target.

PAMSPEC already specifies:

- persistent Memory Objects with stable logical identity;
- immutable logical Memory Versions;
- independently identified, typed and versioned Relationship Objects;
- explicit scope, lifecycle, availability, retention and validation state;
- an authoritative Persistent State Plane separate from a transient Compute Plane;
- embeddings, generated summaries, ranking caches and retrieval caches as **non-authoritative derived state**;
- expected-version optimistic concurrency;
- authorization/scope evaluation while traversing relationships;
- deletion/redaction propagation concerns for derived indexes/caches.

So **canonical object + versioned pointer/relationship + authoritative-vs-derived state + freshness/cache invalidation is no longer a defensible CAVI novelty**, even as a high-level composition. PAMSPEC is an Internet-Draft/work in progress rather than a final standard, but it is enough to kill that broad research novelty framing.

## Confirmed nearby work

| Work | What it already covers | Consequence for CAVI |
|---|---|---|
| PAMSPEC (Infantado & Leroux, Internet-Draft, 2026-07-19) | canonical versioned Memory Objects; versioned relationships; authoritative vs derived state; scope-aware traversal; stale derived caches/indexes; expected-version concurrency | broad CAVI object/version/pointer/derived-state composition is prior-art-collided |
| STALE (arXiv:2605.06527, 2026-05-07) | stale-memory detection, state resolution, premise resistance, write-side state adjudication | “detect stale memory” is not new |
| TEPA (arXiv:2608.07429) | explicit validity/revocation of superseded evidence memory | revocable memory validity is not new |
| FreshCache (arXiv:2607.04281) | freshness-risk gating for cached LLM retrieval reuse | freshness-aware cache reuse is not new |
| SleepGate (arXiv:2603.14517) | conflict/temporal tagging and eviction/compression of stale KV associations | stale neural/cache material is an established problem class |
| MemTX (arXiv:2607.23929, 2026-07-27) | snapshot-isolated belief transactions, validity/provenance, validate-and-commit, cascade repair | transactions/MVCC/validity/commit discipline are not new |
| Commit-Time Authorization (arXiv:2607.10487, 2026-07-11) | freshness/binding/eligibility revalidation at a later durable-effect boundary | final-boundary freshness authorization is not new in itself |
| DKME (Findings ACL 2026.792) | decoupled semantic addressing + partitioned knowledge-memory storage | address/storage separation is not new |
| SERAC | external counterfactual memory + learned scope classifier | external memory + scope routing is not new |
| WISE | dual parametric memory, routing, sharding for lifelong editing | modular/sharded edit memory is not new |
| Knowledge Externalization (ICLR 2026) | removable/editable external memory tokens and reversible knowledge restoration | externalized editable knowledge objects are not new |
| J-Access (arXiv:2608.11408) | Jacobian-lens audit of residual knowledge; optimizing the audit causes evasion | J-space/J-lens stays an independent audit, never a training target |
| generational handles / versioned pointers / MVCC / linearizability | stale-handle rejection, ABA protection, versioned loads and atomic read/update semantics | incarnation/version checks and atomicity are standard systems techniques |
| crypto-shredding / secure deletion | key destruction and versioned secure-deletion graphs | HMACs, keys, epochs, key erasure and “crypto shred” are not new |

Sources used in this update:
- https://ftp.kaist.ac.kr/ietf/draft-infantado-agent-memory-architecture-00.html
- https://arxiv.org/abs/2605.06527
- https://arxiv.org/abs/2608.07429
- https://arxiv.org/abs/2607.04281
- https://arxiv.org/abs/2603.14517
- https://arxiv.org/abs/2607.23929
- https://arxiv.org/abs/2607.10487
- https://aclanthology.org/2026.findings-acl.792/
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e9c2053258b1bdd32ff2654802cd594-Abstract-Conference.html
- https://arxiv.org/abs/2608.11408

## Remaining candidate: CAVI-N

`CAVI` remains the implementation umbrella. The only live novelty subclaim is **CAVI-N / neural-consumption continuity**:

> Memory-derived neural material — Bank rows, routing distributions, route selections, resolved payload vectors, cached activations and serialized intermediate tensors — does not become self-authorizing after a valid resolve. Its original authority lineage, bound to the exact reference/alias incarnation and referent/pod incarnation, remains revocable and is revalidated at any later *actual neural injection/consumption site*. Invalid lineage collapses to exact no-memory execution; fresh current-generation state remains usable. Independent J-space/J-lens is audit-only.

The claim is interesting only if the **tensor boundary creates a real stale-knowledge resurrection** that PAMSPEC-style derived-state classification, a simple pod/version tag, ordinary cache invalidation and cached commit-time authorization do not already close.

## Critical falsification matrix

### F1 — alias relink while both pods stay live

Capture state derived from alias `A -> P`. Relink `A -> Q` while `P` and `Q` remain live and `P`'s incarnation is unchanged.

- no guard: stale `A -> P` state should remain replayable if the threat is real;
- cached commit/export-time authorization: should remain stale after resolve→mutate→consume;
- pod-only version check: should still approve because `P` remains current;
- full reference+referent lineage at neural consumption: must reject the old `A -> P` derivation.

If a simple pod/version check performs identically, CAVI-N is falsified.

### F2 — neural-derived-state bearer attack

After a legitimate live memory read, serialize and later replay each layer independently:

- exported Bank row;
- router/routing distribution;
- selected route/index;
- resolved payload vector;
- post-read hidden activation.

Replay with `bank=None` after authority changes. If stale neural material cannot resurrect stale behavior, there is no reason to add lineage machinery downstream of ordinary cache invalidation.

### F3 — ABA / rollback / restore

Old derived state must stay dead after delete/shred + restore/recreate under the same logical identity. Generational handles are expected to solve plain referent ABA; CAVI-N only gets credit for reference+referent or neural-derived-state cases that stronger ordinary baselines miss.

### F4 — exact scope semantics

- `BYPASS`: exactly the no-memory base path;
- `RESOLVE`: current lineage remains usable;
- `UNKNOWN`: in-scope stale/missing reference does not silently become BYPASS.

### F5 — race / linearization

Mutation in the resolve→inject gap and true concurrent reference/object updates must not allow stale derived state to cross the neural consumption point. Locking itself is not novel; the question is whether the *neural consumption linearization point* is a necessary correctness boundary.

### F6 — ordinary-systems-equivalent baseline

Implement the strongest honest baseline:

- PAMSPEC-like authoritative/derived separation;
- cache invalidation/version tags;
- generational referent handle;
- commit-time authorization;
- normal concurrency/linearizability controls.

CAVI-N survives only if a real stale neural derivation remains dangerous under that baseline and exact reference+referent authority-lineage validation at neural consumption closes it.

### F7 — independent audit

E-000063 J-space/J-lens measurements are independent audit only. They must use NEVER-memory controls and never participate in optimization/routing/authorization.

## Current attack sequence

- **E-000066** — stale exported Bank replay: vulnerability reproduced 20/20.
- **E-000068** — live-incarnation one-use capability: control passed 5/5; known primitive, not novelty.
- **E-000070** — real trained symlink neural consumption, full lineage vs pod-only/commit baselines.
- **E-000071** — actual read-hook TOCTOU/linearization attack.
- **E-000073** — serialized post-read hidden-state replay with `bank=None`, plus relink/ABA/race controls.
- **E-000074** — serialized routing distribution + resolved payload replay with `bank=None`, plus full vs pod-only vs commit-time baselines.
- **E-000072** — staged scope/performance line; J-space routing remains removed.
- **E-000063** — independent NEVER-controlled causal workspace audit.

## Promotion rule

No screening result is a novelty claim. CAVI-N can be promoted only if:

1. stale neural derived state demonstrably resurrects stale knowledge;
2. full alias+pod lineage at neural consumption closes it;
3. pod/version, ordinary cache invalidation/version tags and cached commit-time authorization are materially weaker on a pre-registered case;
4. fresh current-generation capability remains intact;
5. the result survives >=3 seeds and >1 public backbone/materially distinct model setting;
6. rollback/restore, old-alias, serialized tensor, cached router/payload and concurrent replay attacks pass;
7. exact BYPASS/locality and E-000063 independent J-lens audit pass;
8. a final targeted search does not find prior art for lineage-carrying neural intermediate state with consumption-time revocation/revalidation.
