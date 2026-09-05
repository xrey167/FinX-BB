# E-000086 — heterogeneous neural coherence closure

Date: 2026-09-05
Status: **pre-registered candidate experiment; not a novelty claim**

## Why this experiment exists

E-000080 showed object-scoped selective reuse for real KV caches, but with controlled residual payloads rather than a strict-capability real symlink reader. E-000084 and E-000085 move the strong real LINK->Pod->dereference reader into lifecycle and downstream replay attacks.

A fresh prior-art search found a very close 2026 system, **LineageKV**, which performs lineage-guided stale-memory repair directly inside reused Transformer KV caches after records are superseded. That result means the following are explicitly **not** novelty candidates here:

- lineage-guided KV cache repair;
- stale-span suppression/editing in KV;
- selective cache reuse after memory updates;
- version/dependency metadata;
- update-vs-reprefill speedups.

The remaining question is narrower and specifically tied to the Symlink-Pod architecture: can one canonical, non-token knowledge-object generation define a coherence domain across *heterogeneous neural-derived state classes* produced by the same internal memory read?

## Hypothesis

For one real linguistic alias that resolves through an explicit LINK to one canonical Pod, a successful neural memory read produces several reusable artifacts:

1. exported/encoded Bank state;
2. routing distribution / selected route;
3. resolved payload after dereference;
4. post-memory hidden activation;
5. autoregressive KV state derived after that memory injection.

All five artifacts are tagged with the same alias-qualified Pod-generation witness:

`(alias_id, alias_incarnation, pod_id, pod_incarnation)`

After a lifecycle transition, every artifact derived from the invalid generation must be rejected before reuse. Artifacts whose lineage excludes the changed Pod must remain reusable and must match a fresh rebuild.

The proposed systems contract is therefore:

`one canonical knowledge lifecycle transition -> invalidate derivation closure of that generation, preserve unrelated neural state`

This is intentionally stronger than a KV-only lineage repair and intentionally narrower than generic dependency tracking.

## Validity gate — fixed before results

No coherence result is interpretable unless the same trained reader satisfies all of:

- every held-out real-symlink template >= 0.95 correctness;
- target alias resolves through a real LINK+deref path, not a controlled direct payload injection;
- fresh current generation remains answerable after the lifecycle operation when the operation semantics permit it;
- exact BYPASS is checked with memory omitted rather than with a small soft gate.

A seed that fails this gate is recorded but contributes no positive coherence evidence.

## Lifecycle attacks

At minimum run independently:

- alias RELINK from live Pod P to live Pod Q;
- Pod UPDATE;
- REVOKE -> RESTORE;
- SHRED -> RESIGN/RESTORE when defined;
- DELETE;
- ABA/same logical Pod id at a newer incarnation;
- rollback to an authorized newer incarnation without reviving stale witnesses;
- mutation during forward between resolve and later derived-state consumption.

## Artifact replay attacks

Capture and serialize each artifact after one valid real-symlink read. After the lifecycle transition, replay each artifact without silently rebuilding upstream state.

For every artifact class compare:

1. **unguarded** — stale artifact trusted;
2. **global epoch** — any knowledge mutation rejects all artifacts;
3. **artifact-local version only** — local source/version check without alias binding;
4. **Pod-only** — canonical pod generation checked, alias generation omitted;
5. **full object coherence** — alias binding + canonical pod generation validated at the actual consumption boundary;
6. **fresh rebuild** — gold current computation.

A positive row requires unguarded/baseline stale replay to differ materially from current gold where the attack is intended to be observable, while full object coherence rejects/recomputes to the current gold and unrelated artifacts remain exact.

## Heterogeneous closure criterion

Per valid seed, PASS only if all present artifact classes satisfy their class-specific rows:

- Bank stale state rejected;
- router/selected-route stale state rejected;
- resolved-payload stale state rejected;
- post-read hidden state stale replay rejected;
- KV stale replay rejected;
- unrelated artifact reuse equals fresh rebuild within the registered numerical tolerance;
- full object-coherence decision is made from one authority snapshot where multiple dependencies are validated.

A result covering only KV is **not E-000086 PASS**.

## Independent attestation

J-space/J-lens is not used for routing, training, versioning, invalidation or repair.

After the runtime coherence test, independently compare ACTIVE, invalidated and NEVER-memory controls. The audit may support a statement that the runtime-visible lifecycle transition is also reflected in the causal workspace, but it cannot replace the runtime correctness reference and is not itself claimed as novel.

## Utility comparison

The strongest systems comparison is not stale reuse vs full rebuild alone. Compare:

- global cache epoch / full flush;
- ordinary per-artifact dependency/version tags;
- LineageKV-style KV-only stale-memory repair where applicable;
- the heterogeneous object-coherence contract;
- full rebuild / re-prefill.

Measure:

- mutation-to-ready latency;
- normal inference overhead;
- fraction of unaffected neural-derived state retained;
- memory/metadata cost;
- alias fan-out scaling;
- number of lifecycle mutations needed per canonical fact;
- stale-resurrection failure rate.

Prospective major-usefulness bar, fixed before the final systems run:

- >=10x mutation-to-ready advantage over the strongest guarantee-matched baseline on a declared long-lived-agent workload, **or** a comparably large retained-cache/throughput advantage that cannot be obtained with a simpler KV-only contract;
- <=5% normal inference throughput loss;
- matched or explicitly reported memory budget;
- >=3 independent seeds;
- >1 public backbone where CPU/GPU feasible.

## Prior-art boundary

Explicitly excluded from novelty:

- cache invalidation/coherence generally;
- source versioning and generation counters;
- dependency tags and lineage graphs;
- MVCC/snapshot isolation;
- capabilities and effect-boundary authorization;
- prompt/token-span lineage;
- KV-cache editing, pruning, selective recomputation and stale-span repair;
- symlinks/pointers individually;
- external/editable memory individually;
- J-space/J-lens/J-Access individually.

The only remaining candidate seam is the **cross-layer composition**: a canonical, internal, non-token knowledge-object generation whose alias-qualified lifecycle identity propagates as one coherence domain through multiple neural-derived representations and whose causal effect is independently audited.

Even a complete E-000086 PASS would be research evidence for that composed systems property, not proof of legal novelty or 'first ever'.
