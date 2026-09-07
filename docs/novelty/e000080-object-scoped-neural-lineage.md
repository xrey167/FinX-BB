# E-000080 — object-scoped neural-derived-state lineage

Date: 2026-09-05
Status: **useful systems correction; not a breakthrough or novelty claim**

## Question

E-000079 showed that E-000078's first-memory-read -> last-memory-read snapshot boundary is too short for autoregressive generation: old memory can causally influence downstream KV state, a lifecycle mutation can commit before the prefill returns, and a later no-memory token forward can consume that stale KV.

The cheapest repair would be one global cache generation: any pod change invalidates every cache. That closes freshness but destroys practical reuse. E-000080 asks whether the correction can instead be **object-scoped**: a reusable neural-derived object carries the exact alias+pod incarnation witnesses that contributed to it, so only state depending on changed knowledge is rejected.

This is deliberately a systems/correctness prototype. Version tags, dependency sets, selective cache invalidation and recomputation are established techniques and are explicitly excluded from any novelty claim.

## Implementation

`so/derived_lineage.py` adds a conservative `DerivedLineage` over exact CAVI `ResolveWitness` tuples:

`(alias_id, alias_incarnation, pod_id, pod_incarnation)`

A `LineagedState` is reusable only while every dependency witness is still current in the independent authority. Alias qualification is necessary because RELINK can invalidate an alias-derived state while the old pod remains live and unchanged.

Logical metadata for one dependency is 32 bytes (four uint64 fields). This is the packed logical size, not Python object overhead.

## Real-KV experiment

Workflow run: **33965239265**
Backbones: **distilgpt2** and **EleutherAI/pythia-70m-deduped**
Seeds: **0, 1, 2** on each backbone
Unit/contract job: PASS
Both backbone jobs: PASS

For each seed, two independent memory-bearing prefills are built with real HuggingFace `past_key_values`:

- cache A depends on alias A -> pod A;
- cache B depends on alias B -> pod B.

Only pod A is updated. The experiment then checks that A's lineage is stale, B's remains current, unguarded old-A cache reuse diverges from current-A recomputation, reject+recompute repairs A exactly, and unchanged B cache reuse remains identical to a fresh B rebuild.

Controlled residual payloads make the memory effect deterministic. Therefore these runs do **not** satisfy or bypass the >=0.95 real-symlink reader prerequisite for positive CAVI claims.

### Results

| Backbone | Seeds | stale A vs current A | repair A vs current | unchanged B reuse vs rebuild | B rebuild / reuse CPU ratio | lineage metadata |
|---|---:|---:|---:|---:|---:|---:|
| `distilgpt2` | 3/3 | max-abs **27.239786**, KL **6.105278 nats**, top-1 8471 -> 314 | **0.0 max-abs** | **0.0 max-abs** | **2.97x–3.09x** | **32 B** |
| `EleutherAI/pythia-70m-deduped` | 3/3 | max-abs **2.0**, KL **0.135983 nats** | **0.0 max-abs** | **0.0 max-abs** | **3.98x–4.57x** | **32 B** |

On the measured CPU runs, unchanged-state KV reuse took about 12.4 ms versus 36.9–38.7 ms for rebuild+decode on distilgpt2, and about 6.35–7.22 ms versus 28.3–29.0 ms on Pythia-70m.

## Fan-out utility

A separate authority scaling test binds `k = 1, 10, 100, 1,000, 10,000` aliases to one canonical pod, captures old alias-derived witnesses, and updates the pod once.

Across the full range:

- one canonical pod lifecycle operation makes **all k** old alias-derived witnesses stale;
- no alias incarnation is edited;
- the canonical authority mutation stays approximately **0.39–0.43 us median** in this in-process prototype;
- the duplicate-object baseline conceptually requires `k` object edits;
- storing one independent single-dependency lineage per cached alias state costs `32*k` logical bytes.

This does **not** mean total invalidation work is O(1): enumerating or validating all cached states is still O(k). The useful property is that the **mutation itself** need not traverse aliases; stale detection can be lazy at reuse.

## Locality utility

With `N = 2, 10, 100, 1,000, 5,000` independent pods and one cache per pod, updating only pod 1 makes exactly one object-scoped lineage stale while a global epoch would stale all N caches.

Thus the measured recomputation fraction under object-scoped lineage is `1/N` rather than `1.0`: 50%, 10%, 1%, 0.1%, and 0.02% across the tested sizes.

## What this establishes

E-000080 gives the E-000079 architecture correction a concrete engineering benefit: **lifecycle-safe reuse can be selective rather than globally destructive**, and the benefit appears with real autoregressive KV on two public backbones.

It also preserves the symlink fan-out systems property: one canonical lifecycle change can invalidate old neural-derived state reached through many aliases without editing each alias.

## What it does NOT establish

E-000080 is not a breakthrough because:

- dependency/version metadata, selective invalidation and recomputation are established systems techniques;
- the memory payload is controlled, not produced by a >=0.95 three-seed real-symlink reader;
- the dependency certificate is conservative route provenance, not yet an independently measured causal attribution;
- KV is only one derived-state class; hidden/activation, Bank, selected-route and resolved-payload attacks still need one unified contract;
- the joint paraphrase/REVOKE/SHRED/leakage/UNKNOWN/locality gate is not met;
- J-space/J-lens has not yet independently corroborated the runtime dependency;
- direct prior-art searching remains required for neural runtime state carrying mutable knowledge-object dependency lineage.

## Narrowed research seam

The remaining potentially research-worthy seam is **neural derivation closure**, not generic cache invalidation:

> A canonical mutable knowledge pod has many linguistic aliases. When its current incarnation causally contributes to reusable neural runtime state, a compact object-scoped derivation certificate follows that influence across all reusable forms. One lifecycle transition makes every derivation depending on the old incarnation unusable, leaves unrelated neural state reusable, and allows the current incarnation to be resolved again. An independent causal workspace audit must corroborate which pod actually contributed.

That statement is still a hypothesis. E-000080 proves only a controlled KV subset and its selective-reuse utility.
