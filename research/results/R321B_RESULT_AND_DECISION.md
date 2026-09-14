# R321b Result — Generation-Aware Neural Page Table (GANPT)

Status: **real-model KV authority gate passed; not DoD and not a standalone novelty claim.**

## Problem

R321 showed that keeping an expired mutable suffix inside the logical attention sequence and merely masking those positions changed full-vocabulary logits by up to `0.125`, even though corrupting the masked stale K/V values did not worsen the delta. The stale bytes were not influencing attention, but the changed cache geometry itself violated the exact-current oracle.

R321b therefore changes the abstraction: **physical KV residence is not logical neural membership**.

A generation-aware page table admits only current/live KV pages into the logical attention working set. Expired pages may remain physically resident and can be reclaimed asynchronously.

## Frozen Qwen2.5-0.5B result

Seven real-model generation transitions:

- max direct-current splice vs page-table path full-vocabulary logit delta: **0.0**;
- max uncorrupted vs aggressively corrupted stale backing pages delta: **0.0**;
- max full-current recompute vs page-table path delta: **0.0**;
- every stale-backing corruption changed physical K/V bytes: **true**;
- median stale physical bytes deliberately retained but unadmitted: **61,440 bytes**;
- reference-Python page-table materialization median: **666,997 ns** (~0.667 ms).

Report SHA256: `07f1a082e134f987872ec86a9ca59df986533275a574425b062dc7fb9b14d9ea`.
Artifact ZIP SHA256: `5fb05a6dd97871a4573bd759a8621a9cba9a5fa43078f6cf9f47c6d6e772b694`.

## Architectural decision

Promote **GANPT — Generation-Aware Neural Page Table**:

```text
physical KV pages
    │
    ├── old generation pages may remain resident
    └── current generation pages
             │
             ▼
     generation/lifetime authority
             │
             ▼
       logical page table
             │
             ▼
      attention working set
```

The core invariant becomes:

> `physical possession != logical position != semantic authority`.

Revocation/update can first flip generation/page admission and invalidate CLFD descendants. Physical KV memory reclamation can occur later without making stale bytes semantically live.

The Hugging Face reference implementation concatenates admitted ranges to reproduce the exact clean logical geometry. A production implementation should push the same generation-lifetime admission predicate into a paged-attention/block-table kernel so page selection is indirection, not tensor copying.

Paged attention, block tables, cache indirection and MVCC are established. The research target is the generation-lifetime authority semantics plus their composition with CKCA, not page tables by themselves.
