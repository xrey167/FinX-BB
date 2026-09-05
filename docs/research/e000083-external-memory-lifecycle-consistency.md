# E-000083 — External-memory lifecycle consistency

Status: **pre-registered falsification experiment; no novelty claim**

## Motivation

Current prior art closes the broad novelty claims around editable external factual memory, alias-closure deletion, direct KV knowledge injection, generic rollback consistency and source-local/composable caches. The remaining question is narrower: what happens when an **authoritative external-memory lifecycle change occurs after the old knowledge has already causally entered retained neural runtime state**?

E-000083 is designed to falsify that seam against the strongest ordinary baselines before any new mechanism is promoted.

## Hard prerequisite

Do not interpret stale-state attack results unless the same retained reader candidate satisfies all of:

- fresh real-symlink/current-memory correctness >= 0.95 for every one of >=3 seeds;
- held-out paraphrase reading >= 0.95;
- REVOKE >= 0.95;
- SHRED >= 0.95;
- deleted-object leakage <= 0.02;
- missing-key UNKNOWN >= 0.90;
- generic active-memory KL <= 0.05 nats, or exact no-memory BYPASS;
- preferably a second public backbone/materially distinct model setting.

The current E-000077/E-000081 capability line does not yet clear the first condition, so the first E-000083 milestone is still capability qualification.

## Lifecycle states

For an alias `A` and canonical pod `P`, test at least:

1. `CURRENT`: A -> P[g]
2. `UPDATE`: P[g] -> P[g+1]
3. `RELINK`: A[gA] -> P[gP] becomes A[gA+1] -> Q[gQ] while P stays live
4. `REVOKE`: P remains represented but no longer authorizes a read
5. `SHRED`: old payload becomes unavailable by construction
6. `ABA`: delete/recreate or rollback/restore under the same logical identities but new incarnations

## Captured neural-derived state classes

Capture and replay each class independently, then in realistic combinations:

- encoded Bank row/material;
- routing distribution;
- selected route/index;
- resolved payload vector;
- post-read hidden state;
- residual-stream checkpoint;
- KV cache / activation cache;
- generated internal/history tokens or summaries that were themselves downstream of the old pod.

A state class only counts as a vulnerability if stale versus fresh-current execution differs under a controlled causal audit. Structural metadata staleness alone is not enough.

## Primary audit: same-token / different-neural-state

Where technically possible, hold the post-mutation input tokens, decoding settings and authoritative current store identical. Vary only the retained neural state:

- `STALE`: state captured before lifecycle mutation;
- `FRESH`: full recompute from the current authoritative store;
- `RESET`: complete cache/session reset + current store;
- `VERSIONED`: exact object/incarnation dependency tags with lazy invalidation and recompute;
- `TX-RESTORE`: transaction-local clean-prefix restore/truncate/re-prefill;
- `RESOURCE-LOCAL`: source/resource-isolated cache baseline where applicable;
- `CANDIDATE`: only if a proposed neural-specific mechanism exists.

The gold reference is `FRESH`, not the original pre-mutation output.

## Baselines that may NOT be weakened

The strongest ordinary baseline is allowed to use:

- full reset and fresh recomputation;
- alias + pod generations/incarnations;
- complete transitive dependency sets;
- integrity binding between dependency metadata and neural material;
- lazy reuse-time validation;
- selective invalidation/recompute;
- transaction-local restore to a known-clean frontier;
- residual-stream checkpoints rather than full KV where cheaper;
- resource-local/composition-invariant cache blocks;
- regeneration of generated history after the first affected causal frontier.

If this closes the channel with reasonable cost, the novelty candidate is falsified even if a custom mechanism also works.

## Required measurements

### Correctness/security

- stale old-answer rate after UPDATE/RELINK/REVOKE/SHRED/ABA;
- fresh-current answer rate after UPDATE/RELINK;
- deleted-object leakage;
- UNKNOWN on missing/deleted key;
- exact BYPASS or generic KL;
- stale-vs-fresh max-abs logits and KL;
- same-token stale-vs-fresh top-1/effect flips;
- replay/splice attacks that pair fresh metadata with old neural material;
- dependency-omission attacks where a tensor consumed multiple objects but names only one;
- concurrent lifecycle mutation during multi-read inference;
- generated-history contamination after an old fact already changed emitted tokens.

### Utility/performance

For alias fan-out `k` and independent active objects `N`:

- lifecycle mutation latency;
- number/fraction of cached states invalidated;
- recompute FLOPs/tokens/layers;
- rollback/regeneration frontier length;
- TTFT and decode latency overhead;
- memory footprint of dependency metadata/checkpoints;
- one canonical mutation versus O(k) duplicate-object edits;
- retained reuse benefit for unrelated objects;
- throughput under mixed read/update workloads.

Report p50/p95 where repeated timing is meaningful.

## Independent causal audit

J-space/J-lens is audit-only:

- NEVER-memory controls must be included;
- it must not participate in routing, authorization, training or lifecycle decisions;
- the audit should test whether the old pod still has a measurable causal effect after the lifecycle operation and repair;
- disagreement between runtime lineage and independent causal audit must be preserved as a failure/uncertainty, not optimized away.

## Decision table

| Outcome | Decision |
|---|---|
| No stale neural state can resurrect old knowledge after authoritative mutation | cross-layer seam falsified; stop |
| Stale state resurrects old knowledge, but ordinary version/dependency + recompute closes it cheaply | engineering bug/correction only; no novelty |
| Stale state survives ordinary complete dependency handling | inspect control validity first; likely missing dependency, not novelty |
| A neural-specific mechanism matches fresh-current semantics and materially reduces repair cost versus strongest ordinary baselines | continue to 3-seed, >1-backbone, scaling and final prior-art validation |
| Any capability prerequisite fails | do not interpret positive CAVI attack results |

## Promotion threshold

No breakthrough language unless the final retained design simultaneously:

1. clears every capability/security gate above on >=3 seeds;
2. demonstrates a real stale-neural-state failure on a qualified reader;
3. closes every tested Bank/router/route/payload/hidden/residual/KV/history replay path;
4. preserves fresh-current memory capability and exact/no-damage bypass;
5. passes adversarial relink, ABA, splice, dependency-omission and concurrency tests;
6. has a material performance/scaling advantage over full reset, ordinary dependency invalidation, transaction-local restore and resource-local cache baselines;
7. survives independent J-space/J-lens audit;
8. survives a final 2025-2026 papers/standards/patents search for the exact cross-layer property.

Anything weaker remains a negative result, systems correction or research hypothesis.
