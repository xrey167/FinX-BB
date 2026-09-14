# R324 Result — Causal Semantic Object Graph / Virtualized Transcript

Status: **transcript-lifecycle mechanism gate passed; not DoD and not a standalone novelty claim.**

## Problem

Even perfect KV invalidation is insufficient if a previous assistant message containing a governed fact is later copied verbatim into a new prompt. Flat conversation history can become a second stale factual address.

R324 stores governed transcript content as semantic segments instead of treating the old rendered string as future authority:

```text
Text("status = ")
CurrentSlot(PodRef)
HistoricalRef(PodRef, generation)
```

The archival rendered bytes remain available for audit, but active model context is generated from current slots and opaque historical references.

## Result

- Pods: **10,000**
- semantic messages: **50,000**
- current slot references: **200,000**
- world updates: **12,000**
- same-value generation updates: **4,357**
- revocations: **1,481**
- active incremental render vs full rerender mismatches: **0**
- archival snapshots mutated by world updates: **0**
- unreferenced world updates that changed active context: **0**
- average messages patched per update: **20.025**
- average message patch fraction: **0.04005%**
- p99 messages patched per update: **31**
- median Python patch cost: **81.9 µs**

Stale archive attack:

- **516** archival messages were physically corrupted;
- archive digest changed: **true**;
- active model-context digest changed: **false**.

Report SHA256: `17ac65db6c6e4ccb547d7a82c4e9d50fc606bc86e2ab4e29f28000c166ae0ede`.
Artifact ZIP SHA256: `d2e86b8307ed932cc214356d26dfd4d3ff73227f618d6adf7c7dfbec12b1252c`.

## Architectural decision

Promote **CSOG — Causal Semantic Object Graph**:

> Governed factual spans in persistent interaction state are references to lifecycle-controlled semantic objects, not permanently authoritative flattened strings.

This closes a resurrection path outside model KV/hidden state. Historical rendered text can remain physically present for audit without being re-admitted as current factual context.

Structured documents, reactive UIs, materialized views and provenance annotations are established. R324 tests their generation-scoped use in CKCA; component-level novelty is not claimed.
