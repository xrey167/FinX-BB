# R326 Result — Causal Live Semantic Views (CLSV)

Status: **lazy transcript/view lifecycle gate passed; not DoD and not a standalone novelty claim.**

## Idea

A governed answer can persist as a live semantic expression rather than a permanently authoritative string:

```text
LiveView {
  immutable_plan,
  sealed_cached_result,
  lifetime_factor
}
```

A world update only invalidates generation factors. No message is eagerly rewritten. On the next actual access, an expired live view reuses its B plan, executes against current capabilities, commits, reseals and renders.

## Result

- Pods: **10,000**
- live views: **50,000**
- world updates: **12,000**
- reads during workload: **180,000**
- same-value generation updates: **4,791**
- read mismatches: **0**
- stale-cache reuse escapes: **0**
- full final audit mismatches: **0**
- value-update plan recompiles: **0**
- eager message rewrites actually performed: **0**
- theoretical eager view rewrites for the same updates: **119,607**
- lazy recomputes during workload: **78,398**
- cache hits during workload: **101,602**
- deferred work vs eager during workload: **34.45%**
- mean invalidated lifetime nodes per update: **6.39**
- p99 invalidated nodes per update: **14**
- median generation invalidation cost in Python: **4.0 µs**
- median live-view read cost in Python: **3.45 µs**

A final full audit needed **11,752** additional lazy recomputations to make every stored view current and still produced zero mismatches.

Report SHA256: `1a8ed35a47be1f861ebbe5aa77c322850f7929e721d6a1298aad98be705edbe6`.
Artifact ZIP SHA256: `08bb3f981632c01a43f107d936b6231e9af6655223f9ed50329bd252edde22ee`.

## Architectural decision

Promote **Causal Live Semantic Views** as the lazy form of CSOG:

> Persistent language structure can survive while governed derived values behave like generation-aware live materialized views.

This prevents stale transcript resurrection without making every knowledge write proportional to the size of all stored conversations.

Reactive dataflow, materialized views and lazy invalidation are established. The research target is their integration with neural generation lifetimes, World ABI and semantic interaction state.
