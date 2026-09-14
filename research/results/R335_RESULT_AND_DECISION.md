# R335 Result — Transactional Lazy Delta Attention (TLDA)

Status: **transactional lazy attention-lifecycle gate passed; not DoD and not a standalone novelty claim.**

## Core idea

R334's intended stable-K/mutable-V attention rule makes route selection reusable across ordinary value updates. R335 pushes this further: world writes do not fan out into all cached neural views at all.

A retained attention view stores:

- stable selected Pod addresses;
- stable attention weights;
- last-seen value generations and values;
- cached output.

A canonical world write advances only the value and its generation. On the next actual read of a retained view, stale selected pages are caught up exactly by:

`O += alpha_j * (V_current - V_seen)`.

Multiple unseen generations coalesce into one catch-up. Same-value rewrites refresh generation identity while leaving the numeric output unchanged. Immediately before serve, a Neural Commit Barrier rechecks all selected generations; if a race occurred, the local view catches up and retries.

## Result

Scale:

- canonical Pods: **10,000**
- retained attention views: **100,000**
- selected pages/view: **4**
- scheduled writes: **50,000**
- injected race writes: **15,748**
- total writes: **65,748**
- view reads: **200,000**
- same-value generation writes: **27,111**

Correctness:

- expected commit conflicts: **15,748**
- detected commit conflicts: **15,748**
- escaped conflicts: **0**
- retries: **15,748**
- semantic mismatches during serves: **0**
- lifetime mismatches during serves: **0**
- final full-current semantic max error: **9.99e-16**
- final lifetime-mismatched views: **0**
- naive no-commit-barrier false serves under the same races: **15,748**

Update/catch-up behavior:

- lazy slot patches including final audit: **863,633**
- theoretical eager cache patches: **2,647,223**
- maintenance reduction vs eager propagation: **67.38%**
- intermediate generations coalesced away: **1,783,590**
- same-value lifetime refreshes: **155,333**
- median O(1) canonical write in Python: **22.7 µs**
- median transactional view read in Python: **206.8 µs**

Report SHA256: `58a45162f015a35a5348732c9e4d7891fff718310845f99f135a77e47dadfa74`.
Artifact ZIP SHA256: `71089d0912d532cd1baeac8addf6fe4ac8b2d72b213caf957a599e65621a5f74`.

## Architectural decision

Promote **pull-based neural coherence** as a serious CKCA direction:

> Canonical world writes should be cheap and local. Cold derived neural state should not impose write amplification. A hot derived view should prove its own freshness and catch up exactly when it is actually used.

For stable-route attention this yields a particularly clean composition:

```text
O(1) canonical generation write
        -> no cache fanout
        -> on access compare selected generations
        -> exact value-delta catch-up
        -> Neural Commit Barrier
        -> serve current result
```

This is stronger than simple cache invalidation because skipped intermediate generations do not require replay; the current value delta is sufficient for the numeric state while the generation counter preserves temporal identity.

Lazy materialized views, delta propagation, optimistic validation and MVCC are established independently. The research target is their exact composition with generation-scoped neural attention and the broader CKCA/WIT machine.