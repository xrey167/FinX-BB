# R354 Result — Reference-Generating Prospective Neural State (RG-PNS)

Status: **dynamic-reference-generation gate passed; not DoD and not a standalone novelty claim.**

## Problem

R347 covered a fixed set of unresolved references. R353 extended this to world-dependent guarded branches. R354 attacks the harder case in which dereferencing one current world cell **creates the identity of the next knowledge cell**.

A retrospective cache can snapshot an old pointer chain. RG-PNS instead retains only `FOLLOW(start,hops)` and rematerializes the entire current chain lazily.

## Result

- world cells: **8,192**
- retained programs: **6,000**
- pointer hops: **6**
- updates: **60,000**
  - pointer updates: **33,166**
  - payload updates: **26,834**
  - same-target pointer generation rewrites: **9,962**
  - same-value payload generation rewrites: **9,417**
- serves: **180,000**

Correctness:

- RG-PNS false serves: **0**
- final audit mismatches: **0**
- retained-program digest changes after world updates: **0**
- injected generation conflicts expected/detected: **14,514 / 14,514**
- escaped conflicts: **0**
- retry mismatches: **0**

Matched retrospective snapshot-chain control:

- semantic false serves: **173,854**
- update invalidation counterfactual: **305,708**

RG-PNS performed **0 write-time program patches/invalidations**.

Report SHA256: `ef2848bc245c75b85214f9d1872a58eaa147fff1cd9782024b12866a47e19506`.
Artifact ZIP SHA256: `5a3b5ec07ca2148ba98d117162a2a43884e7ebba745aa7f8511102091c863d4f`.

## Architectural decision

Promote **reference-generating state** into the PNS model:

> A future-bindable state need not contain the final knowledge address at all. It may contain an unresolved reference-generating computation whose later dereferences produce new canonical references from the current authoritative world.

This matters for mutable graphs, ownership chains, supplier relations, redirects, entity resolution, indirection and recursive knowledge structures. Future updates can change *which object exists at the next hop* while the retained prospective program remains byte-identical.

The generation read-set is acquired dynamically along the actually traversed current chain and validated before publication.

Pointer chasing, graph traversal, neural RAM and transactional read-sets are established. R354 is an architecture-completeness result for PNS, not a component-level novelty claim.
