# R328 Result — Versioned Symlink Capabilities (VSC)

Status: **Symlink plan-lifetime scalability gate passed; not DoD and not a standalone novelty claim.**

## Problem

A single global alias/namespace epoch is safe but catastrophically coarse. Rebinding one name would invalidate plans that never touched that name.

R328 gives every Symlink binding its own monotonic generation. A B-plan captures capabilities only for the exact aliases it resolved.

## Result

- aliases: **200,000**
- canonical Pods: **50,000**
- compiled plans: **250,000**
- aliases/plan: **3**
- alias rebinds: **20,000**
- same-target generation rebinds: **7,107**
- admission probes: **1,000,000**

Per-Symlink generations:

- stale plans accepted: **0**
- valid plans rejected by reverse index: **0**
- reverse-index vs full capability oracle mismatches: **0**
- total plan invalidations: **71,822**
- mean newly invalidated plans/rebind: **3.5911**
- p99: **9**
- median rebind + reverse invalidation: **1,614 ns** in Python
- median plan capability verify: **2,434 ns** in Python

Global namespace epoch counterfactual:

- theoretical plan invalidations: **5,000,000,000**
- false invalidations observed in the admission sample: **920,302**
- invalidation reduction from per-Symlink generations: **99.99856356%**.

Report SHA256: `4f07c369d0d6ca6619b73783c5e956d5ef7ed655bdd72b593faeb9a0d78fb9a5`.
Artifact ZIP SHA256: `93b91236019831753698020951573ae5b666e40d7e16cdec7100659e1e1a6579`.

## Architectural decision

Promote **Versioned Symlink Capabilities**:

> The linguistic-to-canonical binding is mutable world state and owns its own temporal generation, independently from the Pod value generation.

A plan lifetime therefore includes only the Symlink generations it actually resolved. Rebinding one alias does not poison the whole namespace or all reusable B plans.

Same-target rebinds still create a fresh temporal identity, preserving exact lifecycle semantics rather than treating payload/target equality as lifetime equality.

Versioned symbol tables, symlinks/inodes, capability generations and dependency indexes are established systems concepts. R328 supports their CKCA-specific composition.
