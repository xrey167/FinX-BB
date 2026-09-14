# R331 Result — Lifetime-Carrying Semantic Weave (LCSW)

Status: **semantic-segmentation lifetime gate passed; not DoD and not a standalone novelty claim.**

## Problem

A single mutable fact introduced early in an autoregressive response can conservatively contaminate all later hidden state/tokens. Exact freshness then tends to invalidate a large suffix even when later prose is semantically independent.

R331 inserts explicit semantic/causal reset boundaries. A response is a DAG of independently regenerable clauses. Each clause inherits only the exact world generations it reads plus the lifetimes of explicit parent clauses.

## Result

- Pods: **10,000**
- documents: **20,000**
- clauses/document: **8**
- semantic clauses: **160,000**
- world updates: **12,000**
- same-value generation updates: **4,836**
- full-current oracle mismatches: **0**
- stale generation lifetime survivors: **0**

Invalidation/repair surface:

- Lifetime-Carrying Semantic Weave: **189,371 clauses** cumulatively
- monolithic autoregressive suffix counterfactual: **665,073 clauses**
- whole-document counterfactual: **1,409,400 clauses**
- mean affected clauses/update, weave: **15.781**
- mean affected clauses/update, monolithic suffix: **55.423**
- p99 weave affected clauses/update: **31**
- reduction vs monolithic suffix: **71.53%**
- reduction vs whole-document repair: **86.56%**
- median Python causal repair: **46.6 µs/update**

Report SHA256: `0647e4e61fe9ba7d0255676e35468457808869167c88279e13d3d67b128061ed`.
Artifact ZIP SHA256: `b825eaa3c7c8b00bfd4bc9fe3b863cc461fcf05e0163c6b7aa88dbaf3c3686ee`.

## Architectural decision

Promote **Lifetime-Carrying Semantic Weave** as the free-form output counterpart to GCRD/CSOG:

> Long-form governed output should be composed from independently generated semantic segments whose causal inputs and lifetimes are explicit, rather than relying on one monolithic autoregressive hidden-state chain whenever exact lifecycle repair matters.

This creates explicit neural reset boundaries. A world update can regenerate only affected semantic DAG descendants while preserving unrelated clauses, without weakening generation freshness.

Incremental computation, DAG recomputation, structured generation and modular decoding are established. R331 supports their CKCA-specific use as explicit neural/semantic lifetime boundaries rather than a component-level novelty claim.
