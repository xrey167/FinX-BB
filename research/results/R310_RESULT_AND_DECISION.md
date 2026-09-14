# R310 Result — CKVM Late-Bound K-SSA

Status: **runtime scalability/correctness gate passed; not DoD and not a standalone novelty claim.**

## Scale

- 50,000 Pods
- 200,000 cached knowledge programs
- 20 epochs
- 2,000 total online world updates

## Correctness

- eager-vs-lazy output mismatches: **0**
- lazy validity/output audit mismatches: **0**
- eager validity/output audit mismatches: **0**
- same-value generation rewrite witnesses: **313**

Same semantic value does not preserve an old lifetime: a rewrite creates a new generation and old dependent states remain invalid.

## Dependency and invalidation reduction

Eager execution materialized all five potential operands for every program. CKVM late binding records only values actually read on the executed control-flow path.

- mean eager dependency count: **5.000**
- mean lazy dependency count: **1.9026**
- dependency-edge reduction: **61.9476%**
- mean eager invalidated-state fraction per 100 updates: **0.9942%**
- mean lazy invalidated-state fraction per 100 updates: **0.3782%**
- invalidation-fanout reduction: **61.9619%**
- median eager state-build time: **2.332 s**
- median lazy state-build time: **1.225 s**

Report SHA256: `66512149fa78a04f09f6b3ec64f70ae767c3c01e6098eb40894648eda05e84b9`.
Artifact ZIP SHA256: `9c2db268c3c4c0bda34c9d39137ba58d32d75e2d71ad475cfd9157f1e146c812`.

## Architectural decision

Keep CKVM's **dynamic read-set** rule:

`deps(output) = union(generations actually read on the executed path)`

Do not eagerly materialize every possible branch operand merely because it appears in the plan. Late binding preserves exact outputs while shrinking both the causal lifetime surface and update invalidation fanout.

SSA, lazy evaluation and dynamic read sets are established systems ideas; R310 establishes their lifecycle value inside CKCA, not standalone novelty.
