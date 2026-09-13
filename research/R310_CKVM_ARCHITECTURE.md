# R310 — Causal Knowledge Virtual Machine (CKVM)

Status: architecture candidate inside CKCA. **Not a novelty claim or DoD.**

## Why another architectural layer is necessary

CKCA already separates the immutable language/skill plane from mutable authoritative knowledge. R303/R304 show that many leaf values do not need per-fact neural KV at all; they can remain compact symbolic/model-native values. R292 shows the opposite danger: once mutable information is materialized into downstream neural state, that derived state acquires the source generation's lifetime.

The design implication is stronger than “external memory”:

> **Keep mutable knowledge symbolic and generation-addressable for as long as possible. Materialize neural state only when an operation actually needs it.**

CKVM makes that rule explicit.

## Execution model

The B-plane compiles a natural-language request into a small typed knowledge program. The program references canonical Symlink operands, never copied fact text.

Example conceptual K-SSA:

```text
%a      = RESOLVE(alias_a)
%b      = RESOLVE(alias_b)
%ca     = CAPABILITY(%a)
%flag   = READ_FIELD(%ca, flag)
%value  = SELECT_LAZY(%flag,
                      { READ_VALUE(CAPABILITY(%a)) },
                      { READ_VALUE(CAPABILITY(%b)) })
EMIT_REF(%value)
```

The mutable value does not enter the B-plane prompt or reusable B-plane cache.

## Registers are `(value, type, lifetime_factor)`

Every CKVM register carries:

- a typed value or value handle;
- a semantic type;
- one CLFD lifetime factor.

For a direct generation read:

`factor(READ(pod:g)) = LifetimeLeaf(pod,g)`.

For a pure derived operation:

`factor(f(x,y)) = factor(x) AND factor(y)`.

A register is admissible iff its factor's O(1) validity bit is true.

## Late binding and lazy reads

Ordinary eager memory architectures often materialize all candidate values before the model knows which one it needs. CKVM instead uses lazy knowledge control flow.

For:

`if flag(A) then value(B) else value(C)`

an eager execution reads A, B and C, so the derived state is invalidated by changes to **any** of those generations.

A late-bound CKVM execution reads:

1. A to determine the branch;
2. only the selected B **or** C value.

Its lifetime therefore contains exactly the generations that causally influenced the result, not a static superset.

This has two consequences:

- fewer mutable reads and less neural materialization;
- fewer unrelated lifecycle changes invalidate cached J states.

## Proposed K-SSA instruction families

### Identity / authority

- `RESOLVE(alias) -> pod_id`
- `CAPABILITY(pod_id) -> verified_generation_capability`
- `DEREF(capability) -> capability`

### Typed reads

- `READ_VALUE(capability)`
- `READ_FIELD(capability, field)`
- `READ_RELATION(capability, relation)`

### Pure operations

- `EQ`, `NEQ`, comparisons;
- typed arithmetic;
- Boolean operators;
- set membership;
- projection / selection;
- `SELECT_LAZY`;
- relation joins over canonical references.

### Output/materialization

- `EMIT_REF(value_handle)` for exact symbolic/token-sequence values;
- `MATERIALIZE_J(value_handle)` only when a neural hidden representation is required;
- `SPLICE_DECODE(value_handle)` for late-bound decoder insertion;
- `CALL_TOOL` with lifetime propagation from mutable arguments and governed tool results.

## Causal read-set rule

The runtime records **executed reads**, not merely possible reads in the plan.

`deps(output) = union(deps of instructions actually executed on the dynamic path)`.

The compiler may use a static superset for conservative safety, but CKVM's performance target is dynamic exactness because irrelevant generation dependencies increase invalidation fanout.

## Query-plan versus world-state separation

A cached K-SSA plan depends on:

- language interpretation / operator choice;
- schema version;
- resolver contract.

It does **not** depend on current fact values unless value-dependent specialization has been performed. A fact update can therefore reuse the same plan and B-plane state while rebinding current capabilities at execution time.

## Neural materialization boundary

The preferred order is:

1. resolve canonical identity;
2. verify authority capability;
3. execute as much typed computation as possible over handles;
4. attach exact transitive lifetime factors;
5. materialize model-native J state only for the smallest suffix that genuinely needs neural representation;
6. discard/invalidate only that suffix when a source generation changes.

This is the `late-bound knowledge execution` principle.

## What is not novel by itself

CKVM borrows established ideas from SSA/dataflow execution, lazy evaluation, pointer/copy mechanisms, capability systems, database query execution and virtual-memory abstractions. Context-pointer systems such as CPOS also use virtual-memory language for LLM agent context, and MemMachine explicitly uses “late binding” for multi-hop retrieval where later queries depend on earlier results.

Therefore neither “virtual memory for LLMs”, “late binding”, nor a knowledge instruction set is sufficient as a claim.

The only surviving candidate contribution is their use inside the stronger CKCA neural-generation coherence contract: canonical generation authority, exact lifetime inheritance by neural derivatives, active-only materialization, anti-resurrection across persisted/stale bytes and reusable immutable B-plane execution.

## R310 gate

R310 tests one concrete CKVM benefit at scale: **dynamic lazy read sets** versus eager candidate materialization.

The gate requires:

- identical outputs between eager and lazy execution;
- zero false-valid states after randomized online generation changes;
- smaller dependency sets for lazy execution;
- materially lower invalidation fanout for lazy execution;
- same-value rewrites still invalidate old states because lifetime is generation-based, not semantic-value-based.

This is a runtime architecture test, not a neural-quality benchmark.
