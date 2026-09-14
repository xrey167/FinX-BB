# R338b Result — Proof-Carrying Requalification / Causal Factor Rebasing

Status: **optimized proof-backed requalification gate passed; not DoD and not a standalone novelty claim.**

## New distinction

A generation transition and a physical recomputation do not have to be the same event.

R338b always kills the old temporal factor first. It then asks whether a typed certificate can prove that a derived payload remains semantically identical. If yes, the physical bytes are reused **only under a freshly minted factor** rebased onto current parent factors. The old factor remains permanently dead.

Thus:

```text
old generation expires
    -> old factor dies permanently
    -> prove derived bytes unchanged?
        yes -> reuse bytes + mint fresh current factor
        no  -> recompute bytes + mint fresh current factor
```

This is requalification, not resurrection.

## Result

- canonical inputs: **2,000**
- derived DAG nodes: **18,000**
- updates: **12,000**
- mean affected descendants/update: **98.86**
- p99 affected descendants: **460**

Correctness:

- full-current oracle mismatches: **0**
- dead-factor resurrections: **0**
- current-factor errors: **0**
- parent-factor rebase errors: **0**
- conservative certificate misses: **0**

Efficiency:

- baseline descendant recomputes: **1,186,274**
- actual full recomputes: **291,035**
- certified requalifications: **895,239**
- full recomputes avoided: **895,239**
- recompute reduction: **75.47%**

Of the requalifications:

- pure parent-semantic-equivalence rebases: **826,262**
- operation-specific certificate rebases: **68,977**

Same-value temporal transitions:

- same-value input generation updates: **4,835**
- same-value updates with zero semantic recomputation: **4,835 / 4,835**
- each still received new factors; no old factor was revived.

Report SHA256: `86cc9aafa1523527465b8a40e90735174c38a716ae5c630c2c673547ffe68c70`.
Artifact ZIP SHA256: `9f74fac0b13785be672e5e8b3a7040bdd99e213e13a7cb0f4bcdda0583149e4b`.

## Architectural decision

Promote **Proof-Carrying Requalification (PCR)** as a CKCA research primitive:

> Temporal invalidation is unconditional; physical recomputation is conditional on semantic change. A proof may authorize reuse of bytes, but it may never revive the old temporal identity.

This is especially valuable for same-value ABA writes, changes that do not alter an aggregate/result, and cascades where a parent receives a fresh factor but identical semantic bytes. Descendants can be factor-rebased without rerunning their expensive materialization.

Self-adjusting computation, memoization, incremental change propagation and algebraic invariance certificates have substantial prior art. The CKCA-specific candidate is the combination of **permanent generation death + fresh factor rebasing over reused neural/semantic bytes**. General certificates for arbitrary hidden neural states remain open.