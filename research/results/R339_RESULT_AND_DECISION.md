# R339 Result — Bounded Model Check for Proof-Carrying Requalification

Status: **bounded exhaustive lifecycle model check passed; not a general theorem and not DoD.**

## Model

Finite domain `{0,1,2,3}`, three canonical inputs and a fixed typed derived DAG:

```text
A ---+--> ADD ----+
B ---+            |
     +--> MAX ----+--> XOR ----+--> SELECT_EVEN
C ---+                       C-+
```

Every update first kills the old source/descendant temporal factors. The implementation may then either recompute a changed semantic value or reuse identical physical bytes under a **new** factor rebased onto current parent factors.

All possible initial worlds and all length-three update traces were exhaustively enumerated.

## Result

- initial worlds: **64**
- possible updates/step: **12**
- traces checked: **110,592**
- transition steps checked: **331,776**
- invariant violations: **0**
- semantic full-recompute mismatches: **0**
- old-factor resurrections: **0**
- factor-parent rebase violations: **0**
- same-value generation steps: **82,944**
- same-value steps requiring any semantic recompute: **0**

Repair surface:

- baseline descendant recomputes: **1,105,920**
- actual full recomputes: **580,608**
- proof-backed requalifications: **525,312**
- full recompute reduction: **47.5%**

Report SHA256: `e50a5a51e027c68310cab66685c656f3d00b347f6150f22a2e2d5510388ed10f`.
Artifact ZIP SHA256: `748d1d739cd9125cf550fdc07c67aed38a13a37abd504aafa45d9e2887553ae5`.

## Invariants checked after every transition

1. current semantic values equal from-scratch evaluation;
2. every factor killed by the transition remains dead;
3. every current derived factor is freshly rebased onto current immediate-parent factors;
4. same-value generation transitions never require semantic recomputation in this deterministic typed DAG.

## Architectural decision

The distinction introduced by PCR is coherent in the bounded state space:

> **Temporal invalidation and physical recomputation are separate operations.**

A generation transition always kills the old factor. If a certificate proves that the derived payload is unchanged, the physical bytes may survive only under a freshly minted lifetime that refers to current factors. That is requalification, not resurrection.

This gives CKCA a principled way to make same-value ABA transitions and other operation-invariant changes cheap while retaining exact temporal identity.

The result is exhaustive only for this finite DAG/domain/depth. Self-adjusting computation and change propagation are strong prior art; a general proof and neural-state certificate theory remain open.