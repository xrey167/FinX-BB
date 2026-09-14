# R325 Result — Epistemic Plan Firewall + Trusted Span Linker

Status: **compiler non-interference gate passed; not DoD and not a standalone novelty claim.**

## Problem

A semantic airlock is incomplete if the neural compiler can encode a stale parametric fact into the typed plan itself—for example as a literal, forged PodRef, alternate entity identity, branch constant or hidden memory opcode.

R325 therefore treats the neural compiler as untrusted. It may propose source spans and allowlisted schema/control structure, but only a non-neural trusted linker can map an exact user source span to a canonical PodRef.

## Result

- valid plans: **250,000**
- adversarial mutated plans: **250,000**
- valid-plan false rejects: **0**
- adversarial plan false accepts: **0**
- trusted-linker address errors: **0**
- mean Python verify+link cost: **6,824.7 ns/plan**

The mutation corpus covered nine smuggling classes:

- model factual literals;
- direct Pod forgery;
- invalid source provenance;
- invented identity spans;
- model-origin control constants;
- direct-address substitution;
- unknown model-memory opcodes;
- untrusted schema constants;
- SSA/provenance overwrite.

Report SHA256: `25d020fa52d9931a11f85cd4f0e9c4e60d92b738849ead7848a6a33353db03f9`.
Artifact ZIP SHA256: `03d5029e0be8bfe4e95b3afa5440c03c5230944b866fe9e0bef90ae6bddd81ae`.

## Architectural decision

Promote **EPF — Epistemic Plan Firewall**:

> The model may propose *where in the user's input an entity or literal came from*; it may not directly choose the canonical authority address or inject a new governed factual literal from latent memory.

This makes the compiler side of the Causal Semantic Airlock explicit. The trusted linker and allowlisted IR prevent model-parametric knowledge from crossing the airlock through a non-text side channel.

Capability-safe linking, source-span grounding, SSA and information-flow typing have prior art. R325 supports their CKCA composition rather than a component-level novelty claim.
