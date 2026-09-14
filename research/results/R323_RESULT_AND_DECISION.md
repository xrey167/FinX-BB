# R323 Result — Causal Semantic Airlock + Typed World Slots

Status: **real-model semantic non-interference gate passed; not DoD and not a standalone novelty claim.**

## Problem attacked

Even if an external lifetime system revokes a fact, a pretrained model may still contain the old association in its weights. A prompt-level instruction such as “prefer current memory” is therefore not a sufficiently hard governance boundary.

R323 tests a stronger architecture:

1. a lexical compiler may see the original entity name;
2. only a typed operation + opaque PodRef crosses the **Causal Semantic Airlock**;
3. compiler KV/hidden state is discarded;
4. the governed renderer starts from a fresh neural context with no entity lexeme or stale value lexeme;
5. the final governed literal is a runtime-owned typed World Slot, not unconstrained token sampling;
6. revoked slots materialize as `UNKNOWN`.

## Real Qwen2.5-0.5B result

12 common factual probes were used so the frozen model had realistic parametric associations.

Hard structural checks:

- entity/stale-value lexical leaks into the airlocked renderer: **0**;
- max renderer-logit change after three compiler-side attacks per fact: **0.0**;
- live slot materialization errors: **0**;
- revoked slot materialization errors: **0**.

Descriptive parametric-leakage probe:

- direct lexical stale-value mean log-probability: **-2.2272**;
- leaky revoked renderer, still containing lexical entity identity: **-6.8935**;
- airlocked revoked renderer with opaque identity and fresh state: **-15.1374**;
- mean stale-value log-probability drop, leaky revoked -> airlocked: **8.2439 nats**.

Report SHA256: `2e98cf6a9ed47643af56503f24b53c6796d0be3cdad2501f72e5478ffb606cdc`.
Artifact ZIP SHA256: `8f9ebdafb0165ed7ada3be20a2b3e19b7a0d3fd7f903c1dfaa8a1130f11af628`.

## Architectural decision

Promote **Causal Semantic Airlock (CSA)** and **Typed World Slots (TWS)** into CKCA V3/V4.

The key rule is:

> A neural component that has seen a governed entity lexically is not allowed to carry its hidden/KV state across the authority boundary into governed factual rendering.

Only a typed plan crosses. Governed literal bytes are then owned by the committed World ABI result. This changes stale-parametric recall from an instruction-following problem into an information-flow problem with a structural boundary.

The log-probability reduction is descriptive rather than the safety proof. The safety mechanism is the absence of a compiler-state/lexical channel plus runtime-owned value-slot materialization.

Structured outputs, process isolation, opaque handles, constrained decoding and slot filling are known individually. The research target remains the complete CKCA composition rather than component-level novelty.
