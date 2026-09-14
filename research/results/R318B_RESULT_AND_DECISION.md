# R318b Result — Directional-Gated Causal Dual Rail

Status: **dual-rail repair gate passed; not DoD and not a standalone novelty claim.**

R318 proved exact B-state isolation but lost too much task quality. R318b keeps the one-way information-flow invariant and replaces the weak J parameterization with typed directional gating and operator-specific J experts.

## Result

Three seeds, fully deranged future-world bindings:

Directional-gated dual rail:

- home accuracy: **99.642%**
- deranged future-world accuracy: **99.687%**
- full vs cached-B max logit delta: **0.0**
- B-state max delta under mutable-world changes: **0.0**
- mean cached world-update speedup: **3.170×**

Shared-rail quality control:

- home accuracy: **99.886%**
- deranged future-world accuracy: **99.899%**

Future-world quality gap, dual rail minus shared control: **-0.212 percentage points**.

Report SHA256: `866f081c7b0800d33e43621d3e9f35b0deb801d36d16fee09bb0b417f997cb04`.
Artifact ZIP SHA256: `2eac121c3bb7dc99f7b0736ccd2d99f3e6b6052fbda8013b7c2a4f10d6c7bec6`.

## Architectural decision

Promote the **one-way neural world boundary**:

```text
immutable B / plan rail  ─────────► mutable J / world rail
             ▲                           │
             └──────── NO EDGE ──────────┘
```

The B rail compiles reusable operator/skill state. It is structurally incapable of reading current world values. A typed gate selects the appropriate J execution path, and J reads current value data plus cached B control.

The important invariant is not the exact expert architecture. It is:

> Mutable world information may consume immutable skill state, but may not flow backward into reusable skill state.

That gives exact cache invariance under world edits while preserving essentially all control-model quality in this gate and making the mutable recompute path ~3.17× cheaper than recomputing both rails.

Mixture-of-experts, hard routing, side networks and one-way adapters are known components. The research target is their composition with Binding Equivariance, NLTS, CKT/CLFD and late neural materialization—not component-level novelty.
