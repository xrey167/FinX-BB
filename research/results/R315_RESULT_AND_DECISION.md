# R315 Result — Counterfactual World-Swap Training

Status: **negative mechanism result; training randomization alone is rejected as the main solution.**

## Result

Three seeds, 64 entities, 64 mutable values, four value-dependent operators.

A static-world model fit its home world almost perfectly (**99.83%**) but fell to **44.42%** on unseen world bindings.

Counterfactual world-swap training did **not** solve the problem:

- home accuracy: **44.75%**
- unseen-world accuracy: **44.07%**
- sparse online edit accuracy without gradient updates: **43.72%**
- B-only future accuracy: **31.48%**
- zero-Port future accuracy: **31.90%**
- stale-Port future accuracy: **31.85%**
- gain over static-world future accuracy: **-0.35 percentage points**

Report SHA256: `08386e0e0e8be42251cd305c4f57f3e1b65a65775806e1fed7fe873435e6ac59`.
Artifact ZIP SHA256: `1716c15ffd740eff4c41f41f3f7c3fa7b9d3d12e13b11df46c83629b2ad41d99`.

## Decision

Do **not** claim that random rebinding during training is enough to create a future-bindable neural knowledge substrate.

The stronger architectural inference is that the system should not ask an unconstrained network to *learn* the identity/value separation if the separation can be made a structural invariant. R317 therefore removes the post-resolution identity channel entirely and tests **binding-equivariant execution by construction**.

This is an important kill result: the project moves from a curriculum-only hypothesis to an architecture-level symmetry constraint.
