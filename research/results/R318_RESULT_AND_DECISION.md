# R318 Result — Causal Dual-Rail Neural Execution

Status: **structural isolation and cache-reuse gate passed, but the first dual-rail parametrization fails the quality gate. Do not promote this exact network.**

## Result

Three seeds were trained on a complete static world and evaluated on fully deranged future bindings.

First dual-rail parametrization:

- home accuracy: **57.93%**
- deranged future accuracy: **57.60%**
- full-forward vs cached-B max logit delta: **0.0**
- reusable B-state delta under world changes: **0.0**
- mean cached world-update speedup: **1.92×**

Single-rail control:

- home accuracy: **98.44%**
- deranged future accuracy: **98.43%**
- post-value-injection hidden-state delta under edits: **15.72** mean max-abs units

Report SHA256: `e59bd08cb32747240842c7ce3148093d20f1b9640188beb012e27564f9725f64`.
Artifact ZIP SHA256: `1207cf5fa1a52e3422b38c92d39923804a4b3c992bec8fbea206f4ae521eeb5c`.

## What is established by this gate

The one-way topology does what it is supposed to do mechanically:

`B -> J` is allowed, `J -> B` is absent.

Therefore changing world values leaves the B rail bit-identical, cached-B execution is exactly equivalent to full execution, and mutable information visibly contaminates the single-rail control after its injection point.

## What failed

The first J rail was too weak / badly conditioned for the operator family. Structural purity alone is not acceptable if task quality collapses from ~98% to ~58%.

## Decision

Keep the **one-way information-flow invariant**, reject this specific J parameterization, and test a stronger directional-gated J rail that receives explicit structured value features plus B-plan control without ever writing mutable information back into B.

The architecture criterion is now joint:

1. B-state invariance under mutable-world changes must remain exact;
2. cached-B execution must remain numerically identical to full B execution;
3. quality must match the contaminating control;
4. world-update execution must be cheaper than full recomputation.

R318b is the repair gate.
