# E-000095 — Cross-Context Pod Revision Receipt

Date: 2026-09-05
Status: **preregistered kill-screen; not a novelty claim**

## Question

Can one canonical Pod edit be compiled once into a compact neural-state transport receipt that updates **many already-materialized sessions** to the same counterfactual state as fresh recomputation?

For context/session `x`, old Pod payload `p0`, new payload `p1`, and a frozen backbone:

`H0(x) = F(x,p0)`

`H1(x) = F(x,p1)`

The strongest useful form would have a compact receipt `R(p0,p1)` such that

`T_R(H0(x)) = H1(x)`

for held-out contexts `x`, without rerunning the suffix of `F` per session.

This is different from passive freshness, invalidation, or late binding: the already-materialized mixed neural state is **actively transformed**.

## Prior-art boundary

No credit for:

- full or selective recomputation;
- KV cache editing/erasing as such (KVEraser and related work);
- generic learned cache repair;
- Jacobian/JVP linearization;
- ordinary low-rank adapters;
- memoized dependency propagation;
- cache versioning / lineage / sidecars;
- external late-bound mutable memory.

The experiment only asks whether a **single edit receipt generalizes across contexts** strongly enough to motivate a new mechanism.

## Phase A families

On frozen DistilGPT-2 and Pythia-70M, inject the same old/new controlled Pod payload at one internal read layer across many distinct token contexts and record the final last-token hidden state.

Registered receipt families:

1. **translation receipt**: one vector `delta_bar`, fitted on calibration contexts, applied as `H0 + delta_bar`;
2. **diagonal affine receipt**: per-dimension `a,b`, fitted on calibration contexts, applied as `H0 + a + b*H0`.

The diagonal family is intentionally stronger than a simple average delta but remains compact (`2*d` floats per Pod edit).

Fresh `H1` is always the gold reference.

## Validity / kill rules

For each backbone and seed:

- V1: the old→new Pod edit must materially change fresh final logits on held-out contexts (`maxabs > 1e-4` for >=95% of contexts);
- V2: calibration and held-out contexts are disjoint;
- V3: no held-out target is used to fit a receipt;
- V4: exactness is judged against fresh recomputation, not task accuracy.

A receipt family only survives as an **exact transport** candidate if on every held-out context:

- final-hidden `maxabs <= 1e-6`;
- reconstructed-logit `maxabs <= 1e-5`;
- top-1 equals fresh;
- and this holds on both backbone families and >=3 seeds.

If exactness fails but KL/top-1 remain good, record it as **approximate transport only**; it gets zero deletion/lifecycle guarantee credit.

## Major-usefulness gate (later phase only)

Even an exact receipt is not a major break unless a real LINK→Pod implementation demonstrates a material fleet-level result over the strongest guarantee-matched per-session suffix-recompute/KV-edit baseline, with matched memory:

- target >=10x mutation-to-ready advantage at high session fan-out, or equivalently large retained-throughput advantage;
- <=5% steady-state inference overhead;
- full lifecycle/replay attack battery;
- independent J-space/J-lens content audit only.

## Decision

Phase A is an existence screen. If both compact receipt families fail exactness decisively, do not call approximate state steering a breakthrough. A successor must use genuinely state-dependent nonlinear transport or change the architecture so exact edit transport is structurally available without reducing to late binding.
