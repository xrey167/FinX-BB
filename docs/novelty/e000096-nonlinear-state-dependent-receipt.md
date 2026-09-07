# E-000096 — Nonlinear State-Dependent Revision Receipt Capacity Screen

Date: 2026-09-05
Status: **preregistered falsification screen; not a novelty claim**

## Trigger

E-000095 decisively killed both registered compact cross-context receipt families on DistilGPT-2 and Pythia-70M across three seeds. Translation and per-dimension affine receipts preserved top-1 surprisingly often but failed the exact fresh-state contract by orders of magnitude. The exact old->new correction varied materially across held-out contexts.

The required pivot is therefore to a genuinely state-dependent nonlinear active transport, not another passive freshness tag, mask, invalidation rule, or affine correction.

## Question

Is the exact per-session correction induced by one canonical old->new Pod edit sufficiently regular as a function of the already-materialized mixed old neural state that a flexible edit-level nonlinear receipt, fitted only on calibration sessions, can reproduce fresh-current held-out state without suffix replay?

For a fixed edit `p0 -> p1` and session `x`:

`H0(x) = F(x,p0)`

`H1(x) = F(x,p1)`

`Delta(x) = H1(x) - H0(x)`.

The candidate consumes the actual old mixed state:

`Delta_hat(x) = R_edit(H0(x))`

and transports

`H_hat1(x) = H0(x) + Delta_hat(x)`.

This is active state transport. It is not an accept/reject predicate and it is not late binding.

## Registered nonlinear family

Use Gaussian RBF kernel ridge interpolation over calibration old states.

For calibration anchors `z_i = H0(x_i)` and targets `d_i = Delta(x_i)`, define

`k(z,z_i) = exp(-||z-z_i||^2 / (2*sigma^2))`.

The receipt stores the calibration anchors and fitted output coefficients. This is intentionally a **strong, flexible, non-compact baseline**; its representation receives zero novelty credit.

Hyperparameters are selected without held-out leakage:

- split the registered calibration set deterministically into fit and calibration-validation subsets;
- bandwidth multipliers: `{0.25, 0.5, 1.0, 2.0, 4.0}` times the fit-set median nonzero pairwise distance;
- ridge values: `{0, 1e-10, 1e-8, 1e-6, 1e-4}`;
- choose the pair minimizing calibration-validation final-hidden maxabs;
- refit on the entire calibration set using the selected pair;
- evaluate once on the untouched held-out contexts.

No held-out target may influence bandwidth, ridge, feature scaling, edit strength, layer choice, or stopping.

## Backbones / seeds / contexts

Use the same controlled existence-screen geometry as E-000095 unless a technical incompatibility forces an explicitly recorded validity repair:

- DistilGPT-2;
- EleutherAI/pythia-70m;
- seeds `0,1,2`;
- 64 calibration contexts;
- 64 held-out contexts;
- sequence length 16;
- one fixed old/new payload pair per seed;
- payload RMS 2.0;
- intervention at `len(blocks)-3`, leaving two real nonlinear suffix blocks.

This remains a synthetic internal-Pod existence assay, not a real LINK->Pod qualification.

## Validity gates

For every interpreted backbone/seed cell:

- V1: old->new edit materially changes fresh final logits (`maxabs > 1e-4`) on >=95% held-out contexts;
- V2: calibration and held-out contexts are disjoint;
- V3: held-out targets are never used in model selection;
- V4: at least two real nonlinear blocks remain after the intervention;
- V5: the fitted nonlinear receipt materially depends on the input state: replacing every test state with the test-set mean before receipt evaluation must materially change the predicted correction distribution;
- V6: fresh recomputation remains the gold reference.

## Exact survival bar

A backbone/seed cell survives only if **every held-out context** satisfies:

- final hidden maxabs `<= 1e-6`;
- reconstructed-logit maxabs `<= 1e-5`;
- top-1 agreement `= 1.0`.

The E-000096 nonlinear receipt survives the Phase-A capacity screen only if all three seeds pass on both backbone families.

Approximate KL/top-1 success is diagnostic only and earns zero lifecycle/deletion guarantee credit.

## Strong baselines / zero-credit territory

No novelty credit for the RBF receipt or for generic nonlinear learned cache repair. Also no credit for:

- KVEraser-style steering;
- AgentKVShift-style residual correction;
- generic MLP/cache repair;
- Jacobian/JVP linearization;
- selective recomputation;
- cache blending;
- KV-Direct/residual reconstruction;
- exact delta patch at the last real memory-read site followed by minimal suffix recomputation;
- late-bound external memory;
- passive freshness/version metadata.

If E-000096 passes exactness, the next question is not patentability of kernel regression. The question becomes whether a substantially more structured/bounded operator can retain exactness while beating guarantee-matched suffix recomputation at fleet scale.

## Kill / pivot rule

If the flexible nonlinear receipt fails exact held-out transport on either backbone family or any registered seed, close **calibration-fitted cross-session receipt transport in this registered family** and stop escalating generic predictor capacity merely to chase interpolation.

Then pivot to mechanisms where exactness follows from structure of the computation itself: program-derived/incremental exact transport, algebraically constrained local state transforms, or certified affected-work execution. Such a successor must be compared against minimal exact suffix recomputation and residual/KV reconstruction.

## Major-break gates remain unchanged

No E-000096 result is a major invention without real LINK->Pod reader >=0.95 on every held-out template, >=3 genuine training seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% UNKNOWN in declared missing-key scope, exact bypass or <=0.05 nats generic divergence, stale Bank/router/resolved-payload/Hidden/KV attacks, UPDATE/RELINK/REVOKE/SHRED/DELETE/RESTORE/ABA/rollback/TOCTOU, key/reconstruction attacks, independent J-space/J-lens audit only, <=5% steady-state overhead, matched total memory, and a material fleet-level mutation-to-ready advantage over the strongest guarantee-matched exact baseline.
