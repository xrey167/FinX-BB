# E-000096 result — flexible state-dependent receipts do not generalize exact Pod transport

Date: 2026-09-05
Status: **DECISIVE FALSIFICATION / REGISTERED FAMILY KILL**

## Decision

Kill the registered E-000096 calibration-fitted nonlinear cross-session receipt family.

A genuinely state-dependent Gaussian-RBF receipt was fitted per canonical old->new Pod edit using only calibration sessions. Hyperparameters were selected on a calibration-only validation split. The receipt then transformed already-materialized mixed old final states directly, with no suffix replay.

All validity gates passed in all six backbone/seed cells, including material edits and genuine input-state dependence. Exact transport failed in every cell.

This does **not** prove that every structurally derived state-dependent exact transform is impossible. It closes the registered strategy of learning/interpolating a reusable cross-session correction law from a finite calibration set and then applying it directly to arbitrary held-out mixed states.

## Provenance

Preregistration: `docs/novelty/e000096-nonlinear-state-dependent-receipt.md`.

Executable: `so/experiments/e000096_nonlinear_state_dependent_receipt.py`.

Workflow: `.github/workflows/e000096-nonlinear-receipt.yml`.

Executed commit/run: `6f4dc846d14179541db060f3b734a8332b00fb1d` / GitHub Actions run `33991723855`.

Both matrix jobs completed successfully and uploaded artifacts:

- DistilGPT-2 artifact `9976843829`;
- Pythia-70M artifact `9976860140`.

## Registered family

The receipt used calibration old states as nonlinear anchors:

`Delta_hat(z) = K(z, Z_cal) alpha`

with a Gaussian RBF kernel. Bandwidth multiplier and ridge were chosen over the preregistered 25-pair grid on calibration validation contexts only, then refitted on all 64 calibration contexts.

The receipt is intentionally flexible and large, storing both anchor states and full-dimensional correction coefficients. It receives zero novelty credit; it is a strong capacity baseline.

## DistilGPT-2

All three seeds:

- material edit rate `1.0`;
- two real nonlinear suffix blocks;
- state-dependence control passes strongly (`input_dependence_rms` about `0.397–0.415`);
- exact hidden fraction at `1e-6` = `0`;
- exact logit fraction at `1e-5` = `0`;
- decision `KILL_REGISTERED_NONLINEAR_RECEIPT`.

The most important control is calibration interpolation:

- seed 0 calibration delta maxabs after refit: `0.0`;
- seed 1: `2.27e-13`;
- seed 2: `0.0`.

Thus the family had enough capacity to memorize the calibration correction map essentially exactly, yet did not recover the held-out law.

Held-out results:

| seed | hidden maxabs | logit maxabs | KL mean | top1 | receipt bytes |
|---:|---:|---:|---:|---:|---:|
| 0 | 57.302845 | 63.414181 | 0.000751 | 1.00000 | 393232 |
| 1 | 20.961823 | 15.295452 | 0.001565 | 1.00000 | 393232 |
| 2 | 46.973480 | 35.055565 | 0.833006 | 0.90625 | 393232 |

Again, low KL/top-1 on seeds 0/1 is not exact lifecycle transport.

## Pythia-70M

All three seeds:

- material edit rate `1.0`;
- two real nonlinear suffix blocks;
- state-dependence control passes (`input_dependence_rms` `0.163–0.256`);
- exact hidden fraction at `1e-6` = `0`;
- exact logit fraction at `1e-5` = `0`;
- decision `KILL_REGISTERED_NONLINEAR_RECEIPT`.

Held-out results:

| seed | calibration delta maxabs | hidden maxabs | logit maxabs | KL mean | top1 | receipt bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.008308 | 1.705217 | 3.0 | 0.015146 | 1.00000 | 262160 |
| 1 | 0.007550 | 1.039227 | 2.0 | 0.024320 | 1.00000 | 262160 |
| 2 | 0.002891 | 1.764862 | 3.0 | 0.042356 | 0.96875 | 262160 |

Even this deliberately generous nonlinear receipt is many orders of magnitude outside the exact hidden-state gate.

## Interpretation

E-000095 showed that one fixed edit does not induce a context-independent translation or per-dimension affine correction after nonlinear mixing.

E-000096 now shows something stronger for the registered calibration-fitted strategy: a high-capacity state-dependent interpolator can fit the observed correction examples but still does not provide an exact transport law on unseen sessions.

Therefore do not escalate generic predictor capacity—larger MLPs, kernels, regressors, nearest-neighbor correction pools, or learned offset models—as the major-invention route. Such methods remain useful approximate repair baselines only.

The successor must obtain exactness from **structure of the computation**, not from empirical interpolation alone.

## Decisive 2025 prior-art update: KVCOMM

KVCOMM (Ye et al., NeurIPS 2025; arXiv:2510.12872) is a strong direct baseline for broad cross-context correction claims. It treats divergent-prefix KV reuse as a context-dependent offset problem, maintains a pool of anchors containing observed KV deviations under different contexts, and predicts/combines those deviations to adjust cached KV for new contexts. It reports >70% reuse and up to 7.8x prefill speedup without quality degradation in its workloads.

KVCOMM is not an exact lifecycle mutation system and is not asserted to satisfy this project's fresh-state equality contract. But it means the broad mechanism "store correction examples from multiple contexts and interpolate an adjustment for a new cached context" is already prior art. E-000096's RBF anchor receipt therefore receives no novelty credit even independently of its failure.

Other zero-credit baselines remain AgentKVShift, KVEraser, cache blending/selective recomputation, KV-Direct/residual reconstruction, HCache/KVPR-style state restoration, Jacobian/JVP approximations, generic incremental computation, and late-bound memory.

## Next frontier

The next defensible experiment must be **program/architecture-derived exact active transport** or certified exact affected-work execution.

A candidate must derive its session-specific update from cached computational sufficient state and the exact model operator, rather than predict the update from neighboring examples. It must be compared directly against:

1. exact delta patch at the last real memory-read site + minimal suffix recomputation;
2. residual/KV exact reconstruction;
3. generic incremental/change-propagation execution with the same cached intermediates;
4. matched-memory approximate repair systems only as secondary baselines.

Any claimed advantage must arise from less exact work than these baselines, not from relaxed output equality.

## Programme consequence

**E-000095 and E-000096 close learned cross-session revision receipts as the current major-invention route.**

No major invention is promoted. The programme pivots from learned receipts to structurally exact lifecycle transport / exact affected-work mechanisms under the unchanged real-reader, lifecycle, race, audit, overhead, memory and fleet-level systems gates.
