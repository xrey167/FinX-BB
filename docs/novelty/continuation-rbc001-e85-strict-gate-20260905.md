# Continuation checkpoint — RBC-001 and exact-job E85 gate

Date: 2026-09-05. This record changes no model, core guard, thresholds or historical result files.

## Completed evidence inspected in this continuation

E-000085 run `33979783563`, executed SHA `7f05f16fab8e8e9eee09e2b49f76b49931d95470`: all four result archives were downloaded and SHA-256 verified. Per-template capability on T8/T9/T10/T11:

| Exact job | T8 | T9 | T10 | T11 | Decision |
|---|---:|---:|---:|---:|---|
| hidden seed1 | 1.000 | .960 | 1.000 | 1.000 | capability + structural pass |
| hidden seed2 | .985 | .960 | .995 | .995 | capability + structural pass |
| router/payload seed1 | .990 | .975 | .990 | .990 | capability + structural pass |
| router/payload seed2 | 1.000 | .945 | .995 | 1.000 | EXCLUDE attack interpretation |

Seed2 router's structural flags are true but its exact-job capability fails. Do not round .945, borrow its hidden-job capability, average templates, or count this as three trained seeds. No cause for between-job capability differences was established.

E83 seed0 promotion independently confirmed: consistency .2, alternate supervision .5, BOS, 3000 steps, 100 groups, scores .99/.97/1.00/1.00. Artifact `9973770429`, run `33979228980`. Still remeasure the full strict gate in each integration job.

E86R's generic-dependency reduction is now separately CI-executed and the artifact verified: run `33981787028`, artifact `9974221001`, ZIP SHA-256 `d9002d14dd1a3fea0dfd72d374fb2f8f151a15b8cfe21a90071dcd97680078f6`; 14 traces, 34 prefixes, five artifact labels, 8 tests passed. No major novelty claim.

## New source-level falsification, not a trained-reader attack

Branch `research/rbc001-consumption-boundary-completeness`, evidence commit `98c27be4623e6cbef046f4f0589c5cf7645170d3`, report `docs/novelty/rbc001-consumption-boundary-falsification.md`, full local results `results/rbc001/rbc001-results.json.gz`.

Pinned unmodified authority/guard with real PyTorch hooks and a tiny numerical fixture: 126 schedule cases, 36 stale consumptions for same-thread relevant mutations after mask validation, zero under a precisely scoped conventional late-check comparator; unrelated reuse and other-thread serialization controls pass. 138 local unit tests passed. This falsifies unconditional atomicity when reentrant lifecycle callbacks are permitted; it does not falsify a contract that forbids/defer them or establish a GPT-2 attack rate.

Additional countermodels separate witness freshness from tensor binding, authentication from complete causal lineage, and history-free numerical/Jacobian observations from historical generation identity. Byte-identical observations can accompany different generation-validity labels. This restricts activation-only historical certification; it does not remove independent J-space causal-effect audits.

No production fix, shared-checkpoint all-state battery, new cryptography, new coherence mechanism, >=10x gain, <=5% overhead, leakage/UNKNOWN result, second backbone or complete J-lens certificate was achieved. Keep searching for a genuinely different neural lineage/operator/repair mechanism; ordinary synchronization and authenticated envelopes are correctness baselines only.

RBC-001 CI run `33984317369` was triggered on `b4d5f642ec6a6584d9095c1a6f4759bd91d67db5`; no completed RBC-001 artifact was counted in this checkpoint. The separate E84 and promoted seed0 integration jobs are not declared passed here.
