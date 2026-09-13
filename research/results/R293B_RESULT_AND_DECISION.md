# R293b Result — Functional Quantized Port Fidelity

Status: **codec-fidelity mechanism passed on the bounded probe; base task-quality gate failed; not DoD.**

Backbone: frozen `Qwen/Qwen2.5-0.5B`.

Future held-out Port values: `gamma`, `delta`, `small`, `large`.
Operations: direct read, alternating-set classification, edge-set classification.

## Result

The exact BF16 Port baseline itself reached only **83.33% semantic accuracy** on these tasks. The residual K3/V3 compressed Port reached the exact same **83.33%**.

Crucially, compressed-versus-exact decision top-match was **100% across all executed held-out value/operation cases**. Therefore every semantic error in this bounded gate was already an error of the frozen model using the exact Port, not a new error introduced by compression.

- compressed/exact top-match: **100%**;
- exact semantic accuracy: **83.33%**;
- compressed semantic accuracy: **83.33%**;
- maximum full-vocabulary logit delta: **1.5234375**;
- compressed storage ratio: **21.875%** of BF16 Port bytes.

Artifact ZIP SHA256: `0f067f339cee7b3099a53895401d7fc845c9e359601d451923a715b7592b9c8b`.

## Decision

Do **not** call R293b an end-task pass. The frozen 0.5B model failed the semantic threshold.

Do count it as positive evidence for **query-independent codec fidelity**: one compressed fact capsule preserved the exact model's discrete decision across several operations instead of being specialized to the original read query.

This distinction is important for the architecture. Port representation quality and reasoning-model quality are separate axes. The codec should be judged against the exact-Port oracle, while the complete CKCA system must separately meet real task-quality thresholds on stronger backbones and benchmarks.
