# R290 Result — Quantized Residual Generation Ports

Status: **compactness mechanism gate passed on the narrow held-out read-value test; not DoD.**

## Executed evidence

Real frozen backbone: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, checkpoint SHA256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.

The exact BF16 per-fact two-slot capsule occupies **24,576 bytes**. R290 trained no fact-specific parameters. A shared anchor was built only from the train-value set; future held-out values received no gradient updates and did not refit the anchor.

Best passing variant:

- residual K3 / V3;
- group size 32;
- no sparse correction entries;
- held-out exact-cache top-match: **100%**;
- held-out compressed semantic accuracy: **100%**;
- per-fact storage: **5,376 bytes**;
- per-fact ratio versus BF16 capsule: **21.875%**;
- reduction versus BF16 per-fact capsule: **78.125%**;
- shared anchor: 24,576 bytes total;
- amortized ratio at 1,000 facts: **21.975%**;
- amortized ratio at 1,000,000 facts: **21.8751%**;
- mean tensor relative L2 error: ~0.1071;
- maximum full-vocabulary logit delta: **1.34375**.

Artifact ZIP SHA256: `2f3c472a479456decac631bc19d4f461de65a3abc1ccb382a38bc4b63d7fb583`.
Report SHA256: `2da5b2591e60fcf16a3ccc44e49c7753ae498cb147260877ab8335ca8cc50b59`.

## What changed versus R288

R288's shared low-rank projection failed: even relatively modest reconstruction error destroyed attention behavior on most held-out values. R290 therefore preserves every KV coordinate and reduces precision instead of deleting subspace directions.

The result is qualitatively different: the K3/V3 residual representation retained the exact-cache winner on all held-out read probes while using less than one quarter of the per-fact BF16 bytes.

## What is *not* proved

The 1.34375 maximum vocabulary-logit delta is not small. Therefore R290 does **not** establish logit equivalence or arbitrary-task equivalence. It establishes only decision preservation on the executed held-out read-value probes.

It also does not establish novelty: low-bit KV quantization, mixed key/value precision, residual quantization, and shared-anchor compression all have substantial prior art. The value of R290 is architectural feasibility for a compact Port payload, not a standalone invention claim.

## Decision

- Kill naïve global low-rank Port PCA as the primary representation.
- Keep groupwise residual quantization as the current compact bootstrap representation.
- Do not freeze the codec design yet.
- R293 must test the **same query-independent compressed capsule across multiple operations**.
- Later gates must test larger models, multi-Pod composition, free-form generation, and GPU kernels.
- The eventual novelty claim, if any, remains the full lifecycle architecture (canonical identity + generation authority + causal isolation + lifetime closure + independent verification + compact Ports), not K/V quantization.
