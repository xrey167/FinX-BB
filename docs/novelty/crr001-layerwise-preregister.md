# CRR-001-L — stronger layerwise carrier control

Registered 2026-09-05 AFTER examining the initial aggregate-state results of run 33978377217, and BEFORE running this extension. This is a disclosed follow-up, not part of the original preregistration.

The initial frozen-model screen shows material residual for a rank-16 basis shared over the complete K/V state. That does not by itself reject 16 independently parameterized directions PER LAYER or separate coefficients for keys and values. Give the candidate that stronger representation now.

Use the identical models, two prompts, three direction seeds, injection site, RMS scaling, amplitude grid, float64/eager CPU settings, and source-zero/no-op controls as CRR-001. Pin original resolved revisions explicitly:
- distilbert/distilgpt2: 2290a62682d06624634c1f46a6ad5be0f47f38aa
- EleutherAI/pythia-70m: a39f36b100fe8a5377810d56c3f4789b9c53ac42

Fit a separate oracle response basis for EVERY (layer,key/value) tensor, with separate coefficients per tensor and source amplitude. Ranks 1,2,4,8,16. Oracle training still sees all fresh revision states. This tests a stronger family than the original globally shared basis. Also repeat aggregate K/V metrics for cross-run consistency, without claiming cross-machine bit identity.

Keep all unaffected and early-layer controls, even if they have truly low response rank. In particular block0 K/V must stay unchanged because injection follows block0. Record each tensor's full-rank SVD reconstruction roundoff, spectra and finite residuals. A byte mismatch near numerical roundoff does not establish that a rank bound is insufficient; material residual well above that floor is the numerical obstruction. No fixed tolerance authorizes an approximate repair.

This remains a fixed-prompt latent intervention on frozen models, not native semantic pod editing, an autoregressive lifecycle system, independent J-space auditing, or trained-reader/utility qualification. Source-wise, head-wise, token-wise, revision-dependent, nonlinear-decoder and fallback architectures are not all covered by this particular fixed per-tensor basis family. Do not turn its failure into an unrestricted nonlinear-carrier impossibility claim.
