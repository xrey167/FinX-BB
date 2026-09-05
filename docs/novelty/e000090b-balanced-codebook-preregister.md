# E-000090B — balanced generation-code correction

Date: 2026-09-05
Status: **preregistered protocol correction before execution; not a novelty claim**

## Why E-000090 Phase A is not interpretable as a mechanism kill

The original E-000090 implementation used calibration generations `0..15` and held-out generations `16..23` for an 8-bit binary generation code. That split leaves bits 4–7 constant during calibration; in particular bit 4 is always `-1` in training and `+1` for every held-out generation. A learned linear decoder therefore receives no calibration example for the held-out sign of that code dimension. The observed zero exact-generation accuracy can be caused by this protocol defect and must not be promoted into a falsification of in-band freshness itself.

The original external-metadata-swap control is also malformed: samples are ordered as `(generation, prompt4)`, `(generation, prompt5)`, so the `ix ^ 1` swap exchanges sidecars between two samples carrying the **same generation**. Its reported accuracy is therefore expected to be 1.0 and is not evidence of robustness to a wrong-generation sidecar.

These defects do not alter the already observed facts that the content payload materially affects both models and that the tested signatures stay below/near the registered behavior-distortion bars in some arms. They invalidate only the original generation-generalization and metadata-swap interpretation.

## Frozen correction

E-000090B keeps the E-000090 model loading, residual injection sites, deterministic basis, signature RMS arms, content RMS, linear ridge decoder, held-out prompt/Pod combinations, tensor-only serialization path, stale/current construction, KL metric and top-1 metric unchanged.

Only the generation split is corrected:

- calibration generations: `[0,255,1,254,2,253,4,251,8,247,16,239,32,223,64,191,128,127]`;
- held-out generations: `[17,34,51,68,85,102,153,204]`.

The calibration set contains exactly 9 positive and 9 negative examples for **every one of the 8 generation bits**, and the bit-design matrix plus bias has full rank 9. No held-out generation appears in calibration.

The wrong-generation sidecar control is corrected by cyclically pairing every held-out generation with the *next different held-out generation*. Its accuracy must be 0 by construction; it is a transport-fault baseline control, not part of the neural detector fit.

## Frozen bars

No threshold changes are allowed after execution. Per backbone an arm passes only if all original E-000090 bars pass:

- exact held-out generation accuracy `>=0.99`;
- tensor serialization round-trip exact accuracy `>=0.99`;
- stale/current generation accuracy `>=0.99`;
- false-current rate `<=0.01`;
- max signature-induced output KL `<=0.05` nats;
- top-1 agreement `>=0.98`;
- material unsigned content effect relative to bypass.

Run both DistilGPT-2 and EleutherAI/Pythia-70M. Phase A survives only if at least one common signature-RMS regime satisfies all bars on **both** backbone families. Otherwise kill the in-band freshness direction at this implementation level and do not execute E-000091.

## Novelty boundary

A pass is still not a major invention. Activation watermarking, residual marking, hidden-state probes, ECC, provenance, version tags and cache metadata receive zero novelty credit. E-000091 remains conditional and must still beat co-located sidecars/content-addressed binding under the same transport faults, then survive the full real Symlink-Pod lifecycle and systems gates.
