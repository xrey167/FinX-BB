# E-000090 — In-Band Neural Generation Signature

Status: preregistered falsification experiment; **not a novelty claim**.
Date: 2026-09-05

## Motivation

E-000086R showed that ordinary alias+Pod generation metadata collapses to generic dependency/version tracking. E-000088 showed no useful semantic-dependency elision beyond a stronger field-sensitive baseline. E-000089 tests cross-model causal generation attestation, but external version metadata remains a generic systems mechanism.

The stronger neural-specific question is:

> Can a Pod generation be encoded **inside the neural state itself**, so a materialized Hidden/KV artifact remains self-describing with respect to the knowledge generation that causally produced it, even if external cache metadata is lost, swapped, or detached?

This is a stale-state integrity experiment, not a cryptographic security claim. An attacker with arbitrary tensor-write access is out of scope for the first phase.

## Prior-art boundary

Do **not** claim as novel:

- LLM output watermarking;
- activation watermarking / residual-stream marking;
- cryptographic MACs, version tags, cache metadata, provenance ledgers;
- KV-cache IDs, self-describing cache events, model fingerprints;
- generic error-detecting codes;
- hidden-state probes or linear classifiers.

SLAM (arXiv:2605.05443) already writes watermark signal into residual-stream geometry. Therefore "write a mark into activations" is owned.

The only candidate distinction worth testing is a **knowledge-generation freshness function**: the mark is bound to `(pod_id, generation)` and travels with the actual neural state whose reuse is conditioned on freshness.

## Candidate mechanism

For canonical Pod generation `G=(pod_id, incarnation)`, derive a deterministic generation code `s_G` from a keyed or deterministic codebook. A backbone-specific linker maps `s_G` into a low-energy residual direction at the memory broadcast site.

The memory write becomes conceptually:

`neural_write = content_payload + alpha * generation_signature(G)`

A detector attached to a later neural materialization boundary attempts to recover `G` (or at minimum distinguish current `G` from stale `G-1`) from the tensor alone.

The first experiment uses a deterministic orthogonal/random codebook and model-specific linear readout calibration. This is a screening mechanism, not a final implementation.

## Phase A — existence and utility kill screen

Use at least two frozen public backbones already used by E-000080/E-000089 (DistilGPT-2 and Pythia-70M).

For several generations per Pod and several Pods:

1. inject the same content payload with different generation signatures;
2. materialize post-memory Hidden state and, where feasible, KV state;
3. strip all external generation metadata from the artifact;
4. decode generation identity from the neural artifact alone;
5. test stale/current discrimination after a canonical Pod update;
6. measure the content-behavior change introduced by the signature relative to unsigned memory;
7. test serialization/round-trip of neural artifacts;
8. test unrelated Pod signatures for false-current acceptance;
9. compare against an external-version-tag baseline under normal operation and under deliberate metadata-loss / metadata-swap fault injection.

## Strong baseline

An ordinary external tag `(pod_id,generation)` attached to the cache is the correct baseline. Under normal metadata-preserving operation it should be expected to win on simplicity and exactness.

The in-band mechanism earns any value only under a registered fault model where tensor state survives but metadata is absent or mis-associated.

If the same robustness can be achieved by appending a separate explicit metadata tensor/channel without touching model computation at lower cost, that is also a strong baseline and can kill the neural-signature claim.

## Validity controls

V1. **Material memory effect:** unsigned content payload changes the model output relative to BYPASS/NEVER.

V2. **Signature survives neural transformation:** generation decode is measured from a later materialized neural state, not directly from the injected signature vector before model computation.

V3. **Metadata stripped:** the primary in-band decode receives no generation/version side metadata.

V4. **Current/stale discrimination:** after one Pod update, old neural state must decode as the old generation and fresh state as the new generation.

V5. **Locality:** adding the signature must not materially damage the content behavior or unrelated prompts.

V6. **Cross-model:** the property reproduces on >=2 backbone families using one-time backbone-specific calibration, not per-Pod retraining.

V7. **No audit optimization loophole:** J-space/J-lens is not used to train the signature or define its pass threshold in Phase A.

## Preregistered screening bars

Per backbone, before any novelty interpretation:

- current-vs-stale generation classification >= 0.99 on held-out generation/Pod combinations;
- unrelated-generation false-current rate <= 0.01;
- metadata-stripped state retains >= 0.99 classification accuracy after serialization round-trip;
- signature-induced output KL relative to unsigned same-content memory <= 0.05 nats and top-1 agreement >= 0.98;
- memory content effect relative to BYPASS remains material (registered max-logit difference > 1e-4);
- at least 95% of test artifacts satisfy all bars.

## Decision rule

### KILL

Kill this seam if:

- generation identity is not reliably decodable after neural transformation;
- useful signal requires per-Pod detector training;
- output/locality damage exceeds the bars;
- a separate non-neural metadata channel provides the same registered robustness under the same fault model at lower cost;
- the mechanism works only at the injection layer and not in a later materialized state;
- only one backbone family supports the result.

### SURVIVE AS RESEARCH DIRECTION

Survive only if the same Pod-generation code remains recoverable from model-derived state across >=2 backbones, survives metadata stripping/serialization, distinguishes stale from current generations, and preserves task behavior under the registered bars.

This still does **not** establish novelty. Phase B must then attack direct prior art in activation watermarking/provenance and establish an end-to-end systems benefit.

## Phase B — if A survives

Integrate with the real Symlink->Pod reader and CAVI lifecycle:

- one Pod UPDATE/REVOKE/SHRED changes the in-band signature once;
- every alias-derived state exposes the same generation signature;
- stale Bank/router/payload/hidden/KV states expose the old generation;
- fresh current state exposes the new generation;
- a consumption gate can reject stale state from the neural signature when side metadata is missing;
- independent J-space/J-lens audit checks content lifecycle separately from the signature detector;
- compare normal metadata path, metadata-loss path, metadata-swap path, global epoch and full recomputation.

Major-technology target remains: >=3 seeds, >=2 backbones, >=95% real LINK->Pod capability/lifecycle, <=2% old-generation leakage, exact/near-exact bypass, <=5% normal inference overhead, and a material availability/recompute advantage under realistic distributed-cache fault rates.

## Narrow candidate claim if it survives

> A versioned neural knowledge object carries a generation-bound in-band signature through the neural state it produces, allowing stale model-derived state to identify its originating knowledge generation independently of external cache metadata while preserving the knowledge effect and model behavior.

This is intentionally narrower than watermarking, provenance metadata, cache versioning or shared memory individually.
