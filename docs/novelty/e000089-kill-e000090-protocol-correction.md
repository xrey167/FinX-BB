# E-000089 decision and E-000090 protocol correction

Date: 2026-09-05
Status: **E-000089 novelty seam killed; original E-000090 generalization result quarantined pending preregistered E-000090B correction**

## E-000089 — structural success, novelty failure

GitHub Actions run `33986017770` completed successfully on DistilGPT-2 and EleutherAI/Pythia-70M for seeds 0/1/2. Artifact `9975328150` has GitHub digest `sha256:5c037d017541ca869f7dbd5cde12dce9bdf0a0043e0ea4adb396bea99524dab0`.

All registered structural checks passed. A single canonical Pod generation update invalidated the stale state in both model arms, unrelated state remained reusable, guarded recomputation matched fresh recomputation, and the controlled memory payload produced model-internal hidden-state changes.

That success does **not** survive the preregistered novelty kill rule. In the executed implementation, freshness eligibility is decided entirely by the external lineage predicate:

`not arm.stale_cache.reusable(auth)`

where `LineagedState` contains an ordinary externally captured generation/dependency witness. The hidden-state audit is measured after/beside that decision. It is not used to discover the source generation, validate the witness, distinguish stale from current state, or change whether the cached state may be reused.

Therefore an ordinary external `(object,generation)` check plus the same per-model dependency tag gives the same stale-reuse guarantee. The hidden-state deltas add evidence that the controlled payload had a causal effect inside each model, but they do not add an independent freshness guarantee. Under the programme's explicit rule — kill Cross-Model Causal Generation Attestation if ordinary generation/version checking plus standard per-model dependency tags gives the same guarantee or the causal audit adds no independent freshness information — **E-000089 is killed as a major-invention seam**.

This does not reject cross-model causal auditing as useful validation. It rejects claiming the composition of shared generation metadata + model-specific dependency tags + a hidden-delta audit as the invention.

## E-000090 — original 0% held-out generation result is not a valid mechanism falsification

Original run `33986409115` completed successfully on both backbones. Its artifacts are:

- DistilGPT-2: `9975335281`, digest `sha256:9b9ce35ce0be50825fae72f4506af093f46b38807ca2174428dfd0833f65f4d5`;
- Pythia-70M: `9975314577`, digest `sha256:3e7c737bf615be99c8162c64282fc606081ec97062a556ea8d10a6d0a3664f1c`.

The original implementation used an 8-bit binary code with calibration generations `0..15` and held-out generations `16..23`. That split is defective for the registered learned decoder:

- bits 4–7 are constant during calibration;
- bit 4 is `-1` for every calibration generation and `+1` for every held-out generation;
- consequently the decoder never receives a calibration example for the held-out sign of that dimension.

The resulting 0 exact held-out-generation accuracy on both backbones cannot be promoted into a falsification of in-band generation freshness. It may be fully explained by the train/test-code support defect.

The original metadata-swap control is also invalid. Test examples are emitted in adjacent `(same generation, prompt 4)` / `(same generation, prompt 5)` pairs. Swapping adjacent entries therefore swaps metadata between tensors with the **same generation**, so the reported external swapped-metadata accuracy of 1.0 is expected and does not represent the intended wrong-generation transport fault.

## Frozen correction

Before rerunning, E-000090B was preregistered at `docs/novelty/e000090b-balanced-codebook-preregister.md`. It changes only the generation split and the malformed sidecar-fault control; the neural mechanism, model families, injection sites, decoder family, signature RMS arms, content RMS and all pass thresholds remain unchanged.

E-000090B calibration has both signs for every code bit and full-rank bit design plus bias. Held-out generations are disjoint. The wrong-sidecar control deliberately uses a different held-out generation.

Workflow run `33986688930` was submitted from commit `2398a972eac5980b524f75a9dee823b8b757410b`. No E-000090 mechanism decision is made from that run until its artifacts are complete and inspected.

## Prior-art boundary retained

Nothing here reopens activation watermarking, residual marking, hidden probes, ECC, provenance, version tags, cache metadata, selective invalidation or cross-model KV transfer as novelty. Recent work already strengthens the baseline around latent-state integrity and transport: cryptographic manifests can bind full KV payloads to model/session/context, and cross-model KV reuse/translation is itself an occupied systems area. The only remaining E-000090 question is whether a generation-bound freshness signal carried by the reusable neural tensor creates a systems guarantee that co-located metadata/content-addressed binding cannot provide more cheaply under the same registered fault model.

## Decision

- **E-000089: KILL as major-invention candidate.** Structural behavior is real; novelty reduces to ordinary external generation/dependency validation plus an observational causal audit.
- **Original E-000090: QUARANTINE as a protocol-invalid negative result.** Do not count the 0% held-out result as mechanism evidence.
- **E-000090B: active corrected falsification screen.** E-000091 remains blocked unless the corrected screen survives on both backbone families under unchanged bars.
