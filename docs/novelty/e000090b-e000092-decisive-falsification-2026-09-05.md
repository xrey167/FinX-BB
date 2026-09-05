# E-000090B / E-000092 — decisive falsification and direction change

Date: 2026-09-05
Status: **DECISIVE DIRECTION CHANGE**

This record uses the completed GitHub Actions evidence from the current `research/e000051-clean-bystanders` programme. It does not make a patent-clearance conclusion and does not revive E-000086R or E-000088.

## Decision summary

1. **E-000090 in-band generation watermarking is killed.** The balanced E-000090B correction removes the original decoder-conditioning defect yet still fails the registered generation-recovery/freshness gates on both DistilGPT-2 and Pythia-70M for every signal-strength arm. E-000091 must not be promoted from E-000090/E-000090B.
2. **E-000092 Generation-Keyed Neural Addressing is killed before Phase B.** Although its hard top-1 selector always chooses the current synthetic row in the registered seed-0 screen, the stale row retains material neural routing mass in every alpha arm on both backbones, directly triggering the preregistered kill condition. The zero-KL/top-1 functional result is produced by a hard argmax gate that discards the measured soft routing mass; a co-located sidecar/forwarder can make the same current-row decision with zero stale neural mass and lower complexity.
3. **No major invention is promoted.** The real LINK->Pod reader, >=0.95 every-template capability gate, lifecycle battery, three seeds, two-backbone end-to-end qualification, UNKNOWN/deletion leakage, ABA/rollback/TOCTOU, key/reconstruction attacks, J-space/J-lens content audit, <=5% overhead and systems-advantage gates remain unqualified for any surviving mechanism.

## E-000090B: balanced correction still fails

GitHub Actions run `33986688930` completed successfully as an execution job on both model families. The corrected protocol reports a balanced 8-bit training codebook, full design-plus-bias rank 9, no train/test generation overlap and forced wrong-generation sidecar pairs.

### DistilGPT-2

For signature RMS `0.02, 0.05, 0.1, 0.2`:

- exact held-out generation accuracy: **0.0 in every arm**;
- stale/current accuracy: **0.0 in every arm**;
- roundtrip accuracy: **0.0 in every arm**;
- output KL remains within the 0.05-nat gate and top-1 agreement is 1.0, but the freshness signal is not recoverable.

The backbone decision is `FAIL_BACKBONE_SCREEN`.

### Pythia-70M

For the same registered RMS sweep:

- best exact generation accuracy is **0.0625** at RMS 0.02 and **0.0** for RMS 0.05/0.1/0.2;
- best stale/current accuracy is **0.03125** at RMS 0.02 and **0.0** otherwise;
- RMS 0.02 also violates the <=0.01 false-current gate with **0.0625** false-current rate;
- stronger RMS values do not recover generation identity, and RMS 0.1/0.2 additionally exceed the KL gate or degrade top-1 at the strongest arm.

The backbone decision is also `FAIL_BACKBONE_SCREEN`.

### Consequence

The balanced correction was specifically introduced to determine whether the original E-000090 zero-accuracy result was an artifact of decoder conditioning. Because the corrected, full-rank, balanced codebook still fails on both model families, **decoder conditioning does not rescue the registered watermark mechanism**. Activation watermarking/residual marking remains prior-art territory and no E-000091 promotion is justified.

## E-000092: the reported `pass` does not survive the registered routing-mass kill rule

GitHub Actions run `33987325505` completed on DistilGPT-2 and Pythia-70M, seed 0, 32 prompts, alpha sweep `0.05, 0.1, 0.2, 0.4`.

The implementation constructs, for each prompt, an orthogonal semantic direction `s`, old-generation code `c_old` and current-generation code `c_new`:

```text
k_old = normalize(s + alpha * c_old)
k_new = normalize(s + alpha * c_new)
q_cur = normalize(s + alpha * c_new)
```

Thus `q_cur` is constructed with the same generation component as `k_new`. `_select` computes a softmax routing distribution but returns a **hard argmax** index. The actual output path then uses only that hard index:

```text
sel, p = _select(q_cur, keys)
chosen = [old_payload, new_payload, unrelated_payload][sel]
guarded = run_with_payload(..., chosen)
```

The soft routing distribution `p` is diagnostic only; it is not the mixture used to produce the zero-KL output.

### Material stale routing mass

Both backbones report essentially the same stale routing mass:

| alpha | DistilGPT-2 stale mass | Pythia-70M stale mass |
|---:|---:|---:|
| 0.05 | 0.494925 | 0.494929 |
| 0.10 | 0.480116 | 0.480120 |
| 0.20 | 0.423564 | 0.423570 |
| 0.40 | 0.248902 | 0.248910 |

Every arm is therefore far above a 2% old-generation leakage analogue and, more importantly, is plainly **material routing mass**. The programme's registered rule says to kill E-000092 if stale rows still receive material routing mass. That condition is met on both model families at every registered alpha.

### Why hard top-1 does not rescue novelty

The hard selector chooses the current row in 100% of registered Phase-A prompts, producing 0 KL versus the current-only gold output. But this guarantee comes from discrete row selection after an externally supplied current-generation query code. A correctly co-located sidecar/forwarder that knows the current authority can directly choose the current row and reject the stale row, with zero stale neural routing mass and without requiring a generation subspace inside model-derived addressing keys.

The E-000092 source itself already records this strongest-baseline warning. Phase A does not demonstrate a guarantee or systems advantage beyond that baseline. Because the registered kill condition is already met, **Phase B must not be promoted** merely from the hard-top-1 `pass` field.

The metadata-swap check also does not establish robustness to arbitrary transport faults: the current Phase-A code creates a fake metadata dictionary and never consumes it. The actual current-generation authority is already encoded when `q_cur` is constructed. A real LINK->Pod reader and real stale Bank/router/payload/Hidden/KV tensors were not exercised in E-000092 Phase A.

## E-000089 remains rejected

The E-000089 Cross-Model Causal Generation Attestation seam remains closed under the earlier kill rule: external `(object, generation)` freshness checking plus ordinary per-model dependency tags gives the freshness authorization guarantee, while the model-internal causal audit is observational and does not independently identify or authorize the originating generation.

## Fresh prior-art screen, 2025-2026

The latest targeted search does not supply a single reference that exactly names the narrow E-000092 phrase, but it further tightens the surrounding baseline space:

- TEPA (arXiv:2608.07429) treats stale-memory revocation as keyed validity/retrieval-state management.
- FreshCache (arXiv:2607.04281) explicitly gates reuse on freshness risk while preserving cache savings.
- C2KV (arXiv:2607.17715) uses a learnable sidecar for compressed/composable KV reuse.
- Qualcomm `US20250383989A1` / `WO2025259390A1` covers selected non-contiguous KV vectors through an attention mask.
- Huawei `WO2026086089A1` covers selected KV-cache segment recomputation.
- Microsoft `WO2025071935A1` covers KV-cache streaming for performance/fault tolerance.

These references are not asserted to anticipate the exact failed E-000092 construction. They reinforce that attention selection, cache freshness, sidecars, revocation and selective KV handling receive no standalone novelty credit.

## Direction change

Do not spend further programme credit on:

- heterogeneous generation/dependency metadata (E-000086R closed);
- semantic dependency elision (E-000088 closed);
- cross-model causal attestation as an authorization mechanism (E-000089 closed);
- recoverable in-band generation watermarking (E-000090/E-000090B closed);
- the current generation-keyed hard-address selector (E-000092 closed).

Any successor must change the technical frontier rather than relabeling these mechanisms. In particular, it must beat a guarantee-matched co-located sidecar/forwarder in a measurable system dimension while retaining exact lifecycle safety, and must not obtain its apparent correctness from a hard gate that simply implements the same external authority decision.

**Decision:** no major useful technical novelty survives the current E-000089/E-000090/E-000092 frontier. The immediate research direction changes away from watermark/address-code freshness and back toward mechanisms that can demonstrate a nontrivial guarantee or systems advantage over the strongest co-located authority baseline.