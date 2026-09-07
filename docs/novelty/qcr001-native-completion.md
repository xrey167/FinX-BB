# QCR-001 — completed native-precision result and independent certificate checks

**Restricted carrier falsification, not a major invention.** Observed and audited 2026-09-05. This supplements the earlier queued-status record; it does not rewrite it.

Executed source: `959441c95aaf9a08d41a2e04e8d280d090730664`.
Completed Actions run: **33983919322**. All five jobs completed successfully: scalar controls plus DistilGPT2/Pythia70M in BF16/FP32. The original preregistration and scientific protocol were unchanged. No new model training or additional model run was launched for this completion audit. Main and historical experiment files remain unchanged.

## What the native result establishes

The candidate has a fixed rank-limited source-response basis for each layer's key tensor and value tensor, with independent coefficients per tensor and source revision. The oracle fits to ALL freshly rebuilt target states; it is intentionally favorable, not a deployable method or a speed baseline. Rank is across source-revision response snapshots, not Kamera's token-by-feature factorization axis.

The source is one synthetic scalar injected after block0 at token position3. Two prompts, three direction seeds,33 amplitudes from-1 to3; old source=1, zero source=0. Real native past_key_values is carried through a fixed continuation. Prefill and full continuation cache are checked; NEW continuation slots are also measured separately. There are192 nontrivial revisions per model/precision arm,768 across all four. These are correlated precision/prompt/direction measurements, not768 independent worlds or trained readers.

| Model / native arithmetic | Rank16 whole-trajectory exact | Rank16 NEW-slot exact | Rank32 whole-trajectory exact | Rank32 NEW-slot exact |
|---|---:|---:|---:|---:|
| DistilGPT2 BF16 | 0/192 | 0/192 | 192/192 | 192/192 |
| DistilGPT2 FP32 | 0/192 | 0/192 | 191/192 | 192/192 |
| Pythia70M BF16 | 0/192 | 0/192 | 25/192 | 74/192 |
| Pythia70M FP32 | 0/192 | 0/192 | 192/192 | 192/192 |

Rank1/2/4/8 also give zero exact whole/new reconstructions in all arms. Rank16 source-zero whole/new equality is0/6 in each arm. These compare stored native values exactly, not float64 residuals reinterpreted as BF16 failures. No-op, repeat-forward, absent-hook versus zero-source, pre-injection K/V and retained-prefix controls pass.

**Important rank32 qualification:** only32 nonzero response rows exist in this33-amplitude dataset. Full-rank snapshot reconstruction can therefore fit this finite set in real arithmetic. The rank32 failure maxabs is at most7.11e-15 in DistilGPT2 FP32 and1.51e-14 in Pythia BF16. Those failures are not a material rank-capacity obstruction. Do not turn them into a claim that no32-direction representation could work. Conversely, a finite-grid oracle pass is not generalization, efficient repair, or a full-system gate. Raw target tensors are not archived for diagnosing every tiny mismatch; the pinned script regenerates them.

## Material error in actual new persistent writes

Deepest-layer rank16 NEW-slot maximum error, range across six prompt/direction cases. These are neural-coordinate units, not semantic-error probabilities.

| Model / arithmetic | Keys | Values |
|---|---:|---:|
| DistilGPT2 BF16 | 0.03125–0.03125 | 0.03125–0.03125 |
| DistilGPT2 FP32 | 4.2915e-6–3.5882e-5 | 6.7353e-6–5.0008e-5 |
| Pythia70M BF16 | 1.03125–1.625 | 0.34375–0.5 |
| Pythia70M FP32 | 0.0012531–0.0034550 | 0.0003614–0.0011016 |

These are independently executed native arithmetic arms; changing dtype changes the computation. No monotonic relation between precision and useful repair quality is inferred.

## Stronger than failed least squares:43 exact fixed-basis witnesses

For a PARTICULAR archived rank16 basis U, the stipulated mathematical carrier is `RN_native(h_old + U c)`. Its coefficients may be arbitrary, including outputs of an arbitrarily nonlinear predictor. Each desired native coordinate supplies a rounding interval. If the affine span cannot intersect the joint rounding box, no coefficient vector works under that stipulated one-final-round model.

The numerical LP is only a proposal mechanism. A17-row witness stores16 basis rows, an exactly dependent extra row, exact rational dependency coefficients, shifted rounding intervals and their original dyadic endpoints. Rational interval arithmetic proves that the extra row's required interval is strictly disjoint from the range permitted by the16 anchor intervals. Using CLOSED rounding-cell supersets makes midpoint inclusivity conservative. LP status alone is never counted.

| Arm | Exact witnesses | Inconclusive attempts |
|---|---:|---:|
| DistilGPT2 BF16 | 12/12 | 0 |
| DistilGPT2 FP32 | 9/12 | 3 |
| Pythia70M BF16 | 12/12 | 0 |
| Pythia70M FP32 | 10/12 | 2 |

All43 archived witnesses pass the separate standard-library rational checker. A newly added independent native-cell decoder/encoder additionally verifies **all731 declared rounding cells** against the BF16/FP32 binary grids, without Torch, NumPy or a solver. Checks remain enabled under `python -O`; `--require-witnesses` rejects a vacuous aggregate check. Five unsuccessful witness searches remain INCONCLUSIVE, not proofs of feasibility or impossibility.

**Do not widen the theorem:** this rejects the specified stored basis, not every possible rank16 basis. The witness rows mix prefill/retained and new coordinates; they prove no single permitted coefficient vector satisfies that tensor's whole tested write trajectory, not a separate theorem for new-only carriers. An arbitrary staged floating-point program can leave the represented-real affine span through its arithmetic; that different program is not universally ruled out. The empirical SVD-plus-cast failures and the mathematical fixed-basis certificates are distinct evidence.

The independent checker proves algebra and native rounding-cell geometry for the archived instance. It does not cryptographically authenticate that an untrusted executor actually ran the named model or emitted the declared basis. Model provenance rests on pinned source/checkpoint revisions, workflow evidence and archived hashes, not on the algebra certificate itself. This is not a J-space/J-lens audit.

## Native execution is not lifted evaluation plus final casting

Each trained model's already-quantized parameters were lifted losslessly into float64; parameter-value digests agree before/after lifting. The old and source-zero worlds were evaluated there and cast back to native precision. **None of48 comparisons** matched the complete native write trajectory. This includes both precisions and both backbones. The source injection, nonlinear operators and intermediate arithmetic are part of that numerical-program difference. No task-quality comparison was measured.

Therefore the original QCR distinction stands: a float64 failure cannot automatically be promoted to a BF16 failure, but here actual native execution supplies the missing evidence. Small real-valued errors can disappear on rounding in the scalar positive controls, yet rounding does not rescue these retained rank16 oracle bases.

## Actual test counts

Original29 tests ran in each native job, no skips: DistilGPT2 BF16 1.48s; DistilGPT2 FP32 1.54s; Pythia BF16 1.49s; Pythia FP32 1.06s. These are repeated executions of29 tests, not116 distinct tests.

During this completion audit, the original45-test local suite (29 scientific/control tests plus16 independent-witness checker tests) passed again in2.33s. The32 new native-cell tests passed in1.92s. Running all three files together gave **77 passed in2.71s**, no skips. The native-cell tests include exhaustive roundtrip of finite BF16 encodings excluding negative-zero's lost rational sign, normal/subnormal value controls, native-neighbor comparisons and ties-to-even. The32 new tests are local post-result checker validation, not part of the preregistered model experiment or the original CI run. This is not the full repository suite.

No local pretrained rerun is claimed. The current container lacks transformers and its attempted HF metadata read failed DNS; pretrained measurements are from the completed CI jobs. No access restriction was bypassed.

## Prior-art search this turn

Xiao2608.30198 remains motivation for future-state propagation, not novelty. Kamera2606.23581 already stores canonical clean K/V and low-rank conditioning patches and evaluates functional recovery. Its factorization axis and multimodal workloads differ; this screen neither reproduces nor refutes Kamera. RLibm-MultiRound2504.07409 owns the general rounding-interval/correct-rounding approach. PyTorch2.10 documents numerical differences between mathematically equivalent computations. Classical interval-matrix regularity and rational separation are established mathematics, not a new invention.

A further direct provenance boundary is MemLineage2605.14421v1: cryptographic entry provenance plus weighted derivation lineage and sensitive-action/authority repair. Its threshold-dependent attribution guarantee is not a proof that every downstream neural tensor equals a rebuild. Its discussion explicitly distinguishes prevention from recovery. Attaching lineage and calling it a neural repair certificate would not distinguish our mechanism.

W3C PROV-DM already supplies derivation/revision/invalidation vocabulary. Broad patent searches returned mostly irrelevant biological-lineage results. A Google Patents retrieval attempt for US20260236690A1 and another for US12675453B2 failed; their full claims were NOT reviewed and cannot be treated as cleared prior art. A surfaced MDPI KV-localization result also could not be opened, so no technical conclusion is based on its snippet. No exhaustive novelty search or patent clearance is asserted.

Primary sources read:
https://arxiv.org/html/2608.30198v1
https://arxiv.org/html/2606.23581v1
https://arxiv.org/html/2504.07409v1
https://arxiv.org/html/2605.14421v1
https://arxiv.org/html/2607.27539v2
https://arxiv.org/html/1806.09988v1
https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html
https://www.w3.org/TR/prov-dm/

## Architecture decision and unchanged gates

The stored rank16 affine-chart escape is not rescued by switching its claim to BF16 tolerance or by fitting a better coefficient predictor. Any future candidate must either change the representation, supply a sound native-execution certificate with valid context, or fall back to exact reconstruction. Arbitrary nonlinear decoders, adaptive bases, fine-grained factorization and different write-time representations remain open here; none is promoted as a novel mechanism by name alone.

No full-system gate advances: >=10x complete mutation-to-ready, <=5% inference-throughput loss, matched total memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, trained-reader seeds, independent J-space and generation/publication safety. Two frozen backbones and three direction seeds do not qualify the semantic memory system on two backbones/three training seeds.

## Provenance and reproducibility

All four original model ZIP archives were downloaded and SHA-256 checked against GitHub metadata. Every source/test digest and result JSON digest in their sha256.txt was checked against actual bytes. Exact original archives, JSON/per-revision records, independent checkers, local checker logs and a machine-readable summary are included in the downloadable native-completion bundle. Raw K/V tensors and pretrained weights are not claimed as an archived dataset.

| Artifact ID | Arm | ZIP SHA-256 |
|---:|---|---|
| 9975123051 | DistilGPT2 BF16 | 161e840dd012644d86877945f3916fff227e40e7b310a0bf0ad945ea09652ebf |
| 9975034172 | DistilGPT2 FP32 | cce0e44a4892adeb739856d4374ed0776626c5fe1559139dee0a5e121b1c04ec |
| 9975026307 | Pythia BF16 | 9a0b24788e3ae9b829af054bb1a31ad4170bbd625b44f9e7ae3ba46e4f632c10 |
| 9975058211 | Pythia FP32 | 65267fc0d7c5483d66bec9c3e488164958f71a705eef6022590e62f30df39249 |

New native-cell checker SHA-256: `5f65d0239de951208fd0cc22cee320d39b36018ae78b09c4e5bc80d6192e69ac`.
New tests SHA-256: `08e107a5224d20c48cab2e1bb00b56981b4df2e60160a4971e796a2735f8876d`.

From a repository checkout containing downloaded model-result JSON files:

```bash
python -O tools/qcr001_check_native_cells.py --require-witnesses path/to/model-result.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_qcr001_rounding_cell_repair.py \
  so/tests/test_qcr001_offline_certificate.py so/tests/test_qcr001_native_cells.py -q -ra
```

The first command needs only the standard library and the sibling independent checker. Tests require the dependencies already used by QCR; no pretrained weights are required for those tests. Model reruns use the original pinned script/workflow. No paid compute or model API was purchased in this completion audit.
