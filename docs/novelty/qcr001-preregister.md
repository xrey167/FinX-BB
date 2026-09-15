# QCR-001 — can finite-precision rounding rescue compact exact repair?

Registered BEFORE numerical execution, 2026-09-05. Parent RSI001: 41de8bda822db37d1db445dcfd16c13f87d2340d. Status UNRUN. Candidate/baseline qualification, not a major-invention claim.

## Motivation and distinction

CRR001 tested fixed bases in float64. That cannot by itself reject an implementation promising equality only in its native lower-precision state: nonidentical real responses may round to identical stored values. Conversely, an error described as within bf16 rounding is not a proof of identical tensors. Test the concrete round-after-repair escape rather than extrapolating either way. RLibm/RLibm-MultiRound already uses rounding intervals, and Kamera2606.23581 already combines canonical clean KV with low-rank conditioning patches. Those are prior art, not proposed novelty.

## Fixed execution protocol

Pinned DistilGPT2 revision 2290a62682d06624634c1f46a6ad5be0f47f38aa and Pythia-70M revision a39f36b100fe8a5377810d56c3f4789b9c53ac42. Actual CPU model execution in bfloat16 and float32, eval/eager/one thread; two CRR001 prompts, three direction seeds per model/precision. One scalar injected at token position3 after block0, direction RMS half the source activation RMS, with direction rounded to native dtype before use. Sweep 33 dyadic amplitudes from -1 to3, old source1 and source-absent0. Carry actual K/V through one identical exogenous continuation chunk: ` The recorded result was checked again.` No injection during continuation.

Fit each (layer,key/value) tensor's oracle fixed basis independently using ALL native fresh response tensors, represented losslessly in float64 for analysis. Ranks1,2,4,8,16,32. Project source responses, add original state, and round to the actual native dtype. Copy oracle-observed invariant coordinates and no-op exactly so irrelevant reconstruction roundoff cannot manufacture failure. This uses test targets and is NOT deployable, a learned certificate or a speed baseline. Report coordinate-match fractions, complete-tensor equality, all-tensor equality per revision, and separately newly appended K/V slots. The latter cannot be mistaken for a retained old-prefix mismatch. Preserve early layer controls and any positive admissions. Full-rank cast control locates numerical reconstruction artifacts.

Stronger optional witness: for source-zero at rank16, attempt an exact rational separation between the native rounding box and the affine span of the SVD basis. Select 16 independent anchor rows and one extra row using numerical QR/proposal, then verify the row relation and disjoint interval bound using exact Fractions. A found witness certifies infeasibility for that FIXED represented-real affine basis followed by one final rounding, even if coefficients are not least-squares optimal. Failure to find a witness is inconclusive. It does not rule out differently trained bases, nonlinear decoders, finer granularity, fallback, or arbitrary floating-point programs exploiting internal rounding. Store witness rows and exact certificate data for offline verification.

Arithmetic-target control: lift the native-quantized parameters to float64, with the identical representable source directions and amplitude values. Run old1 and absent0 through the same two-stage protocol, then cast K/V to the native dtype. Compare with genuinely native execution. Check lifted parameter hashes agree exactly after lossless conversion. This separates accumulation/intermediate arithmetic changes from using different weights. No universal same-backend equality assumption.

## Exact interval controls

Implement a small exact rational round-to-nearest-ties-even quantizer and interval-containment checker. Show (a) a nonzero approximation error can nevertheless give exact stored outputs when its proven enclosure lies in one rounding cell; (b) arbitrarily small error can cross a midpoint; (c) staging rounding through a recurrence differs from only rounding the final ideal value. Use independently checked PyTorch bfloat16 examples where applicable. This is classical correct-rounding logic, not a new theorem or method. Native LLM residual bounds are NOT certified by these scalar controls.

## Qualification

Reference contract remains every dependent persistent write, not average error, last logits or a precision-sized tolerance. Positive oracle reconstruction on finite samples is not a sound runtime certificate. Failed SVD reconstruction does not prove infeasibility of every basis/decoder. Do not count direction seeds as training seeds or two frozen architectures as application second-backbone qualification. No trained semantic reader, alias/generation protocol, independent J-space audit, full memory/throughput or >=10x mutation-to-ready claim. Every application gate remains unmeasured here.
