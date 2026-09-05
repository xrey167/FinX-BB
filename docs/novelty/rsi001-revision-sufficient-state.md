# RSI-001 — exact repair needs a revision-sufficient retained interface

**Architecture-contract falsification, not a major invention.** Date: 2026-09-05.
Parent NIC001: `aba7ffd6d91ddbd7db0359bea37cd4dd956a4d16`.
Preregistered before numerical execution at `38d5f326609a5ad92d019b51fb7a61f0d4dd633e`.
Experiment/test source pinned at `85588e1ed44d9f86d3be950908b95cc2efa872f1`.
Main and historical experiments remain unchanged.

## Decision and the interface restriction

CRR001 and NIC001 leave arbitrary joint nonlinear repair decoders open. RSI001 shows why changing decoder capacity alone cannot suffice when its retained inputs have already collapsed histories whose revised rebuilds differ. This rejects every decoder of the declared information-losing interface, not every neural repair architecture.

The interface contains the complete current state of the constructed three-layer matrix memory, the immutable model and exogenous suffix, the canonical source's old payload and requested omission, and complete identity/edge/generation lineage. It does NOT contain the discarded numerical earlier context, original prefix, historical activation values, or a receipt not actually retained. Both earlier contexts remain possible under the same seed/model and identity metadata. The seed fixes two candidate contexts but does not disclose which history occurred.

This is not a claim against full KV caches, a canonical pod store that retains all relevant earlier numerical inputs, or a decoder allowed to retrieve original records. Content-dependent hashes, accessible source references, checkpoints and receipts change the available interface and must be counted. Identity lineage is not assumed to secretly encode the missing values. A different per-history ID would itself be extra information.

## Elementary exact criterion, not a new theorem

Let I(H) be everything available to repair after history H, and T_mu(H) the full rebuilt write trajectory under revision mu. An unrestricted mathematical decoder R_mu with R_mu(I(H))=T_mu(H) exists on the declared domain exactly when

`I(H1)=I(H2) implies T_mu(H1)=T_mu(H2)`.

Necessity follows because a function returns the same output on the same input. Sufficiency defines R_mu by the common target on each interface class. This is elementary function factorization; it proves neither a fast algorithm nor a new information-theoretic result. For a family of allowed revisions the implication must hold for each revision. Identifying that family is part of the contract.

If one interface class has K different required targets, a fixed-width auxiliary record that disambiguates them needs at least ceil(log2 K) bits. Existing current-state bits and identity metadata are held fixed in that statement. This is ordinary counting, not a novel deletion lower bound.

## Exact finite enumeration

Four earlier coordinates, each in {-3,-1,1,3}, give 256 histories. A source overwrite erases the addressed row. The subsequent three-stage rational nonlinear suffix and all current representations are identical across those histories. Omitting the source overwrite produces 256 distinct full trajectories and 256 distinct final states.

- Distinct current interfaces: **1**.
- Distinct required omission targets: **256**.
- Auxiliary information required in this constructed class: **at least 8 bits**.
- A one-byte packed earlier-context receipt repairs all 256 by ordinary rebuilding.
- Exact rational beta=1/2 control remains injective for all 256 inputs.

The one-byte result relies on this four-symbol/four-coordinate domain. It is not a compression result for arbitrary neural memory or an implementation of a new receipt scheme.

## Five-seed stateful nonlinear operator countermodel

Three matrix-memory layers, each 4 keys x 8 values. Two dyadic contexts per parameter seed differ only in the initially addressed row. A unit-key, beta=1, zero-value delta write overwrites that row. Four subsequent steps each update all three layers, using state-dependent tanh values. There are twelve downstream layer writes.

The overwritten histories have byte-identical source-boundary states, all downstream write trajectories, and complete current states. Their identity/edge/generation metadata is also identical. Yet the corresponding source-omitted builds differ in every layer.

| Seed | Maximum final/write difference between the two NEVER histories | Deepest-layer final difference | Error after repairing only the first layer |
|---|---:|---:|---:|
| 0 | 0.8168641901532081 | 0.007619960439116974 | 0.046613157122431303 |
| 1 | 0.7848733081592734 | 0.0016654258972277833 | 0.03949958895065423 |
| 2 | 0.7732231236336212 | 0.007781431558863062 | 0.07114599481119836 |
| 3 | 0.7853922518238151 | 0.008906071854442588 | 0.08360020760221383 |
| 4 | 0.8185527257935921 | 0.008920224798063946 | 0.06792569558783415 |

These are designed neural-operator counterexamples, NOT pretrained/backbone or training-seed experiments. No native-model frequency or task-error rate is inferred from these state units.

Positive reference: saving the eight-coordinate addressed row BEFORE overwrite costs 64 bytes of float64 arrays. In the declared known-zero background, this restores the source-boundary state; ordinary replay of all twelve downstream writes matches every full-rebuild write byte-for-byte for both histories in all five seeds. Arbitrary background requires additional retained information. Current state arrays occupy 768 bytes. This is not matched total-memory qualification, minimum optimal receipt size, or a no-replay speedup. Recoverable earlier canonical inputs could replace this receipt, with their own accounted costs.

Independent NumPy and PyTorch implementations of the recurrence agree within 1.11e-16 locally. That tolerant cross-library check is only an operator diagnostic; actual repair checks require identical shape/dtype/bytes within each execution. Repeat/no-op controls, wrong-receipt rejection and first-layer-only failure are preserved.

## An interior gate does not ensure finite-arithmetic invertibility

A beta=1 overwrite is explicitly destructive in real arithmetic. To avoid treating a nonsingular real formula as a universal escape, the second screen evaluates

`h' = h + 0.5*(1-h)`

on 256 consecutive positive normal representable numbers beginning at 1, with explicit PyTorch CPU operation ordering. Its exact rational counterpart is injective. The floating implementations have collisions:

| dtype | Distinct inputs | Distinct outputs | First distinct-input gap mapping to the same output |
|---|---:|---:|---:|
| bfloat16 | 256 | 161 | 0.0078125 |
| float16 | 256 | 129 | 0.0009765625 |
| float32 | 256 | 129 | 1.1920928955078125e-7 |
| float64 | 256 | 129 | 2.220446049250313e-16 |

In each format, 1 and its next representable neighbor map to the same output. No saturated gate, subnormal input or underflow-only construction is needed. The bfloat16 sweep covers a wider numeric interval, so its distinct-output count need not match the other formats. Each exact rational sweep retains 256 distinct outputs.

This establishes an exact-inversion obstruction for the evaluated finite implementation, not meaningful language degradation at one ULP, a native deployment failure frequency, or a claim that all invertible designs collide. Bit-preserving arithmetic, saved rounding information, exact receipts or original inputs change the conditions.

## Audit boundary

Identical present-state inputs give identical deterministic present-state readouts for both histories even though their omission targets differ. The test includes a nonlinear readout control. It does NOT implement J-space/J-lens or refute an audit that has additional historical/interventional evidence. Independent auditing remains valuable, but cannot infer which discarded history occurred from identical present inputs alone.

The change is upstream of auditing: before a lossy persistent write discards information, the complete retained system must remain sufficient for the promised revision family. This is a correctness requirement, not an invention. Receipts themselves are persistent descendants: they require lifecycle/context lineage and must not become an ungoverned channel for resurrecting invalid source generations. RSI001 does not implement that publication/revocation protocol.

## New primary-paper version changes the prior-art comparison

Ramesh arXiv:2607.27539 is no longer just the July v1 title Subtract or Replay. Version2, revised **13 August 2026**, is **Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory**. It adds the native KDA receipt/forcing study: fixed-input transport succeeds in its control, but native omission changes later transitions/writes and active caches. Its replay audit checks all 80 declared KDA arrays. Those tested receipt classes are not a universal impossibility result for other interfaces.

Its training-free Gemma result uses a conditional retained-key reference; it must not be substituted for full raw-history all-state rebuilding. The revised paper expressly distinguishes those targets and discloses missing corrected-configuration paired decrement/full-repack timing. Future comparisons must use v2's interface and forcing distinctions rather than the older title or a simplified static-subtraction baseline. No result from that paper is claimed as ours.

## Other prior-art boundaries and unsuccessful leads

- MacKay et al., Reversible Recurrent Neural Networks, NeurIPS2018/arXiv1810.10999: storing discarded information bits permits exact reversal with forgetting. Saving lost bits/values is not a new mechanism.
- Hatamizadeh et al., Gated DeltaNet-2, arXiv2605.22791v1: Eq5 explicitly states the unit-key beta=1 overwrite property. RSI001 uses that standard operator, not a novel primitive; its toy is not an evaluation of their trained model.
- State commitment learning, arXiv2606.05201v1: forward-use training after hidden-thought erasure. This is not a guarantee of historical counterfactual state reconstruction.
- Latent State Design for World Models under Sufficiency Constraints, arXiv2605.01694v1: predictive/counterfactual sufficiency distinctions and their standard causal-inference background are already discussed. Generic sufficiency vocabulary receives no novelty credit.
- Xiao arXiv2608.30198 motivates future-memory propagation repair, not the interface result or an invention.
- IBM US20260119893A1, published claims1–20: added KV network/twin-distribution/Gaussian-mixture updates, with insertion/modification/deletion claims. The reviewed claims do not specify this exact full-history counterfactual contract. This is limited technical screening, not patentability advice, exhaustive search or legal clearance.
- W3C PROV-DM supplies derivation/revision/invalidation vocabulary. Identity and lifecycle metadata alone are not asserted to preserve arbitrary discarded numerical data.

Several broader patent keyword queries returned irrelevant results; they are not negative novelty evidence. An attempted arXiv HTML endpoint for the original Gated DeltaNet v2 was unavailable; the accessible Gated DeltaNet-2 v1 supplies the quoted operator. No claim that all relevant 2025–2026 literature or patents was located is made.

Primary URLs:
https://arxiv.org/html/2607.27539v2
https://arxiv.org/abs/1810.10999
https://arxiv.org/html/2605.22791v1
https://arxiv.org/html/2606.05201v1
https://arxiv.org/html/2605.01694v1
https://arxiv.org/abs/2608.30198
https://patents.google.com/patent/US20260119893A1/en
https://www.w3.org/TR/prov-dm/

## Gates, provenance and reproduction

Local selected suite: **36 passed in 1.10s**, no skips. This is not the full repo suite. The dedicated CI run is **33981893332**; completion is recorded separately only after the downloaded artifact is verified.

All full-system gates remain NOT EVALUATED: >=10x complete mutation-to-ready versus the strongest matched baseline, <=5% inference-throughput loss, matched total memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05 nats generic divergence/exact bypass, independent J-space auditing, generation/publication safety, trained seeds and second-backbone qualification. No paid GPU/model API was purchased.

Source SHA-256: `e93bcd79538bc8cb9d5d51609416ac74b704baa042c1e48b52ffca33f528e60c`.
Tests SHA-256: `5614d58c99fa04a4f4ffb07616fc2b641fe473d61799d2a6ff1c6e8eb141901b`.
Original local JSON SHA-256: `e27dcbfcdfba7938a8cb6e7ca15be534ddfdd0dec54c5cf36cbc19186bee5cd2`.
Git blob hashes of both uploaded code files match locally tested bytes. A metadata self-edge in the initial draft was fixed before first numerical execution; no failed numerical result or scientific threshold was hidden by that correction.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.rsi001_revision_sufficient_state --output rsi001-results.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_rsi001_revision_sufficient_state.py -q -ra
```

Dependencies: NumPy2.3.5, PyTorch2.10.0+cpu, pytest9.0.2. Python3.13; exact environment records accompany the results. The downloadable bundle retains the original local JSON/logs, code/tests, reports, CI evidence when verified, and an offline integrity verifier. No archived pretrained weights or trained-reader results are implied.
