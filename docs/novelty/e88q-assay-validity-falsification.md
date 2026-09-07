# E88Q — decisive falsification of the E-000088 kill screen

Date: 2026-09-05.
Inspected source: commit `3e022f432a4acf1d3cba652d5ce786c0db8d8a8b`.
Decision: **INVALID_ASSAY_FOR_GENERAL_PER_ARTIFACT_INFERENCE**. Quarantine E88's general KILL interpretation; neither promote nor abandon semantic dependency elision on that assay.

This is locally executed source-level and analytical evidence, not a newly trained reader, an independent CI reproduction, or a major invention. No original model, experiment, guard, thresholds or historical result files are modified by this checkpoint.

## 1. The current survival gates cannot all pass

The pinned `so/experiments/e000088_semantic_dependency_elision.py` builds a batch containing a positive-control query plus bystanders, compares each complete artifact array with `np.array_equal`, and counts one artifact/Pod pair per kind rather than per query. Its field baseline counts three invalidations per Pod, exempting routing.

Let N be the number of Pods and P the number with responding positive controls. E15.predict computes answers with logits.argmax, and the model computes logits with a fixed deterministic readout of the captured final hidden state. Therefore a responding positive control implies a changed logits row and a changed hidden row. It invalidates both entire batched artifacts even when all bystanders are invariant.

Consequently semantic invalidations >= 2P. The field baseline counts 3N. V1 requires P/N >= .95, so:

`field_sensitive_work_reduction <= 3N/(2P) <= 30/19 < 2`.

For the registered N=8, P must be 8 and the bound is 1.5. The utility gate requires at least 2.0. No reader satisfying the specified deterministic dataflow and validity gate can make all current survival gates pass.

Whole-batch equality is not intrinsically wrong if the entire batch is deliberately the indivisible cache unit. It cannot establish the absence of independently reusable per-query states. Changing measurement granularity does not itself prove physically efficient row-level reuse.

## 2. Executed counterexamples

The full original E88 source was locally verified against Git blob `c65bee4dd221954892a695aaf5985a0fe6401a9f`. Its original run_seed, replace_obj, artifact and exact_equal function bodies were executed unchanged, with explicitly labelled analytical replacements for E15 training/world/prediction dependencies.

Nine fixture jobs: seeds 0/1/2 crossed with independent, coupled and nonresponsive fixtures; eight Pods and three bystanders each; all 256 payload values. Total: 18,432 prediction calls.

For each seed:

| Fixture | Responding positives | Original invariant hidden batches | Invariant bystander hidden states | Original field-count ratio |
|---|---:|---:|---:|---:|
| Independent bystanders | 8/8 | 0/8 | 24/24 | 1.0 |
| Coupled bystanders | 8/8 | 0/8 | 0/24 | 1.5 |
| Nonresponsive reader | 0/8 | 8/8 | 24/24 | 24.0, but V1 fails |

These are diagnostic artifact counts, not measured speedups. The independent fixture is ordinary direct lookup already handled by source-sensitive dependencies: zero novelty credit. Fixture seeds do not count toward the three-trained-seed gate.

## 3. The blanket routing exemption is unsound

E88 exempts the complete routing tensor from payload-update invalidation on the premise that routing only reads keys. This holds for the initial forward read with a fixed query and fixed forward keys, but not for all recorded slots:

`payload -> retrieved value -> DerefBlock query -> dereference routing`

`retrieved value -> HopBlock.apply_read -> next-hop query -> next-hop routing`.

The actual model stores these later distributions in the same routing tensor. Reverse keys can also depend on object values and require separate analysis.

The pinned HopBlock and DerefBlock source definitions were executed locally. Keys were held fixed while one encoded value row changed. Grid: three random parameter seeds, dimensions 8/16/32, float32/float64, eight value rows and 255 alternatives per row: 18 configurations, 36,720 edits.

| Observation | Changed after payload-only edit |
|---|---:|
| First forward route | 0 / 36,720 |
| Dereference route | 36,720 / 36,720 |
| Next-hop route | 36,720 / 36,720 |

These are random-weight operator counterexamples over an explicitly defined encoded-payload domain, not GPT-2 reading, trained-reader attack success, or deleted-object leakage. They falsify a universal structural exemption; they do not assert that every trained route changes.

Source-segment SHA-256 values:
- HopBlock: `f2941751b5404a6990ed764cc65ac249cd708340a6eb02a4cd0e427f6a57adcd`
- DerefBlock: `138969cc216e29ab5baa89638bf0d677a3f25e87ea7395a8b64f950071883cc3`

A valid strongest baseline must be field- and slot-sensitive. Fixing this is correctness engineering, not an invention.

## 4. Implemented measurement repair and evidence scope

The accompanying local evidence bundle implements ArtifactSweep: per-query exact invariance over the complete declared finite domain, separate routing-slot artifacts, independent positive/bystander accounting, rejection of incomplete domains, duplicates, changed shapes/dtypes and nonfinite arrays. A separate exact-job/checkpoint gate checks all templates 8–11 using integer counts: `20*correct >= 19*total`. It does not fabricate capability measurements.

27 local regression tests passed. They include partition/order invariance of the accumulator (not a floating-point kernel guarantee), stable-answer versus changed-hidden separation, routing-slot separation, domain checks, .945 exclusion, job/checkpoint identity and the utility bound for N=1 through 10,000.

The helper is not an E81/GPT-2 integration, production certificate, historical-lineage proof, optimized reconstruction system or independent J-lens audit.

Complete executed code, source snapshots, tests, environment, raw results and hashes are supplied in the conversation evidence archive `FinX-BB-E88Q-evidence.zip`; this Markdown checkpoint does not claim that the archive is stored in GitHub.

Archive SHA-256: `07faffd1891e2f9d9ac8d829ee9a2562ab7445128016799374441c11243e1a5d`.
Raw result JSON SHA-256: `eb59e8d6ca7e921a17437c5b74b9a50c687f81a833f736ae09a6ffc7e93057b6`.
Audit script SHA-256: `ff6d21cb889f3da4eaf16d1bb14f567f34ba3fbc1f2a473490b449749b9795b5`.

Reproduction from the extracted archive:

```sh
python -m unittest -v test_e88q
python e88q_audit.py --e88-source source/e000088_semantic_dependency_elision.py --model-source source/model_blocks.py --output results/reproduced.json
```

The runner also accepts the complete pinned upstream files and verifies E88's blob plus both class source segments. Python/NumPy/PyTorch versions are in the raw result. Execution was local CPU-only; elapsed fixture time is not a performance comparison.

## 5. Direction and preserved gates

Do not interpret E88's existing general KILL result as evidence against the research direction. A corrected test must separate positive controls, bystander queries and routing slots, use a sound guarantee-matched baseline, and use the strict-capability reader in each exact job. Measure real mutation-to-ready latency and memory, including certificate creation and reconstruction, not just artifact counts.

E86R's generic-dependency reduction and RBC-001's provenance/consumption-boundary restrictions remain valid. Nothing here revives tags/coherence as novelty. J-space remains independent audit only. Semantic noninterference and program verification receive no novelty credit.

E85 router/payload seed2 remains excluded at T9=.945. The promoted seed0 configuration remains consistency .2 / alternate supervision .5, but no new seed0/E84/E85 pass is asserted.

No major-invention gates passed here: zero new trained-reader jobs, no full stale-state battery, no leakage/UNKNOWN qualification, no matched-memory >=10x timing gain, no <=5% inference-overhead result, no second backbone and no independent J-lens certificate.
