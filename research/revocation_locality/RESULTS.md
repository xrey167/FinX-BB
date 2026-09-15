# RL-MIX-001: completed controlled cache-interaction screen

2026-09-05. **No breakthrough; no trained symlink qualification.**

## Decision

Reject the drop-in additive-erasure implementation: subtracting a source's standalone KV contribution from an already mixed frozen-model cache does not reproduce the remaining-source execution. This does NOT falsify learned revocation locality in general, nor establish novelty for interaction decomposition.

The next candidate must preserve useful nonlinear computation while making its actual persistent execution graph cheap to repair. Small interaction degree is not a sufficient proxy for repair cost. Compare against complete ordinary dependency tracking and incremental recomputation, not just a weak global-flush baseline.

## Executed and retained

- Run **33966312062**, source `810f4e0ba98dce15bb4bbeeee07f09a8ac9f6d5b`: original float32. GPT-2 completed; Pythia hit legacy-tuple versus Cache incompatibility. The latter is a harness failure, not a scientific result.
- Run **33966550019**, source `13692f2632ba1593d76c04bc8c99c7837a1582da`: explicit fresh DynamicCache and independent cached-versus-full-forward control. GPT-2 passed; all nine float32 Pythia cases exceeded the fixed 0.0005 max-absolute-logit control. They remain invalid under this protocol.
- Run **33966687588**, workflow `aa43cb1ea35f59d91477375347ba98cf5fe5ff0c`: float64 reference arm; both backbones completed and passed the unchanged controls. The workflow checks out `13692f...` and saves an exact asserted precision-only patch, original and executed source, environment and hashes. It changes arithmetic precision, not tolerances or case selection.

The final arm uses frozen `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` and `EleutherAI/pythia-70m` revision `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.

Three **intervention seeds**, three payload amplitudes (RMS 0.1, 1, 4), four controlled sources and all 16 source subsets: **18 cases, 288 clean subset cache states, zero training steps**. Seeds change exogenous prompts and payloads; they are not trained-reader seeds. Source injection uses layers 1 and 3 and token positions 1,4,7,10. No semantic alias resolver is being evaluated.

## Results

For full KV state F(A), the direct candidate after removing P is `F(All) - (F({P}) - F({}))`; the clean reference is `F(All minus P)`. All are decoded through the actual model from independent cache objects.

Maximum over three intervention seeds and four single-source removals at RMS 4:

| Metric | GPT-2 | Pythia-70m |
|---|---:|---:|
| Direct subtraction max absolute logit error | 6.7372629912 | 1.2287491351 |
| Direct subtraction KL to clean-current output, nats | 0.1457133778 | 0.0202951007 |
| Native cached-vs-full-forward control, max over all amplitudes/seeds | 0.0 | 1.4324542e-11 |
| Full interaction reconstruction logit error, max over all amplitudes/seeds | 1.5631940e-13 | 6.3664629e-12 |

Across all amplitudes, **72/72** single-source removals exceeded 0.0005 logit error under direct standalone-contribution subtraction. Low-amplitude cases support non-additivity too. This is not a deleted-fact leakage rate.

The established Boolean-lattice/Moebius decomposition reconstructs F(A) by summing all interaction terms whose source subsets lie inside A. It is an expensive exact reference, not a novel efficient solution. At RMS 4, retaining only singleton/pairwise/third-order terms gives maximum logit errors of `7.8924869 / 2.4312966 / 0.5058517` for GPT-2 and `0.3009752 / 0.1611082 / 0.0255144` for Pythia. These maxima range over all coalitions. Order three is exact for coalitions of at most three sources by construction; do not mislabel its all-four-source residual as a post-single-deletion failure.

The KL here is **approximation-to-current-cache KL**, not the programme's generic-language KL. No generic-language quality, semantic leakage, alias-relink/race protection, independent J-space audit, or speed advantage is established.

## Necessary exclusions and controls

A newly defined low-order model may be exactly revocable relative to itself even if it fails to reproduce its dense teacher. Do not infer universal impossibility from this frozen-model test.

High-order interaction does not imply exponential repair work: a 64-input product can be updated using six gates in an ordinary balanced product tree. Local regression tests cover this counterexample, complete subset reconstruction, source edits, cache serialization, zero-local-Jacobian versus finite-intervention distinction, and evidence gates that reject missing cells, NaN, forged control flags and false breakthrough labels. **19 local tests passed**, separate from model executions.

The novelty search adds/retains these primary-source exclusions:

- HarsanyiNet, https://arxiv.org/abs/2304.01811: neural Harsanyi-interaction structure.
- Eventful Transformers, https://arxiv.org/abs/2308.13494: selective reuse/update of neural computation.
- Self-adjusting computation, https://arxiv.org/abs/1106.0478: memoization, change propagation and from-scratch consistency.
- KVEraser, https://arxiv.org/abs/2606.17034: learned localized KV-context erasure.
- Agentic Unlearning / SBU, https://arxiv.org/abs/2602.17692v2: dependency closure and memory/parameter recontamination.

Pointers, MVCC, capabilities, generations, integrity binding, freshness, external memory and J-space remain excluded individually. The remaining co-design has NOT been shown novel or effective.

## Artifact integrity

Final result archives from run 33966687588:

- Artifact `9969671553` (GPT-2), SHA-256 `c56c43101bec885288fca187b57839b8da4f90fba9c5281beac5a80f13e32dd6`.
- Artifact `9969657414` (Pythia), SHA-256 `deb0c40f83b30eeb23ee59e4b487f69ced4caf09be1dc9f1faf14bb3a718b320`.

These downloaded ZIP digests were verified before reanalysis. Earlier failure artifacts remain available and are preserved in the accompanying research bundle. Full KV tables were hashed, not archived; the frozen models, inputs and code permit regeneration.

The first compact Actions inventory records all 67 runs then present and every listed job status; a later inventory is also retained. E-000080 remains a controlled independent-cache ordinary baseline. The corrected E-000079 race run 33964913121 is separate from the previously inconclusive attempt. E-000052's latest pilot has `use_links=false`, held-out aggregate reading 0.97 but template-9 reading 0.905 and generic KL 0.5282363892. None qualifies the requested joint seam. E-000081 was still running at the last check and screens three weights on seed 2, not three independent training seeds.

## Promotion stays unchanged

Fresh real-symlink and held-out reading >=0.95; REVOKE and SHRED propagation >=0.95; deleted-object leakage <=0.02; missing-key UNKNOWN >=0.90; active-memory generic KL <=0.05 nats. Same retained candidates, >=3 training seeds and >1 backbone where CPU-feasible, with the entire replay/race/reconstruction/independent-audit battery. **RL-MIX-001 does not satisfy or replace any of these positive gates.**
