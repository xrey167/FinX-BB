# NIC-001 — source-local nonlinear repair charts need a valid joint context

**Restricted-family architecture falsification, not a major invention.** Date: 2026-09-05. Parent CRR001: `5996a0074831f152e84f57ad1635ec06f747007b`. Main and historical experiment files remain unchanged.

Preregistered before numerical execution at `c70dd4d7bf33b7db0619df705346a449622de066`. Executed source: `361352d35bd4edcdc2bd36f9cb53ec089fbdf847`. Completed Actions run: **33980515363**.

## Decision and exact restriction

CRR001 rejected inferring a small fixed output basis from low source dimension; it did not reject nonlinear decoders. NIC001 removes that rank restriction and instead tests composition of source-local nonlinear finite responses cached in a common original context. The stronger arm includes every exact pairwise interaction. Both receive oracle fresh tensors, not learned approximations.

A source-indexed response is not necessarily independent of the other sources' revisions. Single-source correctness and even all pairwise corrections do not authorize a three-source endpoint. A generic joint nonlinear decoder, reconditioned chart, state-dependent repair operator, finer factorization, or exact replay can escape this restriction and is NOT falsified here.

The mathematical tool is the classical anchored Boolean Moebius transform, not an invention or a new F-IVM mechanism. This is a representation-validity screen, not an implemented production repair runtime or a systems speed benchmark.

## Oracle protocol

Let H(S) be the fresh state after removing sources in S, with all sources initially present in H(empty). Define I(S)=sum over T subset S of (-1)^(|S|-|T|) H(T). The independent chart uses H(empty)+sum I({i}); the stronger pairwise chart adds every I({i,j}). At the all-three endpoint its missing term is I({1,2,3}). Complete third-order reconstruction is an algebraic consistency/roundoff control, not a compressed repair algorithm: it already uses the target fresh world.

For one source i, the response R_i(c)=H(source i revised, context c)-H(source i old, context c) can change when another source changes c. Storing R_i under only i's own generation loses that dependency. A validity certificate must either bind the required context or prove invariance under its changes. Adding version metadata does not compute the missing interaction, and metadata/provenance itself receives no novelty credit.

## Exact controls

Python Fraction controls include separable nonlinear functions, pairwise-only interactions, a pure triple interaction, and a compact joint nonlinear decoder. In the pure-triple example H(a,b,c)=2+a+2b+3c+11abc with Boolean revision flags, every proper-subset pairwise reconstruction passes, but the joint endpoint is wrong by exactly 11. All six orders of adding the original single-source increments agree while being wrong. Order independence is therefore insufficient for correctness.

The compact joint control H(a,b,c)=1/(1+a+2b+3c) has nonzero third-order interaction despite a short exact nonlinear formula. Higher-order interaction does not establish an exponential general repair lower bound or require storing every subset. All full-order Fraction reconstructions are exact; appropriate single/pair positive controls pass.

## Actual pretrained persistent-state execution

Frozen DistilGPT2 and Pythia-70M, CPU float64, eager attention, deterministic eval, one thread, no training or remote code. Pinned revisions:
- distilbert/distilgpt2: `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
- EleutherAI/pythia-70m: `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.

Two fixed prompts and three direction seeds per backbone. Three synthetic scalar sources enter at separate positions 3,6,9 after block0; each direction has RMS half that position's unmodified activation RMS. Enumerate all eight source-removal worlds. No amplitude or prompt was retuned after results.

Each world carries its actual past_key_values through THREE additional fixed exogenous text chunks. No fresh source injections occur during continuation. The chunks and token IDs are recorded. We examine all layer K/V tensors after prefill and after each continuation, separately isolating NEWLY APPENDED slots from retained prefixes. This tests internal persistent-state propagation, not generated-token feedback or a trained semantic memory updater.

There are 12 model/prompt/direction cells, eight worlds per cell and four stages. Counts over layers/stages are correlated inspections, not additional independent seeds.

### Stronger pairwise chart result: newly written slots

For each backbone, all **144/144 later-layer continuation tensor cases** have material residual even after including every oracle pair correction: six cells x three continuations x four later blocks x K/V. Across both models that is **288/288**, with zero exact pairwise reconstructions in these cases. The prefill's later-block tensors also show the missing higher-order interaction.

Errors in the FINAL continuation's deepest-layer NEW K/V slots, range across six cases per backbone:

| Backbone | Tensor | Independent nonlinear chart maxabs | All-pairs chart maxabs |
|---|---|---:|---:|
| DistilGPT2 | K | 0.016799–0.061102 | 0.001323–0.010417 |
| DistilGPT2 | V | 0.023882–0.092907 | 0.001699–0.020826 |
| Pythia-70M | K | 0.066586–0.278301 | 0.006590–0.053067 |
| Pythia-70M | V | 0.024834–0.079106 | 0.002588–0.023905 |

The maximum all-order reconstruction roundoff over NEW-slot measurements is 3.5527e-15 for DistilGPT2 and 2.8422e-14 for Pythia. Material classification uses the preregistered descriptive floor max(1e-10,100*all-order roundoff). This floor never authorizes approximate reuse. Byte identity remains the exactness test; a near-roundoff mismatch alone is not labelled substantive nonlinear interaction.

Pairwise context drift is also material. Across later-layer continuation tensors and source pairs, cached single-source responses change with another source's removal by 0.000314–0.091101 (DistilGPT2) and 0.000060–0.313696 (Pythia) in maxabs units. These are neural-state units, not factual-error probabilities or task scores.

### Controls locate the effect

Every cell passes complete old/absent rerun byte equality, all-absent zero-hook versus genuinely absent-hook equality, unchanged block0 K/V, and unchanged old cache prefixes when continuation appends slots. Thus the new-slot result is not merely old differences retained in the cache.

For continuation, NEW K/V in blocks0 and1 is exactly source-independent. Pairwise repair there is exact. During prefill, block1 separately encodes the injected positions; its independent/pair residual is only approximately 2.22e-16 and is not counted as material. The higher-order interaction first appears in later persistent layers, after source information is mixed. All early and negative controls are retained in the artifacts.

## Execution and provenance

Local selected suite: **30 passed in 1.05s**, no skips. CI suites: **30 passed in 1.73s** (exact job), **1.12s** (DistilGPT2), **1.13s** (Pythia). These are the same 30 tests, not 90 unique tests or the full repository suite. Exact Fraction output agrees between local and CI executions. Pretrained execution occurred in Actions, not locally.

Three artifacts were downloaded, ZIP SHA-256 checked against GitHub metadata, and every archived source/result digest checked against actual bytes. See `nic001-ci-evidence.md`. Full JSON statistics, all source hashes, exact controls and original ZIPs are in the reproducibility bundle. Raw K/V tensors were temporary and are regenerated by the pinned scripts, not claimed as an archived dataset.

An initial local suite had 29 passing tests. Before publication/execution, snapshot comparison was strengthened to reject unequal nested sequence lengths, and a vacuity regression was added; final source has 30. No scientific arm or threshold changed. A later local analysis command encountered a container transport error; retry completed without changing the experiment.

## Prior-art differentiation checked this turn

- Xiao et al., arXiv:2608.30198v1, explicitly defines a pathway interaction contrast as well as memory-update propagation. Nonadditivity is already in the motivating paper, not our novelty. NIC001 is a controlled latent K/V experiment and not a replication of its MemoryBank/semantic outcome protocol.
- Fumagalli et al., arXiv:2501.16944v1, uses exact any-order Moebius/Shapley interactions for GNNs and spells out the same classical subset transform. Computing the interaction expansion is not a new mechanism.
- Schessl, arXiv:2607.24805v1, studies sequential weight-space Engram edits, order effects and drift in survivor statistics. Its setting differs from fixed-weight persistent K/V; we do not reproduce its numerical findings. Sequential interference or stale contextual calibration cannot be claimed new in broad form. The HTML date labels conflict (July arXiv header/August manuscript date), so no precise publication-day claim is made here.
- MemoRepair, arXiv:2605.07242v1, already specifies barrier-first descendant withdrawal, repair from retained support and validated predecessor-closed republication. Complete influence provenance is a premise; its neural-skill guarantees inherit the chosen parametric operator. Our every-tensor counterfactual contract is not supplied by its store-level guarantee.
- IBM US20260119893A1 claims1–20 were inspected: added KV network layer, twin distribution/Gaussian-mixture updates, insertion/modification/deletion. The reviewed claims do not specify this exact jointly conditioned downstream repair mechanism. This limited comparison is not exhaustive patent search, patentability advice or legal clearance.
- W3C PROV-DM already defines derivation/revision/invalidation vocabulary. Source-version metadata is not an invention.
- A separately explored checksum/syndrome idea remains literature-only and deprioritized: neural ABFT and error-correcting matrix multiplication have direct prior art, including Shinkar/Singh, APPROX/RANDOM2025. No checksum repair code, detection guarantee or performance was measured here.

Two OpenReview PDF leads returned browser challenges and were not inspected. Broad patent queries also returned irrelevant biological material; that is a search limitation, not evidence of absence of direct prior art.

Primary sources:
https://arxiv.org/html/2608.30198v1
https://arxiv.org/html/2501.16944v1
https://arxiv.org/html/2607.24805v1
https://arxiv.org/html/2605.07242v1
https://patents.google.com/patent/US20260119893A1/en
https://www.w3.org/TR/prov-dm/
https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.APPROX/RANDOM.2025.29

## Unchanged gates and architecture consequence

NIC001 does not supply a trained Symlink–Pod reader, canonical alias implementation, full-memory update algorithm, generated-token feedback coverage, publication-race safety, independent J-space audit or benchmark speedup. A source is a synthetic latent injection; source absence is not factual unlearning from pretrained weights. Three direction seeds and two frozen models are not trained-reader/second-backbone qualification.

All full-system gates remain NOT EVALUATED: >=10x complete mutation-to-ready against the strongest guarantee-matched baseline; <=5% throughput loss; matched total memory; >=95% fresh/paraphrase/lifecycle reading; >=90% scoped UNKNOWN; <=0.05 nats generic divergence/exact bypass. There is no speed claim against replay: oracle fresh-world access is a deliberate advantage for a restricted-family falsification, not a deployable baseline.

The change is narrow and actionable: do not treat an exact per-source nonlinear repair chart as context-free. Its numerical validity may depend on other source revisions and on the subsequent persistent computation. Either verify the needed joint context/invariance or repair/recondition the response. An arbitrary joint nonlinear decoder may still be compact, and exact fallback remains valid. Neither provenance scaffolding nor generic interaction-aware recomputation is promoted as the invention.
