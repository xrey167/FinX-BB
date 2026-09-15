# CRR-001 — a scalar source does not imply a small fixed repair basis

**Restricted-family architecture falsification, not a major invention.** Date: 2026-09-05. Parent E91: `bde4c2dda20793d472de8860c0d2c207460f9f2c`. Main and historical experiments are unchanged.

Original preregistration: `b3358be6f1df828895b6543a8e8d96cc20a78d84`.
Initial executed commit/run: `3210ba5ca8b525f8b351f8a231ad7d83c8d3eda2` / **33978377217**.
Stronger layerwise follow-up registered AFTER seeing initial results and BEFORE its own execution: `d1aa84a58341a3f893410d21a2222b0ad4e4beb1`.
Layerwise executed commit/run: `df41fb1a640ca5b78dfb131ade7b5eadfbc6b8f9` / **33978751122**.

## Rejected premise

A proposed carrier stores a fixed output basis U per source/context and learns arbitrary nonlinear coefficients:

`h(p') = h(p) + U g(p,p')`, with `U` having r columns.

All finite changes then lie in the column span of U. A scalar source or rank-one local Jacobian does not bound that finite-response span by one or any source-dimension-only constant. Better training of g cannot escape U's output span.

This is NOT a proof against all bounded nonlinear carriers, a computational lower bound for arbitrary neural computation, or a refutation of LoRA or KV compression. E91's exact fixed-ReLU-region example remains valid; this experiment tests extending fixed output bases to smooth nonlinear computation and pretrained K/V.

## Exact countermodel and positive escape

For `h_i(s)=tanh(a_i+s)`, define `u_i=exp(2a_i)>0`, `v_j=exp(2s_j)>0`. Relative to s=0,

`D_ij = 2*u_i*(v_j-1)/((1+u_i)*(1+u_i*v_j))`

`     = [2/(1+u_i)] * [1/(1/u_i+v_j)] * [v_j-1]`.

With distinct positive u_i and distinct positive v_j !=1, this is an invertible diagonal scaling of a nonsingular Cauchy matrix. Its exact rank is d although the source is scalar and the local Jacobian has rank one. This is classical algebra, not a new theorem claim.

The executable uses rational u_i=i+1 and v_j=j+2. Nonzero determinants modulo 1,000,003, with every denominator invertible, certify full rational rank. Independent Fraction elimination verifies widths through 16. A linear-source positive control has rank one.

| Width | Exact rank | Default NumPy float64 numerical rank |
|---|---:|---:|
| 2 | 2 | 2 |
| 4 | 4 | 4 |
| 8 | 8 | 7 |
| 16 | 16 | 7 |
| 32 | 32 | 8 |
| 64 | 64 | 7 |

Numerical rank is an approximate thresholded description, not an exact certificate or a software defect. These ill-conditioned examples distinguish the two meanings.

Crucially, the same family has an exact nonlinear decoder `h_i(v)=(u_i*v-1)/(u_i*v+1)`, evaluated with one scalar and d stored constants in O(d) scalar arithmetic operations. Rational identity tests pass. Therefore full linear response rank does NOT imply large intrinsic dimension or expensive nonlinear decoding. A nonlinear output map can escape this obstruction.

## Frozen pretrained screen

CPU float64, eval mode, eager attention, frozen weights, no remote model code, one thread. Models pinned to:

- distilbert/distilgpt2: `2290a62682d06624634c1f46a6ad5be0f47f38aa`.
- EleutherAI/pythia-70m: `a39f36b100fe8a5377810d56c3f4789b9c53ac42`.

Two fixed prompts, three injection-direction seeds per model. One scalar source injected at token position 3 after block0, direction RMS half the original activation RMS. Old source=1; 33 amplitudes from -1 to 3 include no-op=1 and absent source=0. Twelve model/prompt/direction cells, 396 amplitudes, 384 nontrivial revisions per screen.

An intentionally favorable oracle sees ALL fresh states before fitting bases of rank1/2/4/8/16. SVD gives the optimal Frobenius approximation for this restricted family, not an L-infinity optimum. It is not deployable or a timing baseline. Reconstruction is checked over all actual K/V coordinates, not final logits. Post-block hidden diagnostics are separately labelled because they are not all persistent in the original backbones.

### Initial aggregate-state result

| Model | Rank16 max K/V residual, six cells | Rank16 source-zero residual | Full-rank SVD reconstruction roundoff |
|---|---:|---:|---:|
| DistilGPT2 | 3.3098e-5–2.9944e-4 | 1.4815e-5–1.5836e-4 | 2.2704e-13–3.8092e-13 |
| Pythia-70M | 1.7740e-4–7.3299e-3 | 9.2503e-5–1.3035e-3 | 1.4211e-13–2.8799e-13 |

Rank16 gives 0/384 exact nontrivial aggregate reconstructions and 0/12 exact source-zero reconstructions. The residuals are far above the full-rank arithmetic floor. Byte mismatch by itself would not prove insufficient representation rank. All ranks and singular spectra are preserved, not just rank16.

### Stronger independent per-layer K/V bases

A single shared basis across the entire state is more restrictive than separate carriers per layer. The initial result alone cannot reject the latter. CRR-001-L therefore grants each (layer,key/value) tensor its OWN oracle basis and its OWN nonlinear coefficients.

| Model | Deepest-layer tensor | Rank16 max residual, six cells | Source-zero residual, six cells |
|---|---|---:|---:|
| DistilGPT2 | key | 1.4709e-5–1.0436e-4 | 5.9498e-6–6.3403e-5 |
| DistilGPT2 | value | 2.0452e-5–1.2804e-4 | 1.0227e-5–7.7512e-5 |
| Pythia-70M | key | 1.3843e-4–5.8606e-3 | 9.6103e-5–1.2166e-3 |
| Pythia-70M | value | 4.6275e-5–1.4266e-3 | 2.8388e-5–4.8602e-4 |

All 96 tensor/case combinations in later blocks2–5 have material rank16 residual. Their 3,072 nontrivial tensor/revision reconstructions and all 96 source-zero tensor cases are inexact. These are not thousands of independent worlds: they inspect eight tensors for the same 384 distinct source revisions.

Positive controls locate the boundary: block0 K/V, computed before source injection, has ZERO response. The first downstream block has numerical response rank TWO at relative 1e-10 in every K and V case; its rank16 residual is near arithmetic roundoff, and is NOT treated as a substantive rank16 obstruction. Later blocks have numerical ranks 23–32 in DistilGPT2 and 32 in Pythia at that threshold. Numerical ranks are diagnostics, not exact analytic rank certificates for these models.

The extension repeated all backbone sweeps. Every aggregate K/V metric and spectrum matches the original run exactly as serialized JSON fields. This is measurement repeatability, not raw-tensor cross-platform bit identity or independent laboratory replication.

## Controls, execution and evidence limits

Every cell passes repeat-forward byte equality, source-zero equality with genuinely absent injection, and unchanged pre-injection K/V. The original exact suite passed **28 tests locally (0.18s) and in CI (0.08s)**, no skips. The combined suite, adding three layerwise controls, passed **31 locally (3.33s), 31 in the DistilGPT2 extension job (1.26s), and 31 in the Pythia extension job (1.24s)**, no skips. This is not the full repository suite.

All five CI artifact ZIPs were downloaded and verified against their published SHA-256 digests. Every archived source hash matches the local file, and every result JSON hash was checked. See `crr001-ci-evidence.md` for IDs and digests. The original exact numerical results, all frozen spectra/error records, controls and environments are preserved in the downloadable reproducibility bundle and original CI artifacts. Full raw K/V tensors are regenerated by the pinned scripts, not included as a saved dataset.

Local model downloads/install failed because outbound DNS was unavailable; pretrained execution is from Actions, not local. No experiment failure was hidden behind continue-on-error. The follow-up protocol was explicitly registered after initial findings; no historical result was rewritten.

Three direction seeds are not trained-reader seeds. Two frozen architectures are not a second-backbone qualification of the Symlink–Pod memory system. The source is synthetic latent injection, not learned semantic pod addressing. No autoregressive token-feedback or multi-turn learned-memory repair, UNKNOWN, generation/publication protocol, leakage test, independent J-space audit, throughput or mutation-to-ready measurement is provided. Source-zero here does not erase pretrained factual knowledge.

All application gates remain NOT EVALUATED: >=10x full mutation-to-ready; <=5% inference-throughput loss; matched complete memory budget; >=95% fresh/paraphrase/lifecycle reading; >=90% scoped UNKNOWN; <=0.05 nats generic divergence/exact bypass; trained seeds and second-backbone qualification.

## Prior-art boundary

Xiao arXiv:2608.30198 motivates memory-update repair, not novelty. FastLRNR (Results Appl Math25,100547,2025; arXiv:2410.04001) and later LRNR work (2510.25123v2) already explore compact nonlinear reduced evaluation and distinguish low-rank components from hidden-state structure. AgentKVShift (2607.21604v1) estimates per-memory residual offsets for approximate K/V repair; KVEraser (2606.17034v2) learns functional steering while leaving the suffix cache unchanged. Neither is tested or refuted by this fixed-basis screen. Their approximation/behavior objectives must not be confused with every-write equality.

LRKV (2601.11471v3) compresses redundancy across heads, a different rank axis. Fiber Fingerprints (2608.15976v1) treats finite future responses versus local linear descriptions in a training-state setting; it is related measurement prior art, not a pod-repair implementation. MemoRepair (2605.07242v1) already describes source-descendant barriers and validated republication. Cauchy algebra, SVD oracles, nonlinear compression and lineage labels receive no novelty credit here.

Published claims1–15 of US20250259042A1 were inspected: agent selection/decomposition, common token-space results, hierarchical memory, validation and fault tolerance; the reviewed claims do not specify this exact all-layer source-response contract. W3C PROV-DM already defines revision/derivation/invalidation. This limited technical screen is neither exhaustive novelty search nor legal clearance.

Primary sources:
https://arxiv.org/html/2608.30198v1
https://arxiv.org/abs/2410.04001
https://arxiv.org/html/2510.25123v2
https://arxiv.org/html/2607.21604v1
https://arxiv.org/html/2606.17034v2
https://arxiv.org/html/2601.11471v3
https://arxiv.org/html/2608.15976v1
https://arxiv.org/html/2605.07242v1
https://patents.google.com/patent/US20250259042A1/en
https://www.w3.org/TR/prov-dm/

## Architecture decision

Do not authorize a fixed low-rank repair basis solely from low source dimension or local Jacobian rank. A nonlinear coefficient learner cannot fix an inadequate fixed output span. Distinguish sparse causal support, exact affine validity regions, and genuinely nonlinear finite-response representations. A nonlinear decoder, adaptive basis, finer factorization or exact fallback changes the candidate family and remains open here; its full execution/memory/certification costs must be paid. None of these broad alternatives is claimed novel or achieved.

## Reproduce

Install NumPy2.3.5, pytest9.0.2, transformers4.57.6, and torch2.10.0 CPU as pinned in the workflows. From repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.crr001_finite_response_rank --output-dir crr001-exact
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_crr001_finite_response_rank.py so/tests/test_crr001_layerwise_extension.py -q -ra
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.crr001_layerwise_extension --model distilbert/distilgpt2 --output-dir crr001-models
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.crr001_layerwise_extension --model EleutherAI/pythia-70m --output-dir crr001-models
```

The layerwise entrypoint pins the recorded checkpoints and repeats aggregate K/V metrics too. The original `--model` entrypoint resolves the model's current revision and records it; a future invocation is not an exact-checkpoint replication unless that revision matches the recorded SHA.
