# BHC-001 — fading influence is not a finite exact repair horizon

**Architecture-contract falsification; not a major invention.** Date: 2026-09-05.
Parent QCR001: `f046318350206899ab1fd68d8b09be4f5cfc4fd4`.
Preregistered BEFORE numerical execution: `273a7e6d511775952017ded055491d997560f039`.
Scientific source/workflow anchor: `b92facf3c640d4e46b95df9d3f809414445eacc4`.
Main and historical experiment files are unchanged.

## Candidate and decision

A hybrid can replay affected persistent writes exactly until the revised COMPLETE future-relevant state equals the corresponding cached old state, then reuse an identical-forcing suffix. That is ordinary exact change propagation and belongs in the strongest baseline.

The shortcut tested here is to infer a finite exact repair horizon from real-valued contractivity/fading memory. That inference fails: a real map with global contraction factor0.5 can have two distinct fixed points in its native floating-point implementation. Repeating that native map never merges those fixed points.

This is not a theorem that all fading memories retain source data, a refutation of published approximate-task results, or an impossibility result for fast repair. Complete-state join checks, engineered synchronizing operators, different arithmetic and exact fallback remain open. Contraction, quantized dynamical effects and ordinary change propagation receive zero novelty credit.

## Exact scalar counterexample

Let u be the spacing above1, b=1+u (odd significand), and a=1+2u (even). Over the reals, `F(h)=b+(h-b)/2` is0.5-Lipschitz and F(a)=b+u/2. At these native inputs, subtraction and division by2 are exactly representable; the final addition reaches the midpoint between b and a. Round-to-nearest-even chooses a. Thus both a and b are native fixed points.

All four formats pass native and independent Fraction checks:

| Format | Distinct fixed-point gap u |
|---|---:|
| BF16 | 0.0078125 |
| FP16 | 0.0009765625 |
| FP32 | 1.1920928955078125e-7 |
| FP64 | 2.220446049250313e-16 |

The infinite-duration conclusion follows from verified distinct fixed points of a deterministic time-homogeneous map, not extrapolation of a finite loop. A changing future input or operator is outside that proof. The even-bias control b=1 instead coalesces after one step. Quantization may merge states, but a real contraction factor alone is not a universal native merge deadline.

The independent standard-library checker validates exact ties-to-even arithmetic on the registered binade, rejects missing/duplicate/tampered witnesses, and remains active under `python -O`. It does not authenticate model execution or certify a neural library's tanh implementation.

## Nonlinear cross-layer screen

Four persistent layers x sixteen coordinates; five parameter seeds x BF16/FP16/FP32/FP64. Nonnegative dyadic matrices W_l have every row summing exactly to1. All seeds share the constant-vector mode by construction: these are not independent estimates of native LLM failure frequency.

Synchronous real transition:

`state'_l = b + 0.5*tanh(W_l*(parent_l-b))`,

where layer0 uses itself and higher layers use the previous-time lower layer. The whole real map is globally0.5-Lipschitz in max norm, from row-stochasticity and tanh's unit Lipschitz bound. Autograd tests are additional diagnostics, not the global proof.

One scalar source adds u only to layer0 at the initial boundary. The NEVER world omits that addition. No later source reads/injections occur.

**All20 parameter/format cells produce distinct stable pairs.** The source difference reaches all four layers. The first repeated complete pair occurs at step4; an extra native transition verifies each is fixed. After256 steps all64 coordinates differ by exactlyu. Repeat-forward controls pass byte equality. Every even-bias nonlinear control instead coalesces completely at step1.

A diagnostic rescaling `(state-b)/u` distinguishes1 from0. It is not learned semantic recall, evidence of meaningful language degradation, a factual-error probability, a pretrained-model result or a J-space audit.

## Positive hybrid and strongest matched baseline

Both old and fresh executions receive the SAME EXOGENOUS event: reset ONLY layer0 to b at step32. The reset is planted, not a capability-preserving proposal or natural-language workload measurement. Then both continue the same native recurrence to512 steps.

Every cell yields:
- First-layer equality at32, with higher layers still different.
- Complete-state equality at35.
- Exact replay of the revised boundary/prefix through35, followed by cached suffix reuse, matches EVERY fresh write.
- Actual transition calls: **35 candidate /35 conventional exact change propagation**. Both get the same state, operators and stopping rule. This is explicitly the same conventional mechanism, not an independently novel algorithm.
- Unsafe first-layer-only stopping at32 leaves **48 incorrect coordinate writes** at subsequent times. All20 such controls fail.

A nominal512-to35 call reduction is not whole-system mutation-to-ready timing and is shared by the strongest baseline. No speed or invention gate advances. Late equality does not clean previously retained snapshots: the old/fresh prefixes before the join remain different and the correct reference repairs them.

## Reuse-authority scope

The entire future-relevant state here is the four-layer tensor. There are no additional KV caches, receipts, RNG, pending generated tokens, tool states or future source reads. Therefore full tensor equality plus identical remaining forcing/execution determines the same suffix.

In a real runtime, every future-relevant component and generation/context premise must be included or proved irrelevant. Current numerical equality is not historical provenance, authority to reuse invalid generations, or publication-race closure. A subsequent source edit may invalidate the identical-forcing premise. Small norms, a quiet diagnostic, a first-layer match or elapsed time cannot substitute for complete native-state equality.

## Primary prior art and search limits

Xiao2608.30198 motivates repair of future-memory error propagation; no claim that its results promise a finite bit-exact decay deadline is made. Gated KalmaNet2511.21016v3 (2026 revision) explicitly studies fading memory via test-time ridge regression; its real/approximate/task claims are not being reinterpreted as universal lifecycle equality.

Cui/Lau1006.3919 (2010) already studies contraction iterations with quantized messages and convergence error. Roth/Wilkinson1912.04241v1 (2019) analyzes numerical coalescence. MathWorks' official limit-cycle example studies fixed-point overflow oscillations, a related but different setting from these floating-point midpoint fixed pairs without overflow. These are established numerical phenomena, not a discovery of BHC001.

Sharir/Anandkumar2307.14988 already formalizes incremental neural reuse. Ramesh2607.27539v2 distinguishes fixed-forcing transport from native omission that changes future writes/other state. Ordinary replay-to-equality and its identical-forcing premise receive no novelty credit.

Further2026 architectural boundaries inspected: **Erase-then-Delta Attention2606.26560v1** learns an erase address independent of its write address and analyzes cross terms; targeted erasure alone is not a new differentiator. **HOLA2607.02303v1** adds a bounded exact-KV store selected from update surprise; exact retained associations/retrieval is not the same as all-descendant counterfactual equality. Neither model was executed or refuted here.

W3C PROV-DM already specifies derivation, revision and invalidation vocabulary, not native numerical deletion. Patent searches mostly returned irrelevant material. Primary retrieval failed for a secondary-index JP2026116654A lead and related US20260187477A1. Their full claims were NOT reviewed, and failed retrieval is not clearance. A latest-arXiv counterfactual-locality phrase led to VoxReason2609.03203, a different speech-planning task; no neural-repair conclusion is drawn from it. No exhaustive prior-art or legal conclusion is claimed.

Primary sources:
https://arxiv.org/html/2608.30198v1
https://arxiv.org/html/2511.21016v3
https://arxiv.org/abs/1006.3919
https://arxiv.org/html/1912.04241v1
https://www.mathworks.com/help/fixedpoint/ug/detect-limit-cycles-in-fixed-point-state-space-systems.html
https://arxiv.org/abs/2307.14988
https://arxiv.org/html/2607.27539v2
https://arxiv.org/html/2606.26560v1
https://arxiv.org/html/2607.02303v1
https://www.w3.org/TR/prov-dm/

## Execution, provenance and unchanged gates

Completed locally: **52 passed in8.04s**, no skips, not the full repo suite. Independent checker validates all four scalar witnesses under python -O. Python3.13.5, NumPy2.3.5, PyTorch2.10.0+cpu, pytest9.0.2, Linux6.18.35/glibc2.41. All three uploaded source/test/checker Git blob hashes match locally tested bytes.

The first shell call failed before code execution because its target directory did not yet exist. After creating it, the first actual numerical run passed. No scientific arm or threshold was retuned. Complete original local results are in `results/bhc001/bhc001-local.json.gz`, plus test/environment records. Witness exports in the downloadable bundle rerun the unchanged functions for packaging; they are not new trained seeds or independent replication.

Scientific CI run **33986978675** was scheduled at the pinned anchor. Observed completion and artifact verification are recorded separately, not inferred from scheduling. No CI pass is asserted in this document before that verification.

SHA-256:
- Experiment: a2b60a165adf00849146b73df7a2c37c263cb6f6e25597fe347227ec33aad09c
- Tests: e9595b9637e0c646066d1c173554c65c43d52f375432e696fc54684450ec1cc8
- Checker: e79c67245e347da44b8351ae7703883c4a76b7f2b817f2f38dcf79b27204705e
- Original local JSON: 472ba5406d285fb8c0d2c10d64430cb402f036c591b123c2200a49f4256ddd0c
- Gzip: 4ee8c6c328dcfbf4096807f2d5fd2125656ec64ebf547ec3f3774a0a22789da7

All full-system gates remain NOT EVALUATED: >=10x complete mutation-to-ready vs strongest baseline, <=5% inference-throughput loss, matched total memory, >=95% fresh/unseen-paraphrase/lifecycle reading, >=90% scoped UNKNOWN, <=0.05nats generic divergence/exact bypass, independent J-space, generation-publication safety, trained seeds and second-backbone qualification. Five operator seeds across four formats are zero trained-reader seeds. No pretrained model, paid GPU or model API was used.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m so.experiments.bhc001_exact_coalescence --output bhc001-results.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest so/tests/test_bhc001_exact_coalescence.py -q -ra
python -O tools/bhc001_check_fixed_points.py bhc001-results.json
```
