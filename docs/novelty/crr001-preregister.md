# CRR-001 — finite nonlinear response rank

Registered before numerical execution, 2026-09-05. Parent E91: bde4c2dda20793d472de8860c0d2c207460f9f2c. Unique name avoids parallel E92/E93 experiments. No invention claim. Status at registration: UNRUN.

## Candidate under test

A source-local finite-response carrier stores a fixed r-dimensional output basis per source/context and allows an arbitrary nonlinear coefficient function of a source revision: h(p')=h(p)+U g(p,p'). Can source dimension or local Jacobian rank bound the r needed for exact downstream persistent writes? This is a restricted candidate family, NOT all bounded nonlinear carriers. A nonlinear decoder can escape a linear output-basis bound.

## Exact countermodel

For scalar s and h_i(s)=tanh(a_i+s), use u_i=exp(2a_i)>0 and v=exp(2s)>0. Relative to s=0 the response is D_ij=2*u_i*(v_j-1)/((1+u_i)*(1+u_i*v_j)). Distinct positive u_i and v_j !=1 yield a diagonally scaled Cauchy matrix. Test widths 2,4,8,16,32,64 with u_i=i+1, v_j=j+2. Certify full rank over the rationals via nonzero modular determinants with all denominators invertible; independently verify small widths using Python Fraction elimination. Floating SVD ranks are descriptive and cannot override this certificate. Include a rank-one linear positive control and an exact rational nonlinear evaluator, so a fixed-basis obstruction is not misreported as an impossibility of compact nonlinear computation. The algebra is classical, not a new theorem claim.

## Frozen pretrained screens

Attempt distilbert/distilgpt2 and EleutherAI/pythia-70m on CPU float64, eval mode, eager attention, one thread, no training or remote code. Resolve immutable HF revision SHAs at the start of execution and record them and dependency versions. Local installation/download failed due network DNS before registration; GitHub Actions will execute the pretrained arm. Failed downloads/imports must be recorded, not replaced by random networks without disclosure.

Three injection-direction seeds (0,1,2), two fixed prompts:
- The engineer checked the updated reference before making a decision.
- A researcher compared the measurements with the previous laboratory notes.

Inject one scalar source along a seeded direction into token position 3 after block 0. Direction RMS is half the unmodified activation RMS at that position. Old source s=1. Sweep 33 amplitudes uniformly from -1 to 3, including no-op s=1 and source-absent s=0. Store and check every layer's actual K/V, and separately post-block hidden diagnostics. Hidden snapshots are not all persistent in the unmodified models. This is fixed-prompt source injection, not learned symlink reading, autoregressive token-feedback coverage, full memory-update trajectories, or semantic erasure.

For each state family, fit an intentionally favorable oracle fixed basis to ALL full fresh response snapshots. Evaluate ranks 1,2,4,8,16 by residual norms, per-revision bit identity after reconstruction, and deletion-to-source-absent equality. An SVD oracle has optimal Frobenius error for this restricted family, not an L-infinity optimality claim. It uses fresh test states and is not a deployable method or a speed baseline. Record spectra, numerical ranks at relative thresholds 1e-6/1e-10/1e-12, dimensions, repeat-forward and first-layer-K/V nondependence controls. No numerical rank is treated as a proof of exact analytic rank.

## Decisions fixed in advance

A nonzero exact rank witness with width>r rejects the universal claim that a scalar source needs only r fixed output directions, even with nonlinear coefficients. Frozen-model measurements establish only membership of their tested interventions, not universal rank or trained task capability. Repeat-forward mismatch invalidates numerical exactness interpretation and must be reported. All inference/utility, memory-budget, >=95% fresh/paraphrase/lifecycle, >=90% UNKNOWN and generic-divergence gates remain NOT EVALUATED. Three direction seeds are not trained-reader seeds; two frozen architectures are not a second-backbone qualification of the proposed memory system.

Prior art search before registration includes Neural Caching (2025), FastLRNR (2025), AgentKVShift (2026), finite-dimensional Koopman/response work and Fiber Fingerprints (2608.15976). Rank growth under nonlinearity and low-rank approximation are not novelty. Xiao 2608.30198 remains motivation only.
