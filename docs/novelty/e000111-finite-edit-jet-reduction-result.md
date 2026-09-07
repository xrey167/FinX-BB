# E-000111 Result — Finite Edit-Jet and Co-computed Higher-Order Sensitivity Reduction

Status: **DECISIVE FALSIFICATION / STANDALONE NOVELTY SEAM CLOSED**

Decision:

`KILL_FINITE_EDIT_JETS_AND_CO_COMPUTED_HIGHER_ORDER_SENSITIVITY_AS_STANDALONE_EXACT_TRANSPORT_NOVELTY`

Major invention promoted: **no**.

## Registered evidence identity

- preregistration + implementation source commit: `432f0d5f4fd25d859f833153a1c3e79a00b76b17`
- GitHub Actions run: `34023068266`
- run conclusion: `success`
- focused regressions: `3 passed`
- artifact id: `9986149245`
- artifact digest: `sha256:14ce72f390d57d5d753cac60859ffc27c87a200e761b6e991636e25a9a9f8d08`
- preregistration source SHA-256: `027b05e5d6761e02f3c032cf8fb5d079809f65c0c78e99c731119fb5b6c4c128`
- experiment source SHA-256: `e19f04d6c36d1130cbce55b905ea0de251f09d5523570e9ebb194f8407ac8548`
- focused-test source SHA-256: `a67ce3e4ef35e94ce162fa8a73be8a8bb89932b2a7dd702225e31ce54027a7eb`
- result JSON SHA-256: `6c9cc7d3c808d073939eaf7a40671cb9d8bd2cec65f945f3b8f5e95b4a5fb93a`

All scientific comparisons used exact `fractions.Fraction` arithmetic. There is no floating-point tolerance in this structural reduction.

## Registered exact result

The candidate tested the strongest natural derivative-receipt escape left after E-000110: instead of constructing a mutation representation only after the edit, co-compute a complete finite higher-order sensitivity object during the original nonlinear forward computation and later evaluate it at the Pod delta.

For the registered bounded-degree nonlinear family, this **works exactly** when the jet order reaches the true Pod degree. But an independently implemented generic Taylor-mode automatic-differentiation baseline produces the same coefficients, same exact fresh state, same coefficient storage, and same counted coefficient-multiplication work.

| Measure | Result |
|---|---:|
| deterministic seeds | **16** |
| nonlinear depths | **1..6** |
| registered seed/depth cells | **96** |
| contexts per cell | **12** |
| edit deltas per context | **3** |
| exact transport cases | **3,456** |
| candidate -> fresh mismatches | **0** |
| generic Taylor-mode -> fresh mismatches | **0** |
| candidate/generic jet mismatches | **0** |
| candidate/generic work mismatches | **0** |
| candidate coefficient slots | **25,344** |
| generic coefficient slots | **25,344** |
| candidate/generic slot ratio | **1.0** |
| candidate counted multiplies | **653,647** |
| generic counted multiplies | **653,647** |
| candidate/generic multiply ratio | **1.0** |
| degree mismatches | **0** |
| maximum exact jet order required | **64** |
| half-order truncated transport failures | **3,456 / 3,456** |

The exact required Pod degree doubled with every registered quadratic nonlinear layer: `2, 4, 8, 16, 32, 64`. This is not asserted as a universal transformer lower bound. It is a constructive exact witness that preserving all finite higher-order sensitivity can cease to be compact as interaction degree grows.

## Independent non-polynomial exactness witness

A second arm tested the exact rational map

`f_a(p) = 1 / (1 - a p)`

around `p0=0`. Its Taylor series has nonzero coefficients at every order for nonzero `a`. Across finite jet orders `1,2,4,8,16,32`, three registered `a` values and two nonzero edit deltas, all **36 / 36** finite-jet transports differed exactly from the fresh rational state. There were **0 exact matches**.

For example, with `a=1/2`, `delta=1/5`, and first-order jet, fresh state is exactly `10/9` while the finite jet yields `11/10`.

This is a scoped counterexample to the claim that storing an arbitrarily sophisticated but fixed finite derivative receipt supplies a **general exact** active-transport guarantee through admissible non-polynomial computation. It is not a claim that every real transformer necessarily requires an infinite jet for every edit.

## Why this is a decisive kill

E-000111 gives the derivative-receipt route both ways it could plausibly win and removes standalone novelty in both:

1. **When a finite exact jet exists**, ordinary Taylor-mode AD on the identical arithmetic program constructs the identical sufficient representation with identical registered work and storage. The benefit is generic automatic differentiation, not Symlink/Pod/J-space semantics.
2. **When the mutable dependence is not bounded-degree polynomial**, a fixed finite Taylor receipt does not in general recover the fresh post-edit state exactly. Higher-order approximation therefore cannot be promoted on KL/top-1 quality under the programme's lifecycle standard.

Standalone claims now receiving zero major-invention credit include:

- co-computing Pod Jacobians/Hessians/higher derivatives as an edit receipt;
- carrying a finite neural jet or truncated Taylor polynomial beside cached session state;
- raising JVP/Jacobian repair to higher order and calling finite-order equality a lifecycle mechanism;
- training only to make a finite Taylor receipt accurate without proving exact fresh-state closure;
- using a bounded-degree polynomial architecture where the same exact jet is obtained by generic Taylor-mode AD with the same resource profile.

A future system may still use derivatives internally, but it must earn an independent neural-specific systems advantage over a guarantee-matched generic higher-order AD engine.

## Fresh literature / patent boundary checked on 2026-09-06

The targeted search again did not surface the narrow FinX target: one knowledge edit producing an exact reusable correction receipt across arbitrary already-cached neural sessions with a fleet-level advantage over guarantee-matched suffix/KV reconstruction. This is a search result, not a freedom-to-operate opinion.

The derivative machinery itself is clearly generic prior art/baseline. JAX's `jax.experimental.jet` propagates truncated Taylor polynomials for higher-order automatic differentiation. `TaylorDiff.jl` implements composable Taylor-mode AD using Taylor-polynomial operator overloading, and `torch-jet` provides Taylor-mode jets in PyTorch. Dangel, Siebert, Zeinhofer and Walther's NeurIPS 2025 `Collapsing Taylor Mode Automatic Differentiation` explicitly optimizes Taylor-mode propagation by compiler-level graph rewriting. Patent publication `US20250013443A1` also describes Taylor-mode versus nested forward AD in a compiler/simulation context and notes the Taylor representation's storage properties.

The exact-memory-mutation boundary remains consistent with `Subtract or Replay? Exact Deletion from Language-Model Memory` (arXiv:2607.27539): addressable influence can be decremented algebraically, while later entangled recurrent writes require checkpointed rewind/replay in the tested native Kimi setting to reproduce omitted state exactly. Generic KV/cache patent territory also remains crowded, including IBM `WO2026087278A1` for direct LLM knowledge injection using a KV-cache network layer and Intel `US20260080217A1` for KV-cache compression using gauge/rank transformations.

## Surviving frontier

E-000111 closes the clean finite **derivative/jet receipt** escape from E-000110.

The remaining invention target is stricter than carrying more local sensitivity information. We need a bounded exact mutation-enabling state whose boundedness is a native invariant of useful neural computation, not a finite truncation of an expansion whose order grows with interaction complexity, and not a representation that a generic AD / algebra / provenance / incremental engine can construct and exploit at the same cost.

A natural next kill screen is the finite-dimensional **observable-lift / Koopman-Carleman closure** escape: lift nonlinear Pod-conditioned computation into a bounded feature state on which mutation acts exactly and cheaply, then compare against a generic dynamical-systems/algebra engine given the identical lift. If exact closure requires an expanding/infinite feature basis or the generic engine inherits the same benefit, that route closes as well. If a bounded native lift survives those baselines and preserves useful neural computation, it becomes worth escalating.

No E-000111 claim advances to the real LINK->Pod / DistilGPT-2 + Pythia / three-seed / leakage / UNKNOWN / generic-divergence / stale-state attack / full lifecycle / J-space and J-lens / <=5% steady-state-overhead promotion gates.
