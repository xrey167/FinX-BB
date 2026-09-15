# E-000111 — Finite Edit-Jet and Co-computed Higher-Order Sensitivity Reduction

Status: **PREREGISTERED STRUCTURAL KILL SCREEN**

Major-invention promotion on this experiment alone: **not allowed**. This experiment can only close or preserve a structural seam. Any survivor must later pass the real LINK->Pod, two-backbone, three-seed, exact-state, lifecycle, attack, leakage, UNKNOWN, J-space/J-lens and <=5% steady-state-overhead gates already registered for the programme.

## Question

After E-000110, the remaining escape is not merely to discover a clever update law for an already-given sufficient representation. A candidate might instead co-compute, during ordinary neural inference, a compact higher-order sensitivity object with respect to the mutable Pod and later use that object as a one-edit exact transport receipt.

E-000111 tests the clean finite-order version of that idea: a **Pod edit jet** (value plus higher-order Taylor coefficients) carried through nonlinear computation and evaluated after a Pod revision.

The experiment asks two independent questions.

1. When a finite jet is sufficient for exact mutation transport, is the advantage neural/lifecycle-specific, or is it exactly generic Taylor-mode automatic differentiation on the same arithmetic program?
2. Does a fixed finite jet provide a general exact transport guarantee once the nonlinear computation is not a bounded-degree polynomial in the Pod?

This is deliberately not a Jacobian/JVP invention claim. Jacobian/JVP linearization is already a programme baseline. E-000111 gives the derivative-receipt route its strongest natural escalation to arbitrary finite order before deciding whether it deserves any remaining invention budget.

## Candidate representation

For session context `x`, mutable scalar Pod `p`, and old Pod value `p0`, the candidate carries coefficients

`J_s = (c_0, c_1, ..., c_K)`

such that, whenever the representation is exact,

`h_s(p0 + d) = sum_k c_k d^k`.

The registered polynomial family starts affine in `x,p` and applies context-dependent quadratic nonlinear layers. The exact Pod degree therefore doubles with depth when the leading coefficient is nonzero. All arithmetic is `fractions.Fraction`; there is no floating-point tolerance.

The candidate path independently propagates the edit jet through each nonlinear layer. The work-matched generic baseline independently executes truncated-power-series/Taylor-mode AD on the identical arithmetic program, with no Pod, Symlink, memory or lifecycle semantics.

Fresh post-edit state is always obtained by directly evaluating the original network at the revised Pod. Exact transport is judged against that fresh state.

## Registered polynomial grid

- deterministic seeds: `0..15`
- nonlinear depths: `1..6`
- contexts per seed/depth cell: `12`
- Pod old value: `0`
- edit deltas: `1/5`, `-1/7`, `2/11`
- exact jet order per depth: `2^depth`
- registered cells: `16 * 6 = 96`
- exact transport cases: `96 * 12 * 3 = 3,456`

For every session the experiment also evaluates a deliberately insufficient jet truncated to half the true polynomial degree.

## Independent non-polynomial exactness witness

The second arm uses the exact rational map

`f_a(p) = 1 / (1 - a p)`

around `p0=0`, whose Taylor series has nonzero coefficients at every order for `a != 0`. Registered finite orders are `1,2,4,8,16,32`, with `a in {1/2,2/3,3/4}` and deltas `{1/5,-1/7}`. This gives 36 exact rational comparisons, away from poles.

This arm is a scoped counterexample to a **general fixed-finite-jet exactness claim**. It is not claimed as a universal transformer lower bound.

## Preregistered kill rule

Close finite edit jets / co-computed higher-order sensitivity as a **standalone FinX exact-transport invention seam** iff all of the following hold:

- 96 registered polynomial cells and 3,456 exact edit cases execute;
- full candidate jets match fresh recomputation in every polynomial case;
- generic Taylor-mode AD jets match fresh recomputation in every polynomial case;
- candidate and generic jets are coefficient-for-coefficient identical in every registered session;
- candidate and generic counted coefficient-multiplication work is identical;
- candidate and generic final jet storage is identical;
- the observed exact Pod degree is `2^depth` in every registered polynomial cell, reaching order 64 at depth 6;
- the half-order truncated jet fails exact fresh-state equality in every registered edit case;
- all 36 finite-order rational non-polynomial witnesses fail exact fresh-state equality.

Registered decision string:

`KILL_FINITE_EDIT_JETS_AND_CO_COMPUTED_HIGHER_ORDER_SENSITIVITY_AS_STANDALONE_EXACT_TRANSPORT_NOVELTY`

## Interpretation if killed

The scoped conclusion will be:

- carrying tangent/Jacobian/Hessian/higher-order Taylor coefficients during neural inference is not itself a FinX invention when a generic Taylor-mode AD engine supplied the same computation constructs the identical sufficient representation with the same measured work and storage;
- in bounded-degree polynomial computation, exact finite edit transport is possible but the required jet order can grow with nonlinear interaction degree;
- a fixed finite jet cannot furnish a general exact mutation guarantee for admissible non-polynomial computation merely because it stores more derivatives;
- approximate Taylor/JVP/Hessian repair may remain useful for task accuracy, but receives zero lifecycle/deletion guarantee credit under the programme's exact-state standard.

This does not rule out a future architecture in which a bounded mutation object is part of the model's **native task state** for reasons stronger than generic AD, nor a different exact algebra whose sufficient representation cannot be constructed or exploited equally by a generic engine. Those remain the frontier.

## Prior-art boundary to check with the result

The result must be interpreted against current Taylor-mode / jet automatic differentiation and exact-memory-mutation literature. In particular, generic Taylor-mode AD and compiler-level higher-order derivative propagation are baselines, not novelty. The exact deletion baseline remains fresh recomputation / checkpoint replay when later nonlinear state is entangled.
