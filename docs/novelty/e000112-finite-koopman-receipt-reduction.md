# E-000112 — Finite Koopman Lift Reusable Revision Receipt Reduction

## Status

Preregistered structural reduction. This experiment does **not** reopen E-000095 or any prior killed seam. It attacks the post-E-000111 escape: deliberately preserve a bounded exact observable lift through nonlinear computation so one canonical Pod revision can be compiled once and transported across many cached sessions.

## Candidate

Use nonlinear original-coordinate dynamics

\[
x' = \lambda x,\qquad c'=c,
\]

\[
y'=\mu y+\nu xc+\rho x^2+b+\eta c,
\]

where `x` is initialized by the mutable Pod and `c` is session context. The six observables

\[
\phi(x,c,y)=(1,x,c,xc,x^2,y)
\]

form an exact finite-dimensional invariant lift. A Pod edit `p_old -> p_new` has lifted initial difference

\[
\Delta\phi_0=A_0+cB_0.
\]

After depth `d`, one shared receipt is

\[
(A_d,B_d)=(K^dA_0,K^dB_0),
\]

and each stale session is repaired exactly by

\[
\phi'_s=\phi_s+A_d+c_sB_d.
\]

This is deliberately stronger than a context-independent translation: `B_d[y] != 0`, so the hidden correction changes with held-out session context while the edit-level receipt remains shared.

## Strong generic baseline

Give an independent generic sparse-linear / Koopman engine the **identical six observables and exact operator K**. It has no Pod, lifecycle, neural-memory, Symlink, or J-space semantics. It independently propagates the same two basis vectors through ordinary sparse matrix-vector multiplication, then applies `A_d + c_s B_d` to each cached state.

If the generic engine matches the candidate's state, memory and normalized mutation work, the benefit belongs to the disclosed exact finite lift rather than to a FinX-specific transport invention.

## Registered domain

- exact `fractions.Fraction` arithmetic;
- 16 deterministic seeds;
- depths `1, 2, 4, 8, 16, 32`;
- 64 distinct cached contexts per seed/depth cell;
- sequential `UPDATE -> DELETE -> RESTORE -> ABA`;
- fresh original-coordinate recomputation is the exact reference;
- full six-dimensional lifted state is audited, not only task output;
- normalized exact work compares one shared two-vector receipt plus per-session application against full original-coordinate replay from the last real Pod-read site.

## Kill rule

Kill **finite exact Koopman/Carleman observable lifting as a standalone lifecycle-transport novelty seam** iff all registered conditions hold:

1. 96 seed/depth cells and 24,576 lifecycle cases execute;
2. candidate -> fresh mismatch count is zero;
3. generic -> fresh mismatch count is zero;
4. candidate and generic receipts/states/work match exactly;
5. every lifecycle case materially changes hidden state;
6. every edit cell has distinct context-dependent hidden corrections across all 64 sessions;
7. candidate and generic use equal per-session lifted-state slots;
8. candidate achieves >8x aggregate normalized replay advantage, proving the mechanism is useful enough that novelty—not mere ineffectiveness—is what kills it.

## Interpretation boundary

A pass does **not** prove that lifecycle-native neural architectures are impossible, nor that every nonlinear system has or lacks a finite Koopman embedding. It establishes a narrower programme rule: once an exact finite observable lift and its operator are disclosed, a generic Koopman/linear-state engine inherits the same reusable exact mutation receipt. Therefore the lift/update law alone receives zero major-invention credit. Any successor must obtain a neural-specific advantage in producing/maintaining the sufficient state under the real <=5% overhead, matched-memory, two-backbone, lifecycle-security and J-space/J-lens gates.
