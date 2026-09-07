# E-000110 — Lifecycle Equivariance and Edit-Group Normalizer Reduction

Status: **PREREGISTERED STRUCTURAL KILL SCREEN**

## Why this experiment exists

E-000109 showed that a reversible nonlinear suffix can transport an edit exactly via
`F o tau o F^-1`, but state-only inverse+forward costs at least the ordinary suffix
recompute in the registered construction, while a precompiled conjugate is just a
generic exact function.

The strongest remaining structural escape is to **design the nonlinear computation
so the edit family is normalized layer by layer**. Then a canonical Pod revision can
stay inside a compact exact action family while crossing nonlinear layers, avoiding
both inverse reconstruction and per-session suffix replay.

This experiment gives that candidate its best clean setting and immediately hands
the identical action/normalizer representation to an independent generic
group-action/compiler baseline.

No result from E-000086R, E-000088, E-000089, E-000090/090B, E-000092,
E-000093, or E-000094 is reopened as an invention claim.

## Registered construction

All arithmetic is exact over `Z_p`, with odd primes `p`.

A session state is `(a,b)`.  The compact edit-action family is

`g_(u,v,w)(a,b) = (a+u, b+v*a+w) mod p`.

A canonical Pod revision at the last real read site starts as a translation

`tau_delta = g_(delta,0,0)`.

Each downstream nonlinear layer is the exact bijection

`F(a,b) = (r*a+t, s*b + alpha*a^2 + beta*a + gamma) mod p`

with nonzero `r`, `s`, and nonzero `alpha`.

This layer normalizes the edit-action family.  If `g=(u,v,w)`, then

`F o g o F^-1 = g_(u',v',w')`

where

- `u' = r*u`
- `v' = (s*v + 2*alpha*u)/r`
- `w' = s*w + alpha*u^2 + beta*u - v'*t`

over `Z_p`.

The important point is that the initially context-independent translation becomes
a genuinely **state-dependent** final-state action (`v != 0`) after a nonlinear
layer, while remaining bounded to three field elements.

## Candidate

Compile the canonical Pod edit through the suffix by propagating `(u,v,w)` through
the registered exact normalizer law once per mutation, then apply the final compact
action directly to every stale cached final state.

Candidate work per edit, in normalized structural units:

`depth group-transport steps + N session action applications`.

No inverse suffix pass and no fresh per-session suffix replay is allowed.

## Strong generic baseline

The generic baseline receives the **identical compact action representation and the
identical layer normalizer metadata**.  It has no Pod, memory, lifecycle, Symlink,
J-space, or neural semantics; it simply propagates a group element through induced
automorphisms and applies the resulting group action to each cached state.

Its normalized work is therefore registered as:

`depth generic group-transport steps + N generic action applications`.

An independent black-box audit additionally reconstructs each one-layer conjugated
action from exact probes of `F o g o F^-1` and verifies held-out probes.  This audit
must agree with both analytic implementations but is not the work-matched baseline.

The ordinary exact baseline remains last-read-site patch plus fresh suffix
recomputation, costing `N * depth` downstream layer evaluations per edit in this
finite-state assay.

## Registered grid

- deterministic seeds: `0..15`
- primes: `11, 13, 17`
- suffix depths: `2, 4, 8, 16`
- canonical edit deltas: `+1, +2, -1`
- sessions: every `(a,b)` in `Z_p^2`
- total network cells: `16 * 3 * 4 = 192`
- total edit cells: `576`

## Required controls

For every registered cell:

1. every nonlinear layer is bijective;
2. candidate final state equals fresh recomputation exactly;
3. work-matched generic group engine equals fresh recomputation exactly;
4. candidate and generic transported actions are exactly identical;
5. the independent probe compiler recovers the same transported action;
6. UPDATE is material on every session;
7. a state-dependent action component must appear after nonlinear transport;
8. inverse edit must restore the exact old final state (ABA);
9. candidate and work-matched generic baseline must use identical normalized
   transport/application work;
10. the candidate must still show a real fleet advantage over per-session suffix
    replay in every cell, so a kill cannot be blamed on a useless candidate.

## Preregistered decision

Return

`KILL_LIFECYCLE_EQUIVARIANCE_AND_GROUP_NORMALIZER_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE`

iff all registered controls pass, candidate and generic exact-state mismatches are
zero, candidate/generic/probe action mismatches are zero, material and
state-dependent-action fractions are `1.0`, ABA failures are zero,
candidate/generic normalized work ratio is exactly `1.0`, and the worst registered
cell still has replay/candidate normalized-work ratio `> 1.9`.

### Interpretation of a kill

A kill means that **equivariance to lifecycle edits, a compact edit group, or
layerwise normalizer closure cannot by themselves carry FinX-BB major-invention
credit**.  Once the exact action and induced automorphisms are the mechanism, an
ordinary group-action/equivariant-program engine given the same representation
inherits the same exact fleet-update advantage.

This does **not** prove that every lifecycle-native architecture reduces to this
group.  A successor can remain live only if it obtains a material exact systems
advantage from additional neural-specific structure that a guarantee-matched
generic group-action/compiler baseline cannot reproduce, and it must later satisfy
the full real LINK->Pod reader, two-backbone, three-seed, leakage/UNKNOWN, generic
divergence, stale-state/lifecycle attack, J-space/J-lens, matched-memory, and
steady-state-overhead gates before major-invention promotion.
