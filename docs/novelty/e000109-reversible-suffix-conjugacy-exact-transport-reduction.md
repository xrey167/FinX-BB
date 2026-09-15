# E-000109 — Reversible Suffix Conjugacy Exact Transport Reduction

Date: 2026-09-06  
Status: **PREREGISTERED**  
Major-invention claim at registration: **NO**

## Why this experiment exists

E-000095/E-000096 killed learned cross-context correction receipts, E-000097 killed generic associative exact recomposition, E-000098 killed a shared exact linear correction quotient, E-000100/E-000101 killed ordinary sparse/layerwise exact delta propagation, E-000102 through E-000106 killed several compact algebraic and precomputed-transition representations by exact reduction to generic computation, and E-000107/E-000108 sharply constrained state-only and centrally preserved exact lineage.

A remaining active-transport escape is to make the nonlinear suffix **invertible/reversible**. If the old mixed final state is

`y = F(x)`

and a canonical Pod revision acts at the last real memory-read site as an exact edit `tau_delta(x)`, then reversibility gives an exact state-only transport law

`T_delta(y) = F(tau_delta(F^-1(y)))`.

This looks superficially attractive because `T_delta` is one edit-keyed operator valid for every cached session. E-000109 asks whether reversibility itself creates a major fleet-level mutation-to-ready advantage over the registered exact suffix-recompute baseline, or whether it merely moves the work into generic inverse/replay or a generic materialized conjugate function.

## Registered candidate

Use exact finite-state nonlinear reversible suffixes built from alternating quadratic shear/coupling layers over `Z_p^2`:

- `b <- b + alpha*a^2 + beta*a + gamma (mod p)`, or
- `a <- a + alpha*b^2 + beta*b + gamma (mod p)`.

Every layer is exactly invertible by subtraction, so the whole nonlinear suffix `F` is bijective.

The Pod revision is an exact translation at the read-site state:

`tau_delta(a,b) = (a + delta mod p, b)`.

The state-only candidate receives only stale final state `y`, the suffix, and `delta`, and computes

`F(tau_delta(F^-1(y)))`.

## Strong exact baselines

### Baseline A — registered last-read suffix recomputation

Give the baseline the exact old state at the last real memory-read site, patch the Pod edit there, then execute the unchanged suffix once:

`F(tau_delta(x))`.

For a depth-`d` reversible suffix, the candidate consumes `2d` reversible-layer evaluations (one inverse traversal plus one forward traversal), while the registered recomputation baseline consumes `d` forward-layer evaluations.

This is intentionally favorable to the baseline because it is the guarantee-matched baseline already required by the programme.

### Baseline B — generic materialized conjugate evaluator

Independently construct the complete one-edit mapping by enumerating read-site states and recording

`F(x) -> F(tau_delta(x))`.

This construction does **not** call the candidate's inverse transport routine. If the candidate materializes the same conjugate map `F o tau_delta o F^-1`, the generic table must match it exactly and has the same state-domain cardinality.

### Baseline C — arbitrary-bijection conjugacy count

For a finite state domain of size `n`, fix `tau` to one full `n`-cycle and exhaustively enumerate suffix bijections `F`. Record all distinct exact transports

`F tau F^-1`.

For an `n`-cycle, the conjugacy class has `(n-1)!` elements. E-000109 exhaustively checks `n = 4,5,6,7,8` rather than assuming that reversibility restricts the edit transport to a small operator family.

## Registered cells

Nonlinear reversible suffix assay:

- deterministic seeds: `0..15`;
- moduli: `p in {11,13,17}`;
- depths: `d in {2,4,8,12}`;
- canonical edit deltas: `{+1,+2,-1}`;
- **every** state in `Z_p^2` is evaluated;
- exact integer modular arithmetic only; no floating tolerance.

This gives `16 × 3 × 4 = 192` suffix cells and evaluates all states for all three edits.

Conjugacy-class assay:

- state-domain sizes: `n in {4,5,6,7,8}`;
- enumerate all `n!` suffix bijections exactly;
- compare observed distinct `F tau F^-1` maps with `(n-1)!`.

## Registered controls

1. **Exact invertibility:** `F^-1(F(x)) == x` for every state in every reversible suffix cell.
2. **Material edit:** every nonzero registered Pod translation must change the final state because both `tau_delta` and `F` are bijections.
3. **Exact candidate equality:** state-only conjugacy transport must equal fresh recomputation for every case.
4. **Exact generic equality:** independent generic transport table must equal fresh recomputation for every case.
5. **Candidate/generic identity:** materialized candidate conjugate and independently compiled generic transport table must match entry-for-entry.
6. **ABA:** applying `delta` then `-delta` through the state-only transport must return the exact old final state.
7. **Work accounting:** count reversible layer evaluations only. Detection, indexing and table-build costs are not charged against the candidate; this makes the kill conservative.

## Preregistered kill rule

Close **reversibility / invertibility and edit-conjugacy as a standalone major exact-transport seam** if all of the following hold:

1. zero candidate/fresh state mismatches;
2. zero generic/fresh state mismatches;
3. zero candidate/generic materialized-map mismatches;
4. zero ABA failures;
5. material edit fraction `1.0`;
6. state-only inverse+forward transport requires at least `2.0×` the registered suffix-layer evaluations of last-read suffix recomputation in every registered depth;
7. exhaustive arbitrary-bijection cells match the full `(n-1)!` conjugacy-class count for all registered `n`.

Decision string on pass:

`KILL_REVERSIBILITY_AND_CONJUGACY_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE`

## Interpretation boundary

A kill does **not** prove that every specially structured reversible architecture is useless. It establishes the narrower but important point that **reversibility alone** supplies exact reconstruction, not a fleet-level mutation advantage:

- if the last-read state is retained, inverse traversal is extra work relative to one exact suffix replay;
- if it is not retained, inverse traversal is simply generic reconstruction of that state before replay;
- if the one-edit conjugate is precompiled/materialized, the same mapping is an ordinary function/table available to a generic evaluator, and reversibility alone does not guarantee that this map belongs to a compact class.

Any successor based on a special normalizer/equivariant/group structure must therefore earn novelty and systems advantage from that additional structure itself, against a generic group-action/compiler baseline using the identical representation. Reversibility, coupling layers, inverse reconstruction, or `F o tau o F^-1` notation receive zero standalone invention credit.

## External-evidence boundary to keep in view

Fresh 2026 work makes the exact baseline unusually strong. Ramesh, *Subtract or Replay? Exact Deletion from Language-Model Memory* (arXiv:2607.27539, v2 August 2026), reports that when later recurrent writes transform a record's influence, exact omission requires rebuilding/replay in the tested Kimi Delta Attention setting; its constructive addressable-memory arm instead makes exact deletion an algebraic property of the memory representation. Ramesh, *Forgetful Attention* (arXiv:2607.12204), uses a reversible incremental/decremental SVM solver for exact deletion in a specially addressable memory, but that mechanism is classical maintained optimization state rather than a generic nonlinear neural-state transport receipt.

Patent search remains crowded around mutable KV/cache machinery, including IBM WO2026087278A1 (direct knowledge injection through a KV-cache network layer), Intel US20260080217A1 (function-preserving/gauge and rank-r KV transformations), and Huawei WO2026086089A1 (KV segment recomputation). E-000109 therefore makes no broad patent claim around reversible or composable cache transformation.

## Major-break gates

Even if this seam unexpectedly survived, no major-invention promotion would occur without the programme's full real-reader, two-backbone, three-seed, exact fresh-state, deletion/leakage/UNKNOWN, stale-state attack, J-space/J-lens, matched-memory systems-advantage and steady-state-overhead gates.
