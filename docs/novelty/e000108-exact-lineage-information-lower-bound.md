# E-000108 — Exact Lifecycle Lineage Information Lower Bound

Date: 2026-09-06
Status: **preregistered scoped information/kill screen; not a novelty claim**

## Trigger

E-000107 showed that exact causal lineage cannot in general be recovered *after the fact* from an already-collapsed neural state. The immediate surviving escape is to preserve an auxiliary neural/latent lineage capsule during computation and decode exact lifecycle consequences from that capsule later.

Before spending backbone budget on a learned capsule, test whether an exact **central** capsule can be materially smaller than the strongest generic sufficient-state ledger when arbitrary Pod contributions remain admissible.

This screen does not reopen E-000086R: ordinary dependency/provenance metadata is the baseline being challenged, not the invention claim.

## Registered family

For `n` Pods, each Pod carries a causal contribution

`v_i in Z_q`.

The task-visible mixed state keeps only a bijective encoding of the aggregate

`H(v) = psi(sum_i v_i mod q)`,

where `psi` is a fixed seeded permutation of `Z_q`. The permutation prevents the assay from relying on a special numeric coding of the aggregate while preserving exact information content.

An exact lifecycle mechanism receives:

- current visible state `H(v)`;
- target Pod identity `i`;
- an auxiliary central capsule `C(v)` created during the original computation;
- the known model / `psi`.

It must return the exact fresh state for arbitrary target DELETE, and the same retained sufficient state is checked for exact UPDATE and ABA.

## Information reduction

Fix one visible aggregate `t = sum_i v_i mod q`.

There are exactly

`q^(n-1)`

different Pod-value vectors with that same aggregate.

For target `i`,

`DELETE_i(v) = psi(t - v_i mod q)`.

Therefore the complete arbitrary-target deletion profile

`(DELETE_1(v), ..., DELETE_n(v))`

uniquely determines every `v_i`, because `psi` is bijective and `t` is already known from the visible state.

Hence, inside a single visible-state bucket, all `q^(n-1)` histories require different exact lifecycle answers for at least one target. By pigeonhole, any auxiliary central capsule that supports exact arbitrary-target deletion must have at least

`|C| >= q^(n-1)`

distinguishable states, i.e.

`log2 |C| >= (n-1) log2 q`

bits of information.

This is an elementary finite-state counting reduction, not a new theorem.

## Strong generic baseline

The generic baseline stores the first `n-1` Pod values. The final value is reconstructed exactly from the current aggregate:

`v_n = t - sum_{i<n} v_i mod q`.

Its auxiliary state space is **exactly** `q^(n-1)`, meeting the lower bound with equality. It then computes exact:

- DELETE: `t' = t - v_i`;
- UPDATE: `t' = t - v_i + v_i'`;
- ABA/RESTORE: reverse the update and recover the exact original aggregate.

Thus a central neural/latent capsule whose sole advantage is compressed exact source lineage cannot asymptotically beat a generic optimally packed ledger on this registered family.

## Registered cells

Exhaustively enumerate:

- `q=2, n=12, seed=1080` — binary membership/dependency case;
- `q=3, n=8, seed=1081`;
- `q=5, n=6, seed=1082`;
- `q=7, n=5, seed=1083`.

For every history and every target:

1. calculate the exact task-visible aggregate state;
2. calculate the entire exact DELETE profile;
3. verify that, conditioned on visible state, every history has a unique DELETE profile;
4. verify the generic `n-1` value ledger reconstructs the complete history exactly;
5. verify generic DELETE equals fresh DELETE;
6. apply one deterministic nontrivial UPDATE and verify generic UPDATE equals fresh UPDATE;
7. reverse it and verify exact ABA restoration.

All arithmetic is exact modular integer arithmetic.

## Kill rule

Kill **compact exact central lineage capsules as a general major-advantage route** for the registered arbitrary-contribution family if:

- every visible-state bucket contains exactly `q^(n-1)` histories;
- every history in every bucket has a distinct arbitrary-target DELETE profile;
- therefore the exact auxiliary state lower bound is `q^(n-1)` in every registered cell;
- the generic `n-1` Pod-value ledger has exactly the same state cardinality;
- generic DELETE, UPDATE and ABA have zero mismatches.

## Consequence if killed

Do not spend major-invention budget on a learned neural code, latent biomarker, source-set embedding, lineage hash, or centralized causal capsule whose only promised advantage is to retain the exact information needed to answer arbitrary Pod lifecycle requests more compactly than a generic sufficient-state ledger.

A successor lineage mechanism must obtain its advantage from additional structure that changes the information/computation frontier, not from renaming exact provenance.

## What is not killed

This screen does **not** rule out:

- restricted/compressible Pod-value families with exploitable structure;
- distributed or ticketed memory where information is moved back to the Pod/user rather than centrally retained;
- an architecture whose normal task-useful state itself preserves lineage and thereby amortizes storage for a separate purpose;
- active intervention or replay;
- approximate attribution;
- a neural causal certificate that supplies a capability generic provenance does not provide at matched information/memory.

Any distributed/ticketed successor must be compared against a generic ticketed-memory baseline carrying the same sufficient information.

## Programme gates unchanged

This is a scoped kill screen, not a major candidate. No major invention may be promoted without the real LINK->Pod reader >=0.95 on every held-out template, >=3 genuine seeds, >=2 backbone families, <=2% old/deleted leakage, >=90% scoped UNKNOWN, exact bypass or <=0.05 nats generic divergence, complete stale-state/lifecycle/race/reconstruction battery, independent J-space/J-lens audit, matched memory, <=5% steady-state inference overhead, and material fleet-level mutation-to-ready advantage over the strongest guarantee-matched baseline.
