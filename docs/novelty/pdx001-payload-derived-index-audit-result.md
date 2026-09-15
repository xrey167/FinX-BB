# PDX-001 result — the E-000028 channel is a property of the pattern, not of `so/model.py`

Date: 2026-09-06
Decision: **PAYLOAD_DERIVED_INDEX_CHANNEL_CONFIRMED_ACROSS_STORE_SHAPES**
Major-invention claim: **NO** — this is an audit result, which is what the novelty statement said it
should be

## What was run

`so/experiments/pdx001_payload_derived_index_audit.py`, stdlib only, exact integer arithmetic,
exhaustive over a payload domain of 256. 24 rows, 12 targets, five deletion policies plus a live
control. Focused regressions: `so/tests/test_pdx001_payload_derived_index_audit.py`, 10 tests.
Reproduce with `make pdxaudit`.

**Validity floor met:** the same attack recovers a live payload at top-1 **1.0000**. The at-chance
readings below are therefore readings, not the artefact of an attack that never worked.

**Instrument self-consistent:** the certificate and the attack agree on all five policies.

## The table

Chance posterior is 1/256 = 0.003906.

| policy | top-1 | candidates left | posterior on true payload | search space cut | certified | channel |
|---|---|---|---|---|---|---|
| `value_gated_shred` | **1.0000** | 1.00 | 1.000000 | 256x | no | `key_rev` |
| `hnsw_tombstone` | 0.0000 | 2.92 | **0.391667** | **88x** | no | `adjacency` |
| `codebook_key` | **1.0000** | 1.00 | 1.000000 | 256x | no | `codebook_key` |
| `revoke_unindex` | 0.0000 | 256.00 | 0.003906 | 1x | **yes** | — |
| `gate_all_derived` | 0.0000 | 256.00 | 0.003906 | 1x | **yes** | — |

## What it says

**The defect is not about `k_rev`.** Restated over an abstract store, E-000028's channel reproduces in
three unrelated shapes drawn from three different literatures. A gated value beside an ungated
payload-derived reverse key, and a cleared value beside a codebook key derived from it, both name the
payload outright at top-1 1.0000. The generalisable rule the novelty statement extracted —
*enumerate every quantity derived from a payload and gate all of them, or take the row out of the
addressable set* — holds on all three, and the two controls that follow it reach chance and are
certified independent over the whole domain.

**The tombstone row is the one worth the ink, and it corrects E-000028's own headline metric.** A
soft-deleted vector-index node that keeps the adjacency list built from its own embedding **never
names the payload**: top-1 is 0.0000, flat. Under E-000028's reported metric that is a clean
deletion. It is not. The retained edge list narrows the domain from 256 candidates to **2.92**, a
posterior of 0.3917 against a chance of 0.0039 — a **88x** cut in the adversary's search space, or
100x on the posterior. The payload is not recovered; it is very nearly recovered, and a top-1 headline
cannot see the difference between that and deletion.

This is the arrangement `docs/so-what-was-found-2026-09-04.md` §8 already flagged in passing —
"any store that keeps a dangling reference after deleting its referent has it, which includes every
soft-deleting vector index that keeps its edges" — now measured rather than asserted, and measured
with the metric that makes it visible.

**The certificate half works, and it is the constructive result.** `revoke_unindex` and
`gate_all_derived` are not "no attack recovered it". Every value the payload could hold leaves the
observables bit-identical, so no attack of this shape exists. The whole 256-value domain remains
admissible, which is the same statement the attack makes from the other side, and the run asserts the
two agree.

## What this does and does not discharge

§5 of `docs/so-novelty-2026-09-04.md` asked for the attack to be run against a *published* system
with its own reported deletion metric. **That has not been done here** and PDX-001 makes no claim
about GRACE, Larimar, or any particular index. There are no checkpoints, no torch and no network in
this environment; the five policies are reconstructions of published *shapes*.

What has changed:

1. The attack is separable from `so/model.py`. It takes a store as an argument. Pointing it at a real
   index is now configuration rather than a rewrite.
2. It carries its own validity floor and a vacuity guard, so an at-chance reading from it means
   something.
3. It reports a metric that survives the generalisation. This matters more than it sounds: run
   against a real HNSW index with a top-1 headline, this audit would have reported a clean deletion
   for the tombstone case and been wrong.

§5's experiment remains the next one, and it is now small: supply a `Store` whose `observe()` reads a
real index, and run.

## Limits

Toy geometry: 256 single-token payloads, an injective linear embedding, exact integer distances, 24
rows and 3 neighbours per node. The adjacency leak in particular will depend on real index parameters
— graph degree, construction algorithm, dimensionality, and how many neighbours survive pruning — and
88x is a number about *this* geometry, not a prediction about any deployed index. What transfers is
the shape of the finding and the metric that exposes it, not the constant.
