# E-000045 — U and T in a representation, counted on readers instead of on rows

Frozen gpt2, layer 7, no training. The intervention is the workspace paper's
own SWAP on J-lens vectors, not an ablation: three earlier attempts to carry the U/T law
into a representation used subspace ablation and measured FAN-IN, which is the wrong count.
U is governed by fan-in; T by fan-out, because a reader still yielding the original value is
a reader retaining evidence of the referent that was replaced.

## The swap, and what follows it

| measure | value |
|---|---|
| entities measured | 10 |
| predicates held per entity (fan-out) | 2.50 |
| **broadcast** — predicates that follow one swap | **0.3438** |
| a random direction of matched norm (control) | 0.0000 |
| swapping an entity for itself (control) | 0.0000 |

## U against T

| measure | value |
|---|---|
| U — one swap makes the attacked predicate stop yielding the original | 1.00 |
| residue — predicates still yielding the original | 0.8375 |
| **T** — 1 + the readers that must also be redirected | **2.95** |
| **T / U** | **2.75** |
| swap pairs where U was ACHIEVED (pre-registered >= 0.50) | 0.3250 |
| entities entering the T/U aggregate | 4 |
| entities where T exceeds U, among those | 1.0000 |
| correlation of T/U with broadcast (predicted negative) | -0.9704 |

TWO THINGS THIS DOES NOT SHOW, stated before the one it does. The correlation above is
NOT independent evidence: T is defined as 1 + residue x fan-out, so T/U is a
deterministic decreasing function of broadcast and the sign was fixed by arithmetic
before any state was read. It is reported because it was pre-registered, and it should
not be quoted as a finding. And `broadcast` FAILS its own attack-validity floor: the
swap redirects 0.16 of off-diagonal predicates where the workspace paper reports 76/192
overall and 42/48 for countries on a far larger model. Both controls sit at exactly
0.0000, so the intervention is real and specific -- but it is weak, and everything here
holds in a weak-intervention regime.

AND A THIRD, WHICH IS THE SERIOUS ONE. `u_rate` FAILS: the attacked predicate stops
yielding the original in only a third of swap pairs, so U is ACHIEVED in a minority of
cases and everything below is computed on the subset where it was. That subset is
selected by whether the intervention worked, which is a selection effect and not a
sample. The honest summary is that GPT-2 small is too weak a model to establish U by
this instrument at the strength registered in advance.

WHAT IS LEFT, on that subset and labelled as such: one swap leaves most of an entity's
readers still yielding the original value, and T exceeds U wherever U was achieved.
That is consistent with the prediction and is not a test of it.

Each predicate that did not follow the swap is a reader retaining evidence of the
referent that was replaced — a dangling reference, and the same residue the workspace
paper reports in its own disconfirming case, where swapping a passage's language vector
left continuation and anomaly detection unmoved while the concept still appeared in the
lens readouts of all four tasks.

## The rule, fixed before the run

T > U for most entities -> the store's law has an activation-space form once T is counted on READERS rather than on rows, and the residue is the dangling reference. T = U throughout -> one swap redirects every reader, the carrier is a true pod, and erasure in a representation is traceless in one move, which would be a stronger result than the one predicted and is recorded as such. A negative correlation between T/U and broadcast is predicted; positive or flat falsifies the identification of broadcast with a reference count. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| broadcast | >= 0.2 | 0.3438 | PASS |
| u_rate | >= 0.5 | 0.3250 | FAIL |
| broadcast_random | <= 0.15 | 0.0000 | PASS |
| broadcast_identity | <= 0.05 | 0.0000 | PASS |
| T_over_U | >= 1.2 | 2.7500 | PASS |
| gap_exists | >= 0.6 | 1.0000 | PASS |
| residue | >= 0.2 | 0.8375 | PASS |
