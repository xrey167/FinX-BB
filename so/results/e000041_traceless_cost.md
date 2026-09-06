# E-000041 — the traceless erasure cost is invariant and canonicalisation cannot touch it

Seeds [0, 1, 2], k from 2 to 8, every mixing ratio from all-copies to a full
pod. No model, no checkpoint, no training: the mechanical resolver and the store's own bank.

## U falls from k to 1 across the spectrum; T does not move

| k access paths | U by number of links (0 = all copies ... k-1 = full pod) | T, same order |
|---|---|---|
| 2 | 2 1 | 2 2 |
| 3 | 3 2 1 | 3 3 3 |
| 4 | 4 3 2 1 | 4 4 4 4 |
| 5 | 5 4 3 2 1 | 5 5 5 5 5 |
| 6 | 6 5 4 3 2 1 | 6 6 6 6 6 6 |
| 7 | 7 6 5 4 3 2 1 | 7 7 7 7 7 7 7 |
| 8 | 8 7 6 5 4 3 2 1 | 8 8 8 8 8 8 8 8 |

`U` is the fact closure — the minimum records to remove before no access path yields the
object. `T` is the minimum to remove so that the fact is unreachable **and** no surviving row
points at anything that is gone. Reading each row left to right is walking from a fully
duplicated store to a fully canonical one.

**U is 1 + (the number of paths that are still copies). T is k, in every cell.**

So canonicalisation is not a reduction in the cost of erasure; it is a move along a
trade-off. It buys the cheap-but-visible regime and gives nothing in the traceless one.
Every claim of the form *normalise and erasure becomes one operation* is a claim about U,
and is true. The same design leaves T exactly where it was — and T is the number a data
subject is promised when they are told a record is gone.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| T_equals_k | >= 1.0 | 1.0000 | PASS |
| U_matches_prediction | >= 1.0 | 1.0000 | PASS |
| all_valid | >= 1.0 | 1.0000 | PASS |
| U_min | <= 1.0 | 1.0000 | PASS |
| U_max | >= 4.0 | 8.0000 | PASS |

## What this is and is not

Codd's modification anomaly is about U. Database resilience — the minimum contingency set —
is U. Raeesi and Roed's proposal to store aliases as pointers into a single canonical record
(arXiv:2607.00605 §9, offered as untested) is about U. None of them is about T. What is not
claimed: that T is invariant in stores unlike this one. A store that compacts its aliases on
deletion, or never exports a target key, has a different T and the law would have to be
restated for it — which is exactly why it is stated as a measurement over a spectrum rather
than as an identity.
