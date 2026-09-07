# WSC-001 — where an accessibility audit of an external memory must read

Pre-registered: `docs/novelty/wsc001-preregister.md`. Seeds [0]; worst seed reported.
Validity: **FAIL: V1,V2,V3**.  Part A: **A1 DEPTH: the write reaches the first read site but no linear readout there attributes it, at any dimension up to the full residual**.  Part B: **VOID (validity)**.

## Part B — the causal side (alias reads, worst seed)

| template | full | dropW | dropW_n | dropT_n | atom | atom_dropW | perm | none |
|---|---|---|---|---|---|---|---|---|

## Part A — the readout side (probe trained on ACTIVE only, worst seed)

| family | direct site8 | direct site9 | direct site10 | direct final | alias site8 | alias site9 | alias site10 | alias final |
|---|---|---|---|---|---|---|---|---|
| jspace | 0.0000 | 0.0000 | 0.9643 | 0.9464 | -0.0089 | -0.0357 | 0.9018 | 0.8839 |
| random | 0.0000 | 0.0000 | 0.8571 | 0.8214 | -0.0089 | -0.0089 | 0.7946 | 0.8214 |
| pca | 0.0000 | 0.0000 | 0.6250 | 0.8036 | -0.0357 | -0.0089 | 0.7946 | 0.8482 |
| unembed | 0.0000 | 0.0000 | 0.9643 | 0.9464 | 0.0089 | -0.0268 | 0.9018 | 0.8839 |
| raw | 0.0000 | 0.0000 | 0.6250 | 0.7500 | 0.0000 | 0.0000 | 0.7946 | 0.7768 |

### The mediator: how far the state moves at each site when the pod is shredded, and when it was never written

| site | direct SHRED | direct NEVER | alias SHRED | alias NEVER |
|---|---|---|---|---|
| site8 | 0.1600 | 0.1652 | 2.2571 | 4.5748 |
| site9 | 0.3904 | 0.1888 | 3.1234 | 6.2648 |
| site10 | 88.1877 | 64.7349 | 74.1002 | 61.2142 |
| final | 21.2091 | 27.4927 | 17.6455 | 22.0051 |

## Summary rows

| measure | mean over seeds | worst seed |
|---|---|---|
| A/write8_shred_moves | 2.2571 | 2.2571 |
| A/write10_shred_moves | 88.8198 | 88.8198 |
| A/site8_max_amn | 0.0089 | 0.0089 |
| A/site10_max_amn | 0.9643 | 0.9643 |
| A/final_raw_amn | 0.7500 | 0.7500 |
| A/alias/site8/shred_moves | 2.2571 | 2.2571 |
| A/alias/site10/shred_moves | 74.1002 | 74.1002 |
| A/alias/final/shred_moves | 17.6455 | 17.6455 |
| A/alias/final/jspace | 0.8839 | 0.8839 |
| A/direct/final/jspace | 0.9464 | 0.9464 |
| A/indirection/jspace | 0.0625 | 0.0625 |
| A/alias/final/raw | 0.7768 | 0.7768 |
| A/direct/final/raw | 0.7500 | 0.7500 |
| A/indirection/raw | -0.0268 | -0.0268 |
| A/shred_residual_direct | 0.2500 | 0.2500 |
| A/shred_residual_alias | -0.0804 | -0.0804 |
| A/shred_residual_asymmetry | 0.3304 | 0.3304 |

## Pre-registered criteria (worst seed)

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| B/full_alias_min | >= 0.8 | - | FAIL |
| B/full_direct_min | >= 0.9 | - | FAIL |
| B/perm_alias_max | <= 0.05 | - | FAIL |
| B/none_alias_max | <= 0.05 | - | FAIL |
| A/final_raw_amn | >= 0.3 | 0.7500 | PASS |
| A/write8_shred_moves | >= 0.0 | 2.2571 | PASS |
| A/site10_max_amn | >= 0.0 | 0.9643 | PASS |
| B/dropW_n_alias_max | <= 0.2 | - | FAIL |
| B/dropT_n_deficit_min | >= -0.2 | - | FAIL |

By construction: keep/drop of the first-order term (ledger 31.39); atom/alias = 1.0 (the arm is handed the ground-truth object); at read layer 10 the atoms are the unembedding rows (31.56).

Not claimed: the Jacobian lens, J-lens auditing, probing, projection, external memory, pods or pointer aliases; no deletion guarantee; the atom arm is an ORACLE and is never a capability comparison
