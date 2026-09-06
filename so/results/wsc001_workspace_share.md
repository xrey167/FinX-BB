# WSC-001 — where an accessibility audit of an external memory must read

Pre-registered: `docs/novelty/wsc001-preregister.md`. Seeds [0, 1, 2]; worst seed reported.
Validity: **FAIL: V1**.  Part A: **A0 SITING: E-000063's capture block is not where the pod's content enters -- the write at the first read site moves by a small fraction of the second's when the pod is shredded -- so no readout there can attribute the memory, and the certificate audits a state the pod has effectively not reached**.  Part B: **VOID (validity)**.

## Part B — the causal side (alias reads, worst seed)

| template | full | dropW | dropW_n | dropT_n | atom | atom_dropW | perm | none |
|---|---|---|---|---|---|---|---|---|
| t0 | 0.9600 | 0.0450 | 0.1100 | 0.9050 | 1.0000 | 0.0100 | 0.0000 | 0.0100 |
| t1 | 0.6900 | 0.0550 | 0.1400 | 0.6700 | 1.0000 | 0.0100 | 0.0000 | 0.0050 |
| t2 | 0.9600 | 0.0500 | 0.0900 | 0.8950 | 1.0000 | 0.0050 | 0.0000 | 0.0100 |
| t3 | 0.9400 | 0.0500 | 0.1100 | 0.8150 | 1.0000 | 0.0150 | 0.0000 | 0.0050 |
| t4 | 0.9650 | 0.0300 | 0.0600 | 0.9050 | 1.0000 | 0.0100 | 0.0000 | 0.0100 |
| t5 | 0.9600 | 0.0450 | 0.0750 | 0.8900 | 1.0000 | 0.0050 | 0.0000 | 0.0150 |
| t6 | 0.9550 | 0.0500 | 0.1150 | 0.8800 | 1.0000 | 0.0050 | 0.0000 | 0.0050 |
| t7 | 0.9100 | 0.0600 | 0.1150 | 0.7850 | 1.0000 | 0.0150 | 0.0000 | 0.0050 |
| t8 | 0.9600 | 0.0700 | 0.1250 | 0.8550 | 1.0000 | 0.0100 | 0.0000 | 0.0150 |
| t9 | 0.8700 | 0.0200 | 0.0700 | 0.6400 | 1.0000 | 0.0200 | 0.0000 | 0.0000 |
| t10 | 0.9550 | 0.0400 | 0.0750 | 0.8800 | 1.0000 | 0.0050 | 0.0000 | 0.0100 |
| t11 | 0.9600 | 0.1050 | 0.1600 | 0.9000 | 1.0000 | 0.0200 | 0.0000 | 0.0100 |

## Part A — the readout side (probe trained on ACTIVE only, worst seed)

| family | direct site8 | direct site9 | direct site10 | direct final | alias site8 | alias site9 | alias site10 | alias final |
|---|---|---|---|---|---|---|---|---|
| jspace | 0.0000 | 0.0000 | 0.8929 | 0.8661 | -0.0134 | -0.0223 | 0.9152 | 0.9152 |
| random | 0.0000 | 0.0000 | 0.7411 | 0.8125 | 0.0000 | -0.0045 | 0.8705 | 0.8348 |
| pca | 0.0000 | 0.0000 | 0.6607 | 0.8571 | -0.0089 | -0.0179 | 0.8170 | 0.8929 |
| unembed | 0.0000 | -0.0089 | 0.8929 | 0.8661 | 0.0000 | -0.0179 | 0.9152 | 0.9152 |
| raw | 0.0000 | 0.0000 | 0.7768 | 0.7500 | -0.0179 | -0.0089 | 0.8616 | 0.8616 |

### The mediator: how far the state moves at each site when the pod is shredded, and when it was never written

| site | direct SHRED | direct NEVER | alias SHRED | alias NEVER |
|---|---|---|---|---|
| site8 | 0.0828 | 0.0564 | 1.5533 | 2.4310 |
| site9 | 0.2021 | 0.0726 | 1.4346 | 3.8453 |
| site10 | 79.1417 | 59.9716 | 76.5166 | 64.4159 |
| final | 21.5871 | 27.0877 | 18.7262 | 24.9274 |

## Summary rows

| measure | mean over seeds | worst seed |
|---|---|---|
| B/full_alias_min | 0.8017 | 0.6900 |
| B/full_direct_min | 0.8833 | 0.8350 |
| B/dropW_alias_max | 0.0450 | 0.1050 |
| B/dropW_n_alias_max | 0.0767 | 0.1600 |
| B/dropT_n_deficit_min | -0.2083 | -0.2600 |
| B/perm_alias_max | 0.0000 | 0.0000 |
| B/none_alias_max | 0.0100 | 0.0150 |
| B/n_templates_dropWn_ge_50 | 0.0000 | 0.0000 |
| B/share_W_mean | 0.2879 | 0.2830 |
| B/atom_dropW_alias_max | 0.0183 | 0.0200 |
| A/write8_shred_moves | 2.8076 | 1.5533 |
| A/write10_shred_moves | 89.6981 | 80.3278 |
| A/write_site_ratio | 0.0318 | 0.0157 |
| A/state_site_ratio | 0.0321 | 0.0158 |
| A/site8_max_amn | 0.0089 | 0.0000 |
| A/site10_max_amn | 0.9360 | 0.9152 |
| A/final_raw_amn | 0.8304 | 0.7500 |
| A/alias/site8/shred_moves | 2.8076 | 1.5533 |
| A/alias/site10/shred_moves | 83.5544 | 76.5166 |
| A/alias/final/shred_moves | 27.0570 | 18.7262 |
| A/alias/final/jspace | 0.9241 | 0.9152 |
| A/direct/final/jspace | 0.9137 | 0.8661 |
| A/indirection/jspace | -0.0104 | -0.0625 |
| A/alias/final/raw | 0.8795 | 0.8616 |
| A/direct/final/raw | 0.8304 | 0.7500 |
| A/indirection/raw | -0.0491 | -0.1116 |
| A/shred_residual_direct | -0.0357 | -0.0982 |
| A/shred_residual_alias | -0.0193 | -0.0625 |
| A/shred_residual_asymmetry | -0.0164 | -0.0357 |

## Pre-registered criteria (worst seed)

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| B/full_alias_min | >= 0.8 | 0.6900 | FAIL |
| B/full_direct_min | >= 0.9 | 0.8350 | FAIL |
| B/perm_alias_max | <= 0.05 | 0.0000 | PASS |
| B/none_alias_max | <= 0.05 | 0.0150 | PASS |
| A/final_raw_amn | >= 0.3 | 0.7500 | PASS |
| A/write8_shred_moves | >= 0.0 | 1.5533 | PASS |
| A/site10_max_amn | >= 0.0 | 0.9152 | PASS |
| B/dropW_n_alias_max | <= 0.2 | 0.1600 | PASS |
| B/dropT_n_deficit_min | >= -0.2 | -0.2600 | FAIL |

By construction: keep/drop of the first-order term (ledger 31.39); atom/alias = 1.0 (the arm is handed the ground-truth object); at read layer 10 the atoms are the unembedding rows (31.56).

Not claimed: the Jacobian lens, J-lens auditing, probing, projection, external memory, pods or pointer aliases; no deletion guarantee; the atom arm is an ORACLE and is never a capability comparison
