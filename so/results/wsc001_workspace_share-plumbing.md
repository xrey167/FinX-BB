# WSC-001 — where an accessibility audit of an external memory must read

Pre-registered: `docs/novelty/wsc001-preregister.md`. Seeds [0]; worst seed reported.
Validity: **PASS**.  Part A: **A1 DEPTH: no linear readout at the write site attributes the memory beyond the prompt, at any dimension up to the full residual**.  Part B: **B1 CONFIRMED: the scored-token atom span is the causal channel of the write and the token-matched null does not reproduce it**.

## Part B — the causal side (alias reads, worst seed)

| template | full | dropW | dropW_n | dropT_n | atom | atom_dropW | perm | none |
|---|---|---|---|---|---|---|---|---|
| t3 | 0.9400 | 0.0500 | 0.1100 | 0.8750 | 1.0000 | 0.0150 | 0.0000 | 0.0050 |

## Part A — the readout side (probe trained on ACTIVE only, worst seed)

| family | direct mid A−N | direct final A−N | alias mid A−N | alias final A−N |
|---|---|---|---|---|
| jspace | 0.0000 | 0.9571 | -0.0071 | 0.8929 |
| random | 0.0000 | 0.8429 | 0.0000 | 0.8000 |
| pca | 0.0000 | 0.9571 | -0.0214 | 0.8714 |
| unembed | 0.0000 | 0.9429 | 0.0000 | 0.8857 |
| raw | 0.0143 | 0.8714 | 0.0071 | 0.8143 |

## Summary rows

| measure | mean over seeds | worst seed |
|---|---|---|
| B/full_alias_min | 0.9400 | 0.9400 |
| B/full_direct_min | 1.0000 | 1.0000 |
| B/dropW_alias_max | 0.0500 | 0.0500 |
| B/dropW_n_alias_max | 0.1100 | 0.1100 |
| B/dropT_n_deficit_min | -0.0650 | -0.0650 |
| B/perm_alias_max | 0.0000 | 0.0000 |
| B/none_alias_max | 0.0050 | 0.0050 |
| B/n_templates_dropWn_ge_50 | 0.0000 | 0.0000 |
| B/share_W_mean | 0.2846 | 0.2846 |
| B/atom_dropW_alias_max | 0.0150 | 0.0150 |
| A/mid_max_active_minus_never | 0.0143 | 0.0143 |
| A/mid_jspace_amn | 0.0000 | 0.0000 |
| A/mid_pca_amn | 0.0000 | 0.0000 |
| A/mid_raw_amn | 0.0143 | 0.0143 |
| A/final_raw_amn | 0.8143 | 0.8143 |
| A/alias/final_jspace_amn | 0.8929 | 0.8929 |
| A/direct/final_jspace_amn | 0.9571 | 0.9571 |
| A/indirection/jspace | 0.0643 | 0.0643 |
| A/alias/final_raw_amn | 0.8143 | 0.8143 |
| A/direct/final_raw_amn | 0.8714 | 0.8714 |
| A/indirection/raw | 0.0571 | 0.0571 |
| A/alias/final_pca_amn | 0.8714 | 0.8714 |
| A/direct/final_pca_amn | 0.9571 | 0.9571 |
| A/indirection/pca | 0.0857 | 0.0857 |
| A/alias/final_raw_shred_minus_never | -0.0214 | -0.0214 |
| A/direct/final_raw_shred_minus_never | 0.4571 | 0.4571 |

## Pre-registered criteria (worst seed)

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| B/full_alias_min | >= 0.8 | 0.9400 | PASS |
| B/full_direct_min | >= 0.9 | 1.0000 | PASS |
| B/perm_alias_max | <= 0.05 | 0.0000 | PASS |
| B/none_alias_max | <= 0.05 | 0.0050 | PASS |
| A/final_raw_amn | >= 0.3 | 0.8143 | PASS |
| B/dropW_n_alias_max | <= 0.2 | 0.1100 | PASS |
| B/dropT_n_deficit_min | >= -0.2 | -0.0650 | PASS |

By construction: keep/drop of the first-order term (ledger 31.39); atom/alias = 1.0 (the arm is handed the ground-truth object); at read layer 10 the atoms are the unembedding rows (31.56).

Not claimed: the Jacobian lens, J-lens auditing, probing, projection, external memory, pods or pointer aliases; no deletion guarantee; the atom arm is an ORACLE and is never a capability comparison
