# E-000051 — the residue against the reader

Readers ['gpt2'], seeds [0, 1, 2], 100 pods per seed, trains nothing. AUCs are five-fold cross-validated Mann-Whitney statistics over held-out pods (worst seed = max for a <= bar, min for a >= bar); the null band at n = 200 is about 0.42-0.58.

## gpt2

| arm (positive vs reference) | AUC (i) deleted keys | AUC (ii) bystanders | AUC (iii) generic | max KL (ii) | top-1 agree (ii) | max KL (iii) |
|---|---|---|---|---|---|---|
| present: live vs never | 1.000 | - | - | - | - | - |
| cascade_soft: cascade vs never | 0.500 | 0.869 | 0.598 | 0.000 | 1.000 | 0.000 |
| blank_matched: blank vs cascade | 0.998 | 0.791 | 0.668 | 4.486 | 0.999 | 1.107 |
| dangle_matched: dangle vs cascade | 0.975 | 0.876 | 0.742 | 3.962 | 0.999 | 0.898 |
| blank_never: blank vs never | - | 0.890 | 0.727 | 4.486 | 0.999 | 1.107 |
| dangle_never: dangle vs never | - | 0.925 | 0.811 | 3.961 | 0.999 | 0.898 |
| perm: perm vs perm2 | - | 0.482 | 0.457 | 0.000 | 1.000 | 0.000 |
| add2: add2 vs perm | - | 0.977 | 0.827 | 0.310 | 0.999 | 0.287 |

| store-level (mean over pods) | exported HI | residue rows | markers equal |
|---|---|---|---|
| cascade | 1.000 | 0.000 | 0.000 |
| blank | 0.000 | 2.000 | 0.000 |
| dangle | 0.000 | 2.000 | 0.000 |

| lifecycle row | alias answers UNKNOWN | alias answers some entity |
|---|---|---|
| BLANK (SET NULL by hand) | 0.825 | 0.175 |
| DANGLE (evict object; = E-000026 delete_target/alias_unknown, reproduction) | 0.970 | 0.030 |

Interface residuals (max |delta| on rows common to both banks, aligned by key): cascade vs never 0.169, blank vs cascade 0.000, add2 vs perm 0.000; hard-gate pipeline check: encoding 0.000, logits 0.000.

## The rule, fixed before the run

Per reader. VOID if present/auc_i < 0.95 (the adversary cannot see a live row against none) or if the hard-gate pipeline check leaves an interface residual. If perm AUC > 0.60 every AUC row is void (the adversary reads summation order) and only KL / top-1 rows are read. If add2 AUC > 0.60 the ROW-COUNT reading is recorded: the reader's off-pod outputs carry the number of active rows, a property of dense routing and not of history, and the residue AUCs are reported against add2's as a calibrated null. If blank/dangle vs matched on class (i) < 0.75 the INVISIBLE reading is recorded. With V1-V5 passing: NEGATIVE (confined) if cascade, blank and dangle AUCs off the pod are all <= 0.60 and KL <= 0.05 nats with top-1 agreement >= 0.98; POSITIVE if any blank/dangle off-pod AUC >= 0.75 with its add2 row <= 0.60, naming the arm and the class, with the KL beside it; MARKER if any cascade-vs-never AUC >= 0.75 (the seeded generator's position is a history channel the reader exposes under the soft gate). Any AUC in (0.60, 0.75) is inconclusive at this n and licenses neither sentence. L1 is read independently: a blanked alias must read UNKNOWN in >= 0.90 of cases and as some entity in <= 0.05; dangle/deleted_key_unknown is E-000026's delete_target/alias_unknown under a new name and is reported, not decided on. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| gpt2/present/auc_i | >= 0.95 | 1.0000 | PASS |
| gpt2/enc/cascade_never_maxabs | >= 1e-06 | 0.0984 | PASS |
| gpt2/perm/auc_ii | <= 0.6 | 0.4974 | PASS |
| gpt2/perm/auc_iii | <= 0.6 | 0.4620 | PASS |
| gpt2/add2/auc_ii | <= 0.6 | 1.0000 | FAIL |
| gpt2/add2/auc_iii | <= 0.6 | 0.9651 | FAIL |
| gpt2/blank_matched/auc_i | >= 0.75 | 0.9961 | PASS |
| gpt2/dangle_matched/auc_i | >= 0.75 | 0.9668 | PASS |
| gpt2/cascade_soft/auc_i | <= 0.6 | 0.5000 | PASS |
| gpt2/cascade_soft/auc_ii | <= 0.6 | 0.8876 | FAIL |
| gpt2/cascade_soft/auc_iii | <= 0.6 | 0.7261 | FAIL |
| gpt2/blank_matched/auc_ii | <= 0.6 | 0.8294 | FAIL |
| gpt2/blank_matched/auc_iii | <= 0.6 | 0.7309 | FAIL |
| gpt2/dangle_matched/auc_ii | <= 0.6 | 0.9108 | FAIL |
| gpt2/dangle_matched/auc_iii | <= 0.6 | 0.8550 | FAIL |
| gpt2/blank_matched/kl_max_ii | <= 0.05 | 4.4861 | FAIL |
| gpt2/dangle_matched/kl_max_ii | <= 0.05 | 3.9619 | FAIL |
| gpt2/blank_matched/top1_agree_ii | >= 0.98 | 0.9992 | PASS |
| gpt2/dangle_matched/top1_agree_ii | >= 0.98 | 0.9992 | PASS |
| gpt2/blank/deleted_key_unknown | >= 0.9 | 0.8250 | FAIL |
| gpt2/blank/deleted_key_wrong_entity | <= 0.05 | 0.1750 | FAIL |
| gpt2/blank_matched/kl_max_iii | <= 0.05 | 1.1069 | FAIL |
| gpt2/dangle_matched/kl_max_iii | <= 0.05 | 0.8981 | FAIL |
