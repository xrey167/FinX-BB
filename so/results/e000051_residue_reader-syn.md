# E-000051 — the residue against the reader

Readers ['syn'], seeds [0, 1, 2], 100 pods per seed, trains nothing. AUCs are five-fold cross-validated Mann-Whitney statistics over held-out pods (worst seed = max for a <= bar, min for a >= bar); the null band at n = 200 is about 0.42-0.58.

## syn

| arm (positive vs reference) | AUC (i) deleted keys | AUC (ii) bystanders | AUC (iii) generic | max KL (ii) | top-1 agree (ii) | max KL (iii) |
|---|---|---|---|---|---|---|
| present: live vs never | 1.000 | - | - | - | - | - |
| cascade_soft: cascade vs never | 0.501 | 0.948 | 0.577 | 0.000 | 1.000 | 0.000 |
| blank_matched: blank vs cascade | 1.000 | 0.817 | 0.670 | 0.229 | 1.000 | 0.000 |
| dangle_matched: dangle vs cascade | 0.999 | 0.869 | 0.685 | 0.035 | 1.000 | 0.000 |
| blank_never: blank vs never | - | 0.951 | 0.671 | 0.229 | 1.000 | 0.000 |
| dangle_never: dangle vs never | - | 0.949 | 0.699 | 0.035 | 1.000 | 0.000 |
| perm: perm vs perm2 | - | 0.499 | 0.500 | 0.000 | 1.000 | 0.000 |
| add2: add2 vs perm | - | 0.965 | 0.749 | 0.000 | 1.000 | 0.000 |

| store-level (mean over pods) | exported HI | residue rows | markers equal |
|---|---|---|---|
| cascade | 1.000 | 0.000 | 0.000 |
| blank | 0.000 | 2.000 | 0.000 |
| dangle | 0.000 | 2.000 | 0.000 |

| lifecycle row | alias answers UNKNOWN | alias answers some entity |
|---|---|---|
| BLANK (SET NULL by hand) | 1.000 | 0.000 |
| DANGLE (evict object; = E-000026 delete_target/alias_unknown, reproduction) | 1.000 | 0.000 |

Interface residuals (max |delta| on rows common to both banks, aligned by key): cascade vs never 0.014, blank vs cascade 0.000, add2 vs perm 0.000; hard-gate pipeline check: encoding 0.000, logits 0.000.

## The rule, fixed before the run

Per reader. VOID if present/auc_i < 0.95 (the adversary cannot see a live row against none) or if the hard-gate pipeline check leaves an interface residual. If perm AUC > 0.60 every AUC row is void (the adversary reads summation order) and only KL / top-1 rows are read. If add2 AUC > 0.60 the ROW-COUNT reading is recorded: the reader's off-pod outputs carry the number of active rows, a property of dense routing and not of history, and the residue AUCs are reported against add2's as a calibrated null. If blank/dangle vs matched on class (i) < 0.75 the INVISIBLE reading is recorded. With V1-V5 passing: NEGATIVE (confined) if cascade, blank and dangle AUCs off the pod are all <= 0.60 and KL <= 0.05 nats with top-1 agreement >= 0.98; POSITIVE if any blank/dangle off-pod AUC >= 0.75 with its add2 row <= 0.60, naming the arm and the class, with the KL beside it; MARKER if any cascade-vs-never AUC >= 0.75 (the seeded generator's position is a history channel the reader exposes under the soft gate). Any AUC in (0.60, 0.75) is inconclusive at this n and licenses neither sentence. L1 is read independently: a blanked alias must read UNKNOWN in >= 0.90 of cases and as some entity in <= 0.05; dangle/deleted_key_unknown is E-000026's delete_target/alias_unknown under a new name and is reported, not decided on. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| syn/present/auc_i | >= 0.95 | 1.0000 | PASS |
| syn/enc/cascade_never_maxabs | >= 1e-06 | 0.0092 | PASS |
| syn/perm/auc_ii | <= 0.6 | 0.5577 | PASS |
| syn/perm/auc_iii | <= 0.6 | 0.5612 | PASS |
| syn/add2/auc_ii | <= 0.6 | 0.9688 | FAIL |
| syn/add2/auc_iii | <= 0.6 | 0.7951 | FAIL |
| syn/blank_matched/auc_i | >= 0.75 | 1.0000 | PASS |
| syn/dangle_matched/auc_i | >= 0.75 | 0.9987 | PASS |
| syn/cascade_soft/auc_i | <= 0.6 | 0.5013 | PASS |
| syn/cascade_soft/auc_ii | <= 0.6 | 0.9758 | FAIL |
| syn/cascade_soft/auc_iii | <= 0.6 | 0.6336 | FAIL |
| syn/blank_matched/auc_ii | <= 0.6 | 0.8451 | FAIL |
| syn/blank_matched/auc_iii | <= 0.6 | 0.6893 | FAIL |
| syn/dangle_matched/auc_ii | <= 0.6 | 0.8743 | FAIL |
| syn/dangle_matched/auc_iii | <= 0.6 | 0.6900 | FAIL |
| syn/blank_matched/kl_max_ii | <= 0.05 | 0.2289 | FAIL |
| syn/dangle_matched/kl_max_ii | <= 0.05 | 0.0353 | PASS |
| syn/blank_matched/top1_agree_ii | >= 0.98 | 1.0000 | PASS |
| syn/dangle_matched/top1_agree_ii | >= 0.98 | 1.0000 | PASS |
| syn/blank/deleted_key_unknown | >= 0.9 | 1.0000 | PASS |
| syn/blank/deleted_key_wrong_entity | <= 0.05 | 0.0000 | PASS |
