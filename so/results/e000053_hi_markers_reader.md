# E-000053 — history-independent markers, measured at the reader

Synthetic E-000015 reader, seeds [0, 1, 2], 100 pods per seed, ``MVCCStore(content_markers=True)``, trains nothing. AUCs are E-000051's five-fold cross-validated Mann-Whitney statistics; 'recorded' is E-000051's value on the same seed with the generator scheme.

| arm (positive vs reference) | AUC (i) | AUC (ii) bystanders | AUC (iii) generic | recorded (ii) | max KL (ii) | top-1 agree (ii) |
|---|---|---|---|---|---|---|
| present: live vs never | 1.000 | - | - | - | - | - |
| cascade_soft: cascade vs never | 0.500 | 0.500 | 0.500 | 0.948 | 0.000 | 1.000 |
| blank_matched: blank vs cascade | 1.000 | 0.800 | 0.670 | 0.817 | 0.229 | 1.000 |
| dangle_matched: dangle vs cascade | 0.999 | 0.839 | 0.681 | 0.869 | 0.035 | 1.000 |
| blank_never: blank vs never | - | 0.800 | 0.670 | - | 0.229 | 1.000 |
| dangle_never: dangle vs never | - | 0.839 | 0.681 | - | 0.035 | 1.000 |
| perm: perm vs perm2 | - | 0.487 | 0.571 | - | 0.000 | 1.000 |
| add2: add2 vs perm | - | 0.974 | 0.770 | 0.965 | 0.000 | 1.000 |

| store-level (mean over pods) | exported HI | residue rows | markers equal |
|---|---|---|---|
| cascade | 1.000 | 0.000 | 1.000 |
| blank | 0.000 | 2.000 | 0.000 |
| dangle | 0.000 | 2.000 | 0.000 |

| R1: the frozen reader on the re-signed live bank | value (worst seed) |
|---|---|
| gate acceptance | 1.000 |
| max |gate delta| | 0.002 |
| top-1 agreement (i) / (ii) | 1.000 / 1.000 |
| max KL (i) / (ii) | 0.000 / 0.000 |
| reading accuracy, derived / generator | 1.000 / 1.000 |

| S1: the duplicate row (identical content) | value |
|---|---|
| pair marker distance, option / generator | 0.000 / 0.135 |
| DUP vs DUPX AUC (ii) | 0.502 |
| max KL (ii) / top-1 | 0.000 / 1.000 |

Interface residual cascade vs never 0.000 (E-000051: 0.0116); hard-gate check encoding 0.000, logits 0.000.

## The rule, fixed before the run

Synthetic reader, three seeds, worst seed. VOID if present/auc_i < 0.95, if perm AUC > 0.60, or if the store half fails (cascade markers_equal < 1 or exported HI < 1: the option is not history independent at the store and nothing downstream is read). NOT USABLE if R1 fails (gate acceptance < 0.99, top-1 agreement < 0.98 or KL > 0.05 nats between the re-signed and the generator-signed live bank, or accuracy moved by more than 0.01): the frozen reader does not accept content-derived markers as its own, and the marker channel closes only by retraining -- that is the negative sentence. NOT COMPARABLE if R2 fails (add2 AUC outside +-0.05 of E-000051's per-seed value or below 0.90): the option changed the reader's arithmetic and the E-000051 table cannot be re-read under it. With V and R passing: the POSITIVE sentence needs M1 (cascade vs never <= 0.60 on every class with interface residual 0.000 -- reported as the pipeline check it is) AND M2 (blank/dangle vs matched within +-0.05 of the record and at or below the add2 floor). If M1 fails with V3 passing, a channel other than content and marker reaches the reader (row order, a placeholder column) and is named. S1 is registered, not decided on: the store-level pair distance is 0.000 by the mechanism; dup/auc_ii <= 0.60 records the identical-marker side effect as invisible at the reader, >= 0.75 as legible, the grey zone as inconclusive at this n. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| present/auc_i | >= 0.95 | 1.0000 | PASS |
| perm/auc_ii | <= 0.6 | 0.5253 | PASS |
| perm/auc_iii | <= 0.6 | 0.5916 | PASS |
| store/cascade/exported_hi | >= 0.999 | 1.0000 | PASS |
| store/cascade/markers_equal | >= 0.999 | 1.0000 | PASS |
| live/marker_absdiff_max | >= 0.001 | 0.2810 | PASS |
| live/gate_accept | >= 0.99 | 1.0000 | PASS |
| live/top1_agree_i | >= 0.98 | 1.0000 | PASS |
| live/top1_agree_ii | >= 0.98 | 1.0000 | PASS |
| live/kl_max_i | <= 0.05 | 0.0001 | PASS |
| live/kl_max_ii | <= 0.05 | 0.0000 | PASS |
| live/acc_absdelta | <= 0.01 | 0.0000 | PASS |
| add2/auc_ii | >= 0.9 | 0.9441 | PASS |
| add2/auc_ii_absdelta | <= 0.05 | 0.0324 | PASS |
| cascade_soft/auc_i | <= 0.6 | 0.5000 | PASS |
| cascade_soft/auc_ii | <= 0.6 | 0.5000 | PASS |
| cascade_soft/auc_iii | <= 0.6 | 0.5000 | PASS |
| enc/cascade_never_maxabs | <= 1e-06 | 0.0000 | PASS |
| blank_matched/auc_ii_absdelta | <= 0.05 | 0.0338 | PASS |
| dangle_matched/auc_ii_absdelta | <= 0.05 | 0.0416 | PASS |
| residue_below_add2 | >= 1.0 | 1.0000 | PASS |
| dup/pair_marker_dist_max | <= 1e-09 | 0.0000 | PASS |
| dup/pair_marker_dist_gen_min | >= 0.05 | 0.1347 | PASS |
