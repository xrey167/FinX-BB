# E-000026 — the symlink lifecycle battery, measured where reading works

Seeds [0, 1, 2]. No training: E-000020's checkpoints, E-000020's battery, run three times at
three phrasings — template 0 (what that record used), template 3 (the trained
template on which E-000017-B's *link-free* adapter reads best) and template 10
(the same rule over the held-out four). The choice comes from a different experiment on a different
adapter, so it cannot be tuned in the link arm's favour.

## The battery at three phrasings

| measure (worst seed) | template0 (t0) | strong_train (t3) | strong_heldout (t10) |
|---|---|---|---|
| direct | 0.5633 | 0.9933 | 0.9967 |
| alias_direct | 0.5000 | 0.8600 | 0.8700 |
| dup_direct | 0.5900 | 0.9950 | 1.0000 |
| shared_update/alias_new_object | 0.5350 | 0.8650 | 0.8850 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_unknown | 0.9950 | 0.9950 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_forced_choice | 0.4650 | 0.4350 | 0.4300 |
| shred_target/alias_probe_top1 | 0.0100 | 0.0100 | 0.0100 |
| active/alias_probe_top1 | 0.4200 | 0.7800 | 0.8000 |
| revoke_alias/sibling_readable | 0.5800 | 0.8800 | 0.8800 |
| delete_target/alias_unknown | 0.9650 | 0.9600 | 0.9100 |

## Pre-registered criteria (E-000020's, unchanged)

### template0 — template 0

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5633 | FAIL |
| alias_direct | >= 0.8 | 0.5000 | FAIL |
| dup_direct | >= 0.85 | 0.5900 | FAIL |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.5350 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9950 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.5900 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5600 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.4200 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.5000 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.5500 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9600 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.5800 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5600 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9650 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: attacks_through_every_alias, attack_validity.

### strong_train — template 3

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.9933 | PASS |
| alias_direct | >= 0.8 | 0.8600 | PASS |
| dup_direct | >= 0.85 | 0.9950 | PASS |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.8650 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.8600 | PASS |
| shred_target/alias_unknown | >= 0.9 | 0.9950 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.9950 | PASS |
| resign_target/alias_direct | >= 0.8 | 0.8600 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.4850 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0150 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.7800 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.7917 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.8100 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.8800 | PASS |
| revoke_alias/target_readable | >= 0.8 | 0.9900 | PASS |
| delete_target/alias_unknown | >= 0.9 | 0.9600 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: one_shred_deletes_every_path, attacks_through_every_alias, attack_validity, alias_lifecycle.

### strong_heldout — template 10

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.9967 | PASS |
| alias_direct | >= 0.8 | 0.8700 | PASS |
| dup_direct | >= 0.85 | 1.0000 | PASS |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.8850 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.8700 | PASS |
| shred_target/alias_unknown | >= 0.9 | 1.0000 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 1.0000 | PASS |
| resign_target/alias_direct | >= 0.8 | 0.8700 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.4750 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0150 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.8000 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.7917 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.8100 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.8800 | PASS |
| revoke_alias/target_readable | >= 0.8 | 1.0000 | PASS |
| delete_target/alias_unknown | >= 0.9 | 0.9100 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: one_shred_deletes_every_path, attacks_through_every_alias, attack_validity, alias_lifecycle.

## How the templates were chosen

| template | link-free reading (E-000017-B) | kind |
|---|---|---|
| 0 | 0.7950 | trained |
| 1 | 0.9917 | trained |
| 2 | 0.7917 | trained |
| 3 | 1.0000 | trained |
| 4 | 1.0000 | trained |
| 5 | 0.9983 | trained |
| 6 | 0.7850 | trained |
| 7 | 0.9967 | trained |
| 8 | 0.5650 | held out |
| 9 | 0.9683 | held out |
| 10 | 1.0000 | held out |
| 11 | 0.4267 | held out |
