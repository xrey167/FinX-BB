# E-000023 — Alias reading in a frozen GPT-2: the 'curriculum' arm against E-000020's budget

E-000020 read direct facts at 57% while the same adapter without links reads this evaluation world at 82.7%, and 84.7% with the machinery attached, so the cost is in learning rather than in the world or the mechanism. This arm attacks the learning and changes nothing else.

| claim group | supported |
|---|---|
| reading_through_an_alias | **no** |
| one_update_reaches_every_path | **no** |
| one_shred_deletes_every_path | **no** |
| attacks_through_every_alias | yes |
| attack_validity | **no** |
| alias_lifecycle | **no** |

| measure | mean over seeds | worst seed | E-000020 baseline |
|---|---|---|---|
| direct | 0.6289 | 0.5767 | 0.5700 |
| alias_direct | 0.0000 | 0.0000 | 0.5067 |
| dup_direct | 0.6867 | 0.6550 | 0.5483 |
| alias_heldout_min | 0.0000 | 0.0000 | 0.3700 |
| probe_calibration_top1 | 0.4833 | 0.4250 | 0.4806 |
| active/alias_probe_top1 | 0.0017 | 0.0000 | 0.3883 |
| active/alias_forced_choice | 0.4567 | 0.3900 | 0.8950 |
| shared_update/alias_new_object | 0.0000 | 0.0000 | 0.5183 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| rollback/alias_direct | 0.0000 | 0.0000 | 0.5067 |
| shred_target/alias_unknown | 0.9950 | 0.9850 | 0.9950 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0017 | 0.0050 | 0.0033 |
| shred_target/alias_forced_choice | 0.4517 | 0.5450 | 0.5100 |
| shred_target/alias_top1_among_entities | 0.0050 | 0.0100 | 0.0067 |
| shred_target/alias_mean_rank | 131.4350 | 120.0300 | 129.6117 |
| dup_shred/copy_direct_acc | 0.6933 | 0.6700 | 0.5583 |
| dup_shred/copy_probe_top1 | 0.5683 | 0.5150 | 0.4983 |
| dup_shred/copy_forced_choice | 0.9783 | 0.9750 | 0.9733 |
| resign_target/alias_direct | 0.0000 | 0.0000 | 0.5067 |
| revoke_alias/alias_unknown | 0.9733 | 0.9600 | 0.9800 |
| revoke_alias/sibling_readable | 0.0000 | 0.0000 | 0.5567 |
| revoke_alias/target_readable | 0.6533 | 0.5600 | 0.5767 |
| delete_target/alias_unknown | 0.9950 | 0.9850 | 0.9917 |
| delete_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |

Pre-registered criteria, identical to E-000020's:

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5767 | FAIL |
| alias_direct | >= 0.8 | 0.0000 | FAIL |
| dup_direct | >= 0.85 | 0.6550 | FAIL |
| alias_heldout_min | >= 0.5 | 0.0000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.0000 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.0000 | FAIL |
| duplicate_update/alias_old_object | >= 0.85 | 0.6550 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.6700 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.0000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0050 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5450 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.0000 | FAIL |
| probe_calibration_top1 | >= 0.2 | 0.4250 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.5150 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9600 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.0000 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5600 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
