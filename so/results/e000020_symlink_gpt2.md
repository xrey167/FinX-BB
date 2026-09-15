# E-000020 — Shared knowledge objects in a frozen GPT-2

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps.

The same world is written twice and read by the same trained adapter from natural-language prompts: in the symlink arm the alias keys are LINK cells over shared targets, in the duplication arm they are ordinary fact cells holding a copy. Every sharing claim is the difference between the arms.

| what is measured | symlink arm | duplication arm |
|---|---|---|
| one UPDATE on the shared object reaches every access path | 51.8% | 0.0% |
| after one SHRED the object is still readable | 0.5% | 55.8% |
| after one SHRED a probe recovers the object | 0.3% | 49.8% |
| operations needed to reach every access path | 1 | 3 |

| claim group | supported |
|---|---|
| reading_through_an_alias | **no** |
| one_update_reaches_every_path | **no** |
| one_shred_deletes_every_path | **no** |
| attacks_through_every_alias | yes |
| attack_validity | yes |
| alias_lifecycle | **no** |

| measure | mean over seeds | worst seed |
|---|---|---|
| direct | 0.5700 | 0.5633 |
| alias_direct | 0.5067 | 0.5000 |
| dup_direct | 0.5483 | 0.5100 |
| alias_heldout_min | 0.3700 | 0.3550 |
| probe_calibration_top1 | 0.4806 | 0.4333 |
| active/alias_probe_top1 | 0.3883 | 0.3600 |
| active/alias_forced_choice | 0.8950 | 0.8800 |
| shared_update/alias_new_object | 0.5183 | 0.5050 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| rollback/alias_direct | 0.5067 | 0.5000 |
| shred_target/alias_unknown | 0.9950 | 0.9850 |
| shred_target/alias_true_object | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0033 | 0.0000 |
| shred_target/alias_forced_choice | 0.5100 | 0.4750 |
| shred_target/alias_top1_among_entities | 0.0067 | 0.0000 |
| shred_target/alias_mean_rank | 129.6117 | 120.3350 |
| dup_shred/copy_direct_acc | 0.5583 | 0.5400 |
| dup_shred/copy_probe_top1 | 0.4983 | 0.4450 |
| dup_shred/copy_forced_choice | 0.9733 | 0.9700 |
| resign_target/alias_direct | 0.5067 | 0.5000 |
| revoke_alias/alias_unknown | 0.9800 | 0.9500 |
| revoke_alias/sibling_readable | 0.5567 | 0.4900 |
| revoke_alias/target_readable | 0.5767 | 0.5200 |
| delete_target/alias_unknown | 0.9917 | 0.9800 |
| delete_target/alias_true_object | 0.0000 | 0.0000 |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 0.5700 | 0.5633 | 900 | 0.5369 | 0.6026 |
| alias_direct | 0.5067 | 0.5000 | 600 | 0.4659 | 0.5474 |
| dup_direct | 0.5483 | 0.5100 | 600 | 0.5075 | 0.5887 |
| shared_update/alias_new_object | 0.5183 | 0.5050 | 600 | 0.4775 | 0.5590 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_unknown | 0.9950 | 0.9850 | 600 | 0.9855 | 0.9990 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_probe_top1 | 0.0033 | 0.0000 | 600 | 0.0004 | 0.0120 |
| dup_shred/copy_direct_acc | 0.5583 | 0.5400 | 600 | 0.5176 | 0.5985 |
| revoke_alias/alias_unknown | 0.9800 | 0.9500 | 300 | 0.9570 | 0.9926 |
| delete_target/alias_unknown | 0.9917 | 0.9800 | 600 | 0.9807 | 0.9973 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5633 | FAIL |
| alias_direct | >= 0.8 | 0.5000 | FAIL |
| dup_direct | >= 0.85 | 0.5100 | FAIL |
| alias_heldout_min | >= 0.5 | 0.3550 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.5050 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.5400 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5300 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.3600 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.4333 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9500 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.4900 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5200 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9800 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

By construction: the store decides which payload a row carries; the bank never exports the target's payload, status or signature, and the model is never told that a value is a pointer; that one operation on a shared object reaches every alias is a property of the store; what is measured is whether the frozen model reports it, and whether the SAME model reports the duplication arm, where it does not, correctly.

Learned: following a pointer inside a frozen pretrained transformer: the dereference query comes from the value just read, and the passthrough column keeps a value that was not a pointer; answering unknown for a dangling pointer after DELETE and for a revoked alias.

Not claimed: chains deeper than one dereference; multi-token entities; anything above 124M parameters.
