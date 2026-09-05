# E-000015 — Explicit symlink cells: several access keys, one knowledge object

Evidence level: **E4** (synthetic system). Deletion level targeted F4, recorded **F3**. Seeds: [0, 1, 2]; 4000 steps.

Two stores hold the SAME world with the SAME ground truth and are read by the SAME trained model: in the symlink arm the 200 alias keys are LINK cells pointing at 100 target cells, in the duplication arm the same keys are ordinary fact cells holding a copy of the object. Every sharing claim is the difference between the arms.

Symlink versus duplication (mean over seeds), the two arms holding identical ground truth:

| what is measured | symlink arm | duplication arm |
|---|---|---|
| one UPDATE on the shared object reaches every access path | 100.0% | 0.0% |
| one SHRED on the shared object leaves nothing readable | 100.0% | 0.0% |
| object recoverable by probe after that one operation | 0.7% | 87.3% |
| operations needed to reach every access path | 1 | 3 |

| claim group | supported |
|---|---|
| reading | yes |
| provenance_through_the_alias | yes |
| dereference_is_what_reads_an_alias | yes |
| one_update_reaches_every_path | yes |
| one_shred_deletes_every_path | yes |
| attacks_through_every_alias | yes |
| alias_lifecycle | **no** |
| capability_limit_of_one_slot | yes |
| no_regression_without_links | yes |

| measure | mean over seeds | worst seed | best seed |
|---|---|---|---|
| direct | 1.0000 | 1.0000 | 1.0000 |
| alias_direct | 1.0000 | 1.0000 | 1.0000 |
| dup_direct | 1.0000 | 1.0000 | 1.0000 |
| hop2 | 1.0000 | 1.0000 | 1.0000 |
| broken1_unknown | 1.0000 | 1.0000 | 1.0000 |
| provenance_direct | 1.0000 | 1.0000 | 1.0000 |
| alias_provenance_pair | 1.0000 | 1.0000 | 1.0000 |
| alias_provenance_len2 | 1.0000 | 1.0000 | 1.0000 |
| deref_disabled/alias_direct | 0.0000 | 0.0000 | 0.0000 |
| deref_disabled/direct | 1.0000 | 1.0000 | 1.0000 |
| shared_update/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| shared_update/target_new_object | 1.0000 | 1.0000 | 1.0000 |
| rollback/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| probe_calibration_top1 | 0.8867 | 0.8400 | 0.9467 |
| shred_target/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0067 | 0.0000 | 0.0100 |
| shred_target/alias_forced_choice | 0.5033 | 0.5000 | 0.5100 |
| shred_target/alias_top1_among_entities | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_mean_rank | 127.4250 | 125.9900 | 130.2250 |
| dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 1.0000 |
| dup_shred/copy_probe_top1 | 0.8733 | 0.8000 | 0.9600 |
| dup_shred/copy_forced_choice | 1.0000 | 1.0000 | 1.0000 |
| resign_target/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/sibling_readable | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/target_readable | 1.0000 | 1.0000 | 1.0000 |
| shred_alias/alias_unknown | 0.9700 | 0.9300 | 1.0000 |
| shred_alias/target_readable | 1.0000 | 1.0000 | 1.0000 |
| relink/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| relink/sibling_unchanged | 1.0000 | 1.0000 | 1.0000 |
| relink_rollback/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| delete_target/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| delete_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| refcount_before_delete | 2.0000 | 2.0000 | 2.0000 |
| chain2/answer_acc | 0.0000 | 0.0000 | 0.0000 |
| chain2/unknown | 1.0000 | 1.0000 | 1.0000 |
| chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| regression/direct | 1.0000 | 1.0000 | 1.0000 |
| regression/hop2 | 1.0000 | 1.0000 | 1.0000 |
| regression/hop3 | 0.9967 | 0.9933 | 1.0000 |
| regression/reverse | 1.0000 | 1.0000 | 1.0000 |
| regression/provenance | 1.0000 | 1.0000 | 1.0000 |
| regression/broken2_unknown | 1.0000 | 1.0000 | 1.0000 |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 1.0000 | 1.0000 | 1800 | 0.9980 | 1.0000 |
| alias_direct | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| dup_direct | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| hop2 | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| broken1_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| provenance_direct | 1.0000 | 1.0000 | 1800 | 0.9980 | 1.0000 |
| alias_provenance_pair | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| shared_update/alias_new_object | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_probe_top1 | 0.0067 | 0.0100 | 600 | 0.0018 | 0.0170 |
| shred_target/alias_top1_among_entities | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| revoke_alias/alias_unknown | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| shred_alias/alias_unknown | 0.9700 | 0.9300 | 300 | 0.9438 | 0.9862 |
| relink/alias_new_object | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| delete_target/alias_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| chain2/answer_acc | 0.0000 | 0.0000 | 300 | 0.0000 | 0.0122 |
| regression/direct | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| regression/hop2 | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| regression/hop3 | 0.9967 | 0.9933 | 900 | 0.9903 | 0.9993 |
| regression/reverse | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.98 | 1.0000 | PASS |
| alias_direct | >= 0.95 | 1.0000 | PASS |
| dup_direct | >= 0.98 | 1.0000 | PASS |
| hop2 | >= 0.95 | 1.0000 | PASS |
| broken1_unknown | >= 0.95 | 1.0000 | PASS |
| provenance_direct | >= 0.95 | 1.0000 | PASS |
| alias_provenance_pair | >= 0.9 | 1.0000 | PASS |
| deref_disabled/alias_direct | <= 0.2 | 0.0000 | PASS |
| deref_disabled/direct | >= 0.9 | 1.0000 | PASS |
| shared_update/alias_new_object | >= 0.95 | 1.0000 | PASS |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| shared_update/target_new_object | >= 0.95 | 1.0000 | PASS |
| rollback/alias_direct | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_unknown | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.95 | 1.0000 | PASS |
| resign_target/alias_direct | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5100 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0000 | PASS |
| revoke_alias/alias_unknown | >= 0.95 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.95 | 1.0000 | PASS |
| revoke_alias/target_readable | >= 0.95 | 1.0000 | PASS |
| shred_alias/alias_unknown | >= 0.95 | 0.9300 | FAIL |
| shred_alias/target_readable | >= 0.95 | 1.0000 | PASS |
| relink/alias_new_object | >= 0.9 | 1.0000 | PASS |
| relink_rollback/alias_direct | >= 0.9 | 1.0000 | PASS |
| delete_target/alias_unknown | >= 0.95 | 1.0000 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| chain2/answer_acc | <= 0.2 | 0.0000 | PASS |
| regression/direct | >= 0.98 | 1.0000 | PASS |
| regression/hop2 | >= 0.95 | 1.0000 | PASS |
| regression/hop3 | >= 0.9 | 0.9933 | PASS |
| regression/reverse | >= 0.95 | 1.0000 | PASS |
| regression/provenance | >= 0.95 | 1.0000 | PASS |
| regression/broken2_unknown | >= 0.95 | 1.0000 | PASS |

Two-slot control (single seed): {'seed': 0, 'chain2/answer_acc': 0.0, 'chain2/depth1_acc': 1.0, 'alias_direct': 1.0, 'direct': 1.0, 'checkpoint_sha256': '34bca206b9d9c8a94d9e16ea82e906f4990e60d87a55aaec3f5e1ff99e05eca7'}

By construction: the store decides which payload a row carries (an alias row carries its target's KEY, a fact row its object), exactly as it decides the marker; the bank never exports the target's payload, its status, its signature or the chain depth; that ONE update or ONE shred on a shared object reaches every alias is a property of the store; what is measured is whether the trained model reports it, and whether the SAME model reports the duplication arm (where it does not) correctly; a deleted target keeps its key as a tombstone, so a dangling pointer stays a pointer and the miss is not pre-resolved by the control plane.

Learned: following a pointer: the dereference slot's query comes from the value just read, not from the question, and the model is never told that a value is a pointer; keeping a value that was not a pointer (the passthrough column) so that fact cells still read correctly, measured as deref_disabled/direct versus deref_disabled/alias_direct; answering UNKNOWN for a dangling pointer, for a revoked or shredded alias and for a shredded target; provenance across the indirection: the routing names the alias AND the cell it points at.

Not claimed: LLM scale (the frozen-GPT-2 chain does not yet carry links); chains deeper than the number of dereference slots; reference counting as a garbage-collection policy.
