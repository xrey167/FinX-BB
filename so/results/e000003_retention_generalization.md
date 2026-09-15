# E-000003 — Retention and generalisation of deletion

Evidence level: **E4** (Controlled neural-network evidence); deletion level **F3** (functional forgetting generalising over paraphrases, multi-hop and reverse access). Seeds: [0, 1, 2, 3, 4]

| measure | mean | min | max |
|---|---|---|---|
| before/target_para_acc | 100.0% | 100.0% | 100.0% |
| before/target_para_unknown | 0.0% | 0.0% | 0.0% |
| before/target_hop2_unknown | 0.0% | 0.0% | 0.0% |
| before/target_hop2_ref_agree | 100.0% | 100.0% | 100.0% |
| before/target_rev_unknown | 0.0% | 0.0% | 0.0% |
| before/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| before/control_para_acc | 100.0% | 100.0% | 100.0% |
| before/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| before/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| revoke/target_para_acc | 0.0% | 0.0% | 0.0% |
| revoke/target_para_unknown | 100.0% | 100.0% | 100.0% |
| revoke/target_hop2_unknown | 100.0% | 100.0% | 100.0% |
| revoke/target_hop2_ref_agree | 100.0% | 100.0% | 100.0% |
| revoke/target_rev_unknown | 100.0% | 100.0% | 100.0% |
| revoke/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| revoke/control_para_acc | 100.0% | 100.0% | 100.0% |
| revoke/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| revoke/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| shred/target_para_acc | 0.0% | 0.0% | 0.0% |
| shred/target_para_unknown | 100.0% | 100.0% | 100.0% |
| shred/target_hop2_unknown | 99.6% | 98.0% | 100.0% |
| shred/target_hop2_ref_agree | 99.6% | 98.0% | 100.0% |
| shred/target_rev_unknown | 95.8% | 90.5% | 100.0% |
| shred/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| shred/control_para_acc | 100.0% | 100.0% | 100.0% |
| shred/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| shred/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| update/target_para_new_obj_acc | 100.0% | 100.0% | 100.0% |
| update/target_para_old_obj_rate | 0.0% | 0.0% | 0.0% |
| update/target_rev_old_obj_ref_agree | 100.0% | 100.0% | 100.0% |
| update/control_para_acc | 100.0% | 100.0% | 100.0% |
| rollback/target_para_acc | 100.0% | 100.0% | 100.0% |
| after_all/identical_to_before | 100.0% | 100.0% | 100.0% |

Pattern required by the ledger (section 16): target high → low, control high → high. 'target_para_unknown' after revoke/shred is the deletion; 'control_para_acc', 'unrelated_para_acc' and 'bypass_hop2_acc' are the retention side ('general_fresh_world_acc' is a by-construction sanity row). Sample sizes per seed: targets 50 x 2 paraphrases, controls 50 x 2, unrelated up to 50 x 2, bypass 300, reverse only where the subject is unique (see n_target_rev).

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| before/target_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/target_para_unknown | >= 0.98 | 1.0000 | PASS |
| revoke/target_hop2_unknown | >= 0.98 | 1.0000 | PASS |
| revoke/control_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/unrelated_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/bypass_hop2_acc | >= 0.98 | 1.0000 | PASS |
| shred/target_para_unknown | >= 0.95 | 1.0000 | PASS |
| shred/control_para_acc | >= 0.98 | 1.0000 | PASS |
| update/target_para_new_obj_acc | >= 0.98 | 1.0000 | PASS |
| update/target_para_old_obj_rate | <= 0.02 | 0.0000 | PASS |
| rollback/target_para_acc | >= 0.98 | 1.0000 | PASS |

REVOKE removes routing by mask (F1), so its effect on every access path (paraphrase, multi-hop, reverse) follows from canonical addressing of one cell; what is learned is that the model answers UNKNOWN instead of using another cell. SHRED leaves the cell routable: refusing it on every path is learned (F3). 'general_fresh_world_acc' uses a separate store and cannot change — it is a sanity row, not evidence of retention.
