# E-000012 — Frozen GPT-2 core: status-gated REVOKE

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps. REVOKE no longer removes routing: the revoked cell stays addressable and the status flag multiplies the gate, so it reads as ' unknown' exactly like an unsigned cell. Only DELETE removes a cell from routing. Motivation: E-000011 seed 0 — SHRED 100% but REVOKE by mask 76% ' unknown' (routing spreads over neighbouring keys once the cell is masked).

| claim group | supported |
|---|---|
| reading | **no** |
| heldout_paraphrases | **no** |
| update_rollback | **no** |
| deletion_behaviour | **no** |
| attacks_after_revoke | yes |
| attacks_after_shred_hard | yes |
| alternative_routes | **no** |
| interventions | yes |

| measure | mean over seeds | worst seed |
|---|---|---|
| prior_direct_acc | 0.4% | 0.2% |
| bank_masked_direct_acc | 0.0% | 0.0% |
| direct | 90.9% | 90.7% |
| template0_train/full_vocab_top1 | 86.2% | 81.2% |
| template1_train/direct | 100.0% | 99.9% |
| template2_heldout/direct | 42.8% | 40.3% |
| template3_heldout/direct | 81.2% | 78.1% |
| template4_heldout/direct | 52.3% | 48.9% |
| template5_heldout/direct | 91.3% | 89.6% |
| direct_heldout_mean | 66.9% | 65.1% |
| provenance_direct | 89.2% | 88.6% |
| hop2 | 91.1% | 90.0% |
| comparator/in_context_both_facts_hop2_acc | 40.3% | 39.0% |
| comparator/in_context_first_fact_only_hop2_acc | 0.9% | 0.0% |
| comparator/adapter_no_context_hop2_acc | 90.2% | 88.7% |
| broken1_unknown | 66.3% | 59.0% |
| broken2_unknown | 62.3% | 53.0% |
| update | 90.0% | 89.0% |
| rollback | 90.3% | 89.0% |
| revoke | 99.0% | 98.0% |
| shred | 99.0% | 98.0% |
| resign | 90.3% | 89.0% |
| update_heldout | 67.9% | 65.2% |
| revoke_heldout | 81.1% | 79.0% |
| revoke_heldout_min | 53.3% | 52.0% |
| shred_heldout | 81.1% | 79.0% |
| shred_heldout_min | 53.3% | 52.0% |
| locality | 99.5% | 99.3% |
| locality_targets_correct | 93.8% | 92.7% |
| alt_route/broken_route_changes | 100.0% | 100.0% |
| alt_route/other_route_survives | 90.7% | 86.0% |
| interventions/pool_correct_rate | 90.2% | 89.5% |

Attacks on 100 targets (mean over seeds):

| attack | active | after REVOKE | after SHRED (soft) | after SHRED (hard) |
|---|---|---|---|---|
| direct_unknown | 0.0900 | 0.9867 | 0.9867 | 0.9833 |
| direct_acc | 0.9000 | 0.0000 | 0.0000 | 0.0000 |
| candidate_other_entity | 0.0100 | 0.0133 | 0.0133 | 0.0167 |
| full_vocab_is_unknown_word | 0.0433 | 0.8300 | 0.8333 | 0.8267 |
| full_vocab_is_true_object | 0.8467 | 0.0000 | 0.0000 | 0.0000 |
| full_vocab_is_other_entity | 0.0067 | 0.0133 | 0.0133 | 0.0167 |
| full_vocab_equals_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_vocab_is_non_entity_token | 0.1033 | 0.1567 | 0.1533 | 0.1567 |
| heldout2_unknown | 0.4000 | 0.8000 | 0.8000 | 0.7967 |
| heldout4_unknown | 0.0500 | 0.5167 | 0.5167 | 0.5167 |
| forced_choice_win | 0.9933 | 0.4300 | 0.4400 | 0.4300 |
| true_obj_top1_among_entities | 0.9500 | 0.0033 | 0.0067 | 0.0033 |
| true_obj_mean_rank | 0.7300 | 136.6733 | 134.8200 | 136.6633 |
| probe_top1 | 0.8300 | 0.0067 | 0.0067 | 0.0067 |
| routing_mass_on_target | 0.8204 | 0.8204 | 0.8204 | 0.8204 |
| gate_on_target | 0.9986 | 0.0000 | 0.0016 | 0.0000 |
| payload_share | 0.8192 | 0.0000 | 0.0012 | 0.0000 |

Causal interventions on correctly answered 2-hop questions (mean / worst seed):

| intervention | mean | worst seed |
|---|---|---|
| localisation_hop1 | 100.0% | 100.0% |
| localisation_hop2 | 99.7% | 99.0% |
| disable_hop1_changes | 99.0% | 99.0% |
| disable_hop1_unknown | 59.3% | 50.0% |
| disable_hop2_changes | 100.0% | 100.0% |
| disable_hop2_unknown | 51.0% | 43.0% |
| disable_random_unchanged | 100.0% | 100.0% |
| swap_hop2 | 99.7% | 99.0% |
| replace_hop2 | 99.7% | 99.0% |

Pre-registered criteria (worst seed; identical to E-000011):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| prior_direct_acc | <= 0.05 | 0.0060 | PASS |
| bank_masked_direct_acc | <= 0.05 | 0.0000 | PASS |
| direct | >= 0.95 | 0.9070 | FAIL |
| template1_train/direct | >= 0.95 | 0.9990 | PASS |
| template2_heldout/direct | >= 0.8 | 0.4030 | FAIL |
| template3_heldout/direct | >= 0.8 | 0.7810 | FAIL |
| template4_heldout/direct | >= 0.7 | 0.4890 | FAIL |
| template5_heldout/direct | >= 0.7 | 0.8960 | PASS |
| update | >= 0.95 | 0.8900 | FAIL |
| rollback | >= 0.95 | 0.8900 | FAIL |
| resign | >= 0.95 | 0.8900 | FAIL |
| revoke | >= 0.95 | 0.9800 | PASS |
| shred | >= 0.9 | 0.9800 | PASS |
| broken1_unknown | >= 0.9 | 0.5900 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.5200 | FAIL |
| shred_heldout_min | >= 0.85 | 0.5200 | FAIL |
| locality | >= 0.98 | 0.9929 | PASS |
| restored/direct_acc | >= 0.95 | 0.8800 | FAIL |
| revoke/probe_top1 | <= 0.05 | 0.0100 | PASS |
| revoke/forced_choice_win | <= 0.6 | 0.4700 | PASS |
| shred_hard/probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_hard/forced_choice_win | <= 0.6 | 0.4700 | PASS |
| shred_hard/true_obj_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| shred_hard/payload_share | <= 0.05 | 0.0000 | PASS |
| shred_hard/direct_unknown | >= 0.9 | 0.9700 | PASS |
| alt_route/broken_route_changes | >= 0.95 | 1.0000 | PASS |
| alt_route/other_route_survives | >= 0.95 | 0.8600 | FAIL |
| interventions/localisation_hop1 | >= 0.9 | 1.0000 | PASS |
| interventions/localisation_hop2 | >= 0.9 | 0.9900 | PASS |
| interventions/disable_hop1_changes | >= 0.95 | 0.9900 | PASS |
| interventions/disable_hop2_changes | >= 0.95 | 1.0000 | PASS |
| interventions/disable_random_unchanged | >= 0.95 | 1.0000 | PASS |
| interventions/swap_hop2 | >= 0.9 | 0.9900 | PASS |
| interventions/replace_hop2 | >= 0.9 | 0.9900 | PASS |

Lenient criteria (secondary):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.9 | 0.9070 | PASS |
| template1_train/direct | >= 0.9 | 0.9990 | PASS |
| revoke | >= 0.9 | 0.9800 | PASS |
| shred | >= 0.85 | 0.9800 | PASS |
| broken1_unknown | >= 0.85 | 0.5900 | FAIL |
