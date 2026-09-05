# E-000013 — Frozen GPT-2 core: prior conflict (override while ACTIVE, fallback to the pretrained distribution after REVOKE / SHRED)

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps.

50 real countries whose capitals GPT-2 small knows receive counterfactual capital cells; 950 prior-free filler facts. The adapter runs in fallback-to-prior mode: an unsigned or revoked cell injects nothing, the null read is a fixed zero.

| claim group | supported |
|---|---|
| copy_bound_by_construction | yes |
| reading_prior_free | **no** |
| override | **no** |
| attack_validity | yes |
| fallback_after_revoke_by_construction | **no** |
| fallback_after_shred_soft | **no** |
| fallback_after_shred_hard | **no** |
| no_key_no_injection | **no** |
| retention_under_deletion | **no** |
| locality_restore | **no** |

| measure | mean over seeds | min | max |
|---|---|---|---|
| prior/restricted_top1 | 0.9600 | 0.9600 | 0.9600 |
| prior/true_capital_prob | 0.0377 | 0.0377 | 0.0377 |
| prior/counterfactual_top1_pooled | 0.0000 | 0.0000 | 0.0000 |
| prior/forced_choice_win | 0.6533 | 0.6000 | 0.7200 |
| prior/probe_top1 | 0.0000 | 0.0000 | 0.0000 |
| prior/counterfactual_mean_rank | 117.2800 | 96.4800 | 133.0400 |
| probe_calibration_top1 | 0.6281 | 0.6053 | 0.6579 |
| masked/kl_to_base | 0.0000 | 0.0000 | 0.0000 |
| direct | 0.9112 | 0.9095 | 0.9137 |
| template1_train/direct | 0.9828 | 0.9811 | 0.9853 |
| direct_heldout_min | 0.1260 | 0.0905 | 0.1463 |
| hop2 | 0.9350 | 0.9150 | 0.9500 |
| provenance_direct | 0.5733 | 0.5653 | 0.5789 |
| broken1/kl_to_base | 0.4570 | 0.3615 | 0.6412 |
| broken1/routing_mass_on_null | 0.5376 | 0.5189 | 0.5687 |
| generic/kl_to_base | 2.2692 | 2.1268 | 2.3482 |
| generic/kl_to_base_worst_prompt | 4.2206 | 3.9478 | 4.5549 |
| generic/routing_mass_on_null | 0.2742 | 0.2236 | 0.3179 |
| override/direct | 1.0000 | 1.0000 | 1.0000 |
| override/full_vocab_top1 | 1.0000 | 1.0000 | 1.0000 |
| override_heldout_min | 0.0000 | 0.0000 | 0.0000 |
| override/true_capital_restricted_top1 | 0.0000 | 0.0000 | 0.0000 |
| agree/direct | 1.0000 | 1.0000 | 1.0000 |
| rollback/direct | 1.0000 | 1.0000 | 1.0000 |
| active/probe_top1 | 0.8733 | 0.7800 | 0.9600 |
| active/forced_choice_excess | 0.3467 | 0.2800 | 0.4000 |
| active/counterfactual_top1_excess | 0.5000 | 0.5000 | 0.5000 |
| active/kl_to_base | 6.7429 | 6.4104 | 7.3498 |
| active/routing_mass_on_target | 0.8706 | 0.8554 | 0.8877 |
| active/gate_on_target | 0.9977 | 0.9975 | 0.9980 |
| revoke/kl_to_base | 0.0004 | 0.0003 | 0.0005 |
| revoke/top1_matches_base_pooled | 0.7617 | 0.7300 | 0.8200 |
| revoke/restricted_matches_base | 1.0000 | 1.0000 | 1.0000 |
| revoke/true_capital_restricted_top1 | 0.9600 | 0.9600 | 0.9600 |
| revoke/counterfactual_top1_pooled | 0.0000 | 0.0000 | 0.0000 |
| revoke/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/probe_top1 | 0.0000 | 0.0000 | 0.0000 |
| revoke/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/forced_choice_win | 0.6533 | 0.6000 | 0.7200 |
| revoke/forced_choice_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/heldout_kl_max | 3.7046 | 3.0526 | 4.4722 |
| revoke/routing_mass_on_target | 0.8709 | 0.8562 | 0.8878 |
| revoke/gate_on_target | 0.0000 | 0.0000 | 0.0000 |
| revoke/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_soft/kl_to_base | 0.0008 | 0.0003 | 0.0018 |
| shred_soft/top1_matches_base_pooled | 0.7600 | 0.7300 | 0.8200 |
| shred_soft/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_soft/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_soft/forced_choice_excess | 0.0067 | 0.0000 | 0.0200 |
| shred_soft/heldout_kl_max | 3.7044 | 3.0526 | 4.4722 |
| shred_soft/injection_rms_share | 0.0012 | 0.0007 | 0.0019 |
| shred_soft/gate_on_unsigned_cells | 0.0014 | 0.0008 | 0.0023 |
| shred_soft/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_hard/kl_to_base | 0.0004 | 0.0003 | 0.0005 |
| shred_hard/top1_matches_base_pooled | 0.7617 | 0.7300 | 0.8200 |
| shred_hard/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/forced_choice_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/heldout_kl_max | 3.7153 | 3.0574 | 4.4867 |
| shred_hard/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_hard/filler_kl_to_active | 0.0000 | 0.0000 | 0.0000 |
| restored/direct | 1.0000 | 1.0000 | 1.0000 |
| resigned/direct | 1.0000 | 1.0000 | 1.0000 |
| locality | 0.9683 | 0.9625 | 0.9738 |
| locality_counterfactual_unchanged | 1.0000 | 1.0000 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| masked/kl_to_base | <= 0.05 | 0.0000 | PASS |
| masked/top1_matches_base | >= 0.95 | 1.0000 | PASS |
| direct | >= 0.95 | 0.9095 | FAIL |
| template1_train/direct | >= 0.95 | 0.9811 | PASS |
| direct_heldout_min | >= 0.7 | 0.0905 | FAIL |
| override/direct | >= 0.9 | 1.0000 | PASS |
| override/full_vocab_top1 | >= 0.8 | 1.0000 | PASS |
| override_heldout_min | >= 0.7 | 0.0000 | FAIL |
| agree/direct | >= 0.95 | 1.0000 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.6053 | PASS |
| active/probe_top1 | >= 0.25 | 0.7800 | PASS |
| active/counterfactual_top1_excess | >= 0.5 | 0.5000 | PASS |
| active/forced_choice_excess | >= 0.1 | 0.2800 | PASS |
| revoke/kl_to_base | <= 0.05 | 0.0005 | PASS |
| revoke/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| revoke/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| revoke/probe_excess | <= 0.05 | 0.0000 | PASS |
| revoke/forced_choice_excess | <= 0.05 | 0.0000 | PASS |
| revoke/heldout_kl_max | <= 0.1 | 4.4722 | FAIL |
| revoke/routing_mass_on_target | >= 0.9 | 0.8562 | FAIL |
| shred_soft/kl_to_base | <= 0.05 | 0.0018 | PASS |
| shred_soft/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| shred_soft/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| shred_soft/probe_excess | <= 0.05 | 0.0000 | PASS |
| shred_soft/forced_choice_excess | <= 0.05 | 0.0200 | PASS |
| shred_soft/heldout_kl_max | <= 0.1 | 4.4722 | FAIL |
| shred_hard/kl_to_base | <= 0.05 | 0.0005 | PASS |
| shred_hard/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| shred_hard/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/probe_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/forced_choice_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/heldout_kl_max | <= 0.1 | 4.4867 | FAIL |
| broken1/kl_to_base | <= 0.05 | 0.6412 | FAIL |
| generic/kl_to_base | <= 0.05 | 2.3482 | FAIL |
| generic/kl_to_base_worst_prompt | <= 0.1 | 4.5549 | FAIL |
| revoke/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_soft/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_hard/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_hard/filler_kl_to_active | <= 0.05 | 0.0000 | PASS |
| locality | >= 0.98 | 0.9625 | FAIL |
| restored/direct | >= 0.9 | 1.0000 | PASS |
| resigned/direct | >= 0.9 | 1.0000 | PASS |
| rollback/direct | >= 0.9 | 1.0000 | PASS |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| prior/counterfactual_top1_pooled | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| template1_train/direct | 0.9828 | 0.9811 | 2850 | 0.9773 | 0.9873 |
| direct_heldout_min | 0.1260 | 0.0905 | 2850 | 0.1140 | 0.1387 |
| hop2 | 0.9350 | 0.9150 | 600 | 0.9122 | 0.9534 |
| override/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| agree/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| rollback/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| active/probe_top1 | 0.8733 | 0.7800 | 150 | 0.8093 | 0.9220 |
| revoke/top1_matches_base_pooled | 0.7617 | 0.7300 | 600 | 0.7255 | 0.7952 |
| revoke/counterfactual_top1_pooled | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| revoke/probe_top1 | 0.0000 | 0.0000 | 150 | 0.0000 | 0.0243 |
| revoke/forced_choice_win | 0.6533 | 0.6000 | 150 | 0.5714 | 0.7291 |
| revoke/filler_direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| shred_soft/top1_matches_base_pooled | 0.7600 | 0.7300 | 600 | 0.7238 | 0.7937 |
| shred_hard/top1_matches_base_pooled | 0.7617 | 0.7300 | 600 | 0.7255 | 0.7952 |
| shred_hard/filler_direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| restored/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| resigned/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| locality | 0.9683 | 0.9625 | 2400 | 0.9605 | 0.9750 |

Sample sizes: 50 counterfactual items per seed cannot resolve a 0.05 bar on their own; the gating rate criteria are therefore pooled over item x template (200 per seed, 600 over three seeds) and the exact binomial intervals below are reported for the pooled counts.

Attack convention: Every attack bar is a PAIRED EXCESS over the frozen model itself, because the counterfactual object is a real capital token the pretrained prior already favours: an absolute forced-choice or probe threshold would measure GPT-2's prior, not leakage from the cell. The floors are recorded as prior/forced_choice_win, prior/probe_top1 and prior/counterfactual_top1_pooled, measured on the same rows with the same distractor draws.

Validity condition: attack_validity: with the cell ACTIVE the same attacks must succeed. If they do not, their failure after deletion is uninformative and the record reports F1 regardless of the fallback groups.

By construction: copy_bound_by_construction: the adapter acts only through the injection; with every cell masked the null read is a fixed zero, so the base distribution is returned exactly (recorded, not learned); fallback_after_revoke_by_construction: with status_gated the status flag multiplies the gate, so a revoked cell's value is exactly zero and the injection vanishes. Exact equality to the base model after REVOKE is therefore arithmetic, not a learned behaviour; the LEARNED residue is that the routing does not spill onto neighbouring ACTIVE cells (kl_to_base, heldout_kl_max) while the revoked cell itself stays addressed (routing_mass_on_target). This group is recorded and does NOT grant a deletion level.; the pretrained fact is never deleted: the weights are frozen; what is measured is that the model answers with it again after REVOKE / SHRED.

Not claimed: unlearning of pretrained facts; LLM scale; multi-token entities.
