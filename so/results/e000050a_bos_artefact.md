# E-000050-A — the paraphrase gap is the position-0 token: a BOS at inference, no training

E-000017-B, E-000013 and E-000020's recorded checkpoints are evaluated as recorded; the only change is a prefix on the prompt at inference.

Seeds [0, 1, 2]; families ['e17', 'e13', 'e20']; 100 targets per seed for the decomposition. Worst seed everywhere.

**Reading: artefact, for reading but not for deletion.**

| claim group | supported |
|---|---|
| record_reproduced | yes |
| bos_restores_heldout_addressing | yes |
| any_token_does_it | yes |
| controls_hold | **no** |
| deletion_follows | **no** |
| no_new_collateral | **no** |
| e13_override_and_revert_on_heldout | **no** |
| e20_lifecycle_at_template0 | yes |

## E-000017-B, per template (read / route_hit, worst seed over seeds)

The subject column is the token index of the subject name read off the tokenizer: none: [0, 4, 0, 4, 5, 6, 0, 4, 0, 2, 5, 0]; bos: [1, 5, 1, 5, 6, 7, 1, 5, 1, 3, 6, 1]; text: [5, 8, 5, 8, 9, 10, 5, 8, 5, 6, 9, 5]; newline: [1, 5, 1, 5, 6, 7, 1, 5, 1, 3, 6, 1]; word: [1, 5, 1, 5, 6, 7, 1, 5, 1, 3, 6, 1]; space: [1, 4, 1, 4, 5, 6, 1, 4, 1, 2, 5, 1]; bos_sp: [2, 5, 2, 5, 6, 7, 2, 5, 2, 3, 6, 2]; text_nosp: [4, 8, 4, 8, 9, 10, 4, 8, 4, 6, 9, 4]

| t | kind | subject | none: read / route | bos: read / route | text: read / route | newline: read / route | word: read / route | space: read / route | bos_sp: read / route | text_nosp: read / route | oracle_read (none) | oracle_read (bos) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | train | initial | 0.76 / 0.87 | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 | 1.00 |
| 1 | train | medial | 1.00 / 1.00 | 0.82 / 0.78 | 0.97 / 0.96 | 0.97 / 0.97 | 0.91 / 0.92 | 0.98 / 0.98 | 0.67 / 0.63 | 0.97 / 0.96 | 1.00 | 0.96 |
| 2 | train | initial | 0.71 / 0.89 | 0.95 / 0.97 | 0.97 / 0.98 | 0.97 / 0.98 | 0.97 / 0.98 | 0.98 / 0.99 | 0.99 / 0.99 | 0.95 / 0.96 | 1.00 | 1.00 |
| 3 | train | medial | 1.00 / 1.00 | 0.99 / 0.96 | 0.99 / 0.95 | 1.00 / 0.99 | 1.00 / 0.96 | 1.00 / 1.00 | 0.99 / 0.95 | 0.99 / 0.95 | 1.00 | 1.00 |
| 4 | train | medial | 1.00 / 1.00 | 1.00 / 0.99 | 0.97 / 0.91 | 1.00 / 0.99 | 1.00 / 0.98 | 1.00 / 0.99 | 1.00 / 0.97 | 0.97 / 0.91 | 1.00 | 1.00 |
| 5 | train | medial | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.99 | 1.00 | 1.00 |
| 6 | train | initial | 0.75 / 0.87 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 1.00 | 0.99 / 0.99 | 0.99 | 1.00 |
| 7 | train | medial | 0.99 / 1.00 | 0.98 / 0.98 | 0.99 / 0.96 | 0.99 / 0.99 | 0.99 / 1.00 | 0.98 / 1.00 | 0.96 / 0.95 | 0.99 / 0.96 | 0.99 | 0.99 |
| 8 | heldout | initial | 0.49 / 0.66 | 0.97 / 0.98 | 1.00 / 0.99 | 0.98 / 0.98 | 0.98 / 0.98 | 0.97 / 0.98 | 0.97 / 0.95 | 1.00 / 1.00 | 0.93 | 0.99 |
| 9 | heldout | medial | 0.95 / 0.94 | 0.70 / 0.64 | 0.69 / 0.45 | 0.70 / 0.69 | 0.75 / 0.70 | 0.94 / 0.95 | 0.67 / 0.62 | 0.69 / 0.45 | 0.99 | 0.88 |
| 10 | heldout | medial | 1.00 / 1.00 | 0.99 / 0.99 | 0.99 / 0.98 | 1.00 / 1.00 | 0.99 / 0.98 | 0.99 / 1.00 | 0.99 / 0.99 | 0.99 / 0.98 | 1.00 | 1.00 |
| 11 | heldout | initial | 0.37 / 0.54 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.99 | 0.99 / 0.99 | 0.98 / 0.98 | 0.99 / 0.98 | 0.99 / 0.99 | 0.91 | 1.00 |

## E-000017-B, the battery

| measure (worst seed) | none | bos | text | newline | word | space | bos_sp | text_nosp |
|---|---|---|---|---|---|---|---|---|
| heldout/active_correct | 0.7288 | 0.9175 | 0.9187 | 0.9313 | 0.9300 | 0.9800 | 0.9100 | 0.9187 |
| train/active_correct | 0.9119 | 0.9719 | 0.9894 | 0.9913 | 0.9850 | 0.9938 | 0.9587 | 0.9875 |
| heldout/read_min | 0.3700 | 0.7000 | 0.6900 | 0.7000 | 0.7500 | 0.9400 | 0.6700 | 0.6900 |
| heldout/route_hit_min | 0.5400 | 0.6400 | 0.4500 | 0.6900 | 0.7000 | 0.9500 | 0.6200 | 0.4500 |
| heldout_initial/read_min | 0.3700 | 0.9700 | 0.9900 | 0.9800 | 0.9800 | 0.9700 | 0.9700 | 0.9900 |
| heldout_initial/route_hit_min | 0.5400 | 0.9800 | 0.9900 | 0.9800 | 0.9800 | 0.9800 | 0.9500 | 0.9900 |
| heldout_initial/oracle_read_min | 0.9100 | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9900 | 0.9800 | 0.9900 |
| heldout_medial/read_min | 0.9500 | 0.7000 | 0.6900 | 0.7000 | 0.7500 | 0.9400 | 0.6700 | 0.6900 |
| heldout_medial/route_hit_min | 0.9400 | 0.6400 | 0.4500 | 0.6900 | 0.7000 | 0.9500 | 0.6200 | 0.4500 |
| train_initial/read_min | 0.7100 | 0.9500 | 0.9700 | 0.9700 | 0.9700 | 0.9800 | 0.9900 | 0.9500 |
| train_medial/read_min | 0.9900 | 0.8200 | 0.9700 | 0.9700 | 0.9100 | 0.9800 | 0.6700 | 0.9700 |
| medial_abs_change_max | - | 0.2500 | 0.2700 | 0.2500 | 0.2000 | 0.0200 | 0.3300 | 0.2700 |
| train_medial_abs_change_max | - | 0.1800 | 0.0300 | 0.0300 | 0.0900 | 0.0200 | 0.3300 | 0.0300 |
| train_read_change_min | - | -0.1800 | -0.0300 | -0.0300 | -0.0900 | -0.0200 | -0.3300 | -0.0300 |
| initial_read_gain_mean | - | 0.3200 | 0.3240 | 0.3220 | 0.3200 | 0.3240 | 0.3260 | 0.3180 |
| heldout/routing_share | 0.8818 | 0.5217 | 0.1818 | 0.6957 | 0.7500 | 0.5714 | 0.4167 | 0.1786 |
| shred_heldout_min | 0.8650 | 0.8400 | 0.9400 | 0.8650 | 0.8100 | 0.9750 | 0.8200 | 0.9400 |
| revoke_heldout_min | 0.8650 | 0.8400 | 0.9400 | 0.8650 | 0.8100 | 0.9750 | 0.8200 | 0.9400 |
| shred_heldout_initial_min | 0.8650 | 0.9900 | 0.9950 | 0.9900 | 0.9850 | 0.9900 | 0.9950 | 0.9950 |
| shred_heldout_medial_min | 0.9650 | 0.8400 | 0.9400 | 0.8650 | 0.8100 | 0.9750 | 0.8200 | 0.9400 |
| shred_train_min | 0.9550 | 0.9700 | 0.9700 | 0.9800 | 0.9850 | 0.9850 | 0.9100 | 0.9500 |
| revoke_train_min | 0.9550 | 0.9700 | 0.9700 | 0.9800 | 0.9850 | 0.9850 | 0.9100 | 0.9500 |
| heldout/revoked_deleted_object | 0.0000 | 0.0013 | 0.0000 | 0.0013 | 0.0013 | 0.0000 | 0.0013 | 0.0000 |
| broken1_unknown | 0.6300 | 0.5200 | 0.6800 | 0.5400 | 0.5700 | 0.5350 | 0.5550 | 0.5800 |
| generic/kl_to_base | 3.6474 | 3.9198 | 3.5267 | 3.8005 | 3.8147 | 3.6770 | 3.8257 | 3.6156 |
| query_cos_between_fact/read1 | 0.2720 | 0.2607 | 0.3107 | 0.2627 | 0.2644 | 0.2623 | 0.2815 | 0.2920 |
| address_collision | 0.1000 | 0.0775 | 0.1375 | 0.0700 | 0.0475 | 0.0075 | 0.0925 | 0.1375 |

## E-000013 (fallback to the prior): override while ACTIVE and KL to the base model after REVOKE

Templates: t0 `The capital of {s} is`; t1 `The capital city of {s} is`; t2 `Q: What is the capital of {s}? A:`; t3 `{s}'s capital city is`; held-out (2, 3); subject positions none: [4, 5, 8, 0]; bos: [5, 6, 9, 1]; text: [8, 9, 12, 5]

| measure (worst seed) | none | bos | text |
|---|---|---|---|
| override/template0_direct | 1.0000 | 1.0000 | 0.6000 |
| override/template1_direct | 1.0000 | 0.8200 | 0.8800 |
| override/template2_direct | 0.0000 | 0.0000 | 0.0000 |
| override/template3_direct | 0.0000 | 0.0000 | 0.3200 |
| override_heldout_min | 0.0000 | 0.0000 | 0.0000 |
| override/template0_route_hit | 0.9600 | 0.8400 | 0.2800 |
| override/template1_route_hit | 0.9800 | 0.2800 | 0.6400 |
| override/template2_route_hit | 0.0000 | 0.0000 | 0.0000 |
| override/template3_route_hit | 0.0000 | 0.0000 | 0.2000 |
| revoke/template0_kl_to_base | 0.0005 | 0.0006 | 0.0170 |
| revoke/template1_kl_to_base | 0.0002 | 0.0010 | 0.0004 |
| revoke/template2_kl_to_base | 4.4722 | 5.6525 | 6.0733 |
| revoke/template3_kl_to_base | 0.0007 | 0.2148 | 0.4137 |
| revoke/heldout_kl_max | 4.4722 | 5.6525 | 6.0733 |
| revoke/top1_matches_base_pooled | 0.7300 | 0.6300 | 0.7100 |
| revoke/counterfactual_top1_pooled | 0.0000 | 0.0000 | 0.0000 |
| generic/kl_to_base | 2.3611 | 2.7388 | 2.2309 |

## E-000020 (link cells): the lifecycle battery at template 0 and the held-out templates

| measure (worst seed) | none | bos |
|---|---|---|
| t0/direct | 0.5633 | 0.9933 |
| t0/alias_direct | 0.5000 | 0.9050 |
| t0/dup_direct | 0.5900 | 0.9950 |
| t0/alias_heldout_min | 0.3000 | 0.7400 |
| t0/shared_update/alias_new_object | 0.5350 | 0.9350 |
| t0/duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| t0/rollback/alias_direct | 0.5000 | 0.9050 |
| t0/shred_target/alias_unknown | 0.9950 | 1.0000 |
| t0/delete_target/alias_unknown | 0.9650 | 0.8750 |
| t8/direct | 0.3933 | 0.9733 |
| t8/alias_direct | 0.3550 | 0.8850 |
| t8/dup_direct | 0.3850 | 0.9800 |
| t8/alias_heldout_min | 0.3000 | 0.7400 |
| t8/shared_update/alias_new_object | 0.3700 | 0.9000 |
| t8/duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| t8/rollback/alias_direct | 0.3550 | 0.8850 |
| t8/shred_target/alias_unknown | 0.9600 | 0.9950 |
| t8/delete_target/alias_unknown | 0.9500 | 0.9000 |
| t9/direct | 0.9300 | 0.7433 |
| t9/alias_direct | 0.8700 | 0.7400 |
| t9/dup_direct | 0.9800 | 0.7900 |
| t9/alias_heldout_min | 0.3000 | 0.7400 |
| t9/shared_update/alias_new_object | 0.8700 | 0.7300 |
| t9/duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| t9/rollback/alias_direct | 0.8700 | 0.7400 |
| t9/shred_target/alias_unknown | 0.9950 | 0.9750 |
| t9/delete_target/alias_unknown | 0.9300 | 0.9450 |
| t10/direct | 0.9967 | 0.9967 |
| t10/alias_direct | 0.8700 | 0.8700 |
| t10/dup_direct | 1.0000 | 0.9950 |
| t10/alias_heldout_min | 0.3000 | 0.7400 |
| t10/shared_update/alias_new_object | 0.8850 | 0.8900 |
| t10/duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| t10/rollback/alias_direct | 0.8700 | 0.8700 |
| t10/shred_target/alias_unknown | 1.0000 | 1.0000 |
| t10/delete_target/alias_unknown | 0.9100 | 0.8950 |
| t11/direct | 0.2933 | 0.9867 |
| t11/alias_direct | 0.3000 | 0.8950 |
| t11/dup_direct | 0.2350 | 0.9950 |
| t11/alias_heldout_min | 0.3000 | 0.7400 |
| t11/shared_update/alias_new_object | 0.2950 | 0.9100 |
| t11/duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| t11/rollback/alias_direct | 0.3000 | 0.8950 |
| t11/shred_target/alias_unknown | 0.9500 | 1.0000 |
| t11/delete_target/alias_unknown | 0.9450 | 0.9050 |

## Pre-registered criteria (worst seed)

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| e17/none/heldout/active_correct | <= 0.8 | 0.7587 | PASS |
| e17/none/heldout_initial/read_min | <= 0.7 | 0.4500 | PASS |
| e17/none/heldout_initial/route_hit_min | <= 0.8 | 0.5900 | PASS |
| e17/bos/heldout_initial/read_min | >= 0.9 | 0.9700 | PASS |
| e17/bos/heldout_initial/route_hit_min | >= 0.9 | 0.9800 | PASS |
| e17/bos/heldout/active_correct | >= 0.9 | 0.9175 | PASS |
| e17/newline/heldout_initial/read_min | >= 0.9 | 0.9800 | PASS |
| e17/newline/heldout_initial/route_hit_min | >= 0.9 | 0.9800 | PASS |
| e17/word/heldout_initial/read_min | >= 0.9 | 0.9800 | PASS |
| e17/word/heldout_initial/route_hit_min | >= 0.9 | 0.9800 | PASS |
| e17/text/heldout_initial/read_min | >= 0.9 | 0.9900 | PASS |
| e17/text/heldout_initial/route_hit_min | >= 0.9 | 0.9900 | PASS |
| e17/bos/medial_abs_change_max | <= 0.05 | 0.2500 | FAIL |
| e17/bos/train_read_change_min | >= -0.05 | -0.1800 | FAIL |
| e17/bos/train/active_correct | >= 0.95 | 0.9719 | PASS |
| e17/bos/shred_heldout_min | >= 0.95 | 0.8400 | FAIL |
| e17/bos/revoke_heldout_min | >= 0.95 | 0.8400 | FAIL |
| e17/bos/heldout/revoked_deleted_object | <= 0.02 | 0.0013 | PASS |
| e17/bos/broken1_unknown | >= 0.63 | 0.5200 | FAIL |
| e17/bos/generic/kl_to_base | <= 3.65 | 3.9198 | FAIL |
| e13/bos/override_heldout_min | >= 0.7 | 0.0000 | FAIL |
| e13/bos/override/direct | >= 0.9 | 1.0000 | PASS |
| e13/bos/revoke/heldout_kl_max | <= 0.1 | 5.6525 | FAIL |
| e20/bos/t0/direct | >= 0.85 | 0.9933 | PASS |
| e20/bos/t0/alias_direct | >= 0.8 | 0.9050 | PASS |
| e20/bos/t0/shared_update/alias_new_object | >= 0.9 | 0.9350 | PASS |

## The rule, fixed before the run

Worst seed on every row. READING 1 (artefact): record_reproduced, bos_restores_heldout_addressing, any_token_does_it and deletion_follows all pass -> the held-out paraphrase gap of this addressable memory on a frozen GPT-2 is the missing-BOS position-0 artefact; the honest held-out numbers for the memory are the BOS-prefixed ones, measured here with no training; kill criterion 5 (E-000017), the bimodality of E-000025, the template selection of E-000026, E-000039-B's negative and the 'behaves like own knowledge: no' row of section 31.36 are re-scoped as measured with the subject at position 0. If controls_hold also passes the fix is free; if controls_hold fails because subject-medial or trained templates MOVE under a BOS, the artefact reading survives (the subject-initial rows are its evidence) but the fix is not free: the adapter learned features of whichever token sat at position 0, and the BOS has to be applied consistently at training time (E-000050-B) before any number is quoted as a ceiling. READING 2 (semantic): the BOS fails its rows where the text prefix passes its own -> the gain is in what the prefix says and not where the subject sits; E-000039-A's prefix finding stands as it is, the position-0 diagnosis is withdrawn, and nothing is re-scoped. If the BOS passes and the newline or the word fails, the effect is BOS-specific (sink-token-specific) rather than position-0-specific and is recorded as such. The E-000013 and E-000020 groups are reported, not decided on: they say how far the switch reaches into the records the target was scored on (section 31.36). Prior art: the diagnosis and the remedy are Yang et al. 2024 (Fall of ROME) for ROME on GPT-2-family models; the mechanism is attention sinks / massive activations; only the memory-adapter measurement, the BOS-specific test, the deletion / override outcome and the controls are this record's. Fixed before the run.

Not claimed: that prepending a BOS is a general law (sink formation is data-dependent; this is GPT-2 124M); that the positional EMBEDDING is the cause (Yang et al. A.4: the first token's self-only attention is the other cause -- this record says position 0, not position embedding); anything about a trained-with-BOS adapter (E-000050-B); LLM scale; multi-token entities.

Prior art: Yang et al. 2024, The Fall of ROME: subject-first prompts break the subject key on GPT-2-family models because of the special distribution of position 0; any prefix or Llama's <s> repairs it; generalisation on the repaired cases stays low (16.88% paraphrase on GPT-2-XL collapse cases) | Xiao et al. 2023 (attention sinks); Sun et al. 2024 (massive activations); TransformerLens prepend_bos and the mechanistic-interpretability folklore of prepending a BOS to GPT-2 | Yang et al. A.4: a position-embedding swap does not remove the first-token anomaly, so this is a position-0 artefact, not a positional-embedding one; Barbero et al. / Gu et al.: sink formation is data-dependent, so the BOS gain is a property of this model family | CounterFact's released paraphrase prompts always carry a generated prefix: the field's paraphrase numbers were measured with the subject never at position 0
