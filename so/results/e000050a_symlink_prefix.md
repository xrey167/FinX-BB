# E-000050-A — the paraphrase gap is the position-0 token: a BOS at inference, no training

E-000017-B, E-000013 and E-000020's recorded checkpoints are evaluated as recorded; the only change is a prefix on the prompt at inference.

Seeds [0, 1, 2]; families ['e20']; 100 targets per seed for the decomposition. Worst seed everywhere.

**Reading: not decided (E-000017-B family not run).**

| claim group | supported |
|---|---|
| record_reproduced | not measured |
| bos_restores_heldout_addressing | not measured |
| any_token_does_it | not measured |
| controls_hold | not measured |
| deletion_follows | not measured |
| no_new_collateral | not measured |
| e13_override_and_revert_on_heldout | not measured |
| e20_lifecycle_at_template0 | yes |

## E-000020 (link cells): the lifecycle battery at template 0 and the held-out templates

| measure (worst seed) | none | space | bos |
|---|---|---|---|
| t0/direct | 0.5633 | 0.9933 | 0.9933 |
| t0/alias_direct | 0.5000 | 0.8850 | 0.9050 |
| t0/dup_direct | 0.5900 | 1.0000 | 0.9950 |
| t0/alias_heldout_min | 0.3000 | 0.8600 | 0.7400 |
| t0/shared_update/alias_new_object | 0.5350 | 0.9150 | 0.9350 |
| t0/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| t0/rollback/alias_direct | 0.5000 | 0.8850 | 0.9050 |
| t0/shred_target/alias_unknown | 0.9950 | 1.0000 | 1.0000 |
| t0/delete_target/alias_unknown | 0.9650 | 0.8700 | 0.8750 |
| t8/direct | 0.3933 | 0.9667 | 0.9733 |
| t8/alias_direct | 0.3550 | 0.8650 | 0.8850 |
| t8/dup_direct | 0.3850 | 0.9750 | 0.9800 |
| t8/alias_heldout_min | 0.3000 | 0.8600 | 0.7400 |
| t8/shared_update/alias_new_object | 0.3700 | 0.8900 | 0.9000 |
| t8/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| t8/rollback/alias_direct | 0.3550 | 0.8650 | 0.8850 |
| t8/shred_target/alias_unknown | 0.9600 | 0.9950 | 0.9950 |
| t8/delete_target/alias_unknown | 0.9500 | 0.9000 | 0.9000 |
| t9/direct | 0.9300 | 0.9433 | 0.7433 |
| t9/alias_direct | 0.8700 | 0.8600 | 0.7400 |
| t9/dup_direct | 0.9800 | 0.9650 | 0.7900 |
| t9/alias_heldout_min | 0.3000 | 0.8600 | 0.7400 |
| t9/shared_update/alias_new_object | 0.8700 | 0.8500 | 0.7300 |
| t9/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| t9/rollback/alias_direct | 0.8700 | 0.8600 | 0.7400 |
| t9/shred_target/alias_unknown | 0.9950 | 1.0000 | 0.9750 |
| t9/delete_target/alias_unknown | 0.9300 | 0.9400 | 0.9450 |
| t10/direct | 0.9967 | 0.9967 | 0.9967 |
| t10/alias_direct | 0.8700 | 0.8700 | 0.8700 |
| t10/dup_direct | 1.0000 | 1.0000 | 0.9950 |
| t10/alias_heldout_min | 0.3000 | 0.8600 | 0.7400 |
| t10/shared_update/alias_new_object | 0.8850 | 0.8850 | 0.8900 |
| t10/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| t10/rollback/alias_direct | 0.8700 | 0.8700 | 0.8700 |
| t10/shred_target/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| t10/delete_target/alias_unknown | 0.9100 | 0.9150 | 0.8950 |
| t11/direct | 0.2933 | 0.9867 | 0.9867 |
| t11/alias_direct | 0.3000 | 0.8900 | 0.8950 |
| t11/dup_direct | 0.2350 | 0.9850 | 0.9950 |
| t11/alias_heldout_min | 0.3000 | 0.8600 | 0.7400 |
| t11/shared_update/alias_new_object | 0.2950 | 0.9100 | 0.9100 |
| t11/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| t11/rollback/alias_direct | 0.3000 | 0.8900 | 0.8950 |
| t11/shred_target/alias_unknown | 0.9500 | 1.0000 | 1.0000 |
| t11/delete_target/alias_unknown | 0.9450 | 0.9150 | 0.9050 |

## Pre-registered criteria (worst seed)

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| e20/bos/t0/direct | >= 0.85 | 0.9933 | PASS |
| e20/bos/t0/alias_direct | >= 0.8 | 0.9050 | PASS |
| e20/bos/t0/shared_update/alias_new_object | >= 0.9 | 0.9350 | PASS |

Not measured in this run (families or variants skipped): e13/bos/override/direct, e13/bos/override_heldout_min, e13/bos/revoke/heldout_kl_max, e17/bos/broken1_unknown, e17/bos/generic/kl_to_base, e17/bos/heldout/active_correct, e17/bos/heldout/revoked_deleted_object, e17/bos/heldout_initial/read_min, e17/bos/heldout_initial/route_hit_min, e17/bos/medial_abs_change_max, e17/bos/revoke_heldout_min, e17/bos/shred_heldout_min, e17/bos/train/active_correct, e17/bos/train_read_change_min, e17/newline/heldout_initial/read_min, e17/newline/heldout_initial/route_hit_min, e17/none/heldout/active_correct, e17/none/heldout_initial/read_min, e17/none/heldout_initial/route_hit_min, e17/text/heldout_initial/read_min, e17/text/heldout_initial/route_hit_min, e17/word/heldout_initial/read_min, e17/word/heldout_initial/route_hit_min

## The rule, fixed before the run

Worst seed on every row. READING 1 (artefact): record_reproduced, bos_restores_heldout_addressing, any_token_does_it and deletion_follows all pass -> the held-out paraphrase gap of this addressable memory on a frozen GPT-2 is the missing-BOS position-0 artefact; the honest held-out numbers for the memory are the BOS-prefixed ones, measured here with no training; kill criterion 5 (E-000017), the bimodality of E-000025, the template selection of E-000026, E-000039-B's negative and the 'behaves like own knowledge: no' row of section 31.36 are re-scoped as measured with the subject at position 0. If controls_hold also passes the fix is free; if controls_hold fails because subject-medial or trained templates MOVE under a BOS, the artefact reading survives (the subject-initial rows are its evidence) but the fix is not free: the adapter learned features of whichever token sat at position 0, and the BOS has to be applied consistently at training time (E-000050-B) before any number is quoted as a ceiling. READING 2 (semantic): the BOS fails its rows where the text prefix passes its own -> the gain is in what the prefix says and not where the subject sits; E-000039-A's prefix finding stands as it is, the position-0 diagnosis is withdrawn, and nothing is re-scoped. If the BOS passes and the newline or the word fails, the effect is BOS-specific (sink-token-specific) rather than position-0-specific and is recorded as such. The E-000013 and E-000020 groups are reported, not decided on: they say how far the switch reaches into the records the target was scored on (section 31.36). Prior art: the diagnosis and the remedy are Yang et al. 2024 (Fall of ROME) for ROME on GPT-2-family models; the mechanism is attention sinks / massive activations; only the memory-adapter measurement, the BOS-specific test, the deletion / override outcome and the controls are this record's. Fixed before the run.

Not claimed: that prepending a BOS is a general law (sink formation is data-dependent; this is GPT-2 124M); that the positional EMBEDDING is the cause (Yang et al. A.4: the first token's self-only attention is the other cause -- this record says position 0, not position embedding); anything about a trained-with-BOS adapter (E-000050-B); LLM scale; multi-token entities.

Prior art: Yang et al. 2024, The Fall of ROME: subject-first prompts break the subject key on GPT-2-family models because of the special distribution of position 0; any prefix or Llama's <s> repairs it; generalisation on the repaired cases stays low (16.88% paraphrase on GPT-2-XL collapse cases) | Xiao et al. 2023 (attention sinks); Sun et al. 2024 (massive activations); TransformerLens prepend_bos and the mechanistic-interpretability folklore of prepending a BOS to GPT-2 | Yang et al. A.4: a position-embedding swap does not remove the first-token anomaly, so this is a position-0 artefact, not a positional-embedding one; Barbero et al. / Gu et al.: sink formation is data-dependent, so the BOS gain is a property of this model family | CounterFact's released paraphrase prompts always carry a generated prefix: the field's paraphrase numbers were measured with the subject never at position 0
