# E-000017-B — Stage-2 template budget: 8 trained, 4 held out, no consistency loss

Roadmap kill criterion 5 fired on a two-template budget. This run gives the stage the budget it prescribes and reports whether the held-out failure survives it.

| claim group | supported |
|---|---|
| reading_generalises | **no** |
| refusal_generalises | yes |
| refusal_on_trained_templates_holds | yes |
| deleted_object_never_returns | yes |
| no_key_no_injection | **no** |

| measure | mean over seeds | worst seed |
|---|---|---|
| train/active_correct | 0.9198 | 0.9119 |
| heldout/active_correct | 0.7400 | 0.7288 |
| train/refusal_given_active_correct | 0.9966 | 0.9952 |
| heldout/refusal_given_active_correct | 0.9928 | 0.9870 |
| heldout/revoked_deleted_object | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0000 | 0.0000 |
| revoke_train_min | 0.9583 | 0.9550 |
| revoke_heldout_min | 0.8983 | 0.8650 |
| shred_train_min | 0.9583 | 0.9550 |
| shred_heldout_min | 0.8983 | 0.8650 |
| broken1_unknown | 0.7183 | 0.6300 |
| generic/kl_to_base | 3.2741 | 2.9591 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| heldout/active_correct | >= 0.9 | 0.7288 | FAIL |
| train/active_correct | >= 0.95 | 0.9119 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8650 | PASS |
| shred_heldout_min | >= 0.85 | 0.8650 | PASS |
| revoke_train_min | >= 0.95 | 0.9550 | PASS |
| shred_train_min | >= 0.9 | 0.9550 | PASS |
| heldout/revoked_deleted_object | <= 0.02 | 0.0000 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0000 | PASS |
| broken1_unknown | >= 0.9 | 0.6300 | FAIL |
| generic/kl_to_base | <= 0.05 | 3.6474 | FAIL |
