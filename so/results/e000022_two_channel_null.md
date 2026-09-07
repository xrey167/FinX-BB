# E-000022 — Splitting the null column: inject a payload only when a cell matches, the unknown direction only when a question found none, and otherwise nothing

E-000018 recorded that its two remedies cut injection into unrelated text fivefold and still missed, because refusing a question and ignoring prose were routed through the same null column. This splits them and changes nothing else.

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | **no** |
| refusal_not_traded_away | yes |
| deleted_object_never_returns | yes |

| measure | mean over seeds | worst seed | E-000018 both | E-000017-B |
|---|---|---|---|---|
| generic/kl_to_base | 0.5508 | 0.3589 | 0.6736 | 3.2741 |
| broken1_unknown | 0.6983 | 0.6000 | 0.7017 | 0.7183 |
| train/active_correct | 0.9050 | 0.8844 | 0.9019 | 0.9198 |
| heldout/active_correct | 0.7254 | 0.7062 | 0.6917 | 0.7400 |
| train/refusal_given_active_correct | 0.9969 | 0.9964 | 0.9960 | - |
| heldout/refusal_given_active_correct | 0.9924 | 0.9894 | 0.9882 | 0.9928 |
| heldout/revoked_deleted_object | 0.0008 | 0.0000 | 0.0004 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0015 | 0.0000 | 0.0000 | - |
| revoke_train_min | 0.9617 | 0.9550 | 0.9533 | - |
| revoke_heldout_min | 0.8833 | 0.8500 | 0.8550 | 0.8983 |
| shred_train_min | 0.9617 | 0.9550 | 0.9533 | - |
| shred_heldout_min | 0.8833 | 0.8500 | 0.8550 | 0.8983 |

Pre-registered criteria (worst seed), identical to E-000018's:

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| generic/kl_to_base | <= 0.05 | 0.8657 | FAIL |
| broken1_unknown | >= 0.9 | 0.6000 | FAIL |
| train/active_correct | >= 0.9 | 0.8844 | FAIL |
| heldout/active_correct | >= 0.7 | 0.7062 | PASS |
| revoke_train_min | >= 0.95 | 0.9550 | PASS |
| revoke_heldout_min | >= 0.85 | 0.8500 | PASS |
| shred_heldout_min | >= 0.85 | 0.8500 | PASS |
| heldout/revoked_deleted_object | <= 0.02 | 0.0013 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0025 | PASS |

By construction: the query-relevance score multiplies only the NULL column's contribution; the cell contribution is still governed by the match gate, so the two channels cannot mask each other; the relevance score is read from the model's own state, not from the cells, so it can fire for a question whose cell is missing.

Learned: both channels: which text counts as a question about a cell, and how well a query has to match a key before its payload is injected.
