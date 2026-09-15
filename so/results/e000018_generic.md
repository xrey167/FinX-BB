# E-000018 — No key, no injection — arm 'generic' (match gate False, generic text share 0.25)

The routing softmax always sums to one, so some cell always wins and the layer injects into text it has no key for. E-000017-B measured 3.27 nats on generic sentences against a 0.05 bar, worse than E-000013's 2.27 with fewer templates.

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | **no** |
| refusal_not_traded_away | **no** |
| deleted_object_never_returns | yes |

| measure | mean over seeds | worst seed | E-000017-B baseline |
|---|---|---|---|
| generic/kl_to_base | 0.6035 | 0.3622 | 3.2741 |
| broken1_unknown | 0.6783 | 0.6250 | 0.7183 |
| train/active_correct | 0.9079 | 0.9019 | 0.9198 |
| heldout/active_correct | 0.6888 | 0.6813 | 0.7400 |
| train/refusal_given_active_correct | 0.9943 | 0.9934 | - |
| heldout/refusal_given_active_correct | 0.9844 | 0.9773 | 0.9928 |
| heldout/revoked_deleted_object | 0.0008 | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0016 | 0.0000 | - |
| revoke_train_min | 0.9500 | 0.9400 | - |
| revoke_heldout_min | 0.8800 | 0.8600 | 0.8983 |
| shred_train_min | 0.9500 | 0.9400 | - |
| shred_heldout_min | 0.8800 | 0.8600 | 0.8983 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| generic/kl_to_base | <= 0.05 | 0.8008 | FAIL |
| broken1_unknown | >= 0.9 | 0.6250 | FAIL |
| train/active_correct | >= 0.9 | 0.9019 | PASS |
| heldout/active_correct | >= 0.7 | 0.6813 | FAIL |
| revoke_train_min | >= 0.95 | 0.9400 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8600 | PASS |
| shred_heldout_min | >= 0.85 | 0.8600 | PASS |
| heldout/revoked_deleted_object | <= 0.02 | 0.0013 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0026 | PASS |

By construction: the match gate adds the CAPACITY to inject nothing (an absolute cosine threshold against the best real cell key); whether the model uses it is learned from the losses; the generic arm trains the behaviour on eight sentence shapes that are disjoint from the five the evaluation uses, so passing by memorising a shape is not available.
