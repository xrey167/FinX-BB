# E-000039-B-address — Tying the routing query across phrasings of the same fact

| claim group | supported |
|---|---|
| reading_generalises | **no** |
| addressing_generalises | **no** |
| deletion_propagates | **no** |
| addresses_do_not_collapse | **no** |
| no_new_collateral | **no** |

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| heldout/active_correct | >= 0.95 | 0.7488 | FAIL |
| train/active_correct | >= 0.95 | 0.9131 | FAIL |
| heldout/route_hit_min | >= 0.95 | 0.5200 | FAIL |
| shred_heldout_min | >= 0.95 | 0.8650 | FAIL |
| revoke_heldout_min | >= 0.95 | 0.8650 | FAIL |
| heldout/revoked_deleted_object | <= 0.02 | 0.0000 | PASS |
| query_cos_between_fact/read1 | <= 0.33 | 0.1683 | PASS |
| address_collision | <= 0.02 | 0.1125 | FAIL |
| broken1_unknown | >= 0.63 | 0.6000 | FAIL |
| generic/kl_to_base | <= 3.65 | 3.1971 | PASS |
