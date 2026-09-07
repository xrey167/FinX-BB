# E-000017-A — Reading versus refusal on held-out phrasings (diagnosis, no training)

Is the held-out failure of E-000011/E-000012 a deletion failure or a reading failure that deletion inherits?

No model was trained for this record: E-000012's three checkpoints are evaluated as they were recorded, so this is a decomposition of an existing result, not a new one.

| measure | mean over seeds | worst seed |
|---|---|---|
| train/active_correct | 0.9608 | 0.9550 |
| heldout/active_correct | 0.6937 | 0.6700 |
| train/refusal_given_active_correct | 0.9991 | 0.9973 |
| heldout/refusal_given_active_correct | 0.9614 | 0.9418 |
| heldout/revoked_deleted_object | 0.0008 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0015 | 0.0000 |

Read the two conditional rows together: `refusal_given_active_correct` is how often the model answers ' unknown' after REVOKE among exactly those targets it read correctly while the cell was ACTIVE, and `deleted_object_given_active_correct` is how often it returns the deleted object instead.
