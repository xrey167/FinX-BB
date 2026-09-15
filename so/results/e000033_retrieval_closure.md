# E-000033 — the deletion closure in a chunked retrieval store

Seeds [0, 1, 2], 150 facts, 4 chunks per fact, frozen gpt2 as the
embedder, cosine retrieval, no training. Both indexes carry the same addressing text for
every chunk; they differ only in whether the ANSWER is repeated or replaced by a pointer.

## Erasure in a vector index

| index | answers before deletion | chunks holding the fact | still retrievable after one deletion | search recall at precision 1.0 | search precision at recall 1.0 | closure known without searching |
|---|---|---|---|---|---|---|
| canonical | 0.0467 | 1.00 | 0.0250 | 0.0000 | 0.0020 | 1.0000 |
| duplicated | 0.1050 | 4.00 | 0.0817 | 0.0000 | 0.0072 | 0.0000 |

`still retrievable after one deletion` removes the chunk the fact's own question retrieves —
what a naive erasure does — and then asks all four phrasings again. The two search columns
give the content-based remedy its best shot: the similarity threshold is swept and the best
achievable point reported. They are the half that is not obvious. Deleting one of k copies
leaving the fact readable is Codd's modification anomaly applied to a delete, and any
practitioner would predict it;
that the search for the remaining copies cannot be both complete and clean is the reason
canonicalisation is not merely tidier.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| control/read_before_deletion | >= 0.8 | 0.0250 | FAIL |
| control/read_gap | <= 0.15 | 0.0800 | PASS |
| canonical/fact_closure_max | <= 1.0 | 1.0000 | PASS |
| duplicated/fact_closure_mean | >= 4.0 | 4.0000 | PASS |
| canonical/still_retrievable_after_one | <= 0.1 | 0.0400 | PASS |
| duplicated/still_retrievable_after_one | >= 0.5 | 0.0550 | FAIL |
| duplicated/search_recall_at_precision_1 | <= 0.9 | 0.0000 | PASS |
| duplicated/search_precision_at_recall_1 | <= 0.9 | 0.0073 | PASS |
| canonical/closure_known_without_search | >= 1.0 | 1.0000 | PASS |
