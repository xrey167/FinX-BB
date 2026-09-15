# E-000016 — Alias chains: how far the indirection carries

Evidence level: **E4** (synthetic system). Seeds: [0, 1, 2]; 4000 steps; 30% of the aliases in training point at another alias.

E-000015 recorded that its two-slot control did not resolve chains. The cause proposed there was the training distribution, not the architecture; this experiment tests that explanation by putting 30% chains into training and changing nothing else.

| claim group | supported |
|---|---|
| two_slots_resolve_a_chain | yes |
| one_slot_refuses_a_chain | yes |
| no_price_paid_elsewhere | yes |
| sharing_still_holds | yes |
| shredding_a_pointer | yes |

| measure | mean over seeds | worst seed | best seed |
|---|---|---|---|
| two/direct | 1.0000 | 1.0000 | 1.0000 |
| two/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| two/chain2/answer_acc | 1.0000 | 1.0000 | 1.0000 |
| two/chain2/unknown | 0.0000 | 0.0000 | 0.0000 |
| two/chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| one/direct | 1.0000 | 1.0000 | 1.0000 |
| one/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| one/chain2/answer_acc | 0.0000 | 0.0000 | 0.0000 |
| one/chain2/unknown | 1.0000 | 1.0000 | 1.0000 |
| one/chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| two/alias_provenance_pair | 1.0000 | 1.0000 | 1.0000 |
| two/hop2 | 1.0000 | 1.0000 | 1.0000 |
| two/shared_update/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| two/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| two/shred_target/alias_unknown | 0.9967 | 0.9900 | 1.0000 |
| two/shred_target/alias_probe_top1 | 0.0100 | 0.0100 | 0.0100 |
| two/dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 1.0000 |
| two/dup_shred/copy_probe_top1 | 0.8867 | 0.8400 | 0.9300 |
| two/shred_alias/alias_unknown | 0.9867 | 0.9700 | 1.0000 |
| one/shred_alias/alias_unknown | 0.9833 | 0.9700 | 1.0000 |
| two/delete_target/alias_unknown | 0.9967 | 0.9900 | 1.0000 |
| two/deref_disabled/alias_direct | 0.0000 | 0.0000 | 0.0000 |
| two/regression/direct | 1.0000 | 1.0000 | 1.0000 |
| two/regression/hop2 | 1.0000 | 1.0000 | 1.0000 |
| two/regression/hop3 | 0.9956 | 0.9900 | 1.0000 |
| two/regression/reverse | 1.0000 | 1.0000 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| two/chain2/answer_acc | >= 0.9 | 1.0000 | PASS |
| one/chain2/answer_acc | <= 0.2 | 0.0000 | PASS |
| one/chain2/unknown | >= 0.9 | 1.0000 | PASS |
| two/direct | >= 0.98 | 1.0000 | PASS |
| two/alias_direct | >= 0.95 | 1.0000 | PASS |
| two/hop2 | >= 0.95 | 1.0000 | PASS |
| two/regression/direct | >= 0.98 | 1.0000 | PASS |
| two/regression/hop2 | >= 0.95 | 1.0000 | PASS |
| two/regression/reverse | >= 0.95 | 1.0000 | PASS |
| two/alias_provenance_pair | >= 0.9 | 1.0000 | PASS |
| two/shared_update/alias_new_object | >= 0.95 | 1.0000 | PASS |
| two/duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| two/shred_target/alias_unknown | >= 0.95 | 0.9900 | PASS |
| two/shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| two/dup_shred/copy_direct_acc | >= 0.95 | 1.0000 | PASS |
| two/shred_alias/alias_unknown | >= 0.95 | 0.9700 | PASS |
| one/shred_alias/alias_unknown | >= 0.95 | 0.9700 | PASS |

By construction: the store resolves a chain by following kids with a depth limit and a cycle check; what is measured is whether the trained model reproduces it from the pointers alone; the one-slot arm CANNOT represent a two-link chain: it has one dereference slot. Its criterion is that it answers unknown rather than inventing an entity..

Learned: following a pointer whose target is itself a pointer, with the query for each dereference coming from the value just read; refusing a chain that does not fit the available slots instead of naming another entity.

Not claimed: chains deeper than the number of slots; LLM scale.
