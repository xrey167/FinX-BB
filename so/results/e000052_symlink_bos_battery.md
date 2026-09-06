# E-000052 — the pointer battery on the BOS-trained symlink adapter, narrowed

Seeds [0, 1, 2], templates [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], link adapter `e000020_gpt2_bos`, link-free `e000050_bos`. Worst seed throughout. Arm C reads the BOS-trained checkpoint with a BOS; arm D reads it without (the reverse control); arm P is E-000025's price against the BOS-trained link-free adapter.

| template | direct | alias | dup | UPDATE reaches alias | SHRED → unknown | DELETE → unknown | BLANK → some entity | RELINK reads | D: alias, no BOS |
|---|---|---|---|---|---|---|---|---|---|
| t0 (initial, trained) | 1.0000 | 0.9200 | 1.0000 | 0.9450 | 1.0000 | 1.0000 | 0.0000 | 0.9200 | 0.0150 |
| t1 (medial, trained) | 0.9767 | 0.8150 | 0.9950 | 0.8250 | 1.0000 | 0.9950 | 0.0000 | 0.8400 | 0.9100 |
| t2 (initial, trained) | 0.9967 | 0.9250 | 1.0000 | 0.9450 | 1.0000 | 1.0000 | 0.0000 | 0.9200 | 0.0300 |
| t3 (medial, trained) | 0.9933 | 0.9000 | 1.0000 | 0.9050 | 1.0000 | 1.0000 | 0.0000 | 0.9200 | 0.9500 |
| t4 (medial, trained) | 1.0000 | 0.9400 | 1.0000 | 0.9550 | 1.0000 | 1.0000 | 0.0000 | 0.9400 | 0.9700 |
| t5 (medial, trained) | 0.9967 | 0.9250 | 1.0000 | 0.9450 | 1.0000 | 0.9950 | 0.0000 | 0.9200 | 0.9650 |
| t6 (initial, trained) | 1.0000 | 0.9300 | 1.0000 | 0.9350 | 1.0000 | 1.0000 | 0.0000 | 0.9300 | 0.0200 |
| t7 (medial, trained) | 0.9967 | 0.8850 | 1.0000 | 0.8950 | 1.0000 | 1.0000 | 0.0000 | 0.8900 | 0.9350 |
| t8 (initial, held out) | 0.9967 | 0.9100 | 1.0000 | 0.9400 | 1.0000 | 1.0000 | 0.0000 | 0.9100 | 0.0100 |
| t9 (medial, held out) | 0.8200 | 0.8300 | 0.8750 | 0.8200 | 0.9950 | 0.9950 | 0.0100 | 0.8200 | 0.9600 |
| t10 (medial, held out) | 0.9967 | 0.9150 | 0.9950 | 0.9300 | 1.0000 | 1.0000 | 0.0000 | 0.9200 | 0.9550 |
| t11 (initial, held out) | 0.9967 | 0.9250 | 0.9950 | 0.9400 | 1.0000 | 1.0000 | 0.0000 | 0.9300 | 0.0100 |

| price (P), BOS regime | train | held out | all |
|---|---|---|---|
| cost_of_sharing | 0.0944 | 0.0750 | 0.0879 |
| cost_of_link_training | 0.0000 | 0.0175 | 0.0054 |

Reproductions, labelled: the trained-template rows against E-000026 (template 3) and the subject-initial held-out recovery against E-000050. Content: the price (P), the BLANK wrong-entity rate (N), and the subject-medial held-out rows t9/t10 (T). The entity-failure rows at t9/t10 are reported (max 0.0000) and never scored.

## The rule, fixed before the run

Worst seed. If the ANCHOR fails (template 3 direct or alias below E-000026's worst seed minus its recorded spread) the reading is REGRESSION: the BOS training changed the adapter and no other row is read. If the REVERSE CONTROL does not fire (subject-initial held-out alias reading above 0.85 without the BOS, or medial below 0.80) the substrate is VOID: the checkpoint does not depend on its BOS. With anchor and control holding, each content row is read on its own and named: PRICE fails if either E-000025 cost exceeds its bar in the BOS regime (E-000025's 0.0954 / 0.0688 do not transfer); SET NULL fails if a blanked alias is answered with an entity at more than 0.05 at any phrasing (the UNKNOWN and neighbour rows must hold for the row to be readable at all); MEDIAL fails if alias_direct or shared UPDATE reach at t9 or t10 is below 0.80 (the entity-failure rows there are reported and never scored, because a routing miss passes them for free). CLEAN if all three hold: the symlink adapter on the corrected substrate meets E-000020's bars at every phrasing, a SET NULL alias is never read as an entity, and the pointer costs the reader no more than it did without a BOS -- a measurement paper's table, every mechanism owned. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| C/anchor/direct | >= 0.9866 | 0.9933 | PASS |
| C/anchor/alias_direct | >= 0.77 | 0.9000 | PASS |
| D/heldout_initial/alias_direct_max | <= 0.85 | 0.0100 | PASS |
| D/heldout_medial/alias_direct_min | >= 0.8 | 0.9150 | PASS |
| C/heldout_initial/alias_direct_min | >= 0.8 | 0.9100 | PASS |
| P/all/cost_of_sharing | <= 0.1 | 0.0879 | PASS |
| P/all/cost_of_link_training | <= 0.25 | 0.0054 | PASS |
| C/blank/alias_wrong_entity_max | <= 0.05 | 0.0100 | PASS |
| C/blank/alias_unknown_min | >= 0.9 | 0.9900 | PASS |
| C/blank/sibling_readable_min | >= 0.8 | 0.7900 | FAIL |
| C/blank/target_readable_min | >= 0.8 | 0.8400 | PASS |
| C/relink/alias_direct_min | >= 0.8 | 0.8200 | PASS |
| C/heldout_medial/alias_direct_min | >= 0.8 | 0.8300 | PASS |
| C/heldout_medial/shared_update_min | >= 0.8 | 0.8200 | PASS |
