# E-000006 — Ablations

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2]; variants trained 2000 steps, full model 3000 steps (E-000001-B). Values are means over seeds.

| variant | direct | hop2 | hop3 | hop2_broken_unknown | provenance | reverse | revoke | shred | update | rollback | locality | alternative_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| full_same_budget | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_marker_gate | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_null_cell | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_routing_loss | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 21.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| no_routing | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 21.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |

'no_routing' and 'no_marker_gate' remove an information path, so their failures (nothing readable / SHRED ineffective) are information-flow necessities, reported to quantify them. 'no_null_cell' and 'no_routing_loss' keep the information paths and test learned behaviour: whether UNKNOWN detection and exact provenance emerge without the dedicated cell / loss. 'full_same_budget' is the fair baseline trained with the variants' step budget.

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| full_same_budget/direct | >= 0.98 | 1.0000 | PASS |
| full_same_budget/shred | >= 0.95 | 0.9800 | PASS |
| no_marker_gate/shred | <= 0.2 | 0.0000 | PASS |
| no_marker_gate/direct | >= 0.98 | 1.0000 | PASS |
| no_null_cell/hop2_broken_unknown | <= 0.5 | 1.0000 | FAIL |
| no_routing/direct | <= 0.1 | 0.0000 | PASS |

Random deletion (revoke another cell, target must stay): 100.0%

Reading the table: for a variant that answers UNKNOWN to everything (no_routing, no_routing_loss) the rows hop2_broken_unknown, revoke, shred and locality are satisfied trivially and carry no information.

Without versioning (UPDATE as in-place replace): rollback impossible (no version to return to) — structural property of the layer, not a learned one.
