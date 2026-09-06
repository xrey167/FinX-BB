# E-000005 — Causal interventions

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2, 3, 4], 100 targets per seed.

| intervention | predicted outcome | observed (mean) | worst seed |
|---|---|---|---|
| disable | UNKNOWN | 100.0% | 100.0% |
| disable_random_other | unchanged | 100.0% | 100.0% |
| swap | partner's object | 100.0% | 100.0% |
| swap_partner | target's object | 100.0% | 100.0% |
| restore | both original | 100.0% | 100.0% |
| replace | new object | 100.0% | 100.0% |
| localization | routed cell == ground-truth cell | 100.0% | 100.0% |
| routed_cell_causal | disabling routed cell -> UNKNOWN | 100.0% | 100.0% |

That the read equation uses the cell's payload is by construction; what is tested is that the trained core actually routes each query to its own cell (localisation), does not draw the answer from anywhere else (disable -> UNKNOWN, random-other -> unchanged) and turns a swapped or replaced payload into exactly the predicted answer. Localisation is a trained objective (routing loss); E-000006 'no_routing_loss' reports how much of it emerges without that supervision.

n = 100 targets per seed. Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| disable | >= 0.98 | 1.0000 | PASS |
| disable_random_other | >= 0.98 | 1.0000 | PASS |
| swap | >= 0.98 | 1.0000 | PASS |
| swap_partner | >= 0.98 | 1.0000 | PASS |
| restore | >= 0.98 | 1.0000 | PASS |
| replace | >= 0.98 | 1.0000 | PASS |
| localization | >= 0.98 | 1.0000 | PASS |
| routed_cell_causal | >= 0.98 | 1.0000 | PASS |
