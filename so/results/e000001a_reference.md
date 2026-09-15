# E-000001-A — Mechanical reference implementation

Evidence level: **E3** (Repeated synthetic evidence). Deletion level exercised: **F1** (routing removal) plus marker shredding at the mechanical level.

Seeds: [0, 1, 2, 3, 4] · cells per seed: 1000 · all tests passed: **True**

| Measure | Mean over seeds | Worst seed |
|---|---|---|
| direct | 100.0% | 100.0% |
| hop2 | 100.0% | 100.0% |
| hop3 | 100.0% | 100.0% |
| hop2_broken_unknown | 100.0% | 100.0% |
| hop3_broken_unknown | 100.0% | 100.0% |
| provenance | 100.0% | 100.0% |
| update_rollback | 100.0% | 100.0% |
| locality | 100.0% | 100.0% |
| locality_targets_changed | 100.0% | 100.0% |
| locality_undo_exact | 100.0% | 100.0% |
| alternative_path | 100.0% | 100.0% |
| replay_deviation | 0 | - |

Per seed:

| seed | direct | hop2 | hop3 | hop2_broken_unknown | hop3_broken_unknown | provenance | update_rollback | locality | locality_targets_changed | locality_undo_exact | alternative_path | replay_deviation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

Interpretation: establishes that the desired semantics are coherent in the controlled reference system. It does not show that a trained neural network reproduces them (that is E-000001-B).
