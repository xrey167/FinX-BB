# E-000001-B — Trained Mini-Transformer over the mutable knowledge layer

Evidence level: **E4** (Controlled neural-network evidence). Deletion levels: REVOKE is routing removal (**F1**, by construction) on which the model has learned to answer UNKNOWN; SHRED is the learned functional-forgetting result (**F3**): the payload stays routable and the model refuses it because its marker is invalid.

Seeds: [0, 1, 2, 3, 4] · training steps: 3000 · parameters: 616,451 · core tests all at 100% in every seed: **False** · pre-registered criteria met: **True**

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 1.0000 | 1.0000 | 5000 | 0.9993 | 1.0000 |
| hop2 | 1.0000 | 1.0000 | 2500 | 0.9985 | 1.0000 |
| hop3 | 0.9988 | 0.9980 | 2500 | 0.9965 | 0.9998 |
| hop2_broken_unknown | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| hop3_broken_unknown | 0.9880 | 0.9700 | 500 | 0.9741 | 0.9956 |
| provenance | 1.0000 | 1.0000 | 10000 | 0.9996 | 1.0000 |
| reverse | 1.0000 | 1.0000 | 1500 | 0.9975 | 1.0000 |
| update | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| update_derived | 1.0000 | 1.0000 | - | - | - |
| rollback | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| revoke | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| restore | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred | 0.9940 | 0.9700 | 500 | 0.9826 | 0.9988 |
| resign | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| update_rollback | 0.9991 | 0.9957 | - | - | - |
| locality | 1.0000 | 1.0000 | 5750 | 0.9994 | 1.0000 |
| locality_targets_correct | 1.0000 | 1.0000 | 750 | 0.9951 | 1.0000 |
| locality_undo_exact | 1.0000 | 1.0000 | - | - | - |
| alternative_path | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| replay_deviation | 0.0000 | 0.0000 | - | - | - |

Pre-registered pass criteria (evaluated on the worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.99 | 1.0000 | PASS |
| hop2 | >= 0.98 | 1.0000 | PASS |
| hop3 | >= 0.95 | 0.9980 | PASS |
| provenance | >= 0.98 | 1.0000 | PASS |
| hop2_broken_unknown | >= 0.95 | 1.0000 | PASS |
| reverse | >= 0.95 | 1.0000 | PASS |
| update | >= 0.98 | 1.0000 | PASS |
| rollback | >= 0.98 | 1.0000 | PASS |
| revoke | >= 0.98 | 1.0000 | PASS |
| restore | >= 0.98 | 1.0000 | PASS |
| shred | >= 0.95 | 0.9700 | PASS |
| resign | >= 0.98 | 1.0000 | PASS |
| locality | >= 0.99 | 1.0000 | PASS |
| alternative_path | >= 0.95 | 1.0000 | PASS |
| replay_deviation | <= 0 | 0.0000 | PASS |

Noise sweep (bank-level Gaussian perturbation of keys and values relative to their RMS, direct queries, mean over seeds; NOT comparable to the architecture document's 0.24 -> 68.4% figure):

| noise | direct accuracy |
|---|---|
| 0.00 | 100.0% |
| 0.05 | 100.0% |
| 0.10 | 100.0% |
| 0.16 | 100.0% |
| 0.20 | 100.0% |
| 0.24 | 100.0% |
| 0.30 | 99.9% |
| 0.40 | 99.1% |
| 0.50 | 96.9% |
| 0.70 | 86.9% |
| 1.00 | 64.7% |
| 1.50 | 33.5% |

Per seed:

| seed | direct | hop2 | hop3 | hop2_broken_unknown | hop3_broken_unknown | provenance | reverse | update | update_derived | rollback | revoke | restore | shred | resign | update_rollback | locality | locality_targets_correct | locality_undo_exact | alternative_path | replay_deviation | train_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 205 |
| 101 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9700 | 1.0000 | 0.9957 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 113 |
| 102 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 115 |
| 103 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 112 |
| 104 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9700 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 113 |

Interpretation: the behaviour is no longer mechanical — a trained neural core operates over the experimental knowledge structure. It is still a synthetic experiment and not proof of LLM-scale editable knowledge.
