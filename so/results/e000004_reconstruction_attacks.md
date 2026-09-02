# E-000004 — Reconstruction attacks

Evidence level: **E4** (Controlled neural-network evidence); deletion level **F4** within the synthetic system (representation-level checks, linear probe). Seeds: [0, 1, 2, 3, 4]. Probe calibration on held-out active cells: top-1 0.949, top-5 0.949. Chance: forced choice 0.5, top-1 among entities 1/256 = 0.0039, mean rank 127.5, probe top-1 0.0039, top-5 0.0195.

| attack (mean over seeds) | active | after REVOKE (mask) | after SHRED (learned) |
|---|---|---|---|
| direct_unknown | 0.0000 | 1.0000 | 1.0000 |
| direct_acc | 1.0000 | 0.0000 | 0.0000 |
| paraphrase_unknown | 0.0000 | 1.0000 | 1.0000 |
| multihop_unknown | 0.0000 | 1.0000 | 1.0000 |
| reverse_unknown | 0.0000 | 1.0000 | 0.9641 |
| forced_choice_win | 1.0000 | 0.5200 | 0.5560 |
| true_obj_top1_among_entities | 1.0000 | 0.0000 | 0.0140 |
| true_obj_mean_rank | 0.0000 | 128.8560 | 114.2580 |
| probe_top1 | 0.9460 | 0.0000 | 0.0400 |
| probe_top5 | 0.9460 | 0.0200 | 0.0880 |
| routing_mass_on_target | 0.9978 | 0.0000 | 0.9978 |
| gated_value_contribution | 13.9243 | 0.0000 | 1.3146 |

After REVOKE the routing mass and value contribution on the target are zero by the mask, not by learning — those two rows are reported for completeness only. After SHRED the cell is still routable, so every row is a measurement of learned behaviour; the SHRED column carries the F4-level evidence.

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| active/direct_acc | >= 0.98 | 1.0000 | PASS |
| active/probe_top1 | >= 0.5 | 0.9200 | PASS |
| revoke/direct_unknown | >= 0.98 | 1.0000 | PASS |
| revoke/probe_top1 | <= 0.05 | 0.0000 | PASS |
| revoke/forced_choice_win | <= 0.6 | 0.6000 | PASS |
| revoke/true_obj_top1_among_entities | <= 0.05 | 0.0000 | PASS |
| shred/direct_unknown | >= 0.95 | 1.0000 | PASS |
| shred/paraphrase_unknown | >= 0.95 | 1.0000 | PASS |
| shred/probe_top1 | <= 0.05 | 0.0800 | FAIL |
| shred/forced_choice_win | <= 0.6 | 0.6900 | FAIL |
| shred/true_obj_top1_among_entities | <= 0.05 | 0.0200 | PASS |
| shred/gated_value_contribution | <= 0.1 | 1.5889 | FAIL |
| restored/direct_acc | >= 0.98 | 1.0000 | PASS |

Sample sizes per seed: 100 targets (probe / forced choice / rank / direct); multi-hop and reverse subsets are smaller (only targets with an outgoing edge or a unique reverse subject).

Dependency reconstruction (K3 derivable from K1 + K2; 'collateral' = 2-hop paths not touching the closure):

| measure | mean |
|---|---|
| n_triples | 30.0000 |
| direct_unknown_after_revoke_K3 | 1.0000 |
| derivable_recovery_after_revoke_K3 | 1.0000 |
| derivable_recovery_after_closure | 0.0000 |
| collateral_bypass_acc_after_closure | 1.0000 |

Context completion: not applicable (symbolic queries, no free text).
