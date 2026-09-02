# E-000009 — Signature-verification gate: closing the SHRED residual

Evidence level: **E4** (Controlled neural-network evidence); deletion level claimed for SHRED with hard verification: **F4** within the synthetic system. Seeds: [0, 1, 2, 3, 4]; 3000 steps; gate loss weight 1.0. Baseline = the E-000001-B models (no gate loss).

Attack battery after SHRED (mean / worst seed):

| attack after SHRED | baseline (soft gate) | verified (soft gate) | verified (hard gate) |
|---|---|---|---|
| direct_unknown | 0.9980 / 0.9900 | 0.9880 / 0.9700 | 0.9700 / 0.9500 |
| direct_acc | 0.0020 / 0.0000 | 0.0120 / 0.0000 | 0.0300 / 0.0200 |
| paraphrase_unknown | 0.9980 / 0.9900 | 0.9880 / 0.9700 | 0.9700 / 0.9500 |
| multihop_unknown | 0.9979 / 0.9897 | 0.9913 / 0.9794 | 0.9717 / 0.9512 |
| reverse_unknown | 0.9277 / 0.8788 | 0.9447 / 0.8857 | 0.9784 / 0.9429 |
| forced_choice_win | 0.5900 / 0.7000 | 0.5620 / 0.6500 | 0.5580 / 0.6200 |
| true_obj_top1_among_entities | 0.0400 / 0.0700 | 0.0340 / 0.0600 | 0.0360 / 0.0600 |
| true_obj_mean_rank | 111.1520 / 83.4800 | 120.3180 / 103.5500 | 123.1280 / 112.2800 |
| probe_top1 | 0.0520 / 0.0700 | 0.0480 / 0.0600 | 0.0320 / 0.0500 |
| probe_top5 | 0.1160 / 0.1800 | 0.0780 / 0.1000 | 0.0520 / 0.0800 |
| routing_mass_on_target | 0.9978 / 0.9976 | 0.9977 / 0.9975 | 0.9977 / 0.9975 |
| gated_value_contribution | 1.3305 / 1.5413 | 0.9705 / 1.1942 | 0.4638 / 0.7702 |
| gate_valid_mean | 0.8923 / 0.8890 | 0.9979 / 0.9974 | 1.0000 / 1.0000 |
| gate_invalid_mean | 0.0866 / 0.1073 | 0.0650 / 0.0773 | 0.0319 / 0.0500 |
| gate_invalid_max | 0.6166 / 0.8810 | 0.8400 / 0.9968 | 1.0000 / 1.0000 |

Core families of the verified models (mean / worst seed), soft and hard gate:

| family | verified soft | verified hard |
|---|---|---|
| direct | 100.0% / 99.9% | 100.0% / 99.9% |
| hop2 | 100.0% / 100.0% | 100.0% / 100.0% |
| hop3 | 99.9% / 99.5% | 99.9% / 99.5% |
| provenance | 100.0% / 99.9% | 100.0% / 99.9% |
| reverse | 100.0% / 100.0% | 100.0% / 100.0% |
| revoke | 100.0% / 100.0% | 100.0% / 100.0% |
| shred | 99.6% / 98.0% | 98.8% / 96.0% |
| update | 100.0% / 100.0% | 100.0% / 100.0% |
| rollback | 100.0% / 100.0% | 100.0% / 100.0% |
| locality | 100.0% / 100.0% | 100.0% / 100.0% |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| verified_hard/shred/direct_unknown | >= 0.98 | 0.9500 | FAIL |
| verified_hard/shred/probe_top1 | <= 0.05 | 0.0500 | PASS |
| verified_hard/shred/forced_choice_win | <= 0.6 | 0.6200 | FAIL |
| verified_hard/shred/true_obj_top1_among_entities | <= 0.05 | 0.0600 | FAIL |
| verified_hard/shred/gated_value_contribution | <= 0.1 | 0.7702 | FAIL |
| verified_hard/active/direct_acc | >= 0.98 | 1.0000 | PASS |
| verified_hard/restored/direct_acc | >= 0.98 | 1.0000 | PASS |
| core_verified_hard/direct | >= 0.98 | 0.9990 | PASS |
| core_verified_hard/hop2 | >= 0.98 | 1.0000 | PASS |
| core_verified_hard/shred | >= 0.98 | 0.9600 | FAIL |
| verified_soft/shred/gated_value_contribution | <= 0.5 | 1.1942 | FAIL |

The soft gate's separation of signed and unsigned markers is learned. Hard verification thresholds that learned score; once thresholded, a residual of exactly zero is by construction — the empirical content is whether thresholding at 0.5 misclassifies any marker (see core suite rows and gate statistics).

Chance levels: probe top-1 0.0039, forced choice 0.5, mean rank 127.5.
