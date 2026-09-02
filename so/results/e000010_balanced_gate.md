# E-000010 — Signature-verification gate: closing the SHRED residual (class-balanced loss)

Evidence level: **E4** (Controlled neural-network evidence); deletion level claimed for SHRED with hard verification: **F4** within the synthetic system. Seeds: [0, 1, 2, 3, 4]; 3000 steps; gate loss weight 5.0, class-balanced. Baseline = the E-000001-B models (no gate loss).

Attack battery after SHRED (mean / worst seed):

| attack after SHRED | baseline (soft gate) | verified (soft gate) | verified (hard gate) |
|---|---|---|---|
| direct_unknown | 0.9980 / 0.9900 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| direct_acc | 0.0020 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| paraphrase_unknown | 0.9980 / 0.9900 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| multihop_unknown | 0.9979 / 0.9897 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| reverse_unknown | 0.9277 / 0.8788 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| forced_choice_win | 0.5900 / 0.7000 | 0.5420 / 0.5900 | 0.5340 / 0.5900 |
| true_obj_top1_among_entities | 0.0400 / 0.0700 | 0.0080 / 0.0200 | 0.0060 / 0.0200 |
| true_obj_mean_rank | 111.1520 / 83.4800 | 125.2440 / 120.9300 | 127.0920 / 121.9600 |
| probe_top1 | 0.0520 / 0.0700 | 0.0040 / 0.0100 | 0.0020 / 0.0100 |
| probe_top5 | 0.1160 / 0.1800 | 0.0140 / 0.0200 | 0.0120 / 0.0200 |
| routing_mass_on_target | 0.9978 / 0.9976 | 0.9980 / 0.9979 | 0.9980 / 0.9979 |
| gated_value_contribution | 1.3305 / 1.5413 | 0.0475 / 0.0721 | 0.0000 / 0.0000 |
| gate_valid_mean | 0.8923 / 0.8890 | 0.9982 / 0.9981 | 1.0000 / 1.0000 |
| gate_invalid_mean | 0.0866 / 0.1073 | 0.0050 / 0.0120 | 0.0020 / 0.0099 |
| gate_invalid_max | 0.6166 / 0.8810 | 0.3321 / 0.9950 | 0.2000 / 1.0000 |

Core families of the verified models (mean / worst seed), soft and hard gate:

| family | verified soft | verified hard |
|---|---|---|
| direct | 100.0% / 99.9% | 100.0% / 99.9% |
| hop2 | 100.0% / 100.0% | 100.0% / 100.0% |
| hop3 | 99.9% / 99.5% | 99.9% / 99.5% |
| provenance | 100.0% / 99.9% | 100.0% / 99.9% |
| reverse | 100.0% / 100.0% | 100.0% / 100.0% |
| revoke | 100.0% / 100.0% | 100.0% / 100.0% |
| shred | 100.0% / 100.0% | 100.0% / 100.0% |
| update | 100.0% / 100.0% | 100.0% / 100.0% |
| rollback | 100.0% / 100.0% | 100.0% / 100.0% |
| locality | 100.0% / 100.0% | 100.0% / 100.0% |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| verified_hard/shred/direct_unknown | >= 0.98 | 1.0000 | PASS |
| verified_hard/shred/probe_top1 | <= 0.05 | 0.0100 | PASS |
| verified_hard/shred/forced_choice_win | <= 0.6 | 0.5900 | PASS |
| verified_hard/shred/true_obj_top1_among_entities | <= 0.05 | 0.0200 | PASS |
| verified_hard/shred/gated_value_contribution | <= 0.1 | 0.0000 | PASS |
| verified_hard/active/direct_acc | >= 0.98 | 1.0000 | PASS |
| verified_hard/restored/direct_acc | >= 0.98 | 1.0000 | PASS |
| core_verified_hard/direct | >= 0.98 | 0.9990 | PASS |
| core_verified_hard/hop2 | >= 0.98 | 1.0000 | PASS |
| core_verified_hard/shred | >= 0.98 | 1.0000 | PASS |
| verified_soft/shred/gated_value_contribution | <= 0.5 | 0.0721 | PASS |

The soft gate's separation of signed and unsigned markers is learned. Hard verification thresholds that learned score; once thresholded, a residual of exactly zero is by construction — the empirical content is whether thresholding at 0.5 misclassifies any marker (see core suite rows and gate statistics).

Chance levels: probe top-1 0.0039, forced choice 0.5, mean rank 127.5.
