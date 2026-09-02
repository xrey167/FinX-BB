# E-000007 — Biomarker: suppression versus representational change

Evidence level: **E4** (Controlled neural-network evidence); deletion level **F4** within the synthetic system. Seeds: [0, 1, 2, 3, 4]. Chance levels: probe top-1 0.0039, top-5 0.0195, mean rank 127.5, forced choice 0.5.

| signal (mean over seeds) | active | revoked | shredded | suppressed |
|---|---|---|---|---|
| target_unknown | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| target_acc | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| control_acc | 1.0000 | 1.0000 | 1.0000 | 0.9840 |
| routing_mass_on_target | 0.9979 | 0.0000 | 0.9979 | 0.6228 |
| gated_value_contribution | 13.9753 | 0.0000 | 1.2678 | 8.2779 |
| probe_top1 | 0.9520 | 0.0000 | 0.0440 | 0.8640 |
| probe_top5 | 0.9520 | 0.0120 | 0.0720 | 0.9080 |
| true_obj_top1_among_entities | 1.0000 | 0.0000 | 0.0160 | 0.8360 |
| true_obj_mean_rank | 0.0000 | 123.9880 | 109.5560 | 9.8600 |
| forced_choice_win | 1.0000 | 0.5520 | 0.6040 | 0.9560 |

Reading: 'suppressed' keeps the biomarker (value contribution) and the probe leak while answering UNKNOWN — output suppression, ledger F0. 'shredded' keeps routing mass (the key is unchanged, by construction) but loses value contribution and probe leak — representational removal, F4, learned. 'revoked' loses both by the mask (F1). The probe used on 'suppressed' is refitted on that model's own active cells.

n = 50 targets, 50 controls per seed. Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| suppressed/target_unknown | >= 0.95 | 1.0000 | PASS |
| suppressed/control_acc | >= 0.95 | 0.9600 | PASS |
| suppressed/gated_value_contribution | >= 0.3 | 7.8362 | PASS |
| suppressed/probe_top1 | >= 0.5 | 0.7400 | PASS |
| shredded/target_unknown | >= 0.95 | 1.0000 | PASS |
| shredded/gated_value_contribution | <= 0.1 | 1.5742 | FAIL |
| shredded/probe_top1 | <= 0.05 | 0.0800 | FAIL |
| revoked/probe_top1 | <= 0.05 | 0.0000 | PASS |
