# E-000019 — Fresh seeds, and the SHRED residual tested against chance

Evidence level: **E4**. Deletion level recorded **F4**. Seeds: [5, 6, 7] — none of them took part in selecting this configuration. Everything else is E-000010's setup, unchanged.

Two objections from the standing audit: that the configuration was selected and confirmed on the same five seeds, and that F4 is a tolerance result with no test against chance.

| claim group | supported |
|---|---|
| f4_criteria_reproduce_on_fresh_seeds | yes |
| core_families_intact | yes |
| residual_is_at_chance | yes |

Attack battery after SHRED (mean over seeds; worst seed for the hard gate):

| attack after SHRED | verified soft | verified hard | hard, worst seed |
|---|---|---|---|
| direct_unknown | 1.0000 | 0.9987 | 1.0000 |
| direct_acc | 0.0000 | 0.0013 | 0.0040 |
| paraphrase_unknown | 1.0000 | 0.9987 | 1.0000 |
| multihop_unknown | 1.0000 | 0.9993 | 1.0000 |
| reverse_unknown | 1.0000 | 1.0000 | 1.0000 |
| forced_choice_win | 0.5040 | 0.5000 | 0.5200 |
| true_obj_top1_among_entities | 0.0093 | 0.0093 | 0.0120 |
| true_obj_mean_rank | 123.7067 | 124.7453 | 126.3440 |
| probe_top1 | 0.0067 | 0.0053 | 0.0120 |
| probe_top5 | 0.0160 | 0.0147 | 0.0280 |
| routing_mass_on_target | 0.9977 | 0.9977 | 0.9980 |
| gated_value_contribution | 0.0517 | 0.0218 | 0.0653 |
| gate_valid_mean | 0.9978 | 1.0000 | 1.0000 |
| gate_invalid_mean | 0.0033 | 0.0013 | 0.0040 |
| gate_invalid_max | 0.3360 | 0.3333 | 1.0000 |

The residual against its chance level, pooled over seeds:

| measure | successes | rate | chance | 95% exact interval | at chance |
|---|---|---|---|---|---|
| probe_top1 | 4/750 | 0.0053 | 0.0039 | [0.0015, 0.0136] | yes |
| true_obj_top1_among_entities | 7/750 | 0.0093 | 0.0039 | [0.0038, 0.0191] | yes |
| forced_choice_win | 375/750 | 0.5000 | 0.5000 | [0.4636, 0.5364] | yes |

A residual counts as being at chance when its pooled exact binomial interval CONTAINS the chance level and its upper end stays within 0.02 of it (forced choice: within 0.05 of 0.5). This is stronger than the F4 bars, which only require the point estimate to fall below a threshold.

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| verified_hard/shred/direct_unknown | >= 0.98 | 0.9960 | PASS |
| verified_hard/shred/probe_top1 | <= 0.05 | 0.0120 | PASS |
| verified_hard/shred/forced_choice_win | <= 0.6 | 0.5200 | PASS |
| verified_hard/shred/true_obj_top1_among_entities | <= 0.05 | 0.0120 | PASS |
| verified_hard/shred/gated_value_contribution | <= 0.1 | 0.0653 | PASS |
| verified_hard/active/direct_acc | >= 0.98 | 1.0000 | PASS |
| verified_hard/restored/direct_acc | >= 0.98 | 1.0000 | PASS |
| core/direct | >= 0.98 | 1.0000 | PASS |
| core/hop2 | >= 0.98 | 1.0000 | PASS |
| core/shred | >= 0.98 | 1.0000 | PASS |
| eq/probe_top1 | >= 1.0 | 1.0000 | PASS |
| eq/true_obj_top1_among_entities | >= 1.0 | 1.0000 | PASS |
| eq/forced_choice_win | >= 1.0 | 1.0000 | PASS |

The soft gate's separation of signed from unsigned markers is learned; hard verification thresholds that learned score, so a residual of exactly zero after thresholding is by construction. What this record adds is that the residual measured on seeds that took no part in choosing the configuration sits where chance puts it, with the interval shown.
