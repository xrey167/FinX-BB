# E-000008 — Frozen pretrained GPT-2 core with the mutable knowledge layer

Evidence level: **E5** (Transformer evidence), partial E6 (a real pretrained LM, GPT-2 small, on CPU). Deletion level within this system: **F4**. Seeds: [0, 1, 2]; adapter steps: 2000; the 124M pretrained weights are frozen.

| measure | mean over seeds | worst seed |
|---|---|---|
| prior_direct_acc | 0.6% | 0.4% |
| bank_masked_direct_acc | 0.0% | 0.0% |
| bank_masked_unknown_rate | 100.0% | 100.0% |
| direct | 88.9% | 88.5% |
| direct_full_vocab_top1 | 83.7% | 80.5% |
| paraphrase | 99.9% | 99.9% |
| provenance_direct | 84.2% | 83.6% |
| hop2 | 75.3% | 72.3% |
| broken1_unknown | 63.7% | 56.0% |
| broken2_unknown | 66.3% | 62.0% |
| update | 95.3% | 95.0% |
| rollback | 96.0% | 96.0% |
| revoke | 56.3% | 49.0% |
| restore | 96.0% | 96.0% |
| shred | 38.0% | 33.0% |
| resign | 96.0% | 96.0% |
| lifecycle_all | 79.6% | 77.5% |
| locality | 99.3% | 98.9% |
| locality_targets_correct | 78.2% | 77.3% |
| locality_undo_exact | 100.0% | 100.0% |

Attacks on 100 targets (mean over seeds; chance: forced choice 0.5, top-1 among entities 0.0039, mean rank 127.5, probe top-1 0.0039 / top-5 0.0195):

| attack | active | after REVOKE | after SHRED |
|---|---|---|---|
| direct_unknown | 0.1000 | 0.5400 | 0.5067 |
| direct_acc | 0.8833 | 0.0000 | 0.1400 |
| paraphrase_unknown | 0.0000 | 0.6067 | 0.3400 |
| forced_choice_win | 1.0000 | 0.4967 | 0.6733 |
| true_obj_top1_among_entities | 0.9467 | 0.0033 | 0.1800 |
| true_obj_mean_rank | 0.2133 | 133.8300 | 77.3567 |
| probe_top1 | 0.7767 | 0.0000 | 0.1500 |
| probe_top5 | 0.8767 | 0.0167 | 0.2100 |
| routing_mass_on_target | 0.7661 | 0.0000 | 0.7661 |
| gated_value_contribution | 13.2832 | 0.0000 | 0.3019 |
| full_vocab_top1_equals_prior | 0.0067 | 0.0333 | 0.0300 |
| full_vocab_top1_is_unknown_word | 0.0100 | 0.1300 | 0.1133 |

Probe calibration on held-out active cells: top-1 0.802, top-5 0.880.

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| prior_direct_acc | <= 0.05 | 0.0100 | PASS |
| bank_masked_direct_acc | <= 0.05 | 0.0000 | PASS |
| direct | >= 0.95 | 0.8850 | FAIL |
| paraphrase | >= 0.95 | 0.9990 | PASS |
| broken1_unknown | >= 0.9 | 0.5600 | FAIL |
| revoke | >= 0.95 | 0.4900 | FAIL |
| restore | >= 0.95 | 0.9600 | PASS |
| update | >= 0.95 | 0.9500 | PASS |
| rollback | >= 0.95 | 0.9600 | PASS |
| shred | >= 0.9 | 0.3300 | FAIL |
| resign | >= 0.95 | 0.9600 | PASS |
| locality | >= 0.98 | 0.9894 | PASS |
| revoke/probe_top1 | <= 0.05 | 0.0000 | PASS |
| revoke/forced_choice_win | <= 0.6 | 0.5300 | PASS |
| shred/probe_top1 | <= 0.05 | 0.1800 | FAIL |
| shred/forced_choice_win | <= 0.6 | 0.7300 | FAIL |
| restored/direct_acc | >= 0.95 | 0.8600 | FAIL |

The frozen core cannot copy a fact by construction; whether the ADAPTER copies is measured by the masked-bank rows (must equal the prior). REVOKE is a mask (F1); what is learned is reading the right cell from natural-language prompts, turning the value into the object token through the unchanged LM head, answering ' unknown' for null reads, and refusing a shredded payload.

Reading: 'prior_direct_acc' is what frozen GPT-2 answers without the layer (chance); 'bank_masked_direct_acc' is the adapter with every cell masked — the copy bound: it must not exceed the prior. 'direct_full_vocab_top1' is the fraction of direct queries where the object token wins over the entire 50,257-token vocabulary, not only among the 257 candidates. 'full_vocab_top1_equals_prior' after REVOKE shows whether the model falls back to its pretrained prior once the cell is gone.
