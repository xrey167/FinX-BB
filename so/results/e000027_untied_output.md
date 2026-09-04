# E-000027 (output) — the knowledge layer on EleutherAI/pythia-160m (untied embeddings), payload from the output embedding

Model `EleutherAI/pythia-160m`, `tie_word_embeddings` = False. Seeds [0, 1], 1500 steps,
read at blocks (8, 10). Everything else is E-000008's protocol unchanged: worlds are
re-sampled every step, the core is frozen, and only the adapter is trained.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.7 | 0.0000 | FAIL |
| paraphrase | >= 0.6 | 0.0000 | FAIL |
| bank_masked_direct_acc | <= 0.01 | 0.0000 | PASS |

The `output` arm is the corrected mechanism. 

## All measures

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| prior_direct_acc | 0.0045 | 0.0020 | 2000 | 0.0021 | 0.0085 |
| bank_masked_direct_acc | 0.0000 | 0.0000 | 2000 | 0.0000 | 0.0018 |
| bank_masked_full_vocab_top1_equals_prior | 0.0000 | 0.0000 | - | - | - |
| direct | 0.4085 | 0.0000 | 2000 | 0.3869 | 0.4304 |
| direct_full_vocab_top1 | 0.0780 | 0.0000 | - | - | - |
| paraphrase | 0.3700 | 0.0000 | 2000 | 0.3488 | 0.3916 |
| provenance_direct | 0.9385 | 0.9230 | 2000 | 0.9271 | 0.9486 |
| hop2 | 0.0000 | 0.0000 | - | - | - |
| broken1_unknown | 0.8700 | 1.0000 | - | - | - |
| broken2_unknown | 0.9750 | 1.0000 | - | - | - |
| lifecycle_all | 0.5733 | 0.3333 | - | - | - |
| update | 0.4200 | 0.0000 | - | - | - |
| revoke | 0.9100 | 0.8200 | - | - | - |
| shred | 0.9400 | 0.8800 | - | - | - |
| locality | 0.9924 | 0.9847 | - | - | - |
| probe_calibration_top1 | 0.1917 | 0.0000 | - | - | - |
