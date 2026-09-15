# E-000050 — the held-out paraphrase gap is the position-0 token, and what a BOS buys

Seeds [0, 1, 2], 3000 steps for the BOS-trained arm, 100 targets per seed for the
decomposition. GPT-2's tokenizer prepends no BOS, so a subject-initial prompt makes the subject the
position-0 token. Worst seed everywhere.

| arm | weights, prompt | held-out reading | held-out subject-initial read / route | held-out subject-medial read / route | trained reading | SHRED reaches worst held-out | generic KL (max) |
|---|---|---|---|---|---|---|---|
| A | recorded, no BOS | 0.7288 | 0.37 / 0.54 | 0.95 / 0.94 | 0.9119 | 0.8650 | 3.647 |
| B | recorded, BOS at inference | 0.9175 | 0.97 / 0.98 | 0.70 / 0.64 | 0.9719 | 0.8400 | 3.920 |
| C | BOS-trained, BOS | 0.9712 | 0.99 / 0.97 | 0.91 / 0.83 | 0.9956 | 0.9900 | 4.224 |
| D | BOS-trained, no BOS | 0.4975 | 0.00 / 0.00 | 0.95 / 0.97 | 0.6225 | 1.0000 | 3.802 |

## The rule, fixed before the run

C passes every row -> the held-out paraphrase gap of this addressable memory was the position-0 token, the honest held-out numbers for the memory are C's, and every held-out number in the record (E-000017's kill criterion, E-000025's bimodality, E-000026's template choice, E-000039-B) is re-scoped as measured without a BOS. C passes the subject-initial rows and fails the medial ones -> the artefact is the subject-initial half only and the remainder is semantic. C fails the trained-template or generic rows -> a BOS at training time costs capability and the finding is B's alone. B fails its initial row -> the probe was one seed's fluke and nothing here is claimed. D not degrading -> position 0 was not the cause. Fixed before the run.

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| A/heldout/active_correct | <= 0.8 | 0.7587 | PASS |
| B/heldout_initial/read_min | >= 0.9 | 0.9700 | PASS |
| B/heldout_medial/read_min | <= 0.9 | 0.7900 | PASS |
| C/heldout/active_correct | >= 0.95 | 0.9712 | PASS |
| C/heldout/route_hit_min | >= 0.95 | 0.8300 | FAIL |
| C/heldout_initial/read_min | >= 0.95 | 0.9900 | PASS |
| C/heldout_medial/read_min | >= 0.95 | 0.9100 | FAIL |
| C/train/active_correct | >= 0.95 | 0.9956 | PASS |
| C/shred_heldout_min | >= 0.95 | 0.9900 | PASS |
| C/revoke_heldout_min | >= 0.95 | 0.9900 | PASS |
| C/heldout/revoked_deleted_object | <= 0.02 | 0.0000 | PASS |
| C/broken1_unknown | >= 0.63 | 0.9350 | PASS |
| C/generic/kl_to_base | <= 3.65 | 4.2238 | FAIL |
| D/heldout_initial/read_min | <= 0.85 | 0.0100 | PASS |
