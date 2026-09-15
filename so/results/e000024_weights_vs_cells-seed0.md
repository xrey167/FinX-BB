# E-000024 — deleting a fact from weights versus deleting it from cells

Seeds [0]; 400 facts, 50 deletion targets, 50 bystanders.
The cells arm is the frozen GPT-2 of E-000012 with its trained adapter; the weights arms are the same
frozen GPT-2 with a rank-16 LoRA fine-tuned on the identical facts and then unlearned two ways.
All three arms are driven to the same surface criterion and attacked identically.

## The comparison (worst seed)

| measure | cells (SHRED) | weights, ascent | weights, relabel | chance |
|---|---|---|---|---|
| direct question answered | 0.0200 | 0.0000 | 0.0200 | - |
| held-out paraphrase answered | 0.0000 | 0.0000 | 0.0200 | - |
| forced choice, true vs random | 0.4400 | 0.7800 | 1.0000 | 0.5000 |
| true object top-1 of 256 | 0.0200 | 0.1200 | 0.8000 | 0.0039 |
| mean rank of true object | 137.2000 | 50.2600 | 0.3400 | 127.5 |
| linear probe top-1 | 0.0000 | 0.0600 | 0.0200 | 0.0039 |
| relearn attack: held-out recovered | 0.0000 | 0.4800 | 0.7200 | - |
| relearn attack: supplied recovered | 0.6000 | 0.9600 | 1.0000 | - |
| bystander facts still answered | 0.9400 | 0.2800 | 0.9400 | - |
| L2 change of model weights | 0.0000 | 56.8141 | 56.8649 | - |
| perplexity on ordinary prose | 42.9338 | 6192810487.1065 | 6388351.9689 | - |
| KL to the pretrained model, generic text | 3.1047 | 15.1225 | 10.3308 | 0.0000 |
| generic top-1 still the pretrained one | 0.0050 | 0.0050 | 0.0300 | 1.0000 |
| seconds to delete 50 facts | 0.0008 | 129.4947 | 334.9378 | - |

## Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| cells/after/direct_acc | <= 0.02 | 0.0200 | PASS |
| ga/after/direct_acc | <= 0.02 | 0.0000 | PASS |
| relabel/after/direct_acc | <= 0.02 | 0.0200 | PASS |
| weights/before/direct_acc | >= 0.8 | 0.9600 | PASS |
| cells/after/forced_choice | <= 0.6 | 0.4400 | PASS |
| cells/after/true_obj_top1 | <= 0.02 | 0.0200 | PASS |
| cells/relearn/heldout_acc | <= 0.05 | 0.0000 | PASS |
| cells/relearn/supplied_acc | >= 0.5 | 0.6000 | PASS |
| ga/relearn/supplied_acc | >= 0.5 | 0.9600 | PASS |
| relabel/relearn/supplied_acc | >= 0.5 | 1.0000 | PASS |
| cells/weight_delta_l2 | <= 0.0 | 0.0000 | PASS |
| cells/ppl_delta | <= 0.0 | 0.0000 | PASS |
| cells/after/bystander_acc | >= 0.7 | 0.9400 | PASS |

## All measures

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| cells/before/direct_acc | 0.9200 | 0.9200 | 50 | 0.8077 | 0.9778 |
| cells/after/direct_acc | 0.0200 | 0.0200 | 50 | 0.0005 | 0.1065 |
| cells/after/paraphrase_acc | 0.0000 | 0.0000 | 50 | 0.0000 | 0.0711 |
| cells/after/forced_choice | 0.4400 | 0.4400 | 50 | 0.2999 | 0.5875 |
| cells/after/true_obj_top1 | 0.0200 | 0.0200 | 50 | 0.0005 | 0.1065 |
| cells/after/true_obj_mean_rank | 137.2000 | 137.2000 | - | - | - |
| cells/after/probe_top1 | 0.0000 | 0.0000 | 50 | 0.0000 | 0.0711 |
| cells/after/probe_top5 | 0.0000 | 0.0000 | 50 | 0.0000 | 0.0711 |
| cells/after/bystander_acc | 0.9400 | 0.9400 | 50 | 0.8345 | 0.9875 |
| cells/ppl_base | 42.9338 | 42.9338 | - | - | - |
| cells/ppl_after | 42.9338 | 42.9338 | - | - | - |
| cells/ppl_delta | 0.0000 | 0.0000 | - | - | - |
| cells/after/generic_kl_mean | 3.1047 | 3.1047 | - | - | - |
| cells/after/generic_top1_matches_base | 0.0050 | 0.0050 | - | - | - |
| cells/relearn/supplied_acc | 0.6000 | 0.6000 | - | - | - |
| cells/relearn/heldout_acc | 0.0000 | 0.0000 | - | - | - |
| cells/relearn/heldout_forced_choice | 0.5600 | 0.5600 | 25 | 0.3493 | 0.7560 |
| cells/relearn/heldout_top1 | 0.0000 | 0.0000 | - | - | - |
| cells/relearn/steps_used | 200.0000 | 200.0000 | - | - | - |
| cells/weight_delta_l2 | 0.0000 | 0.0000 | - | - | - |
| cells/delete_seconds | 0.0008 | 0.0008 | - | - | - |
| weights/before/direct_acc | 0.9600 | 0.9600 | 50 | 0.8629 | 0.9951 |
| ga/after/direct_acc | 0.0000 | 0.0000 | 50 | 0.0000 | 0.0711 |
| ga/after/paraphrase_acc | 0.0000 | 0.0000 | 50 | 0.0000 | 0.0711 |
| ga/after/forced_choice | 0.7800 | 0.7800 | 50 | 0.6404 | 0.8847 |
| ga/after/true_obj_top1 | 0.1200 | 0.1200 | 50 | 0.0453 | 0.2431 |
| ga/after/true_obj_mean_rank | 50.2600 | 50.2600 | - | - | - |
| ga/after/probe_top1 | 0.0600 | 0.0600 | 50 | 0.0125 | 0.1655 |
| ga/after/probe_top5 | 0.0800 | 0.0800 | 50 | 0.0222 | 0.1923 |
| ga/after/bystander_acc | 0.2800 | 0.2800 | 50 | 0.1623 | 0.4249 |
| ga/relearn/supplied_acc | 0.9600 | 0.9600 | - | - | - |
| ga/relearn/heldout_acc | 0.4800 | 0.4800 | - | - | - |
| ga/relearn/heldout_forced_choice | 0.8800 | 0.8800 | 25 | 0.6878 | 0.9745 |
| ga/relearn/heldout_top1 | 0.4800 | 0.4800 | - | - | - |
| ga/relearn/steps_used | 40.0000 | 40.0000 | - | - | - |
| ga/weight_delta_l2 | 56.8141 | 56.8141 | - | - | - |
| ga/delete_seconds | 129.4947 | 129.4947 | - | - | - |
| ga/ppl_after | 6192810487.1065 | 6192810487.1065 | - | - | - |
| ga/after/generic_kl_mean | 15.1225 | 15.1225 | - | - | - |
| ga/after/generic_top1_matches_base | 0.0050 | 0.0050 | - | - | - |
| relabel/after/direct_acc | 0.0200 | 0.0200 | 50 | 0.0005 | 0.1065 |
| relabel/after/paraphrase_acc | 0.0200 | 0.0200 | 50 | 0.0005 | 0.1065 |
| relabel/after/forced_choice | 1.0000 | 1.0000 | 50 | 0.9289 | 1.0000 |
| relabel/after/true_obj_top1 | 0.8000 | 0.8000 | 50 | 0.6628 | 0.8997 |
| relabel/after/true_obj_mean_rank | 0.3400 | 0.3400 | - | - | - |
| relabel/after/probe_top1 | 0.0200 | 0.0200 | 50 | 0.0005 | 0.1065 |
| relabel/after/probe_top5 | 0.0800 | 0.0800 | 50 | 0.0222 | 0.1923 |
| relabel/after/bystander_acc | 0.9400 | 0.9400 | 50 | 0.8345 | 0.9875 |
| relabel/relearn/supplied_acc | 1.0000 | 1.0000 | - | - | - |
| relabel/relearn/heldout_acc | 0.7200 | 0.7200 | - | - | - |
| relabel/relearn/heldout_forced_choice | 1.0000 | 1.0000 | 25 | 0.8628 | 1.0000 |
| relabel/relearn/heldout_top1 | 0.7200 | 0.7200 | - | - | - |
| relabel/relearn/steps_used | 40.0000 | 40.0000 | - | - | - |
| relabel/weight_delta_l2 | 56.8649 | 56.8649 | - | - | - |
| relabel/delete_seconds | 334.9378 | 334.9378 | - | - | - |
| relabel/ppl_after | 6388351.9689 | 6388351.9689 | - | - | - |
| relabel/after/generic_kl_mean | 10.3308 | 10.3308 | - | - | - |
| relabel/after/generic_top1_matches_base | 0.0300 | 0.0300 | - | - | - |
| weights/ppl_base | 42.9338 | 42.9338 | - | - | - |
| weights/ppl_after_learning | 3188915.4781 | 3188915.4781 | - | - | - |
| weights/before/generic_kl_mean | 9.4130 | 9.4130 | - | - | - |
| weights/n_lora_params | 2359296.0000 | 2359296.0000 | - | - | - |
| weights/train_steps_used | 2000.0000 | 2000.0000 | - | - | - |
