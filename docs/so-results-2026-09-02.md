# SO — Session results 2026-09-02

**What this is:** the record of what the code in `so/` actually demonstrated in this session, assembled automatically from `so/results/*.json` by `python -m so.report`. Companion documents: [architecture](so-modular-neural-os.md), [experiment ledger](so-experiment-ledger.md), [roadmap](so-roadmap-2026-09-02.md).

**Evidence scale** (ledger section 4): E0 = Idea only, E1 = Analytical / conceptual support, E2 = Toy implementation, E3 = Repeated synthetic evidence, E4 = Controlled neural-network evidence, E5 = Transformer evidence, E6 = Real pretrained LLM evidence, E7 = Scalable / externally reproduced evidence.

**Deletion levels** (ledger section 6): F0 = Access suppression, F1 = Routing removal, F2 = Component removal, F3 = Functional forgetting, F4 = Representational removal, F5 = Reconstruction-resistant deletion.

## Summary

| experiment | title | evidence level claimed | deletion level | status |
|---|---|---|---|---|
| E-000001-A | Mechanical reference implementation | E3 | F1 | all tests passed |
| E-000001-B | Trained Mini-Transformer over the mutable knowledge layer | E4 | F3 | criteria met |
| e000002_memorization_control | - | - | - | not run |
| E-000003 | Retention and generalisation of deletion | E4 | F3 | criteria met |
| E-000004 | Reconstruction attacks against REVOKE and SHRED | E4 | F4 | **criteria NOT met** |
| E-000005 | Causal interventions on knowledge cells | E4 | - | criteria met |
| e000006_ablations | - | - | - | not run |
| E-000007 | Biomarker: output suppression versus representational change | E4 | F4 | **criteria NOT met** |
| e000008_gpt2_adapter | - | - | - | not run |

## The six breakthrough properties (ledger section 3)

| property | experiments | status | highest level |
|---|---|---|---|
| Selectivity (target disappears) | E-000001-B, E-000003 | E-000001-B: criteria met; E-000003: criteria met | E4 |
| Retention (non-target intact) | E-000001-B, E-000003 | E-000001-B: criteria met; E-000003: criteria met | E4 |
| Generalisation (paraphrases, alternative queries) | E-000003, E-000004 | E-000003: criteria met; E-000004: **criteria NOT met** | E4 |
| Causal isolation (effect follows from the intended structure) | E-000005, E-000007 | E-000005: criteria met; E-000007: **criteria NOT met** | E4 |
| Reconstruction resistance | E-000004, E-000007 | E-000004: **criteria NOT met**; E-000007: **criteria NOT met** | E4 |
| Scalability (path beyond toy models) |  | not run | - |

Scalability is the property this session can least address: E-000008 shows the mechanism attaches to a frozen pretrained transformer on CPU; the path to LLM scale is a roadmap item, not a result.

## Boundary of this evidence

Everything below was produced on 4 CPU cores in one session, with no GPU. It is therefore bounded as follows:

- **No LLM-scale evidence.** The largest neural core used is frozen GPT-2 small (124M parameters, E-000008). Nothing here shows editable knowledge inside a large pretrained model, and nothing here shows unlearning of facts that a pretrained model already encodes in its weights.
- **Synthetic worlds.** Facts are `(subject, relation) → object` triples over 256 entities and 4 relations; queries are symbolic (E-000001 … E-000007) or short natural-language templates (E-000008). Real-world knowledge, multi-token entities and free-text questions are not covered.
- **By construction versus learned.** REVOKE removes routing by a hard mask; that is deletion level F1 by construction. The learned results are: answering UNKNOWN instead of using another cell, refusing a payload whose marker is invalid (SHRED), composing hops, and the probe / forced-choice / rank checks after SHRED. Every result table states which is which.
- **Provenance is trained**, not emergent: the routing loss supervises which cell each hop reads. E-000006 (`no_routing_loss`) measures what remains without it.
- **The outstanding C55–C57 real-model / GPU chain of the ledger is still outstanding.** E-000008 is its CPU-feasible analogue on a small model, not its execution.
- **Noise figures are not comparable** with the architecture document's "noise = 0.24 → 68.4%", whose noise definition is not recorded; the sweep here perturbs bank keys and values relative to their RMS.
- **Seeds.** E-000003 … E-000007 evaluate the same five E-000001-B models on fresh worlds; they are not independent replications of training. E-000002, E-000006 and E-000008 train their own models (3 seeds).


## Reproduction

```bash
pip install -r so/requirements.txt && pip install transformers safetensors
python -m pytest so/tests -q
python -m so.experiments.run_all          # full chain, then: python -m so.report
```

Trained models are cached under `so/results/checkpoints/` (not committed); every JSON record carries the environment, configuration, per-seed numbers, pre-registered criteria and the claim / not-claimed text.

## Per-experiment records

## E-000001-A — Mechanical reference implementation

Evidence level: **E3** (Repeated synthetic evidence). Deletion level exercised: **F1** (routing removal) plus marker shredding at the mechanical level.

Seeds: [0, 1, 2, 3, 4] · cells per seed: 1000 · all tests passed: **True**

| Measure | Mean over seeds | Worst seed |
|---|---|---|
| direct | 100.0% | 100.0% |
| hop2 | 100.0% | 100.0% |
| hop3 | 100.0% | 100.0% |
| hop2_broken_unknown | 100.0% | 100.0% |
| hop3_broken_unknown | 100.0% | 100.0% |
| provenance | 100.0% | 100.0% |
| update_rollback | 100.0% | 100.0% |
| locality | 100.0% | 100.0% |
| locality_targets_changed | 100.0% | 100.0% |
| locality_undo_exact | 100.0% | 100.0% |
| alternative_path | 100.0% | 100.0% |
| replay_deviation | 0 | - |

Per seed:

| seed | direct | hop2 | hop3 | hop2_broken_unknown | hop3_broken_unknown | provenance | update_rollback | locality | locality_targets_changed | locality_undo_exact | alternative_path | replay_deviation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 |

Interpretation: establishes that the desired semantics are coherent in the controlled reference system. It does not show that a trained neural network reproduces them (that is E-000001-B).

## E-000001-B — Trained Mini-Transformer over the mutable knowledge layer

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

## e000002_memorization_control

_not run in this session_

## E-000003 — Retention and generalisation of deletion

Evidence level: **E4** (Controlled neural-network evidence); deletion level **F3** (functional forgetting generalising over paraphrases, multi-hop and reverse access). Seeds: [0, 1, 2, 3, 4]

| measure | mean | min | max |
|---|---|---|---|
| before/target_para_acc | 100.0% | 100.0% | 100.0% |
| before/target_para_unknown | 0.0% | 0.0% | 0.0% |
| before/target_hop2_unknown | 0.0% | 0.0% | 0.0% |
| before/target_hop2_ref_agree | 100.0% | 100.0% | 100.0% |
| before/target_rev_unknown | 0.0% | 0.0% | 0.0% |
| before/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| before/control_para_acc | 100.0% | 100.0% | 100.0% |
| before/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| before/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| revoke/target_para_acc | 0.0% | 0.0% | 0.0% |
| revoke/target_para_unknown | 100.0% | 100.0% | 100.0% |
| revoke/target_hop2_unknown | 100.0% | 100.0% | 100.0% |
| revoke/target_hop2_ref_agree | 100.0% | 100.0% | 100.0% |
| revoke/target_rev_unknown | 100.0% | 100.0% | 100.0% |
| revoke/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| revoke/control_para_acc | 100.0% | 100.0% | 100.0% |
| revoke/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| revoke/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| shred/target_para_acc | 0.0% | 0.0% | 0.0% |
| shred/target_para_unknown | 100.0% | 100.0% | 100.0% |
| shred/target_hop2_unknown | 99.6% | 98.0% | 100.0% |
| shred/target_hop2_ref_agree | 99.6% | 98.0% | 100.0% |
| shred/target_rev_unknown | 95.8% | 90.5% | 100.0% |
| shred/bypass_hop2_acc | 100.0% | 100.0% | 100.0% |
| shred/control_para_acc | 100.0% | 100.0% | 100.0% |
| shred/unrelated_para_acc | 100.0% | 100.0% | 100.0% |
| shred/general_fresh_world_acc | 100.0% | 100.0% | 100.0% |
| update/target_para_new_obj_acc | 100.0% | 100.0% | 100.0% |
| update/target_para_old_obj_rate | 0.0% | 0.0% | 0.0% |
| update/target_rev_old_obj_ref_agree | 100.0% | 100.0% | 100.0% |
| update/control_para_acc | 100.0% | 100.0% | 100.0% |
| rollback/target_para_acc | 100.0% | 100.0% | 100.0% |
| after_all/identical_to_before | 100.0% | 100.0% | 100.0% |

Pattern required by the ledger (section 16): target high → low, control high → high. 'target_para_unknown' after revoke/shred is the deletion; 'control_para_acc', 'unrelated_para_acc' and 'bypass_hop2_acc' are the retention side ('general_fresh_world_acc' is a by-construction sanity row). Sample sizes per seed: targets 50 x 2 paraphrases, controls 50 x 2, unrelated up to 50 x 2, bypass 300, reverse only where the subject is unique (see n_target_rev).

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| before/target_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/target_para_unknown | >= 0.98 | 1.0000 | PASS |
| revoke/target_hop2_unknown | >= 0.98 | 1.0000 | PASS |
| revoke/control_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/unrelated_para_acc | >= 0.98 | 1.0000 | PASS |
| revoke/bypass_hop2_acc | >= 0.98 | 1.0000 | PASS |
| shred/target_para_unknown | >= 0.95 | 1.0000 | PASS |
| shred/control_para_acc | >= 0.98 | 1.0000 | PASS |
| update/target_para_new_obj_acc | >= 0.98 | 1.0000 | PASS |
| update/target_para_old_obj_rate | <= 0.02 | 0.0000 | PASS |
| rollback/target_para_acc | >= 0.98 | 1.0000 | PASS |

REVOKE removes routing by mask (F1), so its effect on every access path (paraphrase, multi-hop, reverse) follows from canonical addressing of one cell; what is learned is that the model answers UNKNOWN instead of using another cell. SHRED leaves the cell routable: refusing it on every path is learned (F3). 'general_fresh_world_acc' uses a separate store and cannot change — it is a sanity row, not evidence of retention.

## E-000004 — Reconstruction attacks

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

## E-000005 — Causal interventions

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2, 3, 4], 100 targets per seed.

| intervention | predicted outcome | observed (mean) | worst seed |
|---|---|---|---|
| disable | UNKNOWN | 100.0% | 100.0% |
| disable_random_other | unchanged | 100.0% | 100.0% |
| swap | partner's object | 100.0% | 100.0% |
| swap_partner | target's object | 100.0% | 100.0% |
| restore | both original | 100.0% | 100.0% |
| replace | new object | 100.0% | 100.0% |
| localization | routed cell == ground-truth cell | 100.0% | 100.0% |
| routed_cell_causal | disabling routed cell -> UNKNOWN | 100.0% | 100.0% |

That the read equation uses the cell's payload is by construction; what is tested is that the trained core actually routes each query to its own cell (localisation), does not draw the answer from anywhere else (disable -> UNKNOWN, random-other -> unchanged) and turns a swapped or replaced payload into exactly the predicted answer. Localisation is a trained objective (routing loss); E-000006 'no_routing_loss' reports how much of it emerges without that supervision.

n = 100 targets per seed. Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| disable | >= 0.98 | 1.0000 | PASS |
| disable_random_other | >= 0.98 | 1.0000 | PASS |
| swap | >= 0.98 | 1.0000 | PASS |
| swap_partner | >= 0.98 | 1.0000 | PASS |
| restore | >= 0.98 | 1.0000 | PASS |
| replace | >= 0.98 | 1.0000 | PASS |
| localization | >= 0.98 | 1.0000 | PASS |
| routed_cell_causal | >= 0.98 | 1.0000 | PASS |

## e000006_ablations

_not run in this session_

## E-000007 — Biomarker: suppression versus representational change

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

## e000008_gpt2_adapter

_not run in this session_
