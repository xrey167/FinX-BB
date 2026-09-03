# SO — Session results 2026-09-02

**What this is:** the record of what the code in `so/` actually demonstrated in this session, assembled automatically from `so/results/*.json` by `python -m so.report`. Companion documents: [architecture](so-modular-neural-os.md), [experiment ledger](so-experiment-ledger.md), [roadmap](so-roadmap-2026-09-02.md).

**Evidence scale** (ledger section 4): E0 = Idea only, E1 = Analytical / conceptual support, E2 = Toy implementation, E3 = Repeated synthetic evidence, E4 = Controlled neural-network evidence, E5 = Transformer evidence, E6 = Real pretrained LLM evidence, E7 = Scalable / externally reproduced evidence.

**Deletion levels** (ledger section 6): F0 = Access suppression, F1 = Routing removal, F2 = Component removal, F3 = Functional forgetting, F4 = Representational removal, F5 = Reconstruction-resistant deletion.

## Summary

| experiment | title | evidence level recorded | deletion level recorded (targeted) | status | recorded at (UTC) |
|---|---|---|---|---|---|
| E-000001-A | Mechanical reference implementation | E3 | F1 | all tests passed | 2026-09-02 22:03 |
| E-000001-B | Trained Mini-Transformer over the mutable knowledge layer | E4 | F3 | criteria met | 2026-09-02 23:59 |
| E-000002 | Weight-memorisation control (copy problem) | E4 | - | criteria met | 2026-09-02 23:59 |
| E-000003 | Retention and generalisation of deletion | E4 | F3 | criteria met | 2026-09-02 23:59 |
| E-000004 | Reconstruction attacks against REVOKE and SHRED | E4 | F3 (F4) | **criteria NOT met** | 2026-09-03 00:00 |
| E-000005 | Causal interventions on knowledge cells | E4 | - | criteria met | 2026-09-03 00:00 |
| E-000006 | Ablations | E4 | - | **criteria NOT met** | 2026-09-03 00:39 |
| E-000007 | Biomarker: output suppression versus representational change | E4 | F3 (F4) | **criteria NOT met** | 2026-09-03 00:01 |
| E-000008 | Frozen pretrained GPT-2 core with the mutable knowledge layer (symlink adapter) | E5 | F1 (F4) | **criteria NOT met** | 2026-09-03 00:44 |
| E-000009 | Signature-verification gate: closing the SHRED residual | E4 | F3 (F4) | **criteria NOT met** | 2026-09-03 00:04 |
| E-000010 | Signature-verification gate: closing the SHRED residual (class-balanced loss) | E4 | F4 (F4) | criteria met | 2026-09-03 00:05 |
| E-000011 | Frozen GPT-2 core v2: verified gate, deletion behaviour, held-out paraphrases, interventions | E5 | F1 (F4) | **criteria NOT met** | 2026-09-03 07:50 |
| e000012_status_gated_revoke | - | - | - | not run | - |
| e000013_prior_conflict | - | - | - | not run | - |
| E-000014 | Addressing at 10,000 cells (2,560 entities), verified gate | E4 | F4 (F4) | criteria met | 2026-09-03 05:58 |
| E-000015 | Explicit symlink cells: several access keys share one knowledge object (symlink arm versus duplication arm) | E4 | F3 (F4) | **criteria NOT met** | 2026-09-03 08:23 |

## The six breakthrough properties (ledger section 3)

| property | experiments | status | highest level |
|---|---|---|---|
| Selectivity (target disappears) | E-000001-B, E-000003, E-000008, E-000011 | E-000001-B: criteria met; E-000003: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met** | E5 |
| Retention (non-target intact) | E-000001-B, E-000003, E-000008, E-000011 | E-000001-B: criteria met; E-000003: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met** | E5 |
| Generalisation (paraphrases, alternative queries) | E-000003, E-000004, E-000008, E-000011 | E-000003: criteria met; E-000004: **criteria NOT met**; E-000008: **criteria NOT met**; E-000011: **criteria NOT met** | E5 |
| Causal isolation (effect follows from the intended structure) | E-000005, E-000006, E-000007, E-000011 | E-000005: criteria met; E-000006: **criteria NOT met**; E-000007: **criteria NOT met**; E-000011: **criteria NOT met** | E5 |
| Reconstruction resistance | E-000004, E-000007, E-000009, E-000010, E-000008, E-000011, E-000014 | E-000004: **criteria NOT met**; E-000007: **criteria NOT met**; E-000009: **criteria NOT met**; E-000010: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000014: criteria met | E5 |
| Scalability (path beyond toy models) | E-000002, E-000008, E-000011, E-000014 | E-000002: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000014: criteria met | E5 |

Scalability is the property this session can least address: E-000008 tests whether the same layer works as an adapter on a frozen pretrained GPT-2 (recorded below); the path to LLM scale is a roadmap item, not a result.

The 'status' column is the status of the whole record's pre-registered criteria; a property can be supported by an experiment whose record fails on a different criterion (E-000004 and E-000007 fail only their F4 rows while their behavioural rows pass — see the split claims inside those records).

## Boundary of this evidence

Everything below was produced on one 4-core CPU box in one session, with no GPU (experiments ran with 2 or 4 torch threads; each record stores the thread count under `environment`). It is therefore bounded as follows:

- **No LLM-scale evidence.** The largest neural core used is frozen GPT-2 small (124M parameters, E-000008). Nothing here shows editable knowledge inside a large pretrained model, and nothing here shows unlearning of facts that a pretrained model already encodes in its weights.
- **Synthetic worlds.** Facts are `(subject, relation) → object` triples over 256 entities and 4 relations; queries are symbolic (E-000001 … E-000007) or short natural-language templates (E-000008). Real-world knowledge, multi-token entities and free-text questions are not covered.
- **By construction versus learned.** REVOKE removes routing by a hard mask; that is deletion level F1 by construction. The learned results are: answering UNKNOWN instead of using another cell, refusing a payload whose marker is invalid (SHRED), composing hops, and the probe / forced-choice / rank checks after SHRED. Every result table states which is which.
- **Provenance is trained**, not emergent: the routing loss supervises which cell each hop reads. E-000006 (`no_routing_loss`) measures what remains without it.
- **The outstanding C55–C57 real-model / GPU chain of the ledger is still outstanding.** E-000008 is its CPU-feasible analogue on a small model, not its execution.
- **Noise figures are not comparable** with the architecture document's "noise = 0.24 → 68.4%", whose noise definition is not recorded; the sweep here perturbs bank keys and values relative to their RMS.
- **Seeds.** E-000003 … E-000007 evaluate the same five E-000001-B models on fresh worlds; they are not independent replications of training. E-000002, E-000006 and E-000008 train their own models (3 seeds); E-000009 trains five.
- **The SHRED residual.** E-000004 and E-000007 found that the marker gate learned without explicit supervision closes to about 9% rather than 0 on unsigned payloads, so a linear probe and forced choice recover a residual; their F4 criteria fail and the records say so. E-000009 (plain verification loss) narrowed it but left an unsigned tail; E-000010 (class-balanced verification loss) closed it to chance in every seed — a recorded result, not an assumption.


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
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 205 |
| 1 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9700 | 1.0000 | 0.9957 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 113 |
| 2 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 115 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 112 |
| 4 | 1.0000 | 1.0000 | 0.9980 | 1.0000 | 0.9700 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 113 |

Interpretation: the behaviour is no longer mechanical — a trained neural core operates over the experimental knowledge structure. It is still a synthetic experiment and not proof of LLM-scale editable knowledge.

## E-000002 — Weight-memorisation control (copy problem)

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2]. Fixed-world regimes trained for 2000 steps; resampled regime = E-000001-B models.

| training regime | direct (layer intact) | layer fully masked | target leak after REVOKE | target UNKNOWN after REVOKE | control after REVOKE |
|---|---|---|---|---|---|
| resampled | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| fixed_routing | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| fixed_no_routing | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |

Reading: 'layer fully masked' is what the weights answer on their own. A leak after REVOKE is knowledge that survived in the weights — the copy problem the ledger warns about (sections 9, 28). The mechanism's deletion guarantee therefore depends on the training regime keeping facts out of the weights. n = 100 targets per seed (leak of 0 in 300 pooled trials -> failure rate below 1.3% at 95%).

Pre-registered criteria (worst seed; leak-type metrics use the max):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| resampled/bank_removed_acc | <= 0.02 | 0.0000 | PASS |
| resampled/target_after_revoke_leak | <= 0.02 | 0.0000 | PASS |
| resampled/control_after_revoke | >= 0.99 | 1.0000 | PASS |
| fixed_no_routing/target_after_revoke_leak | >= 0.5 | 1.0000 | PASS |
| fixed_no_routing/direct | >= 0.5 | 1.0000 | PASS |

Caveats: Only 'fixed_routing' is an empirical control: 'resampled' cannot memorise by construction and 'fixed_no_routing' cannot read the layer by construction. The fixed-world regimes see the same random lifecycle states per step as the re-sampled regime (only the world is held fixed), so the no-routing model receives inconsistent labels for revoked/shredded cells and settles on the majority label. Fixed regimes are trained for fewer steps than the re-sampled E-000001-B models.

Per seed:

**resampled**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

**fixed_routing**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 1 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |

**fixed_no_routing**

| seed | direct | bank_removed_acc | target_before | target_after_revoke_leak | target_after_revoke_unknown | control_after_revoke |
|---|---|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

**Interpretation (post hoc, record unchanged):** Only 'fixed_routing' is an empirical control, and it came out clean: with the layer available, 2000 steps on a fixed world did not copy any fact into the weights (masked-layer accuracy 0%, leak 0%). This is a bound for that budget, not a guarantee for longer training. The no-layer model memorised everything and cannot revoke (leak 100%): the copy problem in its purest form.

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

Evidence level: **E4** (Controlled neural-network evidence); deletion level targeted **F4**, recorded **F3** (behavioural deletion supported, representation-level removal NOT supported). Seeds: [0, 1, 2, 3, 4]. Probe calibration on held-out active cells: top-1 0.949, top-5 0.949. Chance: forced choice 0.5, top-1 among entities 1/256 = 0.0039, mean rank 127.5, probe top-1 0.0039, top-5 0.0195.

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

**Interpretation (post hoc, record unchanged):** SHRED column: behaviourally deleted (direct / paraphrase / multi-hop UNKNOWN 100%), but the gate learned without supervision closes to about 9% of the value norm, so the linear probe (8% worst seed) and forced choice (69% worst seed) recover a residual. Recorded as F3 with a trace; E-000009 is the response.

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

## E-000006 — Ablations

Evidence level: **E4** (Controlled neural-network evidence). Seeds: [0, 1, 2]; variants trained 2000 steps, full model 3000 steps (E-000001-B). Values are means over seeds.

| variant | direct | direct_unknown_rate | hop2 | hop3 | hop2_broken_unknown | provenance | reverse | revoke | shred | update | rollback | locality | alternative_path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| full_same_budget | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.3% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_marker_gate | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_null_cell | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| no_routing_loss | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | 21.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| no_routing | 0.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | 21.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |

'no_routing' and 'no_marker_gate' remove an information path, so their failures (nothing readable / SHRED ineffective) are information-flow necessities, reported to quantify them. 'no_null_cell' and 'no_routing_loss' keep the information paths and test learned behaviour: whether UNKNOWN detection and exact provenance emerge without the dedicated cell / loss. 'full_same_budget' is the fair baseline trained with the variants' step budget.

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| full_same_budget/direct | >= 0.98 | 1.0000 | PASS |
| full_same_budget/shred | >= 0.95 | 0.9800 | PASS |
| no_marker_gate/shred | <= 0.2 | 0.0000 | PASS |
| no_marker_gate/direct | >= 0.98 | 1.0000 | PASS |
| no_null_cell/hop2_broken_unknown | <= 0.5 | 1.0000 | FAIL |
| no_routing/direct | <= 0.1 | 0.0000 | PASS |

Random deletion (revoke another cell, target must stay): 100.0%

Reading the table: for a variant that answers UNKNOWN to everything (no_routing, no_routing_loss) the rows hop2_broken_unknown, revoke, shred and locality are satisfied trivially and carry no information.

Without versioning (UPDATE as in-place replace): rollback impossible (no version to return to) — structural property of the layer, not a learned one.

**Interpretation (post hoc, record unchanged):** Two pre-registered expectations were wrong and are recorded as such. (1) The null cell is NOT essential: without it, broken paths are still answered UNKNOWN at 100% (how the model does this was not measured; a plausible mechanism is a diffuse, low-norm read over non-matching keys). The design claim 'the null cell is what makes broken paths answer UNKNOWN' is withdrawn. (2) Without the routing loss the model collapsed to answering UNKNOWN for everything within 2000 steps (identical numbers to 'no_routing'): routing supervision is necessary for the mechanism to be *learned* at all at this budget, not only for exact provenance. That is an optimisation finding, not a by-construction one. 'no_marker_gate' and 'no_routing' confirm the information-flow necessities (SHRED 0%, nothing readable).

## E-000007 — Biomarker: suppression versus representational change

Evidence level: **E4** (Controlled neural-network evidence); deletion level targeted **F4**, recorded **F3** (suppression/deletion separation supported, SHRED at F4 thresholds NOT supported). Seeds: [0, 1, 2, 3, 4]. Chance levels: probe top-1 0.0039, top-5 0.0195, mean rank 127.5, forced choice 0.5.

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

**Interpretation (post hoc, record unchanged):** The suppression-versus-deletion separation holds in every seed (suppressed: value contribution 8.3, probe 86%, mean rank 10; shredded: 1.3, 4%, 110). The two failed criteria are the same SHRED residual as in E-000004, addressed in E-000009.

## E-000008 — Frozen pretrained GPT-2 core with the mutable knowledge layer

Evidence level recorded: **E5** (E5 = Transformer evidence; a real pretrained LM, GPT-2 small, on CPU — not LLM scale). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; adapter steps: 2000; the 124M pretrained weights are frozen.

Claim parts (each judged on its own pre-registered criteria, worst seed):

| claim | supported |
|---|---|
| With a frozen pretrained transformer as core and natural-language prompts, the adapter reads the right cell and the unchanged LM head emits the object; the pretrained prior is at chance and the adapter with every cell masked adds nothing (copy bound). | **no** |
| UPDATE / ROLLBACK / RESTORE / RESIGN are reproduced against the reference. | yes |
| After REVOKE / SHRED and on broken paths the model answers ' unknown' (behavioural deletion, F3). | **no** |
| After REVOKE nothing is recoverable by probe or forced choice (mask). | yes |
| After SHRED nothing is recoverable by probe or forced choice (representation level, F4). | **no** |

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

## E-000009 — Signature-verification gate: closing the SHRED residual

Evidence level: **E4** (Controlled neural-network evidence); deletion level targeted for SHRED with hard verification: F4, recorded **F3** within the synthetic system. Seeds: [0, 1, 2, 3, 4]; 3000 steps; gate loss weight 1.0. Baseline = the E-000001-B models (no gate loss).

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

**Interpretation (post hoc, record unchanged):** The verification loss sharpened the gate (signed markers 0.89 -> 0.998, unsigned mean 0.087 -> 0.065) but a tail of unsigned markers still scores high (max 0.84 soft; under hard gating 3-5% of shredded payloads pass and answer correctly), so the SHRED residual persists and F4 is still withheld. Cause: the gate loss is averaged over ~1000 cells of which ~5% are unsigned, so the tail receives almost no gradient. E-000010 weights the two classes equally.

## E-000010 — Signature-verification gate: closing the SHRED residual (class-balanced loss)

Evidence level: **E4** (Controlled neural-network evidence); deletion level targeted for SHRED with hard verification: F4, recorded **F4** within the synthetic system. Seeds: [0, 1, 2, 3, 4]; 3000 steps; gate loss weight 5.0, class-balanced. Baseline = the E-000001-B models (no gate loss).

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

**Interpretation (post hoc, record unchanged):** Closes the residual: with signed and unsigned markers weighted equally in the verification loss (weight 5), every reconstruction attack after SHRED is at chance in all five seeds while the payload remains physically present and routed to (routing mass 0.998), and no other family degrades. This is the F4-level result of the session, within the synthetic system. Residual caveat: the soft gate still assigns a high score to a rare unsigned marker in one seed (max 0.995 among all unsigned cells of that bank); none of the 500 shredded targets leaked.

## E-000011 — Frozen GPT-2 core v2

Evidence level: **E5** (substrate: pretrained transformer, 124M frozen). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 adapter steps; verified gate (class-balanced, weight 5); p_revoked 0.20, p_shred 0.10, 20% extra unanswerable queries; templates 0/1 trained, 2/3 (lexical variants), 4 (question form), 5 (prefixed clause) held out. Thresholds are the ones pre-registered for E-000008 (not relaxed); a lenient set (0.90 / 0.85) is reported separately.

Lenient criteria met: **False**. Sample sizes: {'direct/templates': 1000, 'hop2': 300, 'broken': 100, 'lifecycle': 100, 'attacks': 100, 'interventions': 'first 100 correctly answered 2-hop questions from a pool of 400 (pool correct rate recorded)', 'alt_routes': 50, 'comparators': 300}

The frozen core cannot copy a fact; whether the adapter copies is the masked-bank row. REVOKE is a mask (F1). Learned: reading from prompts (including four never-trained templates), composition without the intermediate entity in the text, emitting ' unknown' for a masked or unsigned cell (a trained refusal that ledger §28 would call output suppression if it stood alone — here it is paired with the copy bound, the masked-bank row, the attacks and the answer-category rows that show what is emitted instead), and the gate's selection between payload and ' unknown'. The 2-hop interventions are consistency checks: the read is the only channel through which the adapter can inject the fact, so disabling the cell removing the answer is expected; the informative rows are localisation (does the frozen model's own residual state route to the right cell at each read) and swap / replace (does the answer follow the payload exactly).

Claim parts (each on its own pre-registered criteria, worst seed):

| claim | supported |
|---|---|
| Reading through natural-language prompts on the frozen core; copy bound holds. | **no** |
| Reading generalises to sentence templates never seen in training. | **no** |
| UPDATE / ROLLBACK / RESIGN reproduced against the reference. | **no** |
| After REVOKE / SHRED and on broken paths the model answers ' unknown', also on held-out templates (F3). | **no** |
| After SHRED with hard verification nothing is recoverable by probe, forced choice or rank (F4). | yes |
| Inside the frozen model the two reads of a 2-hop question are causally the two ground-truth cells: disabling either breaks the answer, disabling another cell does not, swapping or replacing the second payload changes the answer as predicted. | yes |

| measure | mean over seeds | worst seed |
|---|---|---|
| prior_direct_acc | 0.4% | 0.3% |
| bank_masked_direct_acc | 0.0% | 0.0% |
| direct | 86.6% | 85.0% |
| template0_train/full_vocab_top1 | 85.2% | 83.7% |
| template1_train/direct | 100.0% | 100.0% |
| template2_heldout/direct | 38.9% | 34.1% |
| template3_heldout/direct | 75.8% | 72.0% |
| template4_heldout/direct | 68.1% | 56.7% |
| template5_heldout/direct | 94.0% | 93.4% |
| direct_heldout_mean | 69.2% | 65.3% |
| provenance_direct | 83.1% | 81.3% |
| hop2 | 83.3% | 82.7% |
| comparator/in_context_both_facts_hop2_acc | 41.7% | 40.0% |
| comparator/in_context_first_fact_only_hop2_acc | 0.6% | 0.3% |
| comparator/adapter_no_context_hop2_acc | 82.2% | 78.7% |
| broken1_unknown | 77.0% | 69.0% |
| broken2_unknown | 72.3% | 68.0% |
| update | 87.7% | 85.0% |
| rollback | 88.3% | 86.0% |
| revoke | 72.7% | 70.0% |
| shred | 99.7% | 99.0% |
| resign | 88.7% | 86.0% |
| update_heldout | 68.2% | 65.8% |
| revoke_heldout | 51.0% | 48.8% |
| revoke_heldout_min | 17.3% | 16.0% |
| shred_heldout | 89.9% | 88.8% |
| shred_heldout_min | 75.0% | 70.0% |
| locality | 98.9% | 98.5% |
| locality_targets_correct | 83.3% | 82.7% |
| alt_route/broken_route_changes | 99.3% | 98.0% |
| alt_route/other_route_survives | 78.0% | 70.0% |
| interventions/pool_correct_rate | 82.0% | 79.0% |

Attacks on 100 targets (mean over seeds; chance: forced choice 0.5, top-1 0.0039, mean rank 127.5, probe 0.0039):

| attack | active | after REVOKE | after SHRED (soft) | after SHRED (hard) |
|---|---|---|---|---|
| direct_unknown | 0.1167 | 0.7033 | 0.9967 | 0.9967 |
| direct_acc | 0.8800 | 0.0067 | 0.0000 | 0.0000 |
| candidate_other_entity | 0.0033 | 0.2900 | 0.0033 | 0.0033 |
| full_vocab_is_unknown_word | 0.1100 | 0.6300 | 0.9867 | 0.9867 |
| full_vocab_is_true_object | 0.8567 | 0.0067 | 0.0000 | 0.0000 |
| full_vocab_is_other_entity | 0.0033 | 0.2767 | 0.0033 | 0.0033 |
| full_vocab_equals_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_vocab_is_non_entity_token | 0.0300 | 0.0867 | 0.0100 | 0.0100 |
| heldout2_unknown | 0.4967 | 0.6500 | 0.8533 | 0.8533 |
| heldout4_unknown | 0.1267 | 0.1667 | 0.7500 | 0.7500 |
| forced_choice_win | 0.9967 | 0.5367 | 0.4933 | 0.4900 |
| true_obj_top1_among_entities | 0.9500 | 0.0067 | 0.0067 | 0.0033 |
| true_obj_mean_rank | 0.3967 | 124.3167 | 125.0633 | 125.9567 |
| probe_top1 | 0.8167 | 0.0067 | 0.0133 | 0.0100 |
| routing_mass_on_target | 0.7644 | 0.0000 | 0.7644 | 0.7644 |
| gate_on_target | 0.9985 | 0.9985 | 0.0014 | 0.0000 |
| payload_share | 0.7633 | 0.0000 | 0.0012 | 0.0000 |

Causal interventions on correctly answered 2-hop questions (mean / worst seed):

| intervention | mean | worst seed |
|---|---|---|
| localisation_hop1 | 100.0% | 100.0% |
| localisation_hop2 | 100.0% | 100.0% |
| disable_hop1_changes | 100.0% | 100.0% |
| disable_hop1_unknown | 69.7% | 62.0% |
| disable_hop2_changes | 99.7% | 99.0% |
| disable_hop2_unknown | 71.3% | 69.0% |
| disable_random_unchanged | 100.0% | 100.0% |
| swap_hop2 | 98.0% | 98.0% |
| replace_hop2 | 97.3% | 96.0% |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| prior_direct_acc | <= 0.05 | 0.0050 | PASS |
| bank_masked_direct_acc | <= 0.05 | 0.0000 | PASS |
| direct | >= 0.95 | 0.8500 | FAIL |
| template1_train/direct | >= 0.95 | 1.0000 | PASS |
| template2_heldout/direct | >= 0.8 | 0.3410 | FAIL |
| template3_heldout/direct | >= 0.8 | 0.7200 | FAIL |
| template4_heldout/direct | >= 0.7 | 0.5670 | FAIL |
| template5_heldout/direct | >= 0.7 | 0.9340 | PASS |
| update | >= 0.95 | 0.8500 | FAIL |
| rollback | >= 0.95 | 0.8600 | FAIL |
| resign | >= 0.95 | 0.8600 | FAIL |
| revoke | >= 0.95 | 0.7000 | FAIL |
| shred | >= 0.9 | 0.9900 | PASS |
| broken1_unknown | >= 0.9 | 0.6900 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.1600 | FAIL |
| shred_heldout_min | >= 0.85 | 0.7000 | FAIL |
| locality | >= 0.98 | 0.9847 | PASS |
| restored/direct_acc | >= 0.95 | 0.8200 | FAIL |
| revoke/probe_top1 | <= 0.05 | 0.0100 | PASS |
| revoke/forced_choice_win | <= 0.6 | 0.5700 | PASS |
| shred_hard/probe_top1 | <= 0.05 | 0.0300 | PASS |
| shred_hard/forced_choice_win | <= 0.6 | 0.5200 | PASS |
| shred_hard/true_obj_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| shred_hard/payload_share | <= 0.05 | 0.0000 | PASS |
| shred_hard/direct_unknown | >= 0.9 | 0.9900 | PASS |
| alt_route/broken_route_changes | >= 0.95 | 0.9800 | PASS |
| alt_route/other_route_survives | >= 0.95 | 0.7000 | FAIL |
| interventions/localisation_hop1 | >= 0.9 | 1.0000 | PASS |
| interventions/localisation_hop2 | >= 0.9 | 1.0000 | PASS |
| interventions/disable_hop1_changes | >= 0.95 | 1.0000 | PASS |
| interventions/disable_hop2_changes | >= 0.95 | 0.9900 | PASS |
| interventions/disable_random_unchanged | >= 0.95 | 1.0000 | PASS |
| interventions/swap_hop2 | >= 0.9 | 0.9796 | PASS |
| interventions/replace_hop2 | >= 0.9 | 0.9600 | PASS |

Lenient criteria (secondary, worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.9 | 0.8500 | FAIL |
| template1_train/direct | >= 0.9 | 1.0000 | PASS |
| revoke | >= 0.9 | 0.7000 | FAIL |
| shred | >= 0.85 | 0.9900 | PASS |
| broken1_unknown | >= 0.85 | 0.6900 | FAIL |

## e000012_status_gated_revoke

_not run in this session_

## e000013_prior_conflict

_not run in this session_

## E-000014 — Addressing at 10,000 cells

Evidence level: **E4**; deletion level targeted F4, recorded **F4**. Seeds: [0, 1, 2]; 3000 steps; banks of 7,000–10,000 cells over 2560 entities; class-balanced verified gate (weight 5).

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 1.0000 | 1.0000 | 30000 | 0.9999 | 1.0000 |
| hop2 | 0.9993 | 0.9980 | 1500 | 0.9963 | 1.0000 |
| hop3 | 0.9947 | 0.9900 | 1500 | 0.9895 | 0.9977 |
| hop2_broken_unknown | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| hop3_broken_unknown | 0.9567 | 0.9500 | 300 | 0.9270 | 0.9767 |
| provenance | 0.9999 | 0.9998 | 33000 | 0.9998 | 1.0000 |
| reverse | 0.9978 | 0.9967 | 900 | 0.9920 | 0.9997 |
| update | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| update_derived | 1.0000 | 1.0000 | - | - | - |
| rollback | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| revoke | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| restore | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| shred | 0.9967 | 0.9900 | 300 | 0.9816 | 0.9999 |
| resign | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| update_rollback | 0.9995 | 0.9986 | - | - | - |
| locality | 1.0000 | 1.0000 | 30450 | 0.9999 | 1.0000 |
| locality_targets_correct | 1.0000 | 1.0000 | 450 | 0.9918 | 1.0000 |
| locality_undo_exact | 1.0000 | 1.0000 | - | - | - |
| alternative_path | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| replay_deviation | 0.0000 | 0.0000 | - | - | - |

Noise (bank perturbation 0.24, direct): seed 0: 0.998, seed 1: 0.998, seed 2: 0.999

Attacks after SHRED on 500 targets (mean over seeds; chance: probe top-1 0.00039, mean rank 1279.5, forced choice 0.5):

Core thresholds (0.98 / 0.95 / 0.90) are lower than E-000001-B's (0.99 / 0.98 / 0.95) because the task is harder in two ways at once: ten times the bank and a ten times larger read-out vocabulary. Attack thresholds are binomially derived for 500 targets at chance 1/2560.

| attack after SHRED | soft gate | hard gate |
|---|---|---|
| direct_unknown | 0.9987 | 0.9980 |
| direct_acc | 0.0013 | 0.0020 |
| paraphrase_unknown | 0.9987 | 0.9980 |
| multihop_unknown | 1.0000 | 0.9986 |
| reverse_unknown | 0.9983 | 0.9983 |
| forced_choice_win | 0.5227 | 0.5273 |
| true_obj_top1_among_entities | 0.0020 | 0.0020 |
| true_obj_mean_rank | 1259.9280 | 1245.9120 |
| probe_top1 | 0.0040 | 0.0033 |
| probe_top5 | 0.0067 | 0.0060 |
| routing_mass_on_target | 0.9953 | 0.9953 |
| gated_value_contribution | 0.0567 | 0.0290 |
| gate_valid_mean | 0.9975 | 1.0000 |
| gate_invalid_mean | 0.0046 | 0.0027 |
| gate_invalid_max | 0.6984 | 0.6667 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.98 | 1.0000 | PASS |
| hop2 | >= 0.95 | 0.9980 | PASS |
| hop3 | >= 0.9 | 0.9900 | PASS |
| hop2_broken_unknown | >= 0.95 | 1.0000 | PASS |
| provenance | >= 0.95 | 0.9998 | PASS |
| reverse | >= 0.95 | 0.9967 | PASS |
| revoke | >= 0.98 | 1.0000 | PASS |
| restore | >= 0.98 | 1.0000 | PASS |
| shred | >= 0.95 | 0.9900 | PASS |
| resign | >= 0.98 | 1.0000 | PASS |
| update | >= 0.98 | 1.0000 | PASS |
| rollback | >= 0.98 | 1.0000 | PASS |
| locality | >= 0.99 | 1.0000 | PASS |
| alternative_path | >= 0.95 | 1.0000 | PASS |
| replay_deviation | <= 0 | 0.0000 | PASS |
| hard/shred/direct_unknown | >= 0.98 | 0.9960 | PASS |
| hard/shred/probe_top1 | <= 0.006 | 0.0060 | PASS |
| hard/shred/true_obj_top1_among_entities | <= 0.006 | 0.0040 | PASS |
| hard/shred/true_obj_mean_rank | >= 1150.0 | 1219.3000 | PASS |
| hard/shred/forced_choice_win | <= 0.56 | 0.5440 | PASS |
| hard/shred/gated_value_contribution | <= 0.1 | 0.0598 | PASS |
| hard/active/direct_acc | >= 0.98 | 1.0000 | PASS |
| hard/restored/direct_acc | >= 0.98 | 1.0000 | PASS |

Scaling curve, same model on fresh worlds (direct accuracy / mean routing max-mass), per seed:

| seed | 1,000 cells | 3,000 cells | 10,000 cells |
|---|---|---|---|
| 0 | 1.000 / 0.995 | 1.000 / 0.995 | 1.000 / 0.995 |
| 1 | 1.000 / 0.995 | 1.000 / 0.995 | 1.000 / 0.995 |
| 2 | 1.000 / 0.996 | 1.000 / 0.995 | 1.000 / 0.996 |

**Interpretation (post hoc, record unchanged):** Ten times the bank and ten times the read-out vocabulary at once: every family stays at the E-000001-B level (direct 100% over 30,000 pooled queries, 3-hop 99.5%, provenance 99.99%), the verified gate keeps SHRED at F4 on 500 targets with thresholds derived for 2,560 entities, and the same model reads 100% at 1,000, 3,000 and 10,000 cells with routing mass 0.995 — addressing does not degrade in this range. Residual: 1 in 500 shredded targets answered (an unsigned marker passing the gate), inside the binomial threshold; the gate's false-accept tail is the quantity to watch at larger scale.

## E-000015 — Explicit symlink cells: several access keys, one knowledge object

Evidence level: **E4** (synthetic system). Deletion level targeted F4, recorded **F3**. Seeds: [0, 1, 2]; 4000 steps.

Two stores hold the SAME world with the SAME ground truth and are read by the SAME trained model: in the symlink arm the 200 alias keys are LINK cells pointing at 100 target cells, in the duplication arm the same keys are ordinary fact cells holding a copy of the object. Every sharing claim is the difference between the arms.

Symlink versus duplication (mean over seeds), the two arms holding identical ground truth:

| what is measured | symlink arm | duplication arm |
|---|---|---|
| one UPDATE on the shared object reaches every access path | 100.0% | 0.0% |
| one SHRED on the shared object leaves nothing readable | 100.0% | 0.0% |
| object recoverable by probe after that one operation | 0.7% | 87.3% |
| operations needed to reach every access path | 1 | 3 |

| claim group | supported |
|---|---|
| reading | yes |
| provenance_through_the_alias | yes |
| dereference_is_what_reads_an_alias | yes |
| one_update_reaches_every_path | yes |
| one_shred_deletes_every_path | yes |
| attacks_through_every_alias | yes |
| alias_lifecycle | **no** |
| capability_limit_of_one_slot | yes |
| no_regression_without_links | yes |

| measure | mean over seeds | worst seed | best seed |
|---|---|---|---|
| direct | 1.0000 | 1.0000 | 1.0000 |
| alias_direct | 1.0000 | 1.0000 | 1.0000 |
| dup_direct | 1.0000 | 1.0000 | 1.0000 |
| hop2 | 1.0000 | 1.0000 | 1.0000 |
| broken1_unknown | 1.0000 | 1.0000 | 1.0000 |
| provenance_direct | 1.0000 | 1.0000 | 1.0000 |
| alias_provenance_pair | 1.0000 | 1.0000 | 1.0000 |
| alias_provenance_len2 | 1.0000 | 1.0000 | 1.0000 |
| deref_disabled/alias_direct | 0.0000 | 0.0000 | 0.0000 |
| deref_disabled/direct | 1.0000 | 1.0000 | 1.0000 |
| shared_update/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| shared_update/target_new_object | 1.0000 | 1.0000 | 1.0000 |
| rollback/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| probe_calibration_top1 | 0.8867 | 0.8400 | 0.9467 |
| shred_target/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0067 | 0.0000 | 0.0100 |
| shred_target/alias_forced_choice | 0.5033 | 0.5000 | 0.5100 |
| shred_target/alias_top1_among_entities | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_mean_rank | 127.4250 | 125.9900 | 130.2250 |
| dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 1.0000 |
| dup_shred/copy_probe_top1 | 0.8733 | 0.8000 | 0.9600 |
| dup_shred/copy_forced_choice | 1.0000 | 1.0000 | 1.0000 |
| resign_target/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/sibling_readable | 1.0000 | 1.0000 | 1.0000 |
| revoke_alias/target_readable | 1.0000 | 1.0000 | 1.0000 |
| shred_alias/alias_unknown | 0.9700 | 0.9300 | 1.0000 |
| shred_alias/target_readable | 1.0000 | 1.0000 | 1.0000 |
| relink/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| relink/sibling_unchanged | 1.0000 | 1.0000 | 1.0000 |
| relink_rollback/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| delete_target/alias_unknown | 1.0000 | 1.0000 | 1.0000 |
| delete_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| refcount_before_delete | 2.0000 | 2.0000 | 2.0000 |
| chain2/answer_acc | 0.0000 | 0.0000 | 0.0000 |
| chain2/unknown | 1.0000 | 1.0000 | 1.0000 |
| chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| regression/direct | 1.0000 | 1.0000 | 1.0000 |
| regression/hop2 | 1.0000 | 1.0000 | 1.0000 |
| regression/hop3 | 0.9967 | 0.9933 | 1.0000 |
| regression/reverse | 1.0000 | 1.0000 | 1.0000 |
| regression/provenance | 1.0000 | 1.0000 | 1.0000 |
| regression/broken2_unknown | 1.0000 | 1.0000 | 1.0000 |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 1.0000 | 1.0000 | 1800 | 0.9980 | 1.0000 |
| alias_direct | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| dup_direct | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| hop2 | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| broken1_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| provenance_direct | 1.0000 | 1.0000 | 1800 | 0.9980 | 1.0000 |
| alias_provenance_pair | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| shared_update/alias_new_object | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_probe_top1 | 0.0067 | 0.0100 | 600 | 0.0018 | 0.0170 |
| shred_target/alias_top1_among_entities | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| revoke_alias/alias_unknown | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| shred_alias/alias_unknown | 0.9700 | 0.9300 | 300 | 0.9438 | 0.9862 |
| relink/alias_new_object | 1.0000 | 1.0000 | 300 | 0.9878 | 1.0000 |
| delete_target/alias_unknown | 1.0000 | 1.0000 | 600 | 0.9939 | 1.0000 |
| chain2/answer_acc | 0.0000 | 0.0000 | 300 | 0.0000 | 0.0122 |
| regression/direct | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| regression/hop2 | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |
| regression/hop3 | 0.9967 | 0.9933 | 900 | 0.9903 | 0.9993 |
| regression/reverse | 1.0000 | 1.0000 | 900 | 0.9959 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.98 | 1.0000 | PASS |
| alias_direct | >= 0.95 | 1.0000 | PASS |
| dup_direct | >= 0.98 | 1.0000 | PASS |
| hop2 | >= 0.95 | 1.0000 | PASS |
| broken1_unknown | >= 0.95 | 1.0000 | PASS |
| provenance_direct | >= 0.95 | 1.0000 | PASS |
| alias_provenance_pair | >= 0.9 | 1.0000 | PASS |
| deref_disabled/alias_direct | <= 0.2 | 0.0000 | PASS |
| deref_disabled/direct | >= 0.9 | 1.0000 | PASS |
| shared_update/alias_new_object | >= 0.95 | 1.0000 | PASS |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| shared_update/target_new_object | >= 0.95 | 1.0000 | PASS |
| rollback/alias_direct | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_unknown | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.95 | 1.0000 | PASS |
| resign_target/alias_direct | >= 0.95 | 1.0000 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5100 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0000 | PASS |
| revoke_alias/alias_unknown | >= 0.95 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.95 | 1.0000 | PASS |
| revoke_alias/target_readable | >= 0.95 | 1.0000 | PASS |
| shred_alias/alias_unknown | >= 0.95 | 0.9300 | FAIL |
| shred_alias/target_readable | >= 0.95 | 1.0000 | PASS |
| relink/alias_new_object | >= 0.9 | 1.0000 | PASS |
| relink_rollback/alias_direct | >= 0.9 | 1.0000 | PASS |
| delete_target/alias_unknown | >= 0.95 | 1.0000 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| chain2/answer_acc | <= 0.2 | 0.0000 | PASS |
| regression/direct | >= 0.98 | 1.0000 | PASS |
| regression/hop2 | >= 0.95 | 1.0000 | PASS |
| regression/hop3 | >= 0.9 | 0.9933 | PASS |
| regression/reverse | >= 0.95 | 1.0000 | PASS |
| regression/provenance | >= 0.95 | 1.0000 | PASS |
| regression/broken2_unknown | >= 0.95 | 1.0000 | PASS |

Two-slot control (single seed): {'seed': 0, 'chain2/answer_acc': 0.0, 'chain2/depth1_acc': 1.0, 'alias_direct': 1.0, 'direct': 1.0, 'checkpoint_sha256': '34bca206b9d9c8a94d9e16ea82e906f4990e60d87a55aaec3f5e1ff99e05eca7'}

By construction: the store decides which payload a row carries (an alias row carries its target's KEY, a fact row its object), exactly as it decides the marker; the bank never exports the target's payload, its status, its signature or the chain depth; that ONE update or ONE shred on a shared object reaches every alias is a property of the store; what is measured is whether the trained model reports it, and whether the SAME model reports the duplication arm (where it does not) correctly; a deleted target keeps its key as a tombstone, so a dangling pointer stays a pointer and the miss is not pre-resolved by the control plane.

Learned: following a pointer: the dereference slot's query comes from the value just read, not from the question, and the model is never told that a value is a pointer; keeping a value that was not a pointer (the passthrough column) so that fact cells still read correctly, measured as deref_disabled/direct versus deref_disabled/alias_direct; answering UNKNOWN for a dangling pointer, for a revoked or shredded alias and for a shredded target; provenance across the indirection: the routing names the alias AND the cell it points at.

Not claimed: LLM scale (the frozen-GPT-2 chain does not yet carry links); chains deeper than the number of dereference slots; reference counting as a garbage-collection policy.

**Interpretation (post hoc, record unchanged):** The first measurement of the Symlink hypothesis as ledger section 7 states it: sharing versus duplicating. Both arms hold the identical world and are read by the identical model, so every number in the contrast is attributable to the storage form alone. One operation on the shared object reaches or deletes every access path; the same operation in the duplication arm reaches one key and leaves the object recoverable through the copies by probe (87.3%) and forced choice (1.000). The dereference ablation is the mechanism control: with the slot disabled, alias reading is 0% and fact reading is 100%. Two results are withheld and recorded as failures: shredding the alias rather than the payload reaches only 93% on the worst seed, and the two-slot control does not resolve two-link chains because chains never occur in the training distribution.
