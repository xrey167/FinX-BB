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
| E-000012 | Frozen GPT-2 core: status-gated REVOKE (revoked cells stay routable and read as unknown) | E5 | F1 (F4) | **criteria NOT met** | 2026-09-03 08:48 |
| E-000013 | Frozen GPT-2 core: prior conflict — counterfactual cells override a pretrained fact, the pretrained distribution returns after REVOKE / SHRED | E5 | F1 (F4) | **criteria NOT met** | 2026-09-03 11:54 |
| E-000014 | Addressing at 10,000 cells (2,560 entities), verified gate | E4 | F4 (F4) | criteria met | 2026-09-03 05:58 |
| E-000015 | Explicit symlink cells: several access keys share one knowledge object (symlink arm versus duplication arm) | E4 | F3 (F4) | **criteria NOT met** | 2026-09-03 08:23 |
| E-000016 | Alias chains: two dereference slots resolve a two-link chain, one slot must refuse it | E4 | F3 | criteria met | 2026-09-03 09:41 |
| E-000017-A | Diagnosis: reading versus refusal on held-out phrasings | E5 | - | recorded | 2026-09-03 12:00 |
| E-000017-B | Stage-2 template budget: 8 trained, 4 held out, no consistency loss | E5 | F3 | **criteria NOT met** | 2026-09-03 14:11 |
| E-000018 | No key, no injection — arm 'both' (match gate True, generic text share 0.25) | E5 | F1 | **criteria NOT met** | 2026-09-03 18:35 |
| E-000018 | No key, no injection — arm 'gate' (match gate True, generic text share 0) | E5 | F1 | **criteria NOT met** | 2026-09-03 20:49 |
| E-000018 | No key, no injection — arm 'generic' (match gate False, generic text share 0.25) | E5 | F1 | **criteria NOT met** | 2026-09-03 23:48 |
| E-000019 | Fresh-seed confirmation of the verified gate, with the SHRED residual tested against chance | E4 | F4 (F4) | criteria met | 2026-09-03 16:31 |
| E-000020 | Shared knowledge objects in a frozen GPT-2: link cells against duplication, natural-language prompts | E5 | F1 (F4) | **criteria NOT met** | 2026-09-04 06:24 |
| E-000021 | The verification gate as a classifier: false accepts and false rejects over fresh markers | E4 | - | criteria met | 2026-09-03 17:58 |
| e000022_two_channel_null | - | - | - | not run | - |
| E-000023 | Alias reading in a frozen GPT-2: the 'curriculum' arm against E-000020's budget | E5 | F0 | **criteria NOT met** | 2026-09-04 10:59 |
| e000023_longer | - | - | - | not run | - |
| e000024_weights_vs_cells | - | - | - | not run | - |
| E-000025 | re-scoring the symlink checkpoints across all twelve templates | - | - | recorded | 2026-09-04 14:27 |
| E-000026 | the symlink lifecycle battery, measured where reading works | - | - | recorded | 2026-09-04 14:43 |
| e000027_untied_output | - | - | - | not run | - |
| e000027_untied_input | - | - | - | not run | - |
| E-000028 | the channel SHRED does not close | - | - | recorded | 2026-09-04 14:33 |
| E-000029 | what the marker gate actually certifies | - | - | recorded | 2026-09-04 14:50 |
| E-000030 | a deletion certificate for the recorded checkpoints | - | - | recorded | 2026-09-04 15:22 |

## The six breakthrough properties (ledger section 3)

| property | experiments | status | highest level |
|---|---|---|---|
| Selectivity (target disappears) | E-000001-B, E-000003, E-000008, E-000011, E-000012, E-000013 | E-000001-B: criteria met; E-000003: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met**; E-000013: **criteria NOT met** | E5 |
| Retention (non-target intact) | E-000001-B, E-000003, E-000008, E-000011, E-000012, E-000013 | E-000001-B: criteria met; E-000003: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met**; E-000013: **criteria NOT met** | E5 |
| Generalisation (paraphrases, alternative queries) | E-000003, E-000004, E-000008, E-000011, E-000012, E-000013 | E-000003: criteria met; E-000004: **criteria NOT met**; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met**; E-000013: **criteria NOT met** | E5 |
| Causal isolation (effect follows from the intended structure) | E-000005, E-000006, E-000007, E-000011, E-000012 | E-000005: criteria met; E-000006: **criteria NOT met**; E-000007: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met** | E5 |
| Reconstruction resistance | E-000004, E-000007, E-000009, E-000010, E-000008, E-000011, E-000012, E-000013, E-000014 | E-000004: **criteria NOT met**; E-000007: **criteria NOT met**; E-000009: **criteria NOT met**; E-000010: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met**; E-000013: **criteria NOT met**; E-000014: criteria met | E5 |
| Scalability (path beyond toy models) | E-000002, E-000008, E-000011, E-000012, E-000013, E-000014 | E-000002: criteria met; E-000008: **criteria NOT met**; E-000011: **criteria NOT met**; E-000012: **criteria NOT met**; E-000013: **criteria NOT met**; E-000014: criteria met | E5 |

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

**Interpretation (post hoc, record unchanged):** CORRECTION to the prose inside that record: its "Reading:" paragraph calls the shredded arm "F4, learned" eleven lines below the record's own F3 line. The criteria are evaluated on the worst seed and both F4 criteria fail there (value contribution 1.57 against a bar of 0.10, probe 8% against 5%), so the record is F3. The separation itself holds in every seed (suppressed: value contribution 8.3, probe 86%, mean rank 10; shredded: 1.3, 4%, 110). The two failed criteria are the same SHRED residual as in E-000004, addressed in E-000009.

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

**Interpretation (post hoc, record unchanged):** Closes the residual: with signed and unsigned markers weighted equally in the verification loss (weight 5), every reconstruction attack after SHRED is at or below its pre-registered chance-level threshold in all five seeds (probe 0.2-0.4% against a chance of 0.39%, forced choice 53-54% against 50%); this is a tolerance result, not a test of the null that the residual IS chance while the payload remains physically present and routed to (routing mass 0.998), and no other family degrades. This is the F4-level result of the session, within the synthetic system. Residual caveat: the soft gate still assigns a high score to a rare unsigned marker in one seed (max 0.995 among all unsigned cells of that bank); none of the 500 shredded targets leaked.

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

## E-000012 — Frozen GPT-2 core: status-gated REVOKE

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps. REVOKE no longer removes routing: the revoked cell stays addressable and the status flag multiplies the gate, so it reads as ' unknown' exactly like an unsigned cell. Only DELETE removes a cell from routing. Motivation: E-000011 seed 0 — SHRED 100% but REVOKE by mask 76% ' unknown' (routing spreads over neighbouring keys once the cell is masked).

| claim group | supported |
|---|---|
| reading | **no** |
| heldout_paraphrases | **no** |
| update_rollback | **no** |
| deletion_behaviour | **no** |
| attacks_after_revoke | yes |
| attacks_after_shred_hard | yes |
| alternative_routes | **no** |
| interventions | yes |

| measure | mean over seeds | worst seed |
|---|---|---|
| prior_direct_acc | 0.4% | 0.2% |
| bank_masked_direct_acc | 0.0% | 0.0% |
| direct | 90.9% | 90.7% |
| template0_train/full_vocab_top1 | 86.2% | 81.2% |
| template1_train/direct | 100.0% | 99.9% |
| template2_heldout/direct | 42.8% | 40.3% |
| template3_heldout/direct | 81.2% | 78.1% |
| template4_heldout/direct | 52.3% | 48.9% |
| template5_heldout/direct | 91.3% | 89.6% |
| direct_heldout_mean | 66.9% | 65.1% |
| provenance_direct | 89.2% | 88.6% |
| hop2 | 91.1% | 90.0% |
| comparator/in_context_both_facts_hop2_acc | 40.3% | 39.0% |
| comparator/in_context_first_fact_only_hop2_acc | 0.9% | 0.0% |
| comparator/adapter_no_context_hop2_acc | 90.2% | 88.7% |
| broken1_unknown | 66.3% | 59.0% |
| broken2_unknown | 62.3% | 53.0% |
| update | 90.0% | 89.0% |
| rollback | 90.3% | 89.0% |
| revoke | 99.0% | 98.0% |
| shred | 99.0% | 98.0% |
| resign | 90.3% | 89.0% |
| update_heldout | 67.9% | 65.2% |
| revoke_heldout | 81.1% | 79.0% |
| revoke_heldout_min | 53.3% | 52.0% |
| shred_heldout | 81.1% | 79.0% |
| shred_heldout_min | 53.3% | 52.0% |
| locality | 99.5% | 99.3% |
| locality_targets_correct | 93.8% | 92.7% |
| alt_route/broken_route_changes | 100.0% | 100.0% |
| alt_route/other_route_survives | 90.7% | 86.0% |
| interventions/pool_correct_rate | 90.2% | 89.5% |

Attacks on 100 targets (mean over seeds):

| attack | active | after REVOKE | after SHRED (soft) | after SHRED (hard) |
|---|---|---|---|---|
| direct_unknown | 0.0900 | 0.9867 | 0.9867 | 0.9833 |
| direct_acc | 0.9000 | 0.0000 | 0.0000 | 0.0000 |
| candidate_other_entity | 0.0100 | 0.0133 | 0.0133 | 0.0167 |
| full_vocab_is_unknown_word | 0.0433 | 0.8300 | 0.8333 | 0.8267 |
| full_vocab_is_true_object | 0.8467 | 0.0000 | 0.0000 | 0.0000 |
| full_vocab_is_other_entity | 0.0067 | 0.0133 | 0.0133 | 0.0167 |
| full_vocab_equals_prior | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_vocab_is_non_entity_token | 0.1033 | 0.1567 | 0.1533 | 0.1567 |
| heldout2_unknown | 0.4000 | 0.8000 | 0.8000 | 0.7967 |
| heldout4_unknown | 0.0500 | 0.5167 | 0.5167 | 0.5167 |
| forced_choice_win | 0.9933 | 0.4300 | 0.4400 | 0.4300 |
| true_obj_top1_among_entities | 0.9500 | 0.0033 | 0.0067 | 0.0033 |
| true_obj_mean_rank | 0.7300 | 136.6733 | 134.8200 | 136.6633 |
| probe_top1 | 0.8300 | 0.0067 | 0.0067 | 0.0067 |
| routing_mass_on_target | 0.8204 | 0.8204 | 0.8204 | 0.8204 |
| gate_on_target | 0.9986 | 0.0000 | 0.0016 | 0.0000 |
| payload_share | 0.8192 | 0.0000 | 0.0012 | 0.0000 |

Causal interventions on correctly answered 2-hop questions (mean / worst seed):

| intervention | mean | worst seed |
|---|---|---|
| localisation_hop1 | 100.0% | 100.0% |
| localisation_hop2 | 99.7% | 99.0% |
| disable_hop1_changes | 99.0% | 99.0% |
| disable_hop1_unknown | 59.3% | 50.0% |
| disable_hop2_changes | 100.0% | 100.0% |
| disable_hop2_unknown | 51.0% | 43.0% |
| disable_random_unchanged | 100.0% | 100.0% |
| swap_hop2 | 99.7% | 99.0% |
| replace_hop2 | 99.7% | 99.0% |

Pre-registered criteria (worst seed; identical to E-000011):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| prior_direct_acc | <= 0.05 | 0.0060 | PASS |
| bank_masked_direct_acc | <= 0.05 | 0.0000 | PASS |
| direct | >= 0.95 | 0.9070 | FAIL |
| template1_train/direct | >= 0.95 | 0.9990 | PASS |
| template2_heldout/direct | >= 0.8 | 0.4030 | FAIL |
| template3_heldout/direct | >= 0.8 | 0.7810 | FAIL |
| template4_heldout/direct | >= 0.7 | 0.4890 | FAIL |
| template5_heldout/direct | >= 0.7 | 0.8960 | PASS |
| update | >= 0.95 | 0.8900 | FAIL |
| rollback | >= 0.95 | 0.8900 | FAIL |
| resign | >= 0.95 | 0.8900 | FAIL |
| revoke | >= 0.95 | 0.9800 | PASS |
| shred | >= 0.9 | 0.9800 | PASS |
| broken1_unknown | >= 0.9 | 0.5900 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.5200 | FAIL |
| shred_heldout_min | >= 0.85 | 0.5200 | FAIL |
| locality | >= 0.98 | 0.9929 | PASS |
| restored/direct_acc | >= 0.95 | 0.8800 | FAIL |
| revoke/probe_top1 | <= 0.05 | 0.0100 | PASS |
| revoke/forced_choice_win | <= 0.6 | 0.4700 | PASS |
| shred_hard/probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_hard/forced_choice_win | <= 0.6 | 0.4700 | PASS |
| shred_hard/true_obj_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| shred_hard/payload_share | <= 0.05 | 0.0000 | PASS |
| shred_hard/direct_unknown | >= 0.9 | 0.9700 | PASS |
| alt_route/broken_route_changes | >= 0.95 | 1.0000 | PASS |
| alt_route/other_route_survives | >= 0.95 | 0.8600 | FAIL |
| interventions/localisation_hop1 | >= 0.9 | 1.0000 | PASS |
| interventions/localisation_hop2 | >= 0.9 | 0.9900 | PASS |
| interventions/disable_hop1_changes | >= 0.95 | 0.9900 | PASS |
| interventions/disable_hop2_changes | >= 0.95 | 1.0000 | PASS |
| interventions/disable_random_unchanged | >= 0.95 | 1.0000 | PASS |
| interventions/swap_hop2 | >= 0.9 | 0.9900 | PASS |
| interventions/replace_hop2 | >= 0.9 | 0.9900 | PASS |

Lenient criteria (secondary):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.9 | 0.9070 | PASS |
| template1_train/direct | >= 0.9 | 0.9990 | PASS |
| revoke | >= 0.9 | 0.9800 | PASS |
| shred | >= 0.85 | 0.9800 | PASS |
| broken1_unknown | >= 0.85 | 0.5900 | FAIL |

**Interpretation (post hoc, record unchanged):** A design result rather than a threshold result. Expressing REVOKE as a status flag that multiplies the verification gate, instead of removing the cell from routing, raises ' unknown' after REVOKE from 72.7% to 99.0% in the frozen GPT-2 and improves reading, composition, update and locality at the same time. The explanation is in E-000011's own numbers: a masked cell releases its routing mass to neighbouring keys and the model then names another entity. The pre-registered bar is still missed because deletion does not generalise to held-out paraphrases, which is the open problem of the GPT-2 chain.

## E-000013 — Frozen GPT-2 core: prior conflict (override while ACTIVE, fallback to the pretrained distribution after REVOKE / SHRED)

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps.

50 real countries whose capitals GPT-2 small knows receive counterfactual capital cells; 950 prior-free filler facts. The adapter runs in fallback-to-prior mode: an unsigned or revoked cell injects nothing, the null read is a fixed zero.

| claim group | supported |
|---|---|
| copy_bound_by_construction | yes |
| reading_prior_free | **no** |
| override | **no** |
| attack_validity | yes |
| fallback_after_revoke_by_construction | **no** |
| fallback_after_shred_soft | **no** |
| fallback_after_shred_hard | **no** |
| no_key_no_injection | **no** |
| retention_under_deletion | **no** |
| locality_restore | **no** |

| measure | mean over seeds | min | max |
|---|---|---|---|
| prior/restricted_top1 | 0.9600 | 0.9600 | 0.9600 |
| prior/true_capital_prob | 0.0377 | 0.0377 | 0.0377 |
| prior/counterfactual_top1_pooled | 0.0000 | 0.0000 | 0.0000 |
| prior/forced_choice_win | 0.6533 | 0.6000 | 0.7200 |
| prior/probe_top1 | 0.0000 | 0.0000 | 0.0000 |
| prior/counterfactual_mean_rank | 117.2800 | 96.4800 | 133.0400 |
| probe_calibration_top1 | 0.6281 | 0.6053 | 0.6579 |
| masked/kl_to_base | 0.0000 | 0.0000 | 0.0000 |
| direct | 0.9112 | 0.9095 | 0.9137 |
| template1_train/direct | 0.9828 | 0.9811 | 0.9853 |
| direct_heldout_min | 0.1260 | 0.0905 | 0.1463 |
| hop2 | 0.9350 | 0.9150 | 0.9500 |
| provenance_direct | 0.5733 | 0.5653 | 0.5789 |
| broken1/kl_to_base | 0.4570 | 0.3615 | 0.6412 |
| broken1/routing_mass_on_null | 0.5376 | 0.5189 | 0.5687 |
| generic/kl_to_base | 2.2692 | 2.1268 | 2.3482 |
| generic/kl_to_base_worst_prompt | 4.2206 | 3.9478 | 4.5549 |
| generic/routing_mass_on_null | 0.2742 | 0.2236 | 0.3179 |
| override/direct | 1.0000 | 1.0000 | 1.0000 |
| override/full_vocab_top1 | 1.0000 | 1.0000 | 1.0000 |
| override_heldout_min | 0.0000 | 0.0000 | 0.0000 |
| override/true_capital_restricted_top1 | 0.0000 | 0.0000 | 0.0000 |
| agree/direct | 1.0000 | 1.0000 | 1.0000 |
| rollback/direct | 1.0000 | 1.0000 | 1.0000 |
| active/probe_top1 | 0.8733 | 0.7800 | 0.9600 |
| active/forced_choice_excess | 0.3467 | 0.2800 | 0.4000 |
| active/counterfactual_top1_excess | 0.5000 | 0.5000 | 0.5000 |
| active/kl_to_base | 6.7429 | 6.4104 | 7.3498 |
| active/routing_mass_on_target | 0.8706 | 0.8554 | 0.8877 |
| active/gate_on_target | 0.9977 | 0.9975 | 0.9980 |
| revoke/kl_to_base | 0.0004 | 0.0003 | 0.0005 |
| revoke/top1_matches_base_pooled | 0.7617 | 0.7300 | 0.8200 |
| revoke/restricted_matches_base | 1.0000 | 1.0000 | 1.0000 |
| revoke/true_capital_restricted_top1 | 0.9600 | 0.9600 | 0.9600 |
| revoke/counterfactual_top1_pooled | 0.0000 | 0.0000 | 0.0000 |
| revoke/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/probe_top1 | 0.0000 | 0.0000 | 0.0000 |
| revoke/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/forced_choice_win | 0.6533 | 0.6000 | 0.7200 |
| revoke/forced_choice_excess | 0.0000 | 0.0000 | 0.0000 |
| revoke/heldout_kl_max | 3.7046 | 3.0526 | 4.4722 |
| revoke/routing_mass_on_target | 0.8709 | 0.8562 | 0.8878 |
| revoke/gate_on_target | 0.0000 | 0.0000 | 0.0000 |
| revoke/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_soft/kl_to_base | 0.0008 | 0.0003 | 0.0018 |
| shred_soft/top1_matches_base_pooled | 0.7600 | 0.7300 | 0.8200 |
| shred_soft/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_soft/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_soft/forced_choice_excess | 0.0067 | 0.0000 | 0.0200 |
| shred_soft/heldout_kl_max | 3.7044 | 3.0526 | 4.4722 |
| shred_soft/injection_rms_share | 0.0012 | 0.0007 | 0.0019 |
| shred_soft/gate_on_unsigned_cells | 0.0014 | 0.0008 | 0.0023 |
| shred_soft/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_hard/kl_to_base | 0.0004 | 0.0003 | 0.0005 |
| shred_hard/top1_matches_base_pooled | 0.7617 | 0.7300 | 0.8200 |
| shred_hard/counterfactual_top1_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/probe_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/forced_choice_excess | 0.0000 | 0.0000 | 0.0000 |
| shred_hard/heldout_kl_max | 3.7153 | 3.0574 | 4.4867 |
| shred_hard/filler_direct | 0.9112 | 0.9095 | 0.9137 |
| shred_hard/filler_kl_to_active | 0.0000 | 0.0000 | 0.0000 |
| restored/direct | 1.0000 | 1.0000 | 1.0000 |
| resigned/direct | 1.0000 | 1.0000 | 1.0000 |
| locality | 0.9683 | 0.9625 | 0.9738 |
| locality_counterfactual_unchanged | 1.0000 | 1.0000 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| masked/kl_to_base | <= 0.05 | 0.0000 | PASS |
| masked/top1_matches_base | >= 0.95 | 1.0000 | PASS |
| direct | >= 0.95 | 0.9095 | FAIL |
| template1_train/direct | >= 0.95 | 0.9811 | PASS |
| direct_heldout_min | >= 0.7 | 0.0905 | FAIL |
| override/direct | >= 0.9 | 1.0000 | PASS |
| override/full_vocab_top1 | >= 0.8 | 1.0000 | PASS |
| override_heldout_min | >= 0.7 | 0.0000 | FAIL |
| agree/direct | >= 0.95 | 1.0000 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.6053 | PASS |
| active/probe_top1 | >= 0.25 | 0.7800 | PASS |
| active/counterfactual_top1_excess | >= 0.5 | 0.5000 | PASS |
| active/forced_choice_excess | >= 0.1 | 0.2800 | PASS |
| revoke/kl_to_base | <= 0.05 | 0.0005 | PASS |
| revoke/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| revoke/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| revoke/probe_excess | <= 0.05 | 0.0000 | PASS |
| revoke/forced_choice_excess | <= 0.05 | 0.0000 | PASS |
| revoke/heldout_kl_max | <= 0.1 | 4.4722 | FAIL |
| revoke/routing_mass_on_target | >= 0.9 | 0.8562 | FAIL |
| shred_soft/kl_to_base | <= 0.05 | 0.0018 | PASS |
| shred_soft/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| shred_soft/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| shred_soft/probe_excess | <= 0.05 | 0.0000 | PASS |
| shred_soft/forced_choice_excess | <= 0.05 | 0.0200 | PASS |
| shred_soft/heldout_kl_max | <= 0.1 | 4.4722 | FAIL |
| shred_hard/kl_to_base | <= 0.05 | 0.0005 | PASS |
| shred_hard/top1_matches_base_pooled | >= 0.95 | 0.7300 | FAIL |
| shred_hard/counterfactual_top1_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/probe_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/forced_choice_excess | <= 0.05 | 0.0000 | PASS |
| shred_hard/heldout_kl_max | <= 0.1 | 4.4867 | FAIL |
| broken1/kl_to_base | <= 0.05 | 0.6412 | FAIL |
| generic/kl_to_base | <= 0.05 | 2.3482 | FAIL |
| generic/kl_to_base_worst_prompt | <= 0.1 | 4.5549 | FAIL |
| revoke/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_soft/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_hard/filler_direct | >= 0.95 | 0.9095 | FAIL |
| shred_hard/filler_kl_to_active | <= 0.05 | 0.0000 | PASS |
| locality | >= 0.98 | 0.9625 | FAIL |
| restored/direct | >= 0.9 | 1.0000 | PASS |
| resigned/direct | >= 0.9 | 1.0000 | PASS |
| rollback/direct | >= 0.9 | 1.0000 | PASS |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| prior/counterfactual_top1_pooled | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| template1_train/direct | 0.9828 | 0.9811 | 2850 | 0.9773 | 0.9873 |
| direct_heldout_min | 0.1260 | 0.0905 | 2850 | 0.1140 | 0.1387 |
| hop2 | 0.9350 | 0.9150 | 600 | 0.9122 | 0.9534 |
| override/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| agree/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| rollback/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| active/probe_top1 | 0.8733 | 0.7800 | 150 | 0.8093 | 0.9220 |
| revoke/top1_matches_base_pooled | 0.7617 | 0.7300 | 600 | 0.7255 | 0.7952 |
| revoke/counterfactual_top1_pooled | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| revoke/probe_top1 | 0.0000 | 0.0000 | 150 | 0.0000 | 0.0243 |
| revoke/forced_choice_win | 0.6533 | 0.6000 | 150 | 0.5714 | 0.7291 |
| revoke/filler_direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| shred_soft/top1_matches_base_pooled | 0.7600 | 0.7300 | 600 | 0.7238 | 0.7937 |
| shred_hard/top1_matches_base_pooled | 0.7617 | 0.7300 | 600 | 0.7255 | 0.7952 |
| shred_hard/filler_direct | 0.9112 | 0.9095 | 2850 | 0.9002 | 0.9214 |
| restored/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| resigned/direct | 1.0000 | 1.0000 | 150 | 0.9757 | 1.0000 |
| locality | 0.9683 | 0.9625 | 2400 | 0.9605 | 0.9750 |

Sample sizes: 50 counterfactual items per seed cannot resolve a 0.05 bar on their own; the gating rate criteria are therefore pooled over item x template (200 per seed, 600 over three seeds) and the exact binomial intervals below are reported for the pooled counts.

Attack convention: Every attack bar is a PAIRED EXCESS over the frozen model itself, because the counterfactual object is a real capital token the pretrained prior already favours: an absolute forced-choice or probe threshold would measure GPT-2's prior, not leakage from the cell. The floors are recorded as prior/forced_choice_win, prior/probe_top1 and prior/counterfactual_top1_pooled, measured on the same rows with the same distractor draws.

Validity condition: attack_validity: with the cell ACTIVE the same attacks must succeed. If they do not, their failure after deletion is uninformative and the record reports F1 regardless of the fallback groups.

By construction: copy_bound_by_construction: the adapter acts only through the injection; with every cell masked the null read is a fixed zero, so the base distribution is returned exactly (recorded, not learned); fallback_after_revoke_by_construction: with status_gated the status flag multiplies the gate, so a revoked cell's value is exactly zero and the injection vanishes. Exact equality to the base model after REVOKE is therefore arithmetic, not a learned behaviour; the LEARNED residue is that the routing does not spill onto neighbouring ACTIVE cells (kl_to_base, heldout_kl_max) while the revoked cell itself stays addressed (routing_mass_on_target). This group is recorded and does NOT grant a deletion level.; the pretrained fact is never deleted: the weights are frozen; what is measured is that the model answers with it again after REVOKE / SHRED.

Not claimed: unlearning of pretrained facts; LLM scale; multi-token entities.

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

## E-000016 — Alias chains: how far the indirection carries

Evidence level: **E4** (synthetic system). Seeds: [0, 1, 2]; 4000 steps; 30% of the aliases in training point at another alias.

E-000015 recorded that its two-slot control did not resolve chains. The cause proposed there was the training distribution, not the architecture; this experiment tests that explanation by putting 30% chains into training and changing nothing else.

| claim group | supported |
|---|---|
| two_slots_resolve_a_chain | yes |
| one_slot_refuses_a_chain | yes |
| no_price_paid_elsewhere | yes |
| sharing_still_holds | yes |
| shredding_a_pointer | yes |

| measure | mean over seeds | worst seed | best seed |
|---|---|---|---|
| two/direct | 1.0000 | 1.0000 | 1.0000 |
| two/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| two/chain2/answer_acc | 1.0000 | 1.0000 | 1.0000 |
| two/chain2/unknown | 0.0000 | 0.0000 | 0.0000 |
| two/chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| one/direct | 1.0000 | 1.0000 | 1.0000 |
| one/alias_direct | 1.0000 | 1.0000 | 1.0000 |
| one/chain2/answer_acc | 0.0000 | 0.0000 | 0.0000 |
| one/chain2/unknown | 1.0000 | 1.0000 | 1.0000 |
| one/chain2/depth1_acc | 1.0000 | 1.0000 | 1.0000 |
| two/alias_provenance_pair | 1.0000 | 1.0000 | 1.0000 |
| two/hop2 | 1.0000 | 1.0000 | 1.0000 |
| two/shared_update/alias_new_object | 1.0000 | 1.0000 | 1.0000 |
| two/duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| two/shred_target/alias_unknown | 0.9967 | 0.9900 | 1.0000 |
| two/shred_target/alias_probe_top1 | 0.0100 | 0.0100 | 0.0100 |
| two/dup_shred/copy_direct_acc | 1.0000 | 1.0000 | 1.0000 |
| two/dup_shred/copy_probe_top1 | 0.8867 | 0.8400 | 0.9300 |
| two/shred_alias/alias_unknown | 0.9867 | 0.9700 | 1.0000 |
| one/shred_alias/alias_unknown | 0.9833 | 0.9700 | 1.0000 |
| two/delete_target/alias_unknown | 0.9967 | 0.9900 | 1.0000 |
| two/deref_disabled/alias_direct | 0.0000 | 0.0000 | 0.0000 |
| two/regression/direct | 1.0000 | 1.0000 | 1.0000 |
| two/regression/hop2 | 1.0000 | 1.0000 | 1.0000 |
| two/regression/hop3 | 0.9956 | 0.9900 | 1.0000 |
| two/regression/reverse | 1.0000 | 1.0000 | 1.0000 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| two/chain2/answer_acc | >= 0.9 | 1.0000 | PASS |
| one/chain2/answer_acc | <= 0.2 | 0.0000 | PASS |
| one/chain2/unknown | >= 0.9 | 1.0000 | PASS |
| two/direct | >= 0.98 | 1.0000 | PASS |
| two/alias_direct | >= 0.95 | 1.0000 | PASS |
| two/hop2 | >= 0.95 | 1.0000 | PASS |
| two/regression/direct | >= 0.98 | 1.0000 | PASS |
| two/regression/hop2 | >= 0.95 | 1.0000 | PASS |
| two/regression/reverse | >= 0.95 | 1.0000 | PASS |
| two/alias_provenance_pair | >= 0.9 | 1.0000 | PASS |
| two/shared_update/alias_new_object | >= 0.95 | 1.0000 | PASS |
| two/duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| two/shred_target/alias_unknown | >= 0.95 | 0.9900 | PASS |
| two/shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| two/dup_shred/copy_direct_acc | >= 0.95 | 1.0000 | PASS |
| two/shred_alias/alias_unknown | >= 0.95 | 0.9700 | PASS |
| one/shred_alias/alias_unknown | >= 0.95 | 0.9700 | PASS |

By construction: the store resolves a chain by following kids with a depth limit and a cycle check; what is measured is whether the trained model reproduces it from the pointers alone; the one-slot arm CANNOT represent a two-link chain: it has one dereference slot. Its criterion is that it answers unknown rather than inventing an entity..

Learned: following a pointer whose target is itself a pointer, with the query for each dereference coming from the value just read; refusing a chain that does not fit the available slots instead of naming another entity.

Not claimed: chains deeper than the number of slots; LLM scale.

**Interpretation (post hoc, record unchanged):** The follow-up that turns E-000015's two recorded failures into an explanation. Both were caused by the training distribution rather than the architecture: with 30% chains in training, two dereference slots resolve a two-link chain completely, a one-slot model refuses it (100% unknown, 0% answered) instead of naming another entity, and shredding the pointer rather than the payload rises from 93% to 97% on the worst seed. The refusal arm is the load-bearing part: it shows the depth the mechanism reaches is set by the number of slots, and that the model reports the limit instead of hiding it.

## E-000017-A — Reading versus refusal on held-out phrasings (diagnosis, no training)

Is the held-out failure of E-000011/E-000012 a deletion failure or a reading failure that deletion inherits?

No model was trained for this record: E-000012's three checkpoints are evaluated as they were recorded, so this is a decomposition of an existing result, not a new one.

| measure | mean over seeds | worst seed |
|---|---|---|
| train/active_correct | 0.9608 | 0.9550 |
| heldout/active_correct | 0.6937 | 0.6700 |
| train/refusal_given_active_correct | 0.9991 | 0.9973 |
| heldout/refusal_given_active_correct | 0.9614 | 0.9418 |
| heldout/revoked_deleted_object | 0.0008 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0015 | 0.0000 |

Read the two conditional rows together: `refusal_given_active_correct` is how often the model answers ' unknown' after REVOKE among exactly those targets it read correctly while the cell was ACTIVE, and `deleted_object_given_active_correct` is how often it returns the deleted object instead.

**Interpretation (post hoc, record unchanged):** The record that reframes the programme's one measured failure. Roadmap kill criterion 5 fired on the unconditional refusal rate, and that stands. Decomposing E-000012's own checkpoints without training anything shows the cause: conditioned on the model having read the fact at all while the cell was active, it refuses after REVOKE 96.1% of the time on held-out phrasings and returns the deleted object in 0.15% of those cases. What does not generalise is reading (69.4% against 96.1%), so the defect sits in the query and routing path. The worst-seed conditional figure is 94.2%, still under the bar, so the remedy run with the template budget the roadmap prescribes is still owed.

## E-000017-B — Stage-2 template budget: 8 trained, 4 held out, no consistency loss

Roadmap kill criterion 5 fired on a two-template budget. This run gives the stage the budget it prescribes and reports whether the held-out failure survives it.

| claim group | supported |
|---|---|
| reading_generalises | **no** |
| refusal_generalises | yes |
| refusal_on_trained_templates_holds | yes |
| deleted_object_never_returns | yes |
| no_key_no_injection | **no** |

| measure | mean over seeds | worst seed |
|---|---|---|
| train/active_correct | 0.9198 | 0.9119 |
| heldout/active_correct | 0.7400 | 0.7288 |
| train/refusal_given_active_correct | 0.9966 | 0.9952 |
| heldout/refusal_given_active_correct | 0.9928 | 0.9870 |
| heldout/revoked_deleted_object | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0000 | 0.0000 |
| revoke_train_min | 0.9583 | 0.9550 |
| revoke_heldout_min | 0.8983 | 0.8650 |
| shred_train_min | 0.9583 | 0.9550 |
| shred_heldout_min | 0.8983 | 0.8650 |
| broken1_unknown | 0.7183 | 0.6300 |
| generic/kl_to_base | 3.2741 | 2.9591 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| heldout/active_correct | >= 0.9 | 0.7288 | FAIL |
| train/active_correct | >= 0.95 | 0.9119 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8650 | PASS |
| shred_heldout_min | >= 0.85 | 0.8650 | PASS |
| revoke_train_min | >= 0.95 | 0.9550 | PASS |
| shred_train_min | >= 0.9 | 0.9550 | PASS |
| heldout/revoked_deleted_object | <= 0.02 | 0.0000 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0000 | PASS |
| broken1_unknown | >= 0.9 | 0.6300 | FAIL |
| generic/kl_to_base | <= 0.05 | 3.6474 | FAIL |

**Interpretation (post hoc, record unchanged):** The remedy run for the fired kill criterion, and it works for the part the criterion is about: at the prescribed budget of eight trained templates, refusal after REVOKE and SHRED on unseen phrasings reaches 89.8% (worst seed 86.5%) against 52% at two templates, the conditional figure reaches 99.3%, and the deleted object returns in exactly 0.0000 of cases. The criterion's own 95% bar is still not met, so it stays fired, but it is no longer evidence against the deletion mechanism. The run also surfaces a worse problem than the one it fixed: injection where there is no key degraded rather than improved (generic text 3.27 nats against a 0.05 bar, above E-000013's 2.27), so more prompt shapes in training mean more shapes that trigger a spurious read. That is the next thing to fix, because it means the layer perturbs the frozen model on unrelated text.

## E-000018 — No key, no injection — arm 'both' (match gate True, generic text share 0.25)

The routing softmax always sums to one, so some cell always wins and the layer injects into text it has no key for. E-000017-B measured 3.27 nats on generic sentences against a 0.05 bar, worse than E-000013's 2.27 with fewer templates.

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | **no** |
| refusal_not_traded_away | **no** |
| deleted_object_never_returns | yes |

| measure | mean over seeds | worst seed | E-000017-B baseline |
|---|---|---|---|
| generic/kl_to_base | 0.6736 | 0.4951 | 3.2741 |
| broken1_unknown | 0.7017 | 0.6650 | 0.7183 |
| train/active_correct | 0.9019 | 0.8869 | 0.9198 |
| heldout/active_correct | 0.6917 | 0.6837 | 0.7400 |
| train/refusal_given_active_correct | 0.9960 | 0.9922 | - |
| heldout/refusal_given_active_correct | 0.9882 | 0.9829 | 0.9928 |
| heldout/revoked_deleted_object | 0.0004 | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0000 | 0.0000 | - |
| revoke_train_min | 0.9533 | 0.9300 | - |
| revoke_heldout_min | 0.8550 | 0.8100 | 0.8983 |
| shred_train_min | 0.9533 | 0.9300 | - |
| shred_heldout_min | 0.8550 | 0.8100 | 0.8983 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| generic/kl_to_base | <= 0.05 | 0.9326 | FAIL |
| broken1_unknown | >= 0.9 | 0.6650 | FAIL |
| train/active_correct | >= 0.9 | 0.8869 | FAIL |
| heldout/active_correct | >= 0.7 | 0.6837 | FAIL |
| revoke_train_min | >= 0.95 | 0.9300 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8100 | FAIL |
| shred_heldout_min | >= 0.85 | 0.8100 | FAIL |
| heldout/revoked_deleted_object | <= 0.02 | 0.0013 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0000 | PASS |

By construction: the match gate adds the CAPACITY to inject nothing (an absolute cosine threshold against the best real cell key); whether the model uses it is learned from the losses; the generic arm trains the behaviour on eight sentence shapes that are disjoint from the five the evaluation uses, so passing by memorising a shape is not available.

**Interpretation (post hoc, record unchanged):** PARTLY WITHDRAWN. The match-gate arm measured nothing because the gate was cancelled by the RMS-matched injection one line later: a scalar that multiplies a vector and is then divided out by that vector's own norm cannot act. The conclusion 'all of the improvement is the training and none is the capacity' therefore does not follow from this record, and the capacity question is reopened in E-000022. What survives: training on generic text brings injection into unrelated text down by a factor of five and no arm gets within a factor of twelve of the bar, and both arms that move the number pay for it in reading and refusal. Original text follows. Read the three arms together rather than one at a time. The match gate alone leaves injection into unrelated text exactly where it was (3.2681 against a baseline of 3.2741); generic text in training alone brings it to 0.6035; both together to 0.6736. All of the improvement is the behavioural training and none is the added capacity, and no arm gets within a factor of twelve of the bar. The reason is that refusing a question and ignoring prose are routed through one null column and pull in opposite directions, which is a design fault rather than a tuning failure.

## E-000018 — No key, no injection — arm 'gate' (match gate True, generic text share 0)

The routing softmax always sums to one, so some cell always wins and the layer injects into text it has no key for. E-000017-B measured 3.27 nats on generic sentences against a 0.05 bar, worse than E-000013's 2.27 with fewer templates.

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | yes |
| refusal_not_traded_away | **no** |
| deleted_object_never_returns | yes |

| measure | mean over seeds | worst seed | E-000017-B baseline |
|---|---|---|---|
| generic/kl_to_base | 3.2681 | 3.0802 | 3.2741 |
| broken1_unknown | 0.7117 | 0.6300 | 0.7183 |
| train/active_correct | 0.9165 | 0.9062 | 0.9198 |
| heldout/active_correct | 0.7379 | 0.7288 | 0.7400 |
| train/refusal_given_active_correct | 0.9964 | 0.9950 | - |
| heldout/refusal_given_active_correct | 0.9908 | 0.9865 | 0.9928 |
| heldout/revoked_deleted_object | 0.0013 | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0025 | 0.0000 | - |
| revoke_train_min | 0.9483 | 0.9350 | - |
| revoke_heldout_min | 0.8667 | 0.8250 | 0.8983 |
| shred_train_min | 0.9483 | 0.9350 | - |
| shred_heldout_min | 0.8667 | 0.8250 | 0.8983 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| generic/kl_to_base | <= 0.05 | 3.3764 | FAIL |
| broken1_unknown | >= 0.9 | 0.6300 | FAIL |
| train/active_correct | >= 0.9 | 0.9062 | PASS |
| heldout/active_correct | >= 0.7 | 0.7288 | PASS |
| revoke_train_min | >= 0.95 | 0.9350 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8250 | FAIL |
| shred_heldout_min | >= 0.85 | 0.8250 | FAIL |
| heldout/revoked_deleted_object | <= 0.02 | 0.0025 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0053 | PASS |

By construction: the match gate adds the CAPACITY to inject nothing (an absolute cosine threshold against the best real cell key); whether the model uses it is learned from the losses; the generic arm trains the behaviour on eight sentence shapes that are disjoint from the five the evaluation uses, so passing by memorising a shape is not available.

## E-000018 — No key, no injection — arm 'generic' (match gate False, generic text share 0.25)

The routing softmax always sums to one, so some cell always wins and the layer injects into text it has no key for. E-000017-B measured 3.27 nats on generic sentences against a 0.05 bar, worse than E-000013's 2.27 with fewer templates.

| claim group | supported |
|---|---|
| no_key_no_injection | **no** |
| reading_not_traded_away | **no** |
| refusal_not_traded_away | **no** |
| deleted_object_never_returns | yes |

| measure | mean over seeds | worst seed | E-000017-B baseline |
|---|---|---|---|
| generic/kl_to_base | 0.6035 | 0.3622 | 3.2741 |
| broken1_unknown | 0.6783 | 0.6250 | 0.7183 |
| train/active_correct | 0.9079 | 0.9019 | 0.9198 |
| heldout/active_correct | 0.6888 | 0.6813 | 0.7400 |
| train/refusal_given_active_correct | 0.9943 | 0.9934 | - |
| heldout/refusal_given_active_correct | 0.9844 | 0.9773 | 0.9928 |
| heldout/revoked_deleted_object | 0.0008 | 0.0000 | 0.0000 |
| heldout/deleted_object_given_active_correct | 0.0016 | 0.0000 | - |
| revoke_train_min | 0.9500 | 0.9400 | - |
| revoke_heldout_min | 0.8800 | 0.8600 | 0.8983 |
| shred_train_min | 0.9500 | 0.9400 | - |
| shred_heldout_min | 0.8800 | 0.8600 | 0.8983 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| generic/kl_to_base | <= 0.05 | 0.8008 | FAIL |
| broken1_unknown | >= 0.9 | 0.6250 | FAIL |
| train/active_correct | >= 0.9 | 0.9019 | PASS |
| heldout/active_correct | >= 0.7 | 0.6813 | FAIL |
| revoke_train_min | >= 0.95 | 0.9400 | FAIL |
| revoke_heldout_min | >= 0.85 | 0.8600 | PASS |
| shred_heldout_min | >= 0.85 | 0.8600 | PASS |
| heldout/revoked_deleted_object | <= 0.02 | 0.0013 | PASS |
| heldout/deleted_object_given_active_correct | <= 0.02 | 0.0026 | PASS |

By construction: the match gate adds the CAPACITY to inject nothing (an absolute cosine threshold against the best real cell key); whether the model uses it is learned from the losses; the generic arm trains the behaviour on eight sentence shapes that are disjoint from the five the evaluation uses, so passing by memorising a shape is not available.

## E-000019 — Fresh seeds, and the SHRED residual tested against chance

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

**Interpretation (post hoc, record unchanged):** The record that turns F4 from a tolerance claim into a chance claim, and does it outside the seeds that chose the configuration. Forced choice lands on exactly 375 of 750 pooled trials, the probe on 4 of 750 against a chance of 1 in 256, the true object top-1 on 7 of 750; every exact interval contains its chance level and stays inside the pre-registered distance. Two objections from the standing audit are answered in one run. What is not answered: the hard gate still admits an unsigned marker in at least one seed, and the top-1 interval only just contains chance with a point estimate about two and a half times the chance rate, so a larger sample could still separate them.

## E-000020 — Shared knowledge objects in a frozen GPT-2

Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **F1**. Seeds: [0, 1, 2]; 3000 steps.

The same world is written twice and read by the same trained adapter from natural-language prompts: in the symlink arm the alias keys are LINK cells over shared targets, in the duplication arm they are ordinary fact cells holding a copy. Every sharing claim is the difference between the arms.

| what is measured | symlink arm | duplication arm |
|---|---|---|
| one UPDATE on the shared object reaches every access path | 51.8% | 0.0% |
| after one SHRED the object is still readable | 0.5% | 55.8% |
| after one SHRED a probe recovers the object | 0.3% | 49.8% |
| operations needed to reach every access path | 1 | 3 |

| claim group | supported |
|---|---|
| reading_through_an_alias | **no** |
| one_update_reaches_every_path | **no** |
| one_shred_deletes_every_path | **no** |
| attacks_through_every_alias | yes |
| attack_validity | yes |
| alias_lifecycle | **no** |

| measure | mean over seeds | worst seed |
|---|---|---|
| direct | 0.5700 | 0.5633 |
| alias_direct | 0.5067 | 0.5000 |
| dup_direct | 0.5483 | 0.5100 |
| alias_heldout_min | 0.3700 | 0.3550 |
| probe_calibration_top1 | 0.4806 | 0.4333 |
| active/alias_probe_top1 | 0.3883 | 0.3600 |
| active/alias_forced_choice | 0.8950 | 0.8800 |
| shared_update/alias_new_object | 0.5183 | 0.5050 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 |
| rollback/alias_direct | 0.5067 | 0.5000 |
| shred_target/alias_unknown | 0.9950 | 0.9850 |
| shred_target/alias_true_object | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0033 | 0.0000 |
| shred_target/alias_forced_choice | 0.5100 | 0.4750 |
| shred_target/alias_top1_among_entities | 0.0067 | 0.0000 |
| shred_target/alias_mean_rank | 129.6117 | 120.3350 |
| dup_shred/copy_direct_acc | 0.5583 | 0.5400 |
| dup_shred/copy_probe_top1 | 0.4983 | 0.4450 |
| dup_shred/copy_forced_choice | 0.9733 | 0.9700 |
| resign_target/alias_direct | 0.5067 | 0.5000 |
| revoke_alias/alias_unknown | 0.9800 | 0.9500 |
| revoke_alias/sibling_readable | 0.5567 | 0.4900 |
| revoke_alias/target_readable | 0.5767 | 0.5200 |
| delete_target/alias_unknown | 0.9917 | 0.9800 |
| delete_target/alias_true_object | 0.0000 | 0.0000 |

Exact binomial intervals (pooled over seeds):

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| direct | 0.5700 | 0.5633 | 900 | 0.5369 | 0.6026 |
| alias_direct | 0.5067 | 0.5000 | 600 | 0.4659 | 0.5474 |
| dup_direct | 0.5483 | 0.5100 | 600 | 0.5075 | 0.5887 |
| shared_update/alias_new_object | 0.5183 | 0.5050 | 600 | 0.4775 | 0.5590 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_unknown | 0.9950 | 0.9850 | 600 | 0.9855 | 0.9990 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 600 | 0.0000 | 0.0061 |
| shred_target/alias_probe_top1 | 0.0033 | 0.0000 | 600 | 0.0004 | 0.0120 |
| dup_shred/copy_direct_acc | 0.5583 | 0.5400 | 600 | 0.5176 | 0.5985 |
| revoke_alias/alias_unknown | 0.9800 | 0.9500 | 300 | 0.9570 | 0.9926 |
| delete_target/alias_unknown | 0.9917 | 0.9800 | 600 | 0.9807 | 0.9973 |

Pre-registered criteria (worst seed):

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5633 | FAIL |
| alias_direct | >= 0.8 | 0.5000 | FAIL |
| dup_direct | >= 0.85 | 0.5100 | FAIL |
| alias_heldout_min | >= 0.5 | 0.3550 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.5050 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.5400 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5300 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.3600 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.4333 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9500 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.4900 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5200 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9800 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

By construction: the store decides which payload a row carries; the bank never exports the target's payload, status or signature, and the model is never told that a value is a pointer; that one operation on a shared object reaches every alias is a property of the store; what is measured is whether the frozen model reports it, and whether the SAME model reports the duplication arm, where it does not, correctly.

Learned: following a pointer inside a frozen pretrained transformer: the dereference query comes from the value just read, and the passthrough column keeps a value that was not a pointer; answering unknown for a dangling pointer after DELETE and for a revoked alias.

Not claimed: chains deeper than one dereference; multi-token entities; anything above 124M parameters.

**Interpretation (post hoc, record unchanged):** The synthetic symlink result carried to a real pretrained model, and it splits cleanly. The sharing and deletion contrast holds: one operation on the shared object leaves every alias path unreadable and unrecoverable (probe 0.33%, forced choice 0.51 at a chance of 0.5) while the duplication arm, built from the same world and read by the same model, stays recoverable at 49.8% and 0.97. Reading does not hold: 57% on direct facts against the 85% this adapter reaches without links. A diagnostic settles where that cost sits — the link-free adapter reads the same evaluation world at 82.7%, and 84.7% with the machinery attached, so the mechanism is free at inference and the price is paid in learning the harder distribution.

## E-000021 — The verification gate as a classifier

The standing audit's objection that the deletion certificate is a learned classifier whose false-accept rate is reported only as a worst-seed maximum.

Nothing was trained. Every recorded checkpoint of the verified-gate family is loaded and its gate scored on freshly drawn markers; the rest of the model is not involved, because the gate is a function of the marker alone.

**Pooled over 11 checkpoints and 2,200,000 markers per class: 1867 false accepts (rate 8.49e-04, 95% interval [8.11e-04, 8.88e-04]) and 0 false rejects (rate 0.00e+00).**

| family | false accepts | rate | 95% interval | false rejects | max score on an unsigned marker | min score on a signed marker |
|---|---|---|---|---|---|---|
| e000010 | 811/1000000 | 8.11e-04 | [7.56e-04, 8.69e-04] | 0/1000000 | 0.8610 | 0.9926 |
| e000014 | 576/600000 | 9.60e-04 | [8.83e-04, 1.04e-03] | 0/600000 | 0.8774 | 0.9902 |
| e000019 | 480/600000 | 8.00e-04 | [7.30e-04, 8.75e-04] | 0/600000 | 0.8929 | 0.9878 |

Pre-registered criteria:

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| false_accept_rate | <= 0.001 | 0.0008 | PASS |
| false_reject_rate | <= 0.001 | 0.0000 | PASS |
| false_accept_ci_upper | <= 0.01 | 0.0009 | PASS |

This is the gate's error rate on markers drawn from the same two distributions the programme uses. It is not a security claim: an adversary who can choose the marker is not modelled here, and a gate that separates two fixed distributions says nothing about one that must resist a search for a passing vector.

**Interpretation (post hoc, record unchanged):** The number the deletion claim needed and did not have. Across 2.2 million fresh unsigned markers and eleven checkpoints the gate admits one in about 1,180, with a tight interval and no false rejects at all. That is the bound on every SHRED result in this programme: behavioural deletion is complete and the residual sits at chance, but roughly one payload per thousand would pass verification. It just clears the pre-registered bar of one in a thousand, and it is a limit rather than a guarantee.

## e000022_two_channel_null

_not run in this session_

## E-000023 — Alias reading in a frozen GPT-2: the 'curriculum' arm against E-000020's budget

E-000020 read direct facts at 57% while the same adapter without links reads this evaluation world at 82.7%, and 84.7% with the machinery attached, so the cost is in learning rather than in the world or the mechanism. This arm attacks the learning and changes nothing else.

| claim group | supported |
|---|---|
| reading_through_an_alias | **no** |
| one_update_reaches_every_path | **no** |
| one_shred_deletes_every_path | **no** |
| attacks_through_every_alias | yes |
| attack_validity | **no** |
| alias_lifecycle | **no** |

| measure | mean over seeds | worst seed | E-000020 baseline |
|---|---|---|---|
| direct | 0.6289 | 0.5767 | 0.5700 |
| alias_direct | 0.0000 | 0.0000 | 0.5067 |
| dup_direct | 0.6867 | 0.6550 | 0.5483 |
| alias_heldout_min | 0.0000 | 0.0000 | 0.3700 |
| probe_calibration_top1 | 0.4833 | 0.4250 | 0.4806 |
| active/alias_probe_top1 | 0.0017 | 0.0000 | 0.3883 |
| active/alias_forced_choice | 0.4567 | 0.3900 | 0.8950 |
| shared_update/alias_new_object | 0.0000 | 0.0000 | 0.5183 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| rollback/alias_direct | 0.0000 | 0.0000 | 0.5067 |
| shred_target/alias_unknown | 0.9950 | 0.9850 | 0.9950 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_probe_top1 | 0.0017 | 0.0050 | 0.0033 |
| shred_target/alias_forced_choice | 0.4517 | 0.5450 | 0.5100 |
| shred_target/alias_top1_among_entities | 0.0050 | 0.0100 | 0.0067 |
| shred_target/alias_mean_rank | 131.4350 | 120.0300 | 129.6117 |
| dup_shred/copy_direct_acc | 0.6933 | 0.6700 | 0.5583 |
| dup_shred/copy_probe_top1 | 0.5683 | 0.5150 | 0.4983 |
| dup_shred/copy_forced_choice | 0.9783 | 0.9750 | 0.9733 |
| resign_target/alias_direct | 0.0000 | 0.0000 | 0.5067 |
| revoke_alias/alias_unknown | 0.9733 | 0.9600 | 0.9800 |
| revoke_alias/sibling_readable | 0.0000 | 0.0000 | 0.5567 |
| revoke_alias/target_readable | 0.6533 | 0.5600 | 0.5767 |
| delete_target/alias_unknown | 0.9950 | 0.9850 | 0.9917 |
| delete_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |

Pre-registered criteria, identical to E-000020's:

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5767 | FAIL |
| alias_direct | >= 0.8 | 0.0000 | FAIL |
| dup_direct | >= 0.85 | 0.6550 | FAIL |
| alias_heldout_min | >= 0.5 | 0.0000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.0000 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.0000 | FAIL |
| duplicate_update/alias_old_object | >= 0.85 | 0.6550 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.6700 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.0000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0050 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5450 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.0000 | FAIL |
| probe_calibration_top1 | >= 0.2 | 0.4250 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.5150 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9600 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.0000 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5600 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9850 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

## e000023_longer

_not run in this session_

## e000024_weights_vs_cells

_not run in this session_

## E-000025 — re-scoring the symlink checkpoints across all twelve templates

No training. The checkpoints of E-000020 (link adapter) and E-000017-B (link-free adapter, eight
trained templates) are loaded from disk and scored at every one of the twelve templates on one
alias world, in both stores.

E-000020's headline numbers — direct 0.5667, alias 0.5067 — are template 0 only, because
`E20._answers` defaults to it. The table below is what the same checkpoints do everywhere else.

### Reading, per template (mean over seeds)

| template | direct (link adapter) | alias, shared object | alias, duplicated | direct (link-free) | alias, duplicated (link-free) |
|---|---|---|---|---|---|
| t0 (trained) | 0.6122 | 0.6167 | 0.6233 | 0.7889 | 0.7900 |
| t1 (trained) | 0.9633 | 0.8367 | 0.9817 | 0.9967 | 0.9933 |
| t2 (trained) | 0.6322 | 0.6017 | 0.6217 | 0.7789 | 0.7650 |
| t3 (trained) | 0.9956 | 0.8933 | 0.9983 | 1.0000 | 1.0000 |
| t4 (trained) | 0.9967 | 0.9350 | 0.9967 | 0.9989 | 1.0000 |
| t5 (trained) | 0.9989 | 0.9250 | 0.9967 | 1.0000 | 1.0000 |
| t6 (trained) | 0.6422 | 0.6317 | 0.6317 | 0.7778 | 0.7633 |
| t7 (trained) | 0.9989 | 0.8917 | 0.9967 | 0.9989 | 0.9950 |
| t8 (held out) | 0.4500 | 0.4283 | 0.4133 | 0.5622 | 0.5200 |
| t9 (held out) | 0.9567 | 0.9033 | 0.9817 | 0.9622 | 0.9650 |
| t10 (held out) | 0.9989 | 0.9250 | 1.0000 | 1.0000 | 1.0000 |
| t11 (held out) | 0.3078 | 0.3433 | 0.2883 | 0.4056 | 0.4200 |

`alias, shared object` and `alias, duplicated` are the *same adapter* answering the *same questions*
against a store that shares one object and a store that holds independent copies: their difference is
the cost of sharing. The difference between `alias, duplicated (link-free)` and `alias, duplicated` is
the cost of having trained on links at all.

### Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| train/alias_max | >= 0.75 | 0.8950 | PASS |
| heldout/alias_mean | >= 0.55 | 0.6013 | PASS |
| all/cost_of_sharing | <= 0.1 | 0.0954 | PASS |
| all/cost_of_link_training | <= 0.25 | 0.0688 | PASS |

Disclosure: alias reading at templates 1, 8 and 9 was already recorded in E-000020, so these
thresholds were set knowing three of the sixty cells above. This record confirms a reading of
existing numbers; it is not an independent prediction.

### Aggregates

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| train/direct_mean | 0.8550 | 0.8313 | - | - | - |
| train/alias_mean | 0.7915 | 0.7325 | - | - | - |
| train/dup_mean | 0.8558 | 0.8387 | - | - | - |
| train/linkfree_direct_mean | 0.9175 | 0.9104 | - | - | - |
| train/linkfree_dup_mean | 0.9133 | 0.8894 | - | - | - |
| heldout/direct_mean | 0.6783 | 0.6617 | - | - | - |
| heldout/alias_mean | 0.6500 | 0.6013 | - | - | - |
| heldout/dup_mean | 0.6708 | 0.6500 | - | - | - |
| heldout/linkfree_direct_mean | 0.7325 | 0.7175 | - | - | - |
| heldout/linkfree_dup_mean | 0.7262 | 0.6987 | - | - | - |
| all/direct_mean | 0.7961 | 0.7747 | - | - | - |
| all/alias_mean | 0.7443 | 0.6888 | - | - | - |
| all/dup_mean | 0.7942 | 0.7812 | - | - | - |
| all/linkfree_direct_mean | 0.8558 | 0.8461 | - | - | - |
| all/linkfree_dup_mean | 0.8510 | 0.8258 | - | - | - |
| train/alias_max | 0.9367 | 0.8950 | - | - | - |
| train/alias_min | 0.6017 | 0.4850 | - | - | - |
| train/cost_of_sharing | 0.0644 | 0.1063 | - | - | - |
| train/cost_of_link_training | 0.0575 | 0.0725 | - | - | - |
| heldout/alias_max | 0.9250 | 0.8700 | - | - | - |
| heldout/alias_min | 0.3433 | 0.3000 | - | - | - |
| heldout/cost_of_sharing | 0.0208 | 0.0737 | - | - | - |
| heldout/cost_of_link_training | 0.0554 | 0.0613 | - | - | - |
| all/alias_max | 0.9383 | 0.8950 | - | - | - |
| all/alias_min | 0.3433 | 0.3000 | - | - | - |
| all/cost_of_sharing | 0.0499 | 0.0954 | - | - | - |
| all/cost_of_link_training | 0.0568 | 0.0688 | - | - | - |

### Provenance

a forced re-run of E-000020 overwrote its seed-0 and seed-1 checkpoints after that record was written; only seed 2 still matches the SHA-256 recorded there. The SHA of every checkpoint scored here is in per_seed.

**Interpretation (post hoc, record unchanged):** Trains nothing: it re-reads E-000020's and E-000017-B's checkpoints at every one of the twelve templates. E-000020's headline pair -- direct 0.5667, alias 0.5067 -- is template 0, because its _answers helper defaults to it. Reading here is bimodal by phrasing and template 0 is one of the weak ones: the same checkpoints read a base fact at 0.9989 and resolve a symlink at 0.9250 on the HELD-OUT template 10, and 0.9350 on trained template 4. Separating the two costs the single number confounded, over all twelve templates and on the worst of three seeds: sharing costs 0.0954 against duplicated copies read by the same adapter, and having trained on links at all costs 0.0688 against the link-free adapter on the same store. The world seed matches E-000020's, so on seed 2 -- the one checkpoint whose SHA still matches that record -- template 0 returns 0.563 / 0.500 against the recorded 0.5633 / 0.50, a reproduction rather than a fresh measurement. Ledger 31.9 carries the correction; the E-000020 record is left as produced.

## E-000026 — the symlink lifecycle battery, measured where reading works

Seeds [0, 1, 2]. No training: E-000020's checkpoints, E-000020's battery, run three times at
three phrasings — template 0 (what that record used), template 3 (the trained
template on which E-000017-B's *link-free* adapter reads best) and template 10
(the same rule over the held-out four). The choice comes from a different experiment on a different
adapter, so it cannot be tuned in the link arm's favour.

### The battery at three phrasings

| measure (worst seed) | template0 (t0) | strong_train (t3) | strong_heldout (t10) |
|---|---|---|---|
| direct | 0.5633 | 0.9933 | 0.9967 |
| alias_direct | 0.5000 | 0.8600 | 0.8700 |
| dup_direct | 0.5900 | 0.9950 | 1.0000 |
| shared_update/alias_new_object | 0.5350 | 0.8650 | 0.8850 |
| duplicate_update/alias_new_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_unknown | 0.9950 | 0.9950 | 1.0000 |
| shred_target/alias_true_object | 0.0000 | 0.0000 | 0.0000 |
| shred_target/alias_forced_choice | 0.4650 | 0.4350 | 0.4300 |
| shred_target/alias_probe_top1 | 0.0100 | 0.0100 | 0.0100 |
| active/alias_probe_top1 | 0.4200 | 0.7800 | 0.8000 |
| revoke_alias/sibling_readable | 0.5800 | 0.8800 | 0.8800 |
| delete_target/alias_unknown | 0.9650 | 0.9600 | 0.9100 |

### Pre-registered criteria (E-000020's, unchanged)

#### template0 — template 0

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.5633 | FAIL |
| alias_direct | >= 0.8 | 0.5000 | FAIL |
| dup_direct | >= 0.85 | 0.5900 | FAIL |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.5350 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_unknown | >= 0.9 | 0.9950 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.5900 | FAIL |
| resign_target/alias_direct | >= 0.8 | 0.5000 | FAIL |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.5600 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0100 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.4200 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.5000 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.5500 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 0.9600 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.5800 | FAIL |
| revoke_alias/target_readable | >= 0.8 | 0.5600 | FAIL |
| delete_target/alias_unknown | >= 0.9 | 0.9650 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: attacks_through_every_alias, attack_validity.

#### strong_train — template 3

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.9933 | PASS |
| alias_direct | >= 0.8 | 0.8600 | PASS |
| dup_direct | >= 0.85 | 0.9950 | PASS |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.8650 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.8600 | PASS |
| shred_target/alias_unknown | >= 0.9 | 0.9950 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 0.9950 | PASS |
| resign_target/alias_direct | >= 0.8 | 0.8600 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.4850 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0150 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.7800 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.7917 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.8100 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.8800 | PASS |
| revoke_alias/target_readable | >= 0.8 | 0.9900 | PASS |
| delete_target/alias_unknown | >= 0.9 | 0.9600 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: one_shred_deletes_every_path, attacks_through_every_alias, attack_validity, alias_lifecycle.

#### strong_heldout — template 10

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| direct | >= 0.85 | 0.9967 | PASS |
| alias_direct | >= 0.8 | 0.8700 | PASS |
| dup_direct | >= 0.85 | 1.0000 | PASS |
| alias_heldout_min | >= 0.5 | 0.3000 | FAIL |
| shared_update/alias_new_object | >= 0.9 | 0.8850 | FAIL |
| duplicate_update/alias_new_object | <= 0.05 | 0.0000 | PASS |
| rollback/alias_direct | >= 0.8 | 0.8700 | PASS |
| shred_target/alias_unknown | >= 0.9 | 1.0000 | PASS |
| shred_target/alias_true_object | <= 0.05 | 0.0000 | PASS |
| dup_shred/copy_direct_acc | >= 0.85 | 1.0000 | PASS |
| resign_target/alias_direct | >= 0.8 | 0.8700 | PASS |
| shred_target/alias_probe_top1 | <= 0.05 | 0.0100 | PASS |
| shred_target/alias_forced_choice | <= 0.6 | 0.4750 | PASS |
| shred_target/alias_top1_among_entities | <= 0.05 | 0.0150 | PASS |
| active/alias_probe_top1 | >= 0.25 | 0.8000 | PASS |
| probe_calibration_top1 | >= 0.2 | 0.7917 | PASS |
| dup_shred/copy_probe_top1 | >= 0.2 | 0.8100 | PASS |
| revoke_alias/alias_unknown | >= 0.9 | 1.0000 | PASS |
| revoke_alias/sibling_readable | >= 0.8 | 0.8800 | PASS |
| revoke_alias/target_readable | >= 0.8 | 1.0000 | PASS |
| delete_target/alias_unknown | >= 0.9 | 0.9100 | PASS |
| delete_target/alias_true_object | <= 0.05 | 0.0000 | PASS |

Groups passed: one_shred_deletes_every_path, attacks_through_every_alias, attack_validity, alias_lifecycle.

### How the templates were chosen

| template | link-free reading (E-000017-B) | kind |
|---|---|---|
| 0 | 0.7950 | trained |
| 1 | 0.9917 | trained |
| 2 | 0.7917 | trained |
| 3 | 1.0000 | trained |
| 4 | 1.0000 | trained |
| 5 | 0.9983 | trained |
| 6 | 0.7850 | trained |
| 7 | 0.9967 | trained |
| 8 | 0.5650 | held out |
| 9 | 0.9683 | held out |
| 10 | 1.0000 | held out |
| 11 | 0.4267 | held out |

**Interpretation (post hoc, record unchanged):** E-000020 ran its whole battery at template 0, so update, rollback, shred, revoke, delete and both attacks were statements about a phrasing the model reads at 0.5633. This re-runs the battery unchanged at three phrasings, the two strong ones chosen at run time from E-000017-B's record of the LINK-FREE adapter so the choice cannot favour the link arm; it picks trained template 3 and held-out template 10. Criteria groups passed out of six: 2 at template 0, 4 at template 3, 4 at template 10. The deletion claim gains the most, because its attack becomes valid: the probe recovers 0.42 of LIVE aliases at template 0 and 0.80 at template 10, so 0.01 after one SHRED of the shared object is a real result there and a weak one here. The sharing contrast is phrasing-independent -- an update to one duplicate never reaches the others (0.0000 everywhere) -- while the shared-object side rises from 0.5350 to 0.8850 and still misses its pre-registered 0.90 on the worst seed, recorded as a miss. reading_through_an_alias fails at all three because its alias_heldout_min criterion is the WEAKEST held-out template, where even the link-free adapter reads 0.42. Ledger 31.11.

## e000027_untied_output

_not run in this session_

## e000027_untied_input

_not run in this session_

## E-000028 — the channel SHRED does not close

Seeds [0, 1, 2, 3, 4], 100 targets, the recorded E-000010 checkpoints, no training.
The attacker knows a cell's subject and relation, finds its column from the routing of the
ordinary forward question, then sweeps every candidate object through a REVERSE query and
takes the candidate that steers the read onto that column.

### Object recovery through the key channel (worst seed)

| condition | object recovered top-1 | top-5 | mean rank | chance top-1 | chance rank |
|---|---|---|---|---|---|
| active | 1.0000 | 1.0000 | 0.0 | 0.0039 | 127.5 |
| shred | 1.0000 | 1.0000 | 0.0 | 0.0039 | 127.5 |
| revoke | 0.0200 | 0.0400 | 120.6 | 0.0039 | 127.5 |
| delete | 0.0200 | 0.0400 | 120.6 | 0.0039 | 127.5 |

`active` is the validity control: if the sweep cannot recover a live object, the shredded
number means nothing. `revoke` and `delete` remove the row from routing altogether.

### Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| column_located | >= 0.9 | 1.0000 | PASS |
| active/object_top1 | >= 0.2 | 1.0000 | PASS |

### What F4 would require of this channel

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| shred/object_top1 | <= 0.00390625 | 1.0000 | FAIL |

### All measures

| measure | mean over seeds | worst seed | pooled n | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| column_located | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/column_located | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/column_mass | 0.9980 | 0.9977 | - | - | - |
| active/object_top1 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/object_top5 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| active/object_mean_rank | 0.0000 | 0.0000 | - | - | - |
| active/margin_mean | 0.6195 | 0.5968 | - | - | - |
| shred/object_top1 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/object_top5 | 1.0000 | 1.0000 | 500 | 0.9926 | 1.0000 |
| shred/object_mean_rank | 0.0000 | 0.0000 | - | - | - |
| shred/margin_mean | 0.6195 | 0.5968 | - | - | - |
| revoke/object_top1 | 0.0040 | 0.0200 | 500 | 0.0005 | 0.0144 |
| revoke/object_top5 | 0.0220 | 0.0400 | 500 | 0.0110 | 0.0390 |
| revoke/object_mean_rank | 128.0200 | 120.6300 | - | - | - |
| revoke/margin_mean | 0.0022 | 0.0016 | - | - | - |
| delete/object_top1 | 0.0040 | 0.0200 | 500 | 0.0005 | 0.0144 |
| delete/object_top5 | 0.0220 | 0.0400 | 500 | 0.0110 | 0.0390 |
| delete/object_mean_rank | 128.0200 | 120.6300 | - | - | - |
| delete/margin_mean | 0.0022 | 0.0016 | - | - | - |

**Interpretation (post hoc, record unchanged):** The strongest deletion claim here -- F4 for SHRED with the verified gate -- was only ever measured on the value channel: the answer, the logits, the hidden state, the linear probe. shred() writes only the marker and leaves the row ACTIVE, and in encode_bank the routing keys are computed before the gate and are never gated, so a shredded cell's reverse key k_rev(LN(object + relation)) still names the object. An attacker given the cell's subject and relation locates its column from the routing of the ordinary forward question and sweeps candidate objects through a reverse query: over 500 targets on five seeds the shredded object comes back at 1.0000 against a chance of 0.0039, with numbers identical to the live cell's to four decimals -- the keys are the same tensors before and after. REVOKE and DELETE remove the row from routing and land on chance (0.0040, mean rank 128.0 against 127.5). The E-000010 and E-000019 records stand: every number in them is about the value channel and every one is still correct. What is withdrawn is the unqualified reading of F4 for SHRED. This is a synthetic-model defect: the GPT-2 adapter's key is k_proj(ln_key(subject + relation)) and carries no object, which two tests assert as a property. ModelConfig.gate_reverse_key repairs it and is unevaluated -- it needs its own training run. Ledger 31.10.

## E-000029 — what the marker gate actually certifies

11 recorded checkpoints, no training.

E-000021 reported the gate's false-accept rate as 8.49e-04 and called it the bound on the deletion
guarantee. Its unsigned class comes from `invalid_markers`, which rejects every draw within 0.7 of
the centre, while the store calls everything beyond 0.35 deleted. The band in between was
measured by nothing. These are the three distributions side by side.

### The gate's accept rate, by where the marker is

| marker distribution | accepted | of | rate | 95% CI lower | 95% CI upper |
|---|---|---|---|---|---|
| uniform | 5843 | 5500000 | 1.062e-03 | 1.035e-03 | 1.090e-03 |
| annulus | 2199996 | 2200000 | 1.000e+00 | 1.000e+00 | 1.000e+00 |
| rejection_sampled | 1881 | 2200000 | 8.550e-04 | 8.168e-04 | 8.945e-04 |

### Accept rate by distance from the centre

| distance from the centre | accept rate (mean over checkpoints) | mean gate score |
|---|---|---|
| 0.10 | 1.0000 | 0.9985 |
| 0.20 | 1.0000 | 0.9978 |
| 0.30 | 1.0000 | 0.9962 |
| 0.40 | 1.0000 | 0.9917 |
| 0.50 | 1.0000 | 0.9777 |
| 0.60 | 1.0000 | 0.9300 |
| 0.70 | 0.9999 | 0.7725 |
| 0.80 | 0.2191 | 0.4320 |
| 0.90 | 0.0000 | 0.1299 |
| 1.00 | 0.0000 | 0.0254 |
| 1.10 | 0.0000 | 0.0044 |
| 1.20 | 0.0000 | 0.0008 |
| 1.30 | 0.0000 | 0.0002 |
| 1.40 | 0.0000 | 0.0000 |
| 1.50 | 0.0000 | 0.0000 |
| 1.60 | 0.0000 | 0.0000 |
| 1.70 | 0.0000 | 0.0000 |
| 1.80 | 0.0000 | 0.0000 |
| 1.90 | 0.0000 | 0.0000 |
| 2.00 | 0.0000 | 0.0000 |

Operational radius (smallest shell with an accept rate of exactly zero), worst checkpoint: 0.90 against a declared 0.35.

### The centre is not a secret

A signed marker is `normalise(centre + N(0, 0.05^2))`, so the mean of 950 of them
estimates the centre to 0.0076 on the worst checkpoint, well inside
the 0.35 radius. Markers minted from that estimate pass `marker_valid` at 0.9999 and the learned gate at 1.0000. `make_centre` derives the centre from `10_000 + seed`,
and every checkpoint stores it verbatim, so shipping a model ships the ability to mint signatures.

### Pre-registered criteria

| criterion (worst checkpoint) | required | observed | result |
|---|---|---|---|
| annulus_accept_rate | <= 0.001 | 1.0000 | FAIL |
| uniform_accept_rate | <= 0.001 | 0.0013 | FAIL |

**Interpretation (post hoc, record unchanged):** E-000021 put the gate's false-accept rate at 8.49e-04 and the programme called that the bound on the deletion guarantee. The rate is right; the distribution is not the one the guarantee is about. marker_valid deletes everything beyond 0.35 of the centre, but invalid_markers -- E-000021's unsigned class -- rejects every draw within 0.7, and new_invalid_markers, which is what shred() writes, rejects the same band. The annulus between was measured by nothing. Over eleven recorded checkpoints, with markers placed on each shell by construction because rejection sampling cannot reach the near shells in sixteen dimensions: the gate accepts EVERYTHING out to 0.70 at 1.0000, 0.2191 at 0.80, and first reaches zero at 0.90. Its operational radius is 0.90 on every checkpoint against a declared 0.35, and the deleted annulus is accepted at 1.0000 (2,200,000 markers). The rejection-sampled column reproduces E-000021 at 8.550e-04, which is how one knows the re-measurement is faithful before looking at what it missed. The boundary sits at 0.8 because training shows signed markers at 0.194 and unsigned ones at 0.7 and nothing between, so the classifier learned the MARGIN rather than the predicate -- and the recorded false-accept rate is just the tail of the shred sampler crossing it. The centre is not a secret either: the mean of 950 signed markers estimates it to 0.0076, minted markers pass marker_valid at 0.9999 and the gate at 1.0000, make_centre derives it from 10_000 + seed, and every checkpoint stores it verbatim. The gate is an integrity check, not a signature. Ledger 31.12.

## E-000030 — a deletion certificate for the recorded checkpoints

Seeds [0, 1, 2], 3 targets, the recorded E-000010 checkpoints, no training.
For each lifecycle operation, every value the deleted payload could hold is swept and what the
model computes is compared. The payload domain has 256 values, so the sweep
is every case rather than a sample.

### What survives the deletion

| operation | certified for every query (interface) | certified on the swept queries | mediation premise | first quantity that moves | encodings swept |
|---|---|---|---|---|---|
| revoke | no | yes | consistent | encode_bank[v_f] | 2 |
| shred | no | no | consistent | encode_bank[v_f] | 2 |
| delete | yes (structural) | yes (structural) | n/a | the row is not in the bank | 0 |
| revoke (GPT-2, soft gate) | yes | - | - | - | 782 |
| shred (GPT-2, soft gate) | no | - | - | encode_bank[values] | 2  (residual 1.39e-02) |
| revoke (GPT-2, hard gate) | yes | - | - | - | 782 |
| shred (GPT-2, hard gate) | yes | - | - | - | 782 |

`interface` compares `encode_bank`'s output. The forward reads the bank only there, so an
invariant encoding means an invariant computation FOR EVERY POSSIBLE QUERY, not just the swept
ones. `outputs` compares the returned logits over an exhaustive single-hop query domain
`mediation` is the falsification check on the premise the interface column rests on: it looks
for an output that moves while the encoding does not, and voids the certificate if it finds one.

### Pre-registered criteria

| criterion (worst seed) | required | observed | result |
|---|---|---|---|
| shred/outputs_certified | >= 1.0 | 0.0000 | FAIL |
| revoke/outputs_certified | >= 1.0 | 1.0000 | PASS |
| revoke/mediation_consistent | >= 1.0 | 1.0000 | PASS |
| shred/mediation_consistent | >= 1.0 | 1.0000 | PASS |

**Interpretation (post hoc, record unchanged):** The first deletions here that are not 'no attack recovered it'. For each lifecycle operation it sweeps EVERY value the deleted payload could hold -- an entity id, so 256 values, every case rather than a sample -- and checks whether the model computes anything different. The interface level compares encode_bank, which both models read the store through exactly once (so/model.py:246, so/llm_adapter.py:323), so an invariant encoding means an invariant computation for EVERY POSSIBLE QUERY and not just a swept set, at one cheap encoding per value and without running the core. Synthetic: REVOKE is certified on the 838 swept questions and not at the interface (v_f = v_fwd(o) * g still carries the object, masked downstream); SHRED is certified at neither, which is E-000028 restated as a proof rather than an attack; DELETE is structural, the row being absent from the bank. Frozen GPT-2: REVOKE CERTIFIED under both gate modes, SHRED CERTIFIED under the HARD gate and not under the soft one, where a sigmoid never returns zero and 1.390e-02 of the payload survives in the value -- the first precise statement of what the hard gate buys. Two guards keep the instrument honest and both were needed: check_mediation looks for an output that moves while the encoding holds still, and the first adapter arm compared values_payload, an ungated diagnostic forward never reads, reporting a 3.49 residual through a tensor the model does not look at. Ledger 31.14.
