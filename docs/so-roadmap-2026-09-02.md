# SO — Roadmap from the current evidence to the breakthrough definition

**Status date:** 2026-09-02  
**Companion documents:** [architecture](so-modular-neural-os.md) · [experiment ledger](so-experiment-ledger.md) · [session results](so-results-2026-09-02.md) · code in [`so/`](../so/README.md)  
**Purpose:** say exactly what is established, what is not, which experiment closes which gap, at which evidence level, and what would kill the idea.

---

## 1. What is established today

The mechanism that the code implements and that the session's experiments test is:

> The neural core never stores facts. It is trained on worlds that are re-sampled every step, so the only stable signal is *how to read* addressable cells. A lifecycle operation on a cell (UPDATE, REVOKE, RESTORE, ROLLBACK, SHRED) is therefore a lifecycle operation on the model's behaviour, without touching any weight.

The numbers are in the [session results](so-results-2026-09-02.md); this section only states what kind of evidence each result is.

| Property (ledger §3) | What is shown | Kind | Level |
|---|---|---|---|
| Selectivity | The targeted cell's answer disappears for direct, paraphrased, multi-hop and reverse access; controls unchanged | REVOKE: routing removal by mask (**by construction**, F1). SHRED: payload stays routable, the *learned* marker gate refuses it (**learned**, F3; F4 with the verified gate of E-000010) | E4 synthetic |
| Retention | Controls, unrelated cells, bypass paths unchanged after mutation; targets restored exactly by RESTORE / ROLLBACK | learned (the model must not draw the answer from another cell) | E4 |
| Generalisation | Deletion holds across all surface paraphrases and access directions | partly by construction (one canonical cell per fact) — the learned part is that no other route produces the answer; only 2 surface forms per relation in the synthetic system, 2 natural-language templates in E-000008 | E4 / E5, **weakest property** — now measured in the real model: E-000011 REVOKE 72.7% on the trained template, 17.3% on the weakest held-out one |
| Causal isolation | disable / swap / replace / restore on a cell change the answer exactly as predicted; the routed cell is the ground-truth cell | localisation is a *trained* objective (routing loss); in this architecture the cell read is the only channel, so this is a consistency check rather than a discovery | E4 |
| Reconstruction resistance | With the unsupervised gate, SHRED leaves a residual that a probe and forced choice exploit (E-000004); with the class-balanced verified gate (E-000010) probe, forced choice and logit rank are at chance and the value contribution is zero while the payload is still present and routed to; dependency closure is required for derivable facts | learned (SHRED column); after REVOKE the activation rows are the mask | E4, F4 within the synthetic system (E-000010) |
| Shared knowledge objects | E-000015: link cells give several access keys ONE object. One UPDATE or one SHRED reaches every path (100% / 100% unknown) against 0% / copies fully readable in the duplication arm; after that one SHRED the object is unrecoverable through every alias (probe 0.7%, forced choice 0.503) and fully recoverable through the copies (probe 87.3%, forced choice 1.000); provenance names alias and target | synthetic system only; shredding the alias itself reaches 93% on the worst seed; chains resolve once chains are in training (E-000016: two slots 100%, one slot refuses 100%) | E4 / **F3** |
| Scalability | E-000014: addressing, lifecycle and verified deletion hold unchanged at 10,000 cells / 2,560 entities (scaling curve flat at 100% from 1k to 10k). E-000008: a frozen pretrained GPT-2 reads the layer from natural-language prompts (direct 88.9%, 83.7% over the full vocabulary, 2-hop 75%, update/rollback 95–96%, copy bound 0%); answering ' unknown' after REVOKE / SHRED only 56% / 38% at 2,000 adapter steps | first step only; 124M parameters, CPU, single-token entities; deletion behaviour is the part that did not transfer for free | E5 as substrate; reading and update supported, deletion behaviour not at the pre-registered thresholds |

What is **not** established: anything about knowledge already encoded in pretrained weights (the design avoids that regime by construction rather than solving it), free-text paraphrases, multi-token entities, models above 124M parameters, GPU-scale runs, external reproduction.

---

## 2. Gap analysis

| Breakthrough requirement | Gap | Closed by |
|---|---|---|
| Selectivity, retention at real-model scale | only 124M / CPU | Stages 2, 4, 5 |
| Generalisation over language | 2 templates per relation; single-token entities | Stage 2 |
| "Knowledge participates in computation" (not RAG) | shown as multi-hop composition inside the synthetic core; in GPT-2 only as a 2-hop metric without an in-context baseline | Stage 2 (2-hop through a cell must beat the single-fact in-context baseline) |
| Causal isolation in a model with its own knowledge channels | the synthetic core has no other channel | Stage 3 |
| Reconstruction resistance beyond linear probes | linear probe, symbolic queries | Stage 3 (MLP probes, context completion, few-shot elicitation, near-name attacks) |
| F5 | never claimable from the inside | Stage 6 (external red team) |
| E7 | — | Stage 6 |

---

## 3. Staged roadmap

Compute column: what runs on 4 CPU cores now versus what needs a GPU.

### Stage 0 — Close the ledger gap on the synthetic core (this session)

Run E-000001-B … E-000007 on the corrected objective, record every number with pre-registered criteria, per-seed worst case and exact binomial intervals, and generate the results document from the JSON records only. Load-bearing control: E-000002 (masked-layer accuracy at chance; leak after REVOKE ≈ 0 for re-sampled training, > 0 for fixed-world training). **Kill:** masked-layer accuracy above chance or REVOKE leak above 2% in any seed — the "no copy" foundation is false and every F3/F4 claim collapses to F1. *Compute: CPU, ~2 hours.*

### Stage 1 — E-000008: frozen pretrained core, single-token prior-free facts (this session — recorded)

Frozen GPT-2 small; symlink adapter reading cells at blocks 8 and 10; keys from the model's own token embedding of the subject plus a learned relation embedding; values from the object's token embedding through the unchanged LM head. Controls: prior baseline (no layer), masked-bank copy bound (must equal the prior), lifecycle against the mechanical reference, attacks after REVOKE / SHRED. **Kill:** masked-bank accuracy above the prior by more than 5 points (the adapter copies); direct accuracy below 50% of what the mechanism reaches in the synthetic core; leak after REVOKE above 5 points. *Compute: CPU, ~20 min per seed.* **Recorded outcome:** none of the kill criteria triggered (copy bound 0%, direct 88.9%, leak after REVOKE 0% by probe / forced choice); the open item is the ' unknown' behaviour after deletion (56% / 38%), which becomes the first task of stage 2: longer adapter training, a larger share of unanswerable queries, and the fallback-to-prior formulation instead of a trained ' unknown' token.

### Stage 2 — Language: multi-token entities, held-out paraphrases, 2-hop versus in-context, prior conflict

1. Multi-token objects: the cell value becomes up to 6 token directions; report first-token accuracy and full-string exact match separately.
2. Paraphrases: 12 templates per relation, 8 train / 4 held-out, plus hand-written free paraphrases as a final held-out set. Deletion must hold on every held-out template.
3. Multi-hop: "The capital of the country where X was born is" with two routed reads and an intermediate entity that never appears in the text; the in-context baseline with only the first fact in the prompt cannot answer, so 2-hop accuracy above that baseline shows the value participates in downstream computation. Revoking the first-hop cell must break the answer; an alternative route must survive.
4. Prior conflict: real entities with counterfactual objects (Berlin → Switzerland). Measure override during ACTIVE and fallback to the pretrained distribution after REVOKE (KL to the base model below 0.05 nats). State explicitly that the pretrained fact is not deleted.
5. Placement: sweep the read block; choose by 2-hop support with the smallest injection norm; check WikiText perplexity with the adapter on versus off (< 2% change).

**Kill:** held-out-template deletion below 95% while train-template deletion ≈ 100% (surface overfitting, the ledger's "one exact prompt forgotten"); 2-hop not above the in-context baseline at any placement (the read is RAG with a vector interface); override of moderate priors impossible without perplexity damage above 5%. *Compute: CPU borderline (30–60-token prompts); a single 16 GB GPU turns days into hours.*

### Stage 3 — Attacks, causal interventions, biomarker and ablations inside GPT-2

Replicate E-000004 … E-000007 in the real model, where the base has knowledge channels of its own: context completion and few-shot elicitation (now applicable), MLP probes at every block above the read, near-name attacks (a misspelt revoked subject must not leak while misspelt active subjects still route), dependency closure; disable / swap / replace / restore with prediction tables; the suppression control fine-tunes the adapter only; ablations without marker gate, without null key, without routing loss, with keys from a per-subject table instead of the model's representation (must fail on fresh names), with a randomly initialised base, and with a logit-level adapter as the RAG-equivalent baseline. **Kill:** any attack recovering a revoked prior-free object above prior + 5 points (then F3 only); a probe above chance at any block above the read after REVOKE; suppressed and revoked not separable by any internal signal. *Compute: inference-heavy, CPU-feasible in hours; ablation retraining ~2 GPU-hours or 1–2 CPU-days.*

### Stage 4 — C55 (proposed definition): scaling curve on one tokenizer family

Pythia 70M → 1.4B with the same code, placement chosen by causal tracing on training facts only (rule fixed before seeing test numbers), 5 seeds per size, general-capability retention (perplexity, small benchmarks) with the adapter on versus off, interference with 100k cells (exact top-k, no approximate retrieval yet; 10k cells is recorded for the synthetic core in E-000014). **Kill:** any metric trending down with size; addressing collapsing above 10k cells; 2-hop not improving with size. *Compute: one 24 GB GPU, ~100–200 GPU-hours.*

### Stage 5 — C56 (proposed definition): 7B base and instruction-tuned model

Llama-3.1-8B or Qwen2.5-7B, base and instruct. What changes and must be measured: much stronger priors (the prior-conflict regime becomes the main override test with an injection-norm / perplexity trade-off curve), 32 layers (verify with a logit lens that the injected direction survives to the output), native "I don't know" behaviour and jailbreak-style paraphrases in the attack set, multi-token entities as the norm, 100k-cell banks. Control: a LoRA fine-tuned on the same facts and "deleted" by disabling the LoRA — the ledger's "adapter disablement alone" false positive — to show what cells add: per-fact routing, provenance, causal interventions, a copy bound. **Kill:** override impossible without perplexity damage above 5%; leak after REVOKE above prior + 5 points under any attack; 2-hop-through-cell below the in-context baseline; general benchmarks dropping more than 1 point. *Compute: 1× 80 GB GPU, ~10–20 GPU-hours training, ~50 GPU-hours attacks.*

### Stage 6 — C57 (proposed definition): external reproduction and red team

Publish code, seeds, world generators, template sets, pre-registered criteria and two challenge sets: 1,000 revoked prior-free facts (goal: recover any object above prior + 5 points by any input or white-box probe) and 1,000 counterfactual facts (goal: show a residual deviation from the base model after REVOKE). An independent group re-runs Stage 5 on a different 7B family. **Result levels:** E7 if reproduced; F5 only if the red team fails, and then only for knowledge that entered through cells. *Compute: on the reproducer's side.*

---

## 4. Kill criteria (consolidated)

1. Masked-layer accuracy above chance, or REVOKE leak above 2%, in the re-sampled synthetic core (Stage 0).
2. Masked-bank accuracy above the prior by more than 5 points in GPT-2 (Stage 1): the adapter copies.
3. Leak after REVOKE above 5 points for prior-free facts (Stages 1–3).
4. 2-hop through a cell not above the single-fact in-context baseline (Stage 2): the mechanism is RAG with a vector interface.
5. Held-out-template deletion below 95% with train-template deletion ≈ 100% (Stage 2): surface overfitting.
6. Override of moderate priors only with perplexity damage above 5% (Stages 2, 5).
7. Any reconstruction attack above prior + 5 points, or a probe above chance above the read block (Stage 3).
8. Suppressed and revoked not separable by any internal signal (Stage 3): the biomarker programme stops.
9. Any core metric trending down with model size, or addressing collapsing above 10k cells (Stage 4).
10. External recovery of revoked facts or failure to reproduce (Stage 6): claims capped at E6 / F4.
11. Process: if recorded Stage 0 numbers contradict section 20 of the architecture document, the document is corrected before any E-000008 result is reported.

---

## 5. What an external reproduction (E7) requires

- Fixed package: code at a tagged commit, seeds, world generators, template sets (train / held-out / free), pre-registered pass thresholds, per-fact prior table.
- Two challenge sets with a published leak threshold (prior + 5 points) and the exact attack rules.
- A second model family on the reproducer's side; their numbers reported unedited next to ours.
- Everything above is CPU-reproducible up to Stage 1 with the commands in `so/README.md`.
