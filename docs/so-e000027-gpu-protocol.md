<!--
Written 2026-09-04 as the output of a design panel, then checked against the repository before being
committed. Every recorded number it cites was re-read from so/results/*.json and matched exactly:
E-000011 train_seconds 2707.1 / 2797.8 / 2930.4 (mean 2811.7); E-000013 active/injection_rms_share
0.8537 / 0.8666 / 0.8855 (mean 0.8686), generic/kl_to_base mean 2.2692, revoke/kl_to_base 0.0004,
revoke/heldout_kl_max 3.7046; E-000017-B generic/kl_to_base mean 3.2741, worst 3.6474; E-000018
generic 0.6035, gate 3.2681, both 0.6736. The experiment id was moved from E-000026 to E-000027,
because E-000024 (weights vs cells), E-000025 (template re-scoring) and E-000026 (the lifecycle
battery at a readable template) already exist in so/experiments/.

This is a PLAN. Nothing in it has been run. It states what would be measured, what it would cost and
what result would stop the spending -- it does not report any outcome.
-->

# E-000027 — Pre-registered GPU protocol: from E5 to E6

**Proposed path:** `/home/user/FinX-BB/docs/so-e000027-gpu-protocol.md`
**Status date:** 2026-09-04 · **Companion:** [roadmap](docs/so-roadmap-2026-09-02.md) · [ledger](docs/so-experiment-ledger.md) §4, §28 · [results](docs/so-results-2026-09-02.md)
**Experiment id:** E-000027. **E-000024, E-000025 and E-000026 are taken** — `/home/user/FinX-BB/so/experiments/e000024_weights_vs_cells.py` (653 lines, LoRA-vs-cells, unrun) and `/home/user/FinX-BB/so/experiments/e000025_template_rescoring.py` (unrun) already exist. Three of the four merged plans proposed "E-000024"; all three would have collided.

---

## 0. The argument in one page

The ceiling is not 124M parameters. It is four measured defects, and only one of them is a scale problem:

| Defect | Measured | Bar | Fixed by scale? |
|---|---|---|---|
| Injection where there is no key | `generic/kl_to_base` 3.2741, worst seed 3.6474 (`e000017b_templates8.json`); best arm 0.6035 / worst 0.8008 (`e000018_generic.json`) | 0.05 | **No** — it got *worse* with more templates (2.2692 at 2 templates in `e000013`, 3.2741 at 8) |
| Reading on unseen phrasings | `heldout/active_correct` 0.7400, worst 0.7288 | 0.90 | Plausibly |
| Single-token entities everywhere | `llm_adapter.py:75-76`, `:149-150`, `:266` | — | **No** — needs code |
| Deletion is template-local | `revoke/kl_to_base` 0.0004 on the trained template, `revoke/heldout_kl_max` 3.7046 / worst 4.4722 (`e000013`) | 0.10 | Unknown |

So the money is spent kill-first, and the first two kills cost **nothing** and **$3–6** respectively.

There is one more fact that reorders everything, and no merged plan used it. **The match gate — the only mechanism in the codebase that can say "no cell matches, inject nothing" — was arithmetically inert when E-000018 measured it, and the bug has since been fixed.** `e000018_gate` recorded `generic/kl_to_base` 3.2681 against a 3.2741 baseline: no effect at all. `docs/so-results-2026-09-02.md:71` withdraws that conclusion; `llm_adapter.py:236-243` carries the fix (the RMS normaliser is now taken from the **ungated** read `ref = self.o_proj(val)`, so a closed gate genuinely injects less). `so/experiments/e000022_two_channel_null.py` is written, wired, and **has never been run** — `so/results/` has no `e000022` record, `docs/so-results-2026-09-02.md:38` reads `not run`, and the ledger says the capacity question "is now E-000022's to answer".

Refusing to book a GPU before that question is answered for free on four CPU cores is the correct sequencing, and it is Stage 0 below.

---

## 1. Staged budget

Costs are derived, not asserted. **Anchor:** `so/results/e000011_gpt2_v2.json` `train_seconds` mean 2811.7 s (2707.1 / 2797.8 / 2930.4) for 3,000 steps at batch 32 on 3 threads = **0.937 s/step** for GPT-2 small. At batch 32 × ~14 tokens the step is ~2.2e11 FLOP (forward 1.1e11, activation-grad above block 8 ~0.7e11, LM head over all positions ~3.5e10), so the box sustains ~235 GFLOP/s — a sane AVX2 figure, which validates the model rather than the wish.

**Projection, Qwen2.5-7B** (d=3584, L=28, V=152064, read at block ~19), batch 64 × ~20 tokens: forward 1.95e13 + activation-grad over the 9 blocks above the read 1.06e13 + LM head fwd/bwd 4.2e12 + bank encode 2.6e10 ≈ **3.4e13 FLOP/step**. A100-80GB bf16 peak 312 TFLOP/s; at a deliberately conservative **20 % MFU** (short sequences, forward hooks, fp32 adapter) = 62 TFLOP/s → **0.55 s/step**, 6,000 steps = **0.9 GPU-h/seed** train, +0.6 GPU-h eval. H100 SXM ≈ 2–2.5× faster at ≈1.7× the price, i.e. roughly cost-neutral and it buys wall-clock.

Memory is not a constraint: 7.6B bf16 = 15.2 GB frozen; the adapter at d=3584 is v_proj 12.85M + 2×o_proj 25.7M + k_proj 0.92M + 2×q_proj 1.84M ≈ **41.3M trainable** (0.54 %), 0.5 GB of fp32 AdamW state. Three seeds fit concurrently on one 80 GB card. A multi-GPU box buys wall-clock, not capability.

### Stage 0 — ZERO GPU-HOURS, ZERO DOLLARS (prerequisite, ~3 CPU-hours)

Run `python -m so.experiments.e000022_two_channel_null --seeds 0 1 2` on the four cores already owned. Its sibling `e000018_both` recorded 3607.9 / 3623.5 / 3637.2 s per seed, so three seeds is about three hours. It re-uses `E18.train_arm` with `GENERIC_SHARE = 0.25` and pre-registers E-000018's bars unchanged (`e000022_two_channel_null.py:63-73`). One seed checkpoint exists (`e000022_seed0.pt`); the seed-2 NaN it hit is the one the `clamp_min` at `llm_adapter.py:242` fixes.

**It decides:** with the match gate able to act for the first time and the null column split into payload-absence and query-relevance channels (`llm_adapter.py:95-99`, `:214-219`), does `generic/kl_to_base` fall?

- **< 0.10** → the defect was a missing mechanism, not the architecture. Book Stage 1 with `two_channel_null=True, match_gate=True` as the default.
- **0.10 – 0.50** → proceed, but Stage 2's absolute inertness bar becomes the sole gate and Stage 3 is contingent.
- **> 0.50** → the layer perturbs text it has no key for regardless of mechanism. **Book Stage 1 only** (2 h, $6) to record the logit-lens result, publish the negative, and do not spend the rest.

Also free, also Stage 0: run the refactored adapter (Stage 1's code changes, `--model gpt2`) on CPU and reproduce `e000008_gpt2_adapter.json` exactly — direct 0.8890, full-vocab 0.8373, `bank_masked_direct_acc` 0.0000, `provenance_direct` 0.842. **A port that moves those numbers is a bug, and finding it here costs nothing.** No plan budgeted this; it is the strongest available control on the ~250 lines that are the real constraint.

### Stage 1 — TWO GPU-HOURS (~$3–6). Try to kill the port before buying a result.

No training. Everything here is a measurement or a diagnostic.

1. **(25 min) Pre-flight, every candidate model.** `' unknown'` tokenises to exactly one id (`e000008_gpt2_adapter.py:120-122` already asserts this — it must now run per tokenizer and fail at construction); count of vocabulary entries matching `Ġ[A-Z][a-z]{3,}` minus `STOP` (`e000008_gpt2_adapter.py:57-67`) — **record the number before anything else**, because `world.py:77-79` caps the bank at `n_entities × n_relations` and every bank-size claim in this document is bounded by it; mean row-wise cosine between input and output embedding rows; per-block last-token residual RMS on 256 WikiText-103 sentences, plus the ratio (full mean-square RMS)/(RMS with the top 1 % of dimensions removed).
2. **(45 min) THE LOGIT LENS — the decisive test, before any training hour.** Add `α · w_out[t]` to the residual stream at block `round(0.67·L)` for random tokens `t`, `α ∈ {0.1, 0.3, 0.87, 1.0} × rms_h`, and record the rank of `t` in the output distribution and the perplexity cost. `0.87` is not arbitrary: `e000013_prior_conflict.json` records `active/injection_rms_share` = 0.8686, i.e. the working GPT-2 read overwrites 87 % of the residual RMS at the read position — with 2 blocks downstream. Qwen2.5-7B has 9. This tests roadmap Stage 5's own requirement ("verify with a logit lens that the injected direction survives to the output"), which no recorded experiment has ever done.
3. **(25 min) The null-column diagnostic, at fixed weights and stated as such.** Distribution of (null score − best cell score) on generic text as C grows, and the routing mass on the null column per read. `e000013` records 0.8805 at the first read and 0.2742 after the second at C = 1,000 — consistent with, but not proof of, a log C squeeze. The main read at `llm_adapter.py:185-190` has **no** size compensation while the dereference passthrough at `:205` adds `float(np.log(n_cells))`, with the comment at `:113-116` recording that a plain +5 bias left only 16 % of the mass on the passthrough at 800 cells. Add `null_log_c: bool = False` to `AdapterConfig` mirroring `:205`.
4. **(25 min) gpt2-xl smoke.** `e000008` unchanged, one seed, 800 steps, read layers (32, 40) of 48 — tied embeddings, so it needs none of the port.

**GATE.** If no injection magnitude brings the target token to rank 1 on the untied 7B without wrecking perplexity, **and** the read-placement sweep finds no block that does, the mechanism is not physically available in a deep untied model. Publish that, stop, and spend the remaining budget on the tied ladder instead. **This is a real architectural finding and it is worth $6.**

Caveat that must be recorded, not hidden: item 3 evaluates fixed weights outside their training bank size. It is a diagnostic that motivates Stage 2's *train-at-C* arm; it is **not** evidence that the architecture cannot be trained to be inert at large C, and no kill criterion fires on it alone.

### Stage 2 — TWENTY GPU-HOURS (~$30–50). The decisive single-model run.

| Item | GPU-h |
|---|---|
| Pre-flight and read-placement sweep on Qwen2.5-7B (rule fixed in writing first, whole sweep reported) | 1.5 |
| **E-000027 main arm**, Qwen2.5-7B, multi-token subjects and objects, 3 seeds, 6,000 steps, 8 trained / 4 held-out templates, `match_gate + two_channel_null + null_log_c` | 6.0 |
| **The tied/untied contrast** — Qwen2.5-1.5B (see §2), 3 seeds, identical script | 2.0 |
| **The artificial-untie control** on gpt2-small: give the adapter a `w_out` that is a fixed random rotation of `wte`, hold corpus, tokenizer, architecture, entity ids and read layers constant, move only the tie | 0.5 |
| Bank-size ladder **trained at C**, C ∈ {1e3, 4e3, 1.6e4}, 1 seed each, 2,000 steps | 2.0 |
| WikiText-103 perplexity, adapter on vs off, every configuration | 1.0 |
| Reserve (`so/results/` literally contains `e000020_symlink_gpt2_attempt1.json`) | 7.0 |
| **Total** | **20.0** |

**What Stage 2 decides:** kills K1–K4 below can all fire here. **What it cannot claim:** E6. It has no scaling trend, no real-knowledge arm, no LoRA control, no full attack battery, and the roadmap's Stage-5 kill on general benchmarks is unmeasured. Stage 2 is recorded as **E5 + real pretrained substrate**, and the record says so in one sentence.

### Stage 3 — TWO HUNDRED GPU-HOURS (~$300–450). The E6 run.

| Item | GPU-h | Why |
|---|---|---|
| Everything in Stage 2, as the anchor point | 20 | |
| **Real-knowledge arm** (§3): PopQA/Wikidata triples split KNOWN / UNKNOWN / BORDERLINE / FABRICATED before any training; write-then-erase, override-then-erase, suppress-then-verify; 5 seeds | 35 | The only arm that reaches the erasure motivation |
| Qwen2.5 ladder 0.5B → 1.5B → 7B → 14B, 3 seeds, core metrics | 30 | Roadmap Stage 4 / kill 9, on one tokenizer |
| Full attack battery: MLP probes at every block above the read, near-name, few-shot elicitation, context completion, dependency closure, prefix/suffix leak | 20 | Roadmap Stage 3; `so/attacks.py:6-7` records context completion as "not applicable" — that stops being true here |
| **LoRA control** — extend `/home/user/FinX-BB/so/experiments/e000024_weights_vs_cells.py`, which already implements `LoRAConv1D`, gradient-ascent and relabel unlearning, a relearning attack, bystander accuracy and frozen-core perplexity. It needs porting from GPT-2 `Conv1D` to `nn.Linear`, not writing. | 8 | Ledger §28 "Adapter disablement alone" |
| Qwen2.5-7B-Instruct, 3 seeds, refusal reported as paired excess over the same model's own baseline refusal | 10 | Roadmap Stage 5 |
| Benchmarks: MMLU 5-shot, ARC-easy, HellaSwag, adapter on vs off, ~20 configurations | 8 | Roadmap Stage 5 kill (>1 point) |
| Bank-scale interference, **trained at C**, up to the tokenizer-imposed ceiling (§4), exact top-k | 8 | Roadmap Stage 4 |
| Symlink cells at 7B — scoped as **diagnosis, not claim** (`e000020`: alias 0.5067 vs direct 0.5700; `e000023` curriculum collapsed alias reading to 0.0000 on every seed) | 10 | |
| Ablations at 1.5B: no marker gate, no null key, no routing loss, per-subject key table (must fail on fresh names), random base, logit-level RAG-equivalent | 12 | Roadmap Stage 3 |
| 5 seeds instead of 3 on the 7B headline | 6 | Ledger §28 "One successful seed" |
| **Subtotal** | **167** | |
| Reserve, 20 % | 33 | |
| **Total** | **200** | |

**Money.** 200 A100-80GB-hours at $1.50/h = $300; at $2.06/h = $412. H100 SXM does the same work in ~90 h at ~$2.50/h ≈ $225. Add ~$30 for weight storage and egress. **Budget $450, hard cap $600.** Prices must be re-checked at booking; the arithmetic above is what to re-check them against. The expensive resource is engineer time on §4's code changes, and that cost is paid before any card is rented.

---

## 2. Model and data, with licences

### Substrate — the Qwen2.5 family, chosen for a reason none of the merged plans found

Every open model at 7B and above **unties input and output embeddings**, and the mechanism this programme rests on assumes they are tied. `llm_adapter.py:18-21` states it: *"Because the value is built from the model's own (tied) token embedding of the object, adding it into the residual stream raises that token's logit through the unchanged LM head."* It is implemented as `nn.init.eye_(self.v_proj.weight)` at `:82` ("start as write the object's own embedding direction"), `null_value = wte[unknown_token_id]` at `:120-121`, the `' unknown'` direction at `:168`, and the read-out `full[:, self.candidate_ids]` at `:266`. In GPT-2, `wte` **is** the LM head. In an untied model it is not, and an identity-initialised `v_proj` writes into input-embedding space with near-zero projection onto the object's output row.

Verified at the config level today:

| Model | d | Layers | `tie_word_embeddings` | Vocab | Licence |
|---|---|---|---|---|---|
| gpt2 / medium / large / xl | 768–1600 | 12–48 | **true** | 50257 | MIT |
| Qwen2.5-0.5B | 896 | 24 | **true** | 151936 | apache-2.0 |
| **Qwen2.5-1.5B** | 1536 | **28** | **true** | 151936 | apache-2.0 |
| **Qwen2.5-7B** | 3584 | **28** | **false** | 152064 | apache-2.0 |
| Qwen2.5-14B | 5120 | 48 | false | 152064 | apache-2.0 |
| Pythia-6.9B-deduped | 4096 | 32 | false | 50432 | apache-2.0 |

**Qwen2.5-1.5B and Qwen2.5-7B have the same depth (28 blocks), the same tokenizer, the same architecture and the same pretraining family, and differ in width and in whether the embeddings are tied.** That is the cleanest available isolation of the untying hazard, and it costs about 2 GPU-hours. It is strictly better than the Pythia-410M-vs-GPT-2-medium pair one merged plan proposed, which confounds untying with corpus (Pile vs WebText), tokenizer (so the entity *set* differs, not just its size), rotary vs learned positions, and NeoX's parallel attention/MLP residual. The tied 0.5B and 1.5B rungs also give a **tied ladder inside the same family**, so "bigger" and "untied" are separable twice over.

Qwen2 uses byte-level BPE with the `Ġ` word-boundary marker, so `select_entities` (`e000008_gpt2_adapter.py:57-67`, regex `Ġ[A-Z][a-z]{3,}`) works unmodified. A SentencePiece model (Llama-2, Mistral) uses `▁` and would return **zero** entities. This must be verified by the Stage-1 tokenizer scan, not assumed.

**Not chosen as primary, with reasons.** Llama-3.1-8B: community licence with an acceptable-use rider, gated download, and roadmap §5 requires a package an independent group can re-run without a click-through. It belongs in Stage 6 as an external-reproduction family. OLMo-2-7B (Apache-2.0 weights, ODC-BY searchable corpus) is the right primary **if and only if** the real-knowledge arm's per-fact prior claim becomes the headline; it is named as the Stage-3 alternate for exactly that reason. Verify every per-size `LICENSE` file at the pinned commit and **record its SHA-256 in the result JSON** — do not take a licence from this document.

### Data

**Synthetic worlds, unchanged and load-bearing.** `World.sample` re-sampled every step inside the training loop (`e000011_gpt2_v2.py:71-72`, `e000018_no_key_no_injection.py:81-83`), banks from `bank_from_world` (`so/data.py:87`) with `p_revoked=0.20, p_shred=0.10, p_stale=0.05` and 20 % deliberately unanswerable queries. This is the anti-memorisation foundation (E-000002: with the layer present, masked-layer accuracy 0 % and REVOKE leak 0 % in every seed) and **nothing in this protocol weakens it**.

**Entities, now multi-token.** 4,096 subject strings and 4,096 object strings, each 2–6 tokens under the target tokenizer, length distribution recorded in the JSON. A 256-entry single-token subset is retained as an explicit **control arm** so the new numbers sit beside the recorded ones rather than replacing them with something incomparable.

**Templates.** `TEMPLATES12` (`e000017_paraphrase_gap.py:54-72`), 8 trained / 4 held out — the budget roadmap Stage 2 prescribes and that E-000017-B supplied. Plus a hand-written free-paraphrase set, **written and committed before any training run** so it cannot be tuned against (roadmap line 60; it has never been built).

**Generic text.** `TRAIN_GENERIC` (8 shapes, `e000018_no_key_no_injection.py:47-50`) for training, `E17.GENERIC` (5 different shapes, `e000017_paraphrase_gap.py:75-76`) for evaluation — disjoint by construction, so memorising a sentence shape does not pass. Extended with 2,000 held-out WikiText-103 sentences containing no bank entity and 2,000 containing a bank entity in an off-relation context. The second set is the hard case: `e000013` records `generic/prompt2_kl_to_base` 3.7541 and `prompt4` 3.7413 against `prompt0` 0.6039, and the two worst prompts are the ones ending on the subject token — half a key.

**Real facts (Stage 3 only).** Wikidata triples over ~16 relations via PopQA (`akariasai/PopQA`: subj/obj strings, QIDs, `s_aliases`/`o_aliases`, `s_pop`/`o_pop` pageviews). **The HF card states no licence — verify at the pinned commit; the fallback that removes the question is to rebuild the same relations from a Wikidata dump (CC0).** Filtering before any prior is measured: single unambiguous object, object ≤ 6 tokens, at least one alias, subject not a bare number or date.

**The prior split, fixed in writing before the adapter exists.** Eight templates per relation, 4 used to measure the prior and 4 to score deletion, **disjoint**, so "known" is never defined on the phrasings used to score erasure. Measured on the base model with the adapter absent — `llm_adapter.py:260-261` sets `_ctx = None` and the hook at `:177-178` returns `None`, so the base forward is bit-identical.
- **KNOWN** = exact match on ≥ 6 of 8 AND restricted 50-way rank 1 on ≥ 6 of 8.
- **UNKNOWN** = exact match 0 of 8 AND restricted rank worse than the set median on ≥ 6 of 8.
- **BORDERLINE** = everything else, **excluded from every primary criterion and reported with its size**. Without this band the split is a knob the experimenter turns after seeing results.
- **FABRICATED** = 2,000 pseudo-names verified at corpus count 0 — the only band on which F3/F4 can honestly be claimed.

**And the thing not to get wrong: training never sees the true triples.** Training re-samples counterfactual object assignments over the real entity vocabulary; the true configuration appears only at evaluation. If training saw it, the adapter could memorise real facts and every F-level collapses to F1.

### What is trained, what is frozen

Frozen: every base weight, enforced at `llm_adapter.py:70-71`. Trained: exactly `adapter_parameters()` (`llm_adapter.py:133-134`) — `rel_emb`, `ln_key`, `k_proj`, `v_proj`, per-read-layer `q_ln`/`q_proj`/`o_proj`, `marker_gate`, `null_key`, `inject_gain`, `scale`, plus config-dependent `match_tau`/`match_temp`, `query_relevance`, `v_link`, `q_deref`/`deref_*`, plus one new `(Lmax, d)` slot embedding. Under `fallback="prior"`, `null_value` is fixed at zero and non-trainable (`llm_adapter.py:122-124`) so no constant shortcut can be learned. **The only arm in this protocol that unfreezes anything is the LoRA control, and it is a control, not a claim.**

At d=3584 the adapter has ~41M trainable parameters against ~1.8M at d=768 — a 23× growth in capacity to memorise. That is precisely why the copy bound (C1) is the first criterion measured.

---

## 3. The three arms, and the scoping sentence that must travel with them

- **Arm A — WRITE-THEN-ERASE on facts the model does not know** (UNKNOWN and FABRICATED bands). The programme's actual claim. F3/F4 reachable: nothing was in the weights, the layer held it, the layer gave it up.
- **Arm B — OVERRIDE-THEN-ERASE on facts the model does know** (the `e000013` regime, scaled). **F1 by construction** and the record must say so: `status_gated` multiplies the gate, so a revoked cell's value is arithmetically zero. The *learned* residue is that routing does not spill onto neighbours — and that residue currently **fails**: `revoke/heldout_kl_max` 3.7046, worst seed 4.4722, against 0.10.
- **Arm C — SUPPRESS-THEN-VERIFY on facts the model does know.** A tombstone cell must drive refusal on held-out phrasings without a prompt filter. **F0/F3 at best — the frozen weights still contain the fact**, and `e000013` says so in its own record. Arm C additionally runs the `e000007_biomarker.py` battery: if suppressed and absent are not separable by any internal signal, Arm C is ledger §28 "output refusal" and is reported as such.

**Any write-up that blurs Arm B or Arm C into "we deleted a pretrained fact" is false.** Ledger §28 names both false positives.

Mechanism note: Arms B and C need **per-cell fallback**. `AdapterConfig.fallback` is a single global string (`llm_adapter.py:59-60`), branched on for the whole bank at `:162` and again for the injection normalisation at `:233-243`; `prior` and `unknown` are mutually exclusive settings with two different injection paths. Making fallback per-cell is a real code change (§4 item 8), not a config edit. Until it exists, the three arms are three separately trained adapters and the record must not claim that one resident layer supports both erasure regimes.

---

## 4. Exact code changes, by file

Nothing below is a new mechanism. `so/mvcc.py` (the entire lifecycle, op log, deterministic replay, `state_hash`), `so/world.py`, `so/reference.py`, `so/train.py`'s `routing_loss`, `so/interventions.py`, `so/evaluation.py` and `so/report.py` need **no edits at all**. That is the strongest evidence that the single-token limit was never architectural: `obj` is an integer id everywhere (`so/mvcc.py`, `so/world.py`), and only its *rendering* to tokens is single-token.

**C1 — Model plumbing (~15 lines).** `llm_adapter.py:73` `lm.config.n_embd` → `config.hidden_size`; `:127` `lm.transformer.h[l]` → `lm.model.layers` (Qwen) / `lm.gpt_neox.layers` (Pythia); `:120` and `:131` `lm.transformer.wte.weight`. One `_resolve(lm) -> (blocks, w_in, w_out, d)` dispatching on `lm.config.model_type`.

**C2 — The untied split (the load-bearing two lines).** Split the single `wte` property (`:130-131`) into `w_in`, used for **keys** (`:149`, `:157`), and `w_out`, used for **values** and the `' unknown'` direction (`:150`, `:168`, `:120-121`). Keep `nn.init.eye_(v_proj.weight)` — it is now identity in the correct basis. Record `‖v_proj − I‖_F / ‖I‖_F` per model after training, so how far the value path had to move is a **measurement**, not an assumption. Verify by logit lens that the direction survives Qwen2's RMSNorm and the 9 blocks above the read.

**C3 — Multi-token values and keys (the largest change).** Objects: `:150` and `:266`. **Subjects too** — `:149` builds the key from the same one-id-per-entity buffer (`:75`), and PopQA subjects and the FABRICATED floor arm are overwhelmingly multi-token, so without a multi-token *key* there is no addressing and no floor arm. Keys become a length-masked **mean** of `w_in` over the subject's tokens (pre-registered; last-token pooling is a recorded ablation, not a free choice made after seeing numbers). Values become `(C, Lmax, d)`: `v_proj` per slot plus a learned slot embedding, pad slots carrying the terminator. `:266` `cand = full[:, candidate_ids]` is **kept as an optional return** — `cand` is consumed at ~55 sites across eight experiment files, including `e000011_gpt2_v2.py:291-292` where the in-context comparators are computed. Deleting it silently destroys the anti-RAG criterion.

**C4 — The training objective, which is the part that must learn.** `e000011_gpt2_v2.py:83` `F.cross_entropy(cand, target)` becomes token-level teacher-forced cross-entropy over the full vocabulary at the answer positions, masked past the terminator, **averaged per sequence then over the batch** so a 5-token name does not outweigh a 2-token one. `loss_route` (`:84`) and the class-balanced `loss_gate` (`:85-88`) are copied unchanged.

**C5 — The write positions, and the RMS trap.** `:245` `delta[ar, ctx["last_idx"]] = read` becomes a scatter into `ctx["write_idx"]` of shape `(B, Lmax)` with a validity mask. **`rms_h` at `:232` must be computed per write position from that position's own hidden state, not from `hl`** — the residual norm at answer position 3 is not the norm at the prompt's last token, and using `hl` for all of them mis-scales every slot but the first into a plausible-looking degraded number. Unit test required. Routing is computed **once** at the prompt's last token and cached across slots; the routing tensor shape `(B, R, C+1)` is preserved so `so/train.py:67-82` is untouched. The `n_deref` path (`:193-211`) needs an explicit broadcast — `qd = q_deref(deref_ln(val))` with `val` now `(B, Lmax, d)` would otherwise produce `(B, Lmax, C+1)` and break that shape.

**C6 — Generation. Nothing in this repository has ever decoded more than one token.** `grep` over `so/` for `generate` / `max_new_tokens` / `past_key_values` returns nothing. `exact_match`, `prefix_leak@k` and `suffix_completion_excess` all need greedy multi-token decoding, and under HF cached decoding the hooked tensor holds only the new token so `h[ar, ctx["prompt_last"]]` does not exist. Either write the incremental-decoding path (recommended) or run `Lmax` full re-forwards per query — **the latter is ~6× the evaluation cost the budget in §1 is built on, and the budget assumes the former.**

**C7 — Device, dtype and the bf16 routing hazard.** The repository is CPU-only end to end: `Bank.tensors(device="cpu")` (`so/data.py:68`) is called as `bank.tensors()` at `e000008:131`, `e000011`, `e000017:113`, `e000018:99`; `predict` calls `.numpy()` on model outputs (`e000008:138-144`, ~23 such sites); `LinearProbe` (`so/attacks.py:41-65`) is CPU-only; `ledger.environment()` writes `info["device"] = "cpu"` unconditionally (`so/ledger.py:65`); `requirements.txt` points at the CPU wheel. This is a change touching every experiment file and is comparable in size to all the mechanism work combined. **Budget it.** Separately: the frozen 7B must be bf16, the adapter fp32. `torch.softmax` over C+1 columns at `:190` in bf16 (8 mantissa bits) destroys the routing distribution, and `so/train.py:79` then takes `log(p.clamp_min(1e-9))` of it. The failure is not an exception — it is a quietly worse routing loss that reads as *"the bigger model routes less well"*. Cast `hl = h[ar, idx].float()` at the top of the hook, everything downstream fp32, `h + delta.to(h.dtype)` at `:248`, and **assert** the routing tensor is fp32.

**C8 — Per-cell fallback** (§3): `fallback` moves from a global string to a per-row selector, with the two injection paths at `:162-171` and `:233-243` selected per cell.

**C9 — `null_log_c`** (§1 Stage 1 item 3): a flag mirroring `:205` on the main read at `:185-190`, kept as a flag so it is A/B-able. **Both an evaluation-time A/B on existing checkpoints and a trained arm** — the softmax temperature `scale` (`:125`) and `null_key` (`:118`) are trained parameters that already absorbed a calibration at C ≈ 700–1,000, so an evaluation-only A/B on those checkpoints moves reading as well as inertness and cannot support a kill on its own.

**C10 — Checkpoint paths do not name the model (1 line × 3 sites) — a present bug this protocol would trip immediately.** `e000008_gpt2_adapter.py:207` `f"e000008_gpt2{CKPT_SUFFIX}_seed{seed}.pt"` and `e000011_gpt2_v2.py:104` carry no model identifier, and the loads at `:210` / `:107` use `load_state_dict(..., strict=False)`. Two models with the same `d` and read-layer indices produce **identical state-dict keys**, not merely compatible shapes. `so/results/checkpoints/` already holds `e000008_gpt2_seed{0,1,2}.pt`, `e000011_gpt2_seed*.pt`, `e000017_t8_c0_seed*.pt` and `e000018_{gate,generic,both}_seed*.pt`. **Fix this before a single ladder run starts, or the first Qwen run silently loads a GPT-2 adapter and the failure reads as a scale result.**

**C11 — Relation count and the world sampler.** `AdapterConfig.n_relations` (`:43`) is not a one-line change: the experiments hardcode 4 at `e000008_gpt2_adapter.py:174`, `e000011_gpt2_v2.py:72` and `e000018_no_key_no_injection.py:82`.

**C12 — `so/ledger.py`.** `environment()` (`:59-69`) must record GPU name, driver, torch build, dtype and whether the top-k bank path was used; otherwise a GPU record is indistinguishable from a CPU one. `check_criteria` (`:207-221`), `clopper_pearson` (`:110-167`), `ci_rows` (`:170`), `worst` (`:195`) and `aggregate` (`:230`) are reused **as they stand** — `check_criteria` already evaluates the **worst seed** (min for `>=`, max for `<=`) and already records a missing metric as `observed: None, pass: False`, which is why §5's metric names must exist in the arms that are actually run.

**C13 — Attacks.** Add `forced_choice_seq` (summed teacher-forced log-probability, **token-length-matched distractors** — an unmatched pool turns forced choice into a length test), `string_rank` over 64 sampled distractors + truth (chance top-1 1/65, unnormalised: length-normalisation lets short distractors win), `prefix_leak(k)`, `suffix_completion_leak(j)`, normalised edit distance, and `MLPProbe` alongside `LinearProbe`. **Every leak metric is a paired excess over the no-bank arm on identical prompts** — the tokenizer completes names on its own ("Ber" → "lin"), and an absolute rate scores the frozen model's own behaviour as a leak. The convention already exists at `e000013`'s `counterfactual_top1_excess` and `forced_choice_excess`.

**C14 — The experiment file.** `so/experiments/e000026_real_substrate.py`, forked from `e000013_prior_conflict.py`, whose `evaluate()` is already the right shape and whose `criteria_groups()` is the template for §6.

---

## 5. Measurement list

Every metric per seed (3 at Stage 2, 5 on the Stage-3 headline), per template class (8 trained / 4 held out / free paraphrase), per prior band where applicable, with exact Clopper-Pearson intervals on pooled counts (`ledger.py:110`) and **worst-seed reporting** as every existing record does.

**Sample sizes, stated so the criteria are decidable.** `EVAL = dict(n_cells=1000, n_targets=200, n_broken=200, n_generic=200)` (`e000017_paraphrase_gap.py:74`). At n = 200 the 95 % Clopper-Pearson half-width at a 2 % rate is about ±2 points, so a "≤ 0.02 on the worst seed" criterion cannot be distinguished from 0.04. **Every criterion below at the 0.02 level therefore runs at n_targets = 1,000, pooled across seeds to n = 3,000.** This costs GPU minutes and is budgeted.

1. **Copy bound and prior.** `prior_direct_acc` (recorded 0.0037), `bank_masked_direct_acc` (0.0000), `bank_masked_unknown_rate` (1.0000). Re-measured at every d, because adapter capacity grew 23×.
2. **Reading.** `first_token_acc`, **`exact_match`** (full string, the headline, no precedent in `so/results/`), `full_vocab_top1` (recorded 0.8523 / worst 0.8370 — the 2,049-way candidate softmax at `:266` flatters the mechanism; the full-vocabulary number is what "the unchanged LM head emits the object" means), `restricted_top1` over 64 length-matched distractors. Trained, held-out, free paraphrase.
3. **Refusal and deletion.** `revoke_train_min`, `revoke_heldout_min`, `shred_train_min`, `shred_heldout_min` (worst template, not the mean — the weakest held-out template is where roadmap kill 5 fired); `refusal_given_active_correct` and `deleted_object_given_active_correct` (the E-000017-A decomposition); `heldout/revoked_deleted_object`.
4. **No key, no injection.** `generic/kl_to_base` computed **in fp32 from the logits** (`e000017_paraphrase_gap.py:307-316`), pooled mean **and** `kl_to_base_worst_prompt` (recorded 4.2206 — a pooled mean hides the two prompts that end on the subject token), on generic prose, on held-out natural corpus, and on off-relation sentences containing a bank entity. Plus `broken1_unknown` (0.7183 / worst 0.63) and `routing_mass_on_null` per read (0.8805 first read, 0.2742 after the second).
5. **Multi-token leak — new, and the reason this is not the old experiment on a bigger model.** `prefix_leak@k` for k = 1..Lmax; `suffix_completion_excess` (force-feed the first j tokens of a SHREDded object and measure P(token j+1) against the no-bank arm); normalised edit distance against the distance to a random other object; per-slot token probes. **Each requires an ACTIVE-cell positive control** — with the cell live, the forced-prefix attack must recover the object at a high rate, or "no leak" and "the attack does not work" are indistinguishable. This is the same discipline as `e000020_symlink_gpt2.py:293-295`'s `attack_validity` group and it is what the merged plans omitted for exactly these two new metrics.
6. **Reconstruction, all as paired excess.** Linear probe, MLP probes at every block above the read, forced choice, object rank, few-shot elicitation, context completion, near-name, dependency closure. Validity gate first: with the cell ACTIVE every one must succeed (`e000013` records `active/probe_top1` 0.8733, `active/forced_choice_excess` 0.3467 — the gate passes today, which is why its deletion numbers mean anything).
7. **Composition vs RAG.** `hop2` against **both** comparators, which exist only in `e000011_gpt2_v2.py:275-297` and are recorded at 0.0056 (first fact only, cannot answer by construction) and 0.4167 (both facts), with the adapter at 0.8333. **`e000011` must therefore be in the run list of every tier that asserts this criterion** — one merged plan defined its RAG kill on a metric no tier it scheduled emits. Stated as a **margin over the first-fact-only baseline**, so a generally stronger base model cannot pass by being better at 2-hop.
8. **Provenance and causal isolation.** `provenance_direct` (0.8307 / worst 0.8130); `alt_route/broken_route_changes` (0.9933) and `alt_route/other_route_survives` (0.7800); the intervention table (`localisation_hop2` 1.0000, `swap_hop2` 0.9797, `replace_hop2` 0.9733).
9. **Retention.** WikiText-103 validation perplexity, adapter on vs off, 2,048-token windows; MMLU 5-shot, ARC-easy, HellaSwag; filler-fact accuracy (`active/filler_direct` 0.9112) and filler KL before and after every lifecycle op; exact undo after rollback/restore. **`grep` shows "wikitext" appears nowhere in `so/` and `requirements.txt` carries neither `datasets` nor `lm_eval` — this harness does not exist and is budgeted as new work, not as free.**
10. **Gate error rates.** `e000021` re-run on every new checkpoint: 2.2M markers per class, false accepts 8.486e-04 [8.106e-04, 8.880e-04], false rejects 0 [0, 1.68e-06]. This is a property of the 16-dimensional marker alone and **must not move with model size**. The bank-level rate is different and must be reported separately: 1 − (1 − p)^C is ~57 % at 1,000 cells and effectively 1 at 100,000, so the criterion is stated **per targeted cell** and the bank-level expectation reported alongside. `e000021`'s own `interpretation_limit` — that an adversary who can choose the marker is not modelled — travels with it.
11. **Scale trend.** Every headline metric across Qwen2.5 0.5B → 1.5B → 7B → 14B, plus the tied/untied contrast at matched depth and the artificial-untie control on GPT-2.
12. **Diagnostics that turn assumptions into measurements.** `‖v_proj − I‖_F / ‖I‖_F`; `injection_rms_share` per read layer (0.8686 recorded); `inject_gain`; `match_tau` / `match_temp`; `query_relevance` firing rate on prose vs questions; embed-in/embed-out cosine; top-k exactness residual if approximate retrieval is ever used.

---

## 6. Pre-registered criteria and the level rule

**The level rule — which no merged plan supplied, and without which "a pass" is undefined.** Criteria are grouped as in `e000013_prior_conflict.py:458-508` and `e000022_two_channel_null.py:63-73`, and the F-level is a function over groups:

- **All of G1–G5 met** on the worst of ≥ 3 seeds, on a 7B frozen pretrained model, with multi-token entities, from **one checkpoint per configuration** → **E6 / F3**.
- **G1–G5 plus G6 (reconstruction at chance with a passing validity gate) on the FABRICATED band** → **E6 / F4**, for knowledge that entered through cells only.
- **G1–G4 met, G5 failed** → **E6 substrate, F1** — a reading-and-routing result, not a deletion result.
- **Any of G1 failed** → the claim is capped at **F1** regardless of everything else.
- G7 (scale trend) and G8 (retention) are **gating, not level-conferring**: failing either caps the claim at E5 with the reason recorded.
- Arm B is **F1 by construction**; Arm C is **F0/F3 at best**. Neither can raise the level, and both must carry the scoping sentence from §3.

| Group | Metric | Op | Threshold | Justification |
|---|---|---|---|---|
| **G1 copy bound** | `bank_masked_direct_acc` − prior | ≤ | 0.05 | Roadmap kill 2. Recorded 0.0000 at 124M; adapter capacity grew 23× at d=3584 and only re-sampled worlds prevent memorisation |
| **G1** | per-subject-key-table ablation on fresh names | — | must FAIL | Roadmap Stage 3. If a lookup table works, addressing is not from the model's representation |
| **G2 reading** | `train/active_correct` | ≥ | 0.95 | `e000017_paraphrase_gap.py:328`, unchanged. Recorded 0.9198 / worst 0.9119 — already a fail, so a pass is itself a scale result |
| **G2** | `heldout/active_correct`, first token | ≥ | 0.90 | Roadmap Stage 2, unrelaxed. Recorded 0.7400 / worst 0.7288 with the prescribed 8-template budget |
| **G2** | `heldout/exact_match`, full string | ≥ | 0.80 | **No prior number exists.** This is the metric that retires the single-token limit; set below first-token accuracy because spelling a whole name is strictly harder |
| **G2** | `full_vocab_top1`, trained templates | ≥ | 0.90 | Recorded 0.8523 / worst 0.8370. Reported beside the candidate-softmax number, never instead of it |
| **G3 inertness** | `generic/kl_to_base`, worst seed | ≤ | 0.05 | The roadmap's own bar. Recorded 3.2741 (worst 3.6474); best arm 0.6035 (worst 0.8008). **Reported from the same checkpoint as G2** |
| **G3** | `generic/kl_to_base_worst_prompt` | ≤ | 0.10 | Recorded 4.2206. Two of the five generic prompts end on the subject token — a half key — and a pooled mean hides that family |
| **G3** | `broken1_unknown` | ≥ | 0.90 | `e000017:333`, unchanged. Recorded 0.7183 / worst 0.63 |
| **G3** | `wikitext103_ppl_ratio_on_over_off` | ≤ | 1.02 | Roadmap Stage 2 item 5. Never measured anywhere in this repository |
| **G4 refusal** | `revoke_heldout_min` | ≥ | 0.85 | `e000017:329`, unchanged. Recorded 0.8983 / worst 0.8650; the roadmap's own kill-5 bar is 0.95 and is reported against separately |
| **G4** | `shred_heldout_min` | ≥ | 0.85 | The learned-deletion arm: payload present and routable, only the marker gate refuses |
| **G4** | `revoke_train_min` | ≥ | 0.95 | Guards against buying generalisation with a regression |
| **G5 deletion** | `heldout/revoked_deleted_object` | ≤ | 0.02 | `e000017:330-331`. Recorded **0.0000 in every seed and every held-out template** — currently met, must not be traded away |
| **G5** | `heldout/deleted_object_given_active_correct` | ≤ | 0.02 | The E-000017-A decomposition. Recorded 0.0000 |
| **G5** | `shred/prefix_leak_at_1_excess` | ≤ | 0.02 | **New.** Catches deletion that suppresses the name but emits its first token |
| **G5** | `shred/suffix_completion_excess` | ≤ | 0.02 | **New, and the strongest attack multi-token creates.** An adversary who knows half a name is realistic |
| **G5** | `revoke/kl_to_base_heldout_max` (Arm B) | ≤ | 0.10 | Recorded 0.0004 on the trained template and 3.7046 / worst 4.4722 on held-out. "After deletion the model returns to itself" is currently a claim about two sentence shapes |
| **G6 reconstruction** | `shred/probe_excess`, `forced_choice_excess`, `string_rank_top1_excess`, FABRICATED band | ≤ | 0.05 | Roadmap kills 3 and 7, paired over the base on the same rows with the same distractor draws. Recorded 0.0000 / 0.0000 at `e000013` |
| **G6 validity** | `active/probe_top1` ≥ 0.25, `probe_calibration_top1` ≥ 0.20, ACTIVE prefix/suffix recovery ≥ 0.50 | ≥ | — | Verbatim from `e000020_symlink_gpt2.py:293-295`. Without a floor a merely weak probe prints "3 % vs 4 %" and reads as a deletion result |
| **G7 scale** | worst per-metric delta between adjacent Qwen2.5 rungs | ≥ | −0.02 | Roadmap kill 9. Stated as **any adjacent-rung decrement**, not "monotonic over four points" — a metric going 0.85 → 0.60 → 0.62 → 0.55 must fire it |
| **G8 anti-RAG** | `hop2` − `comparator/in_context_first_fact_only_hop2_acc` | ≥ | 0.20 | Roadmap kill 4. Margin 0.8277 at 124M (0.8333 vs 0.0056). At 7B the both-facts baseline rises sharply and that is expected and reported; the first-fact-only baseline **cannot** rise |
| **G8** | `provenance_direct` | ≥ | 0.85 | Recorded 0.8307 / worst 0.8130. Provenance is what separates this from a soft associative memory |
| **G8** | worst benchmark drop, MMLU / ARC-e / HellaSwag | ≤ | 1.0 pt | Roadmap Stage 5's own kill, which appears in no merged plan's criteria list |

**Two rules that close the tempting escape hatches.** (i) `e000018` bought `generic/kl_to_base` 3.2741 → 0.6035 by trading `heldout/active_correct` 0.7400 → 0.6888 and `revoke_heldout_min` 0.8983 → 0.8800. **Every configuration reports its full metric set from one checkpoint; the inertness bar and the reading bar must be met simultaneously or neither is met.** (ii) Read placement and the entity/template sets are fixed **before** any test number is seen, and the whole placement sweep is reported including the blocks not chosen.

---

## 7. Kill criteria

Architecture-falsifying unless marked otherwise. Each fires on the worst of ≥ 3 seeds, in the best arm, with every implemented mitigation enabled.

**K1 — The mechanism is not physically available in a deep untied model.** No injection magnitude at any candidate read block brings the target token to rank 1 without perplexity damage above 5 %, and the degradation tracks the **number of blocks above the read** rather than model size. Then "the unchanged LM head emits the object" is a property of shallow tied models, no adapter training helps, and the frozen stack is what erases it. *Fires at Stage 1 for $6.*

**K2 — Injection without a key survives scale.** `generic/kl_to_base` stays above 0.50 nats with the match gate (now able to act), the two-channel null and generic-text training all on, at 7B. Then the layer changes the frozen model everywhere, the deletion effect is smaller than the noise the layer itself adds, and this is a perturbation story rather than a knowledge interface. *Most likely of the eight to fire: the number got worse with more templates, and the fix that helps costs reading.*

**K3 — The copy bound is an artifact of re-sampling.** Masked-bank accuracy exceeds the prior by more than 5 points at any `bank_refresh_every ∈ {1, 8, 64}`. The recorded 0.0000 was measured with `World.sample` called every step (`e000011_gpt2_v2.py:71-72`) — the one regime no deployment has. If it survives only under re-sampling, "the neural core never stores facts" is a property of the training schedule and **every F3/F4 claim in the ledger collapses to F1**. *Roadmap kills 1 and 2; four lines of change; ~1.5 GPU-h.*

**K4 — Addressing is bound to surface form.** `heldout/active_correct` stays below 0.80 at 7B with 8 trained templates and multi-token entities. Then per-fact routing — the property that separates this from a LoRA and that carries provenance, causal intervention and the copy bound — is not real, and every claim caps at F1.

**K5 — The value cannot be spelled.** Routing mass on the ground-truth cell above 0.90 while `direct/exact_match` is below 0.20. The cell is found and its value cannot be emitted; every recorded number is then an artefact of the 257-way read-out at `llm_adapter.py:266`. *Testable on GPT-2 in Stage 1 for a few dollars — but note that GPT-2 is tied, so this test cannot detect K1 and the two must not be conflated.*

**K6 — Deletion is first-token suppression.** `shred/suffix_completion_excess` above 5 points while slot-0 refusal is above 95 %, with the ACTIVE positive control passing. The F4 claim (representational removal, `e000019`: forced choice exactly 375/750 with CI [0.4636, 0.5364], probe 4/750 against chance 1/256) collapses to F0/F3 for multi-token objects.

**K7 — It is RAG.** `hop2` − `in_context_first_fact_only` below 0.20 at 7B at any read placement. That prompt cannot answer by construction — the second fact is not in the text — so if a routed 2-hop cannot beat it while costing perplexity, the vector interface buys nothing over putting the fact in the prompt. *The GPT-2 margin was measured against a model too weak to do in-context composition; this is the first run where the criterion can genuinely fire.*

**K8 — No operating point exists.** At the injection gain needed for counterfactual override ≥ 0.90 on held-out templates, benchmark drop exceeds 1 point or WikiText perplexity rises more than 2 %, at every read placement. *Roadmap kill 6. A layer that must be switched off between queries is a retrieval pipeline with a residual-stream API.*

**SCOPE REDUCTION, NOT FALSIFICATION — record it as such.** Arm A passes every bar and Arm C fails at every gain and placement. That does not falsify the architecture; it says the architecture governs knowledge that **entered through cells** and cannot reach knowledge already in the weights. The programme then puts that sentence at the top of every future claim, and the right-to-erasure motivation applies only to facts written after deployment.

---

## 8. What this plan will NOT establish

1. **Not F5, and not from the inside.** F5 requires an external red team (roadmap Stage 6). Nothing here can claim it.
2. **Not E7.** External reproduction is on the reproducer's side. This protocol produces the *package* Stage 6 needs — tagged commit, seeds, world generators, template sets, per-fact prior table, published leak threshold — but not the reproduction.
3. **Not deletion of pretrained knowledge.** Arms B and C do not remove anything from the frozen weights, by construction, and the record says so in every arm's `not_claimed` field.
4. **Not security.** `e000021`'s own `interpretation_limit` applies unchanged: an adversary who can *choose* the marker is not modelled, and a gate that separates two fixed distributions says nothing about one that must resist a search for a passing vector.
5. **Not approximate retrieval, and therefore not deployment-scale banks.** Every bank claim uses exact top-k. The tokenizer scan caps `n_entities`, and `world.py:77-79` caps the bank at `n_entities × n_relations`; the honest reachable C is whatever that scan yields, stated in the record, and it is **not** 10^6 without multi-token entities in the *key* path plus more relations. A 10^6-cell claim is not made.
6. **Not free-text queries.** Eight trained templates plus four held out plus a committed free-paraphrase set is a much better budget than two, and it is still not open language.
7. **Not causal-tracing-grade mechanistic understanding.** The logit lens and the placement sweep say the direction survives; they do not say what the frozen model does with it.
8. **Not a claim about instruction-tuned refusal.** An instruct model's own "I don't know" behaviour inflates the metric the whole deletion story is measured in. The instruct arm reports refusal as a paired excess over that same model's baseline, and **the base model, not the instruct model, carries the headline claim.**
9. **Not cross-family generality.** One family (Qwen2.5) plus one tied control family (GPT-2). Pythia and Llama-3.1 are Stage-6 reproduction targets, not evidence here.
10. **Not "the model is no longer a toy world".** Stage 2 replaces a 124M substrate with a 7B one and keeps synthetic re-sampled worlds. Only the Stage-3 real-knowledge arm engages a substrate's own knowledge, and it does so on ~16 Wikidata relations, not on open-domain text.

---

## 9. Risks, and what is done about them

- **The port bug that reads as a scale failure.** A silently wrong `w_out` (C2) or a mis-scaled per-position `rms_h` (C5) produces exactly the signature of "the mechanism does not scale". Mitigations: the free CPU reproduction of `e000008_gpt2_adapter.json` before any card is rented; the GPT-2 ladder on the unchanged code path; the artificial-untie control that moves only the tie; `‖v_proj − I‖_F` recorded per model; the logit lens before the first training step; unit tests on per-position RMS and on fp32 routing.
- **Checkpoint collision.** C10. Present bug, would fire on the first Qwen run, would look like a scale result.
- **The inertness bar probably gets worse, not better.** A larger residual stream with high-magnitude outlier channels is a harder place to be surgical. The trimmed-RMS variant (mean of `hl**2` excluding the top 1 % of dimensions) is the only untested lever and is one line; Stage 1's RMS-ratio measurement says whether it is mandatory.
- **Passing a kill test for the wrong reason.** An adapter regularised until it is inert also never reads. Every inertness criterion is paired with a reading floor in the **same arm and the same seed**, and the one-checkpoint rule in §6 forbids reporting the two sides from different runs.
- **The prior split is a knob.** Band thresholds in writing before the adapter exists, BORDERLINE excluded and its size reported, prior templates disjoint from deletion-scoring templates, and the split required to correlate with corpus counts and pageviews — **if it does not, the run reports that finding instead of a deletion result.**
- **The CPU side starves the GPU, and the fix endangers the control.** `so/data.py:87-117` builds three C-sized Python dicts every step and `:30-39` rejection-samples invalid markers in a Python loop — invisible at 0.937 s/step, 30–50 % of wall clock at 0.25 s/step. A prefetch thread fixes throughput but must keep the per-step RNG seeded per step, or the re-sampled-world control is quietly compromised and the copy bound passes for the wrong reason. Unit test: step-for-step identical banks with and without prefetch.
- **The budget rests on FLOP scaling from a CPU measurement.** Short prompts and small batches are exactly where MFU disappoints. Stage 1 measures it, and the gate is explicit: **under 10 % measured MFU, the primary model drops from Qwen2.5-7B to Qwen2.5-1.5B and the record says so. The budget does not rise.**

---

## 10. What was dropped from each merged plan, and why

**From the "real knowledge" plan (kept as Stage 3's centrepiece).**
*Dropped:* OLMo-2-7B as primary — demoted to alternate, because the Qwen2.5 family supplies a tied→untied transition on one tokenizer that no other family offers, and that hazard is more urgent than corpus transparency for Stages 1–2. *Dropped:* its `e000024_real_knowledge.py` filename — the id is taken. *Dropped:* its claim that the tie is "small, mechanical" while multi-token is "the largest change" — inverted; C2 decides whether the port works at all. *Dropped:* its assertion that per-cell tombstones are "no new mechanism" — `fallback` is a global string with two incompatible injection paths, and making it per-cell is C8. *Dropped:* its 3 GPU-h for a from-scratch LoRA control — `e000024_weights_vs_cells.py` already implements it. *Corrected:* `generic/kl_to_base` 3.2741 is `e000017b`, not `e000013` (which is 2.2692). *Kept, and it is the best thing in it:* the free CPU pre-gate, the BORDERLINE band, the paired-excess convention, the FABRICATED floor arm, "training never sees the true triples", and the calibration gate that drops the model rather than raising the budget.

**From the "faithful / scale only the substrate" plan.**
*Dropped:* the premise that only the substrate should change — the judges are right that this produces a bigger E5, and the roadmap's own Stage 5 names multi-token entities as constitutive. Multi-token moves into Stage 2. *Dropped:* Pythia-410M vs GPT-2-medium as the untying isolator — confounded by corpus, tokenizer, positions and parallel residual; replaced by Qwen2.5-1.5B vs 7B at matched depth plus the artificial-untie control on GPT-2. *Dropped:* the 100k-cell arm as specified and the `encode_bank` refactor justified by it — at `n_entities × n_relations` = 8,192 that arm cannot run, and factoring through the entity vocabulary is a *pessimisation* at the bank sizes actually reached. *Dropped:* its 20 GPU-h Tier-2 arithmetic (each script trains its own adapter; the real figure is ~35 h) and its LM-head FLOP term (missing a factor of T). *Corrected:* `e000011` is required in every tier asserting the anti-RAG criterion and the full-vocab metric. *Kept, and these are excellent:* C2's untied diagnosis traced to `:82`, `:120-121`, `:150`, `:168`; the $3 logit lens as a hard gate; the checkpoint-collision bug; the bf16 routing hazard traced to `train.py:79`; the per-cell vs bank-level gate distinction.

**From the "kill first" plan (whose ordering this document adopts).**
*Dropped:* its entire Tier A as written. It names `e000017b_seed{0,1,2}.pt`, which do not exist (the files are `e000017_t8_c0_seed{0,1,2}.pt`); it sweeps C past the 1,024-pair ceiling on frozen checkpoints whose `entity_token_ids` and `candidate_ids` buffers would size-mismatch on load; and it raises `n_relations` 4 → 16 into untrained rows of `rel_emb` with no templates for them. *Dropped:* its `null_log_c` evaluation-only A/B as a *decision gate* — `scale` and `null_key` are trained parameters calibrated at C ≈ 1,000, so the test is confounded; it survives as a Stage-1 **diagnostic** with a trained arm at Stage 2. *Dropped:* its claim that "a softmax always injects, therefore stop the programme" — false as a description of the code: `match_gate` (`:220-230`) and `two_channel_null` (`:214-219`) both sit outside the softmax, the match gate was **inert by a since-fixed arithmetic bug**, and `e000022` is the repository's own untested mitigation. Ending the programme on a defect whose implemented fix was abandoned mid-run is the one-sentence objection a hostile reviewer would use. *Dropped:* its 10^6-cell routing criteria (unreachable). *Kept, and it is the single best code observation across all four plans:* the `+log(C)` asymmetry between `:205` and `:185-190`, with the author's own comment at `:113-116` as corroboration; the persistent-bank copy-bound arm, now K3; and pairing every inertness threshold with a reading floor in the same seed.

**From the "multi-token" plan.**
*Dropped:* its E-000024 id (taken) and its assertion that `:266` "is DELETED" — `cand` is consumed at ~55 sites and its removal breaks the very comparator its own anti-RAG criterion needs. *Dropped:* the claim that only objects need lifting — subjects are single-token too, at `:149`, and the FABRICATED floor arm is multi-token by construction. *Dropped:* its "the deref path needs only a broadcast" — `:201` with `val` of shape `(B, Lmax, d)` breaks the routing tensor shape. *Dropped:* its power-free `≤ 0.02` criteria at n = 200; §5 raises n_targets to 1,000. *Corrected:* `e000011` direct is 0.8657 (min 0.85), not "0.88 mean / 0.82 worst" — that is `active/direct_acc` — and 0.90 was `e000011`'s own *lenient* bar, which it failed. *Added, because the plan's design would otherwise be near-vacuous:* the exact-match criterion is paired with an **ACTIVE-cell positive control** for the prefix and suffix attacks, so "no leak" and "the attack does not work" are distinguishable. *Kept, and it is right:* the bounded-edit argument (`obj` is an integer id everywhere; `so/mvcc.py`, `world.py`, `reference.py`, `data.py`, `ledger.py` need no edits), length-matched distractors, paired excess over the no-bank arm, the correction that roadmap line 37 is stale (the comparators exist at `e000011_gpt2_v2.py:275-297` and are recorded), and the observation that WikiText retention has never been measured.

**Added by this merge, present in none of the four:** the Qwen2.5 tied→untied ladder at matched depth; the artificial-untie control on GPT-2; the free CPU reproduction of `e000008` before renting anything; the discovery that the match gate was inert by a since-fixed bug and that `e000022` is therefore the free prerequisite; the explicit **level rule** over criterion groups; the sample-size/power statement; the generation-path gap (nothing in the repository decodes more than one token); the ACTIVE positive control for the two new leak attacks; and the correct experiment id, E-000027.

---

**Sources for the model facts in §2:** [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json) · [Qwen2.5-1.5B config](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json) · [Qwen2.5-0.5B config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json) · [Qwen2.5-14B config](https://huggingface.co/Qwen/Qwen2.5-14B/raw/main/config.json) · [Pythia-6.9B-deduped config](https://huggingface.co/EleutherAI/pythia-6.9b-deduped/raw/main/config.json) · [Qwen2.5-7B licence metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-7B) · [Qwen2.5-1.5B licence metadata](https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B)