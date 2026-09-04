# What is new here, and what is not

*2026-09-04. This document exists to stop the programme claiming things that are already published.*

## The verdict, in my own words

**The architecture is re-invention.** An external addressable knowledge store read by a frozen core,
with per-entry lifecycle operations and a forgetting primitive, is SERAC, GRACE, Larimar, SILO, LMLM
and MUNKEY. All published. Several at larger scale. Several with explicit forgetting. The copy bound
is LMLM's masked-value training and SILO's corpus isolation; provenance-by-routing is definitional for
memory-augmented architectures and goes back to Memory Networks with supporting-fact supervision. None
of that is ours, and the word "provably" in the copy-bound claim is not earned: absence of a fact is
not demonstrable from weights, only from the training algorithm, and the algorithmic argument belongs
to SILO and LMLM.

**What is ours is the audit, and it is two negative results about our own system.**

1. **A gate on values is not a deletion primitive when another term in the computation is a function
   of the same payload.** SHRED passed a calibrated linear probe, forced choice, logit rank and top-1
   across five seeds and 750 pooled trials, every one at chance. An attack written afterwards
   recovered the deleted object at 1.0000 with numbers identical to the live cell to four decimals,
   through the ungated reverse key `k_rev(LN(object + relation))` that no attack in the battery read.
   This is not "soft-deleted data still on the medium": the payload *was* gated, the value channel *is*
   at chance, and the recovery runs through a derived index the deletion primitive never touched. The
   same question can be asked of GRACE's codebook keys, Larimar's memory addressing, SERAC's scope
   classifier, MUNKEY's instance keys, and any vector store with a soft delete.

2. **A learned gate certifies the margin between the classes it was shown, not the predicate it was
   written to implement** — and here that is measured as a swept geometry rather than asserted. The
   store declares a radius of 0.35; the gate's operational radius is 0.90 on all eleven checkpoints,
   and the annulus the store calls deleted is accepted at 2,199,996 of 2,200,000. The published
   false-accept rate of 8.49e-04 is reproduced at 8.550e-04 on the distribution it was measured on,
   and is not the false-accept rate of the thing being claimed. That learned detectors fail
   adversarially is old (Carlini and Wagner; learned index structures needing an exact backup filter);
   demonstrating the specification-versus-boundary gap on a deletion mechanism, with both rates side
   by side, is what I could not find.

So the honest framing is not "we built an architecture that deletes." It is: **here is a
completeness check that the published memory-augmented systems have not run, and here is what it finds
when you run it on a system built specifically to get deletion right.** E-000030 is the constructive
half of that — independence proven over the entire payload domain instead of an attack that failed.

## Provenance and what I did not verify

The statement below is the output of a 41-agent literature workflow. Its claims about **this
repository** I checked myself against `so/results/*.json` before committing, and they match. Its
claims about the **external literature** — paper titles, authors, venues, arXiv identifiers, dates and
reported numbers — I did **not** independently verify, and several cited works postdate my training
data. Treat every external citation as a lead to check, not as a checked fact. Where it says
"verified", that is the workflow's word, not mine.

---

## NOVELTY STATEMENT — SO (Modular Neural Operating System)

**Verdict in one line.** As a mechanism, this is re-invention: an external addressable knowledge store read by a frozen core, with per-entry lifecycle operations, is SERAC / GRACE / Larimar / SILO / LMLM, all published, all at larger scale, several with explicit forgetting primitives. As an audit, it contains two findings that no work I could verify reports — and both are *negative*, both were produced by this repository's own last two experiments, and neither appears in the six claims as they were handed to me. The claim list is a version of this programme that its own ledger has already overtaken.

---

### 1. What is genuinely new, and why the nearest prior art does not cover it

**(a) A gate on values is not a deletion primitive when another term in the computation is a function of the same payload. (E-000028, `so/results/e000028_key_channel.json`, ledger §31.10.)**

SHRED is the programme's only F4 result. E-000028 gives an attacker the same thing every other attack in the battery gets — a cell's subject and relation — locates its routing column, then sweeps candidate objects through a *reverse* query. Five seeds, 500 pooled targets, no training:

| condition | object top-1 | mean rank | winning margin |
|---|---|---|---|
| active (validity control) | 1.0000 | 0.0 | 0.6195 |
| **shred** | **1.0000** | **0.0** | **0.6195** |
| revoke / delete | 0.0040 | 128.02 | 0.0022 |
| chance | 0.0039 | 127.5 | — |

The shredded row is not leaky, it is *unchanged to four decimals including the margin*, because `encode_bank` computes `k_r = k_rev(LN(o + r))` from the object **before** the gate and never gates it. The record's own criteria block carries `"key_channel_leak": {"claim_supported": false}`.

Nothing I verified covers this. Ghost Vectors (Chakraborttii, García Alvarado, Abdulofizova & Dwivedi, arXiv:2606.18497, 16 Jun 2026 — verified) is the nearest: soft-deleted HNSW embeddings stay on disk and Vec2Text recovers 25.5% of exact names, 99% face identity. But that is *undeleted plaintext on a medium*. E-000028 is different in kind: the payload **was** gated, every value-channel attack **is** at chance, and the recovery runs through a *derived index* that the deletion primitive never touched. The generalisable statement — enumerate every quantity derived from a payload and gate all of them, or take the row out of the addressable set — applies directly to GRACE codebook keys, Larimar memory addressing, SERAC's scope classifier, MUNKEY's instance keys and any RAG store with soft delete. That is the most transferable thing this programme has produced, and the claim summary omits it.

**(b) A learned gate certifies the margin between the classes it was shown, not the predicate it was written to implement — measured, with the geometry swept. (E-000029, ledger §31.12.)**

`MVCCStore.marker_valid` declares a radius of 0.35. Sweeping shells over 11 checkpoints: accept rate is 1.0000 out to ‖m−κ‖ = 0.70, 0.2191 at 0.80, and first reaches 0 at 0.90. **The gate's operational radius is 0.90 against a declared 0.35, on every checkpoint.** The annulus 0.35–0.70 — which the store's own predicate calls deleted, and which no training or evaluation distribution ever populates — is accepted at 2,199,996 of 2,200,000. Measured on the sampled distributions instead, the same gate reads 8.550e-04, reproducing E-000021's 8.49e-04 inside its interval. So the headline false-accept rate is correct and is not the false-accept rate of the thing being claimed.

The adversarial-ML literature says this qualitatively — Carlini & Wagner (AISec@CCS 2017) for learned detectors; Kraska, Beutel, Chi, Dean & Polyzotis (SIGMOD 2018) and Mitzenmacher (NeurIPS 2018) for learned membership structures, where the classifier's error is a property of the query distribution and a worst-case bound needs an exact backup filter. I found no work that demonstrates the specification-versus-learned-boundary gap as a *swept geometry* on a deletion mechanism, with the sampled-distribution rate and the predicate-region rate reported side by side. §31.12 adds the corollary: κ is estimable to 0.0076 from ~950 signed markers, `make_centre` derives it from the seed, every checkpoint serialises it — shipping a trained model ships the ability to mint signatures. The ledger correctly withdraws the crypto-shredding analogy to "an integrity check against unprivileged or accidental modification".

**(c) The residual-equivalence protocol (small, methodological).** E-000019 accepts a chance null with a *pre-registered equivalence margin* (δ=0.02; forced-choice band 0.05), on three seeds excluded from configuration selection, with a *calibrated attack floor* — the probe reads live cells at 0.893–0.927, so "at chance after SHRED" is not the artefact of a probe that never worked. TOFU (Maini, Feng, Schwarzschild, Lipton & Kolter, COLM 2024) already makes non-rejection the criterion (Forget Quality, p > 0.05) — and is criticised for exactly the missing half by Thaker, Hu et al. (arXiv:2410.02879, SaTML 2025). Sommer, Song, Wagh & Mittal (arXiv:2003.04247) do the power analysis but *reject a retention null*. LEACE (Belrose et al., NeurIPS 2023) proves linear guardedness in closed form, for concept erasure. The margin-plus-floor-plus-held-out-seeds combination is unclaimed. It is a protocol, not a mechanism.

---

### 2. What is known and must be cited, not claimed

**Claim 1 (COPY BOUND) — known, and misdescribed.** Structurally preventing a fact from becoming a gradient target is LMLM (Zhao, Zalouk, Belardi, Lovelace, Zhou, Noonan, Go, Weinberger, Artzi & Sun, arXiv:2505.15962 — verified: "strategically masks externally retrieved factual values from the training loss"), descending from the Goldfish loss (Hans et al., NeurIPS 2024). Corpus-level isolation with opt-out by store removal is SILO (Min et al., ICLR 2024). "Unlearning by design" with a memory-augmented transformer where forgetting is key deletion is MUNKEY (Laguna, da Silva Gonçalves, Vandenhirtz, Ryser, Cannistraci & Vogt, arXiv:2603.15033, 16 Mar 2026 — verified, vision-only). The corresponding *audit* is already published at 40× this scale: Raeesi & Roed (arXiv:2607.00605, 1 Jul 2026 — verified) run 12,228 alias-closure deletions over thirteen databases, four adversarial retrieval topologies and six prompt formulations, find parametric leakage near zero everywhere, and locate the 0.7–13.6% residual in the retrieval graph.

Internally: "provably" is not earned — E-000002's own caveat says only `fixed_routing` is an empirical arm, and the roadmap's Stage 0 line states that **the separating variable is the layer, not the re-sampled training regime**, correcting an earlier summary that said otherwise. The leak is 0 of 300 pooled trials, i.e. ≤1.3% at 95%, not 0%. Thudi, Jia, Shumailov & Papernot (USENIX Security 2022) is why the word cannot be repaired empirically: absence is not provable from weights, only from the algorithm — and the algorithmic argument here is SILO's and LMLM's.

**Claim 3 (PROVENANCE) — known.** Reading off which record was accessed is definitional for memory-augmented and retrieval architectures: Memory Networks with supporting-fact supervision (Weston, Chopra & Bordes, 2015), kNN-LM (Khandelwal et al., ICLR 2020), product-key memory layers (Lample et al., NeurIPS 2019), Entities as Experts (Févry et al., EMNLP 2020), Facts as Experts / FILM (Verga, Sun, Baldini Soares & Cohen, 2020/NAACL 2021), SILO's sentence-level attribution. Two internal corrections the claim drops: provenance is a *trained* objective (E-000006: 0% without the routing loss; the roadmap calls it "a consistency check rather than a discovery"), and 100% / 99.99% are synthetic-core only — in the frozen GPT-2 the same metric is 84% (E-000008) and 83.1% (E-000011), computed at template 0, which §31.9 shows is one of the weak phrasings.

**Claim 6 (OVERRIDE AND RETURN) — known, and overstated.** ~100% efficacy on a known fact is ROME (Meng, Bau, Andonian & Belinkov, NeurIPS 2022) and MEMIT (ICLR 2023). Exact return to base on revoke is the defining property of every frozen-base external editor — SERAC (Mitchell, Lin, Bosselut, Manning & Finn, ICML 2022), GRACE (Hartvigsen et al., NeurIPS 2023), Larimar (Das et al., ICML 2024). The record agrees: `fallback_after_revoke_by_construction … does NOT grant a deletion level`, `claim_supported: false`, F1. And the 100% and the 0.0004 nats are **one trained template**: `override_heldout_min` is 0.0000 in all three seeds, and injection into generic text is 2.27 nats against a 0.05 bar, rising to 3.27 with more training templates (E-000017-B). ROME reports ~96% paraphrase generalisation on GPT-2 XL; this is 0% on unseen phrasings.

**Claim 5 (COMPOSITION BEATS IN-CONTEXT) — not established, and the numbers cross two runs.** E-000012 is 91.1% versus 40.3% (its `comparator/in_context_both_facts_hop2_acc`); the 41.7% quoted belongs to E-000011, whose paired figure is 83.3%. The routed arm is a 2.37M adapter trained 3,000 steps with 30% two-hop batches; the comparator is the untouched frozen model, zero-shot, no demonstrations, no CoT. The pre-registered kill criterion 4 is written against the *single-fact* baseline (0.9%), which no working system could fail. E-000012 records `claim_supported: false`, F1.

**Claim 4's counting argument — known for fifty years.** One-operation-reaches-all versus one-per-copy is Codd's update anomaly (CACM 1970), the Unix inode and symlink (Ritchie & Thompson, CACM 1974; 4.2BSD 1983), and dedup-plus-crypto-erase. That copies survive deletion is now also a published unlearning result: Ye, Zhu, Li, Gao, Liu, Zhang, Zhou & Zhang, "Data Duplication: A Novel Multi-Purpose Attack Paradigm in Machine Unlearning" (USENIX Security 2025, arXiv:2501.16663 — verified). Learned pointer dereference in external memory is NRAM (Kurach, Andrychowicz & Sutskever, ICLR 2016), the DNC (Graves et al., Nature 2016) and PANM (Le et al., arXiv:2404.11870).

---

### 3. What is new only in combination

- **Claim 2's mechanism.** Per-payload credential + learned verifier in the read path + payload deliberately left resident and still routed to + the verifier's false-accept rate reported as a deletion-relevant quantity. Every ingredient is old (SERAC/GRACE/WISE for a learned gate in the read path; NIST SP 800-88 Cryptographic Erase for destroy-the-key-keep-the-ciphertext; Larimar for selective fact forgetting; learned Bloom filters for FAR-as-headline). The assembly is unclaimed — and is the thing §1(a) and §1(b) have just undermined from inside.
- **Claim 4's experimental artifact.** The same synthetic world written twice — as LINK cells over one shared object, and as duplicated fact cells — held to identical ground truth, read by the **same** trained model, with the **same** calibrated probe applied to both arms after one lifecycle operation (E-000015: probe 0.7% vs 87.3%, forced choice 0.503 vs 1.000, calibration 88.7%). Ye et al. establish the duplication half with no sharing arm; Facts-as-Experts and Larimar establish the sharing half with no duplication control. The paired arms in one reader I could not find. Scope: F3, synthetic; at GPT-2 scale only E-000026's held-out template 10 makes it meaningful (alias read 0.87, probe 0.80 live → 0.01 after one SHRED), while `one_update_reaches_every_path` still misses at 0.8850 against 0.90, and the key-channel sweep of E-000028 has **never been run on the symlink arms**.
- **Claims 1+3 together** give exhaustive rather than indicative attribution, because there is no parametric channel to compete with the trace. SILO has the same property for the same reason. Not an independent contribution.

---

### 4. The single strongest objection, and what would answer it

**The objection.** *The only claim in this programme that is not arithmetic has been falsified by the programme's own last experiment, and the claims that survive are the ones it admits are by construction.* REVOKE, RESTORE, ROLLBACK and the return-to-prior are removals from routing or a zeroed injection — the records label them `by_construction` and say they grant no deletion level. SHRED is the one *learned* deletion primitive, the one F4, and the entire justification for the marker-and-gate apparatus. E-000028 shows it recovers the destroyed object at top-1 1.0000 with numbers identical to a live cell, while REVOKE and DELETE — the by-construction operations — are at chance. E-000029 shows the gate whose 1-in-1,180 error rate was quoted as the bound has an operational boundary 2.6× its declared radius and accepts the store's entire deleted annulus at rate 1.0000, with the centre estimable from any bank and serialised in every checkpoint. So: the sound deletion is trivial, and the non-trivial deletion is not sound. Add that §31.13 shows REVOKE and SHRED are literally the same number in 36 of 36 comparable cells after the status flag — one measurement reported as two mechanisms agreeing.

**What would answer it.** Not a rebuttal — a run. (i) Train `ModelConfig.gate_reverse_key` (already written, off by default, needs its own training run) and re-run E-000028 with a *complete enumeration* of every quantity derived from a payload, showing key-channel recovery at chance while the payload stays resident and value-channel behaviour is unchanged. (ii) Run that same sweep on the symlink arms, where it has never been run, so claim 4's "nothing recoverable" stops being value-channel-only. (iii) Retrain the gate with unsigned markers drawn from the whole region beyond the declared 0.35 — §31.12 says this is a change to the data, not the architecture — and re-measure the annulus rate. (iv) Then answer the question that makes the construction worth having at all: with the key channel closed, what does resident-but-gated buy over REVOKE, which already closes both channels? If nothing, the marker and the gate have no job and the honest system is a store with an addressability flag.

---

### 5. The smallest concrete result that would make this matter outside the project

**Port E-000028 out of this repository.** Take one published external-memory or edit-memory system — GRACE's codebook, Larimar's episodic memory, or a soft-deleting HNSW RAG store — and run the same attack: after its own deletion or edit-removal operation, sweep candidates through whichever index term is a function of the removed payload, and report top-1 recovery against that system's own reported deletion metric. No training. Inference hours on existing checkpoints. Two possible outcomes, both worth publishing: either a real recovery channel in a system people cite, which makes "enumerate every payload-derived quantity" a finding of general force rather than a note about one repository's `k_rev`; or a null, which localises the defect to this key construction and tells the programme its flagship failure is idiosyncratic. Ghost Vectors did the analogous thing for *undeleted* embeddings and is a verified 2026 paper; nobody has done it for an entry a system reports as deleted.

**Second, and it decides claim 5 rather than extending it:** one CPU adapter run with a *matched-capacity, matched-budget* in-context reader — identical frozen GPT-2, identical worlds, identical 3,000 steps and 2.37M parameter budget, spent on a soft prefix that composes two facts placed in the prompt — evaluated on the exact E-000012 two-hop rows, alongside the logit-level RAG-equivalent adapter the roadmap's own Stage 3 already specifies. Until that exists, "composition beats in-context" is a trained module beating an untrained one on the task the module was trained for, and the honest form of the claim is the single-fact baseline, which is vacuous.

**What would not move anyone:** more seeds, more cells, another synthetic battery, or any further result at 124M on single-token entities.

**Sources:** [MUNKEY](https://arxiv.org/abs/2603.15033) · [Auditing Forgetting in LMLMs](https://arxiv.org/abs/2607.00605) · [Subtract, Transport, or Replay?](https://arxiv.org/abs/2607.27539) · [Ghost Vectors](https://arxiv.org/abs/2606.18497) · [LMLM](https://arxiv.org/abs/2505.15962) · [Data Duplication (USENIX Sec 2025)](https://arxiv.org/abs/2501.16663) · [Agentic Unlearning](https://arxiv.org/abs/2602.17692) · [Kathleen Remembers](https://arxiv.org/abs/2608.30376)

Primary records read: `/home/user/FinX-BB/docs/so-experiment-ledger.md` §§31.2, 31.4–31.13, 31.8; `/home/user/FinX-BB/docs/so-roadmap-2026-09-02.md`; `/home/user/FinX-BB/so/results/{e000019_fresh_seed_chance,e000021_gate_error_rates,e000028_key_channel,e000029_marker_geometry,e000011_gpt2_v2,e000012_status_gated_revoke,e000013_prior_conflict,e000015_symlink_cells,e000020_symlink_gpt2,e000025_template_rescoring,e000026_lifecycle_readable_template,e000016_alias_chains}.json`.