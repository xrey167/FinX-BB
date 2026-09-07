# References for the deletion-certificates draft — verification status

**All thirteen clusters verified 2026-09-07 with live network access from the drafting environment.**
Clusters 1-8 were checked first; **clusters 9-13 were added later the same day by NOV-005**, the audit
that calibrated this programme's literature search and then ran it against its own novelty claims.
Three of the five change a claim rather than support one. Until this date the
draft carried a blanket warning that every external citation was an unchecked lead, because the
environment was believed to have no network. That belief was never tested and was wrong: `pip`
reaches an index, `huggingface.co` answers, and web search and fetch both work. The warning was
true when written and false by the time it was repeated.

Each entry below records **what was checked and how**. A claim verified by reading the paper's own
full text is marked `FULL TEXT`; one verified only from search metadata and abstract is `METADATA`;
one still unchecked is `UNVERIFIED`.

---

## Verified

### 1. Larimar — memory-augmented LM with per-entry lifecycle operations `METADATA`

Das et al., *Larimar: Large Language Models with Episodic Memory Control*, ICML 2024.
arXiv:[2403.11901](https://arxiv.org/abs/2403.11901).

Supports §1's "prior art at larger scale". Confirmed: a distributed episodic memory supporting
one-shot updates without retraining, explicitly providing "selective fact forgetting" and
"information leakage prevention". The draft's concession that the architecture is re-invention
stands.

### 2. GRACE — the discrete codebook whose key is derived from the embedding `METADATA`

Hartvigsen et al., *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors*,
NeurIPS 2023. arXiv:[2211.11031](https://arxiv.org/abs/2211.11031).

**Load-bearing for PDX-001's `codebook_key` policy.** Confirmed: GRACE caches edits in a discrete
codebook in which "the original embedding is referred to as a key, and the learned embedding as a
value", with ε-balls around cached keys. A key computed *from* the content it indexes is exactly the
shape the reconstruction models, so PDX-001's third policy is faithful to a real design rather than
invented.

### 3. Ghost Vectors — the nearest prior work, and different in kind `METADATA`

*Ghost Vectors: Soft-Deleted Embeddings Remain Reconstructible in HNSW Vector Databases*, 2026.
arXiv:[2606.18497](https://arxiv.org/abs/2606.18497).

**§2's comparative claim is confirmed correct.** Ghost Vectors recovers embeddings that a soft
delete left *physically on disk*, by reading raw index files beneath the API. That is a retention
failure: the payload was never gated. §2's channel is different in kind — the payload *was* gated,
the value channel *is* at chance, and recovery runs through an index term derived from the payload.
The draft's "different in kind" is accurate and should be kept.

### 4. Auditing Forgetting in Limited Memory Language Models — the canonicalisation proposal `FULL TEXT`

Arya Raeesi and Hanna Roed, *Auditing Forgetting in Limited Memory Language Models*, 1 July 2026.
arXiv:[2607.00605](https://arxiv.org/abs/2607.00605).

**The draft's most delicate related-work claim, verified in full and holding in every part.**

* The quoted phrase is real and is in **§9 (Future Work)**, as the draft says:
  > "Both approaches are directly testable within our framework: re-running the audit on the
  > modified database and measuring whether R(f) and the retrieval artifact rate fall below their
  > current ranges would tell us how much of the residual is recoverable through database design
  > alone."
* It **proposes** storing "aliases and paraphrastic forms as pointers into a single canonical
  record" and **does not implement or evaluate it** ✓
* It **does not measure deletion closure size, nor the cost of finding the closure** ✓ — which is
  precisely what §6 claims as ours.

Its own result — that the unlearning boundary "is drawn primarily by the database administrator
rather than by the model", with post-deletion correctness dominated by near-neighbour retrieval
artifacts rather than parametric memory — is independent support for §6's record/fact distinction
and should be cited as such rather than only as the unrun proposal.

### 5. Resilience and minimum contingency sets `METADATA`

The database literature §6 attributes the quantity to. The problem is a variant of deletion
propagation: given a database satisfying a Boolean query, the minimum number of tuples to remove —
a **contingency set** — to make the query false. Dichotomy results (PTIME vs NP-complete) exist for
self-join-free conjunctive queries, for conjunctive queries with inequalities, and for unions of
conjunctive digraph queries; see for example arXiv:[1907.01129](https://arxiv.org/abs/1907.01129)
and arXiv:[2601.05346](https://arxiv.org/abs/2601.05346).

§6's attribution is correct: the definition matches the closure this repository computes, and the
terminology is the literature's, not ours.

### 6. Adversarial failure of learned predicates `METADATA`

Carlini and Wagner, *Towards Evaluating the Robustness of Neural Networks*, IEEE S&P 2017.
arXiv:[1608.04644](https://arxiv.org/abs/1608.04644).

Kraska et al., *The Case for Learned Index Structures*, SIGMOD 2018.
arXiv:[1712.01208](https://arxiv.org/abs/1712.01208).

§5's analogy is confirmed: a learned Bloom filter keeps an **exact backup filter** built over the
keys the model scores below threshold, precisely so that no false negative survives. A learned
predicate backed by an exact structure is the established remedy, which is what §5 says.

### 7. Attack-based unlearning benchmarks, and a published critique `FULL TEXT` (critique)

The benchmarks: Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*
(arXiv:[2401.06121](https://arxiv.org/abs/2401.06121)); Li et al., *The WMDP Benchmark: Measuring and
Reducing Malicious Use With Unlearning* (arXiv:[2403.03218](https://arxiv.org/abs/2403.03218));
Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*
(arXiv:[2407.06460](https://arxiv.org/abs/2407.06460)).

**This is the cluster the draft's framing of "the standard" depends on, and it is real.** MUSE
enumerates six desiderata for an unlearned model, one of which is *no privacy leakage* — an
attack-based criterion. Evaluation by "delete, attack, report the attack failed" is the convention,
not a straw man.

The critique, read in the source: Noam Diamant, Neta Glazer and Ethan Fetaya, *Stress Testing
Unlearning Algorithms*, 23 August 2026 (arXiv:[2608.22527](https://arxiv.org/abs/2608.22527)). From
the abstract:

> "they do not actively test whether unlearned information can still be forcibly extracted, and
> (2) they fail to evaluate performance preservation on boundary questions"

**And the relationship to F1 is worth stating precisely, because it is not agreement.** That critique
argues the benchmarks *do not attack hard enough*, and proposes attacking harder (WMDP++). The
draft's F1 is the sharper claim: attacking harder is not the fix, because the space of attacks is not
closed. §2's fifth attack was written *after* the four had returned at chance, and no amount of
adversarial effort closes that gap — only a proof over the payload domain does. The critique is
therefore independent evidence that the standard is the standard, and simultaneously an instance of
the response F1 says is insufficient.

### 8. Masked-value training and corpus isolation for the copy bound `METADATA`

*Provably Confidential Language Modelling*, arXiv:[2205.01863](https://arxiv.org/abs/2205.01863) —
Confidentially Redacted Training (CRT), which screens the corpus into a public set and a private set
and replaces repeated sentences with a `MASK` token from the second occurrence, yielding a provable
confidentiality guarantee.

Supports §10's concession exactly: the guarantee comes from **the training algorithm**, not from
inspecting the weights afterwards. That is why §10 declines the word "provably" for the copy bound
and attributes the argument to prior work.

---


### 9. The paper that reached F1 first, and takes N3 with it `FULL TEXT`

Sen Yang and Yuen-Hei Yeung, *Unlearning as Distribution Restoration: A Controlled Counterfactual
Study, a Validated Selective Screen, and the Limits of Oracle-Free Certification*, 21 July 2026.
arXiv:[2607.19442](https://arxiv.org/abs/2607.19442).

**The most consequential citation in this file, and it was found by looking for prior art against
our own nulls rather than for support.** Two things.

*It has F1.* Its section heading is "the adversarial boundary: forward-only certification is not
sound", and it earns it: a fixed-magnitude logit-suppression penalty "lands the forget-answer NLL
within family tolerance, and the entire forward battery accepts a suppressed model" in **12 of 45
cells**, on a model whose knowledge is intact. Independently, two months before this draft, over
five architecture families. F1 is therefore corroborated and not owned; §12 says so.

*It has all three parts of the residual-equivalence protocol §13 used to call unclaimed.* A declared
tolerance built from retraining redraws, with survivors tested for equivalence against it — "only
28.0% of survivors (CI [22.6%,34.0%]) are TOST-equivalent to never-learned at that tolerance"; a
**sealed challenge panel of known-label models** as the floor — "the screen rejects Minj in 45/45
cells and accepts the reference in 44/45", which is a two-sided control where E-000019 has one; and
an analysis rule "fixed before evaluating held-out families". E-000019 is three seeds on one
synthetic model. The claim is withdrawn.

*What it leaves open, and it is this paper's remaining half.* It declines the constructive step in
its own words: its result is "an empirical selective test for methods-as-produced, not an
adversarially sound certificate".

### 10. Derived artifacts surviving a plaintext-layer delete `FULL TEXT`

Chao Yao et al., *Forgetting Without Restarting: Execution-State Unlearning for Stateful LLM
Agents*, 4 September 2026. arXiv:[2609.04875](https://arxiv.org/abs/2609.04875).

**Narrows §2's most transferable claim to a measurement.** Deployed stacks "offer only a forgetting
affordance that operates on plaintext at a single layer: delete the memory record, edit the Markdown
file, drop the message from retrieval", and deleting the persistent memory record "leaves leakage
*exactly* unchanged from doing nothing". Their derived artifacts are compressed summaries, authored
plans, and KV-cache tensors; behavioural extraction succeeds in 80-100% of episodes with zero string
matches.

So *enumerate every quantity derived from a payload and gate all of them* is published, three days
before this audit. Two gaps remain and both are narrow: their artifacts **carry** the content, where
`k_rev(LN(o + r))` holds no payload and names none yet still discriminates; and they measure presence
and behaviour, not a candidate-set posterior, so they would read PDX-001's tombstone row exactly as a
top-1 audit does — 0.0000, with the search space cut 88x.

### 11. The composition, proved in 2020 `FULL TEXT` (Garg et al.) / `METADATA` (Cohen et al.)

Sanjam Garg, Shafi Goldwasser and Prashant Nalini Vasudevan, *Formalizing Data Deletion in the
Context of the Right to be Forgotten*, Eurocrypt 2020.
[eprint 2020/254](https://eprint.iacr.org/2020/254). And Aloni Cohen, Adam D. Smith, Marika Swanberg
and Prashant Nalini Vasudevan, *Control, Confidentiality, and the Right to be Forgotten*, CCS 2023.
arXiv:[2210.07876](https://arxiv.org/abs/2210.07876).

**Kills §6's composition claim, which the programme had already withdrawn internally and re-asserted
in the draft.** Read in the source: the data collector of their Fig. 5 "maintains a dataset as a
history-independent dictionary Dict", takes "any learning algorithm with such a deletion operation",
and on a deletion request "updates model to be the output of delete(Dict, model, key)". Theorem 3.4
bounds it: "The data collector (M, pi, piD) as described in Fig. 5 has 1-representative
deletion-compliance error at most (1/lambda + poly(lambda)/2^lambda)." That is a store-side guarantee
composed with a learned reader's certified deletion, proved, six years earlier. Cohen et al. then
"build a unified formalism for deletion that encompasses previous approaches as special cases".

### 12. The nearest prior art to F2's certificate `METADATA`

Vishwajith Ramesh, *Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory*,
30 July 2026 (revised 13 August). arXiv:[2607.27539](https://arxiv.org/abs/2607.27539). With
*Verifiable and Provably Secure Machine Unlearning* (arXiv:[2210.09126](https://arxiv.org/abs/2210.09126))
and *Proof of Unlearning: Definitions and Instantiation*
(arXiv:[2210.11334](https://arxiv.org/abs/2210.11334)).

**F2 is not alone in its line and the draft must stop implying it is.** Ramesh audits deletion from a
frozen LM's persistent memory with intermediate arrays inspected — "zero residual on final logits and
all 80 audited KDA arrays verifies restoration across the declared checkpoint surface" — at 1B, 4B
and 12B. The distinction F2 keeps is the *kind* of certificate: that work verifies a **recomputation**
(this state is what replay without the record produces) and still leans on behavioural attacks
reaching never-stored baselines for the residual argument, while §4 sweeps the **whole payload
domain** and checks nothing downstream moves for any value it could have held. The cryptographic line
(2210.09126, 2210.11334) proves the deleted point is absent from the *training set*, which is a claim
about provenance rather than about the computation.

### 13. Stale edges as a cost, and nobody calling them a channel `METADATA`

*MERIT: Efficient In-Place Deletion for Dynamic Graph-Based Approximate Nearest Neighbor Indexes*,
July 2026. arXiv:[2607.29173](https://arxiv.org/abs/2607.29173).

**Checked as prior art against §8 and PDX-001, and it is not.** The same structure is the subject —
after deleting a vertex the index "does not know which other vertices contain outgoing edges to it",
and stale incoming edges survive — but entirely as a performance problem: they "consume search
capacity" and degrade recall. There is no privacy or disclosure claim anywhere in it. That a proximity
graph's surviving adjacency is an *information* channel about the deleted row, and that it is
invisible to top-1, is not said here or, as far as these searches reach, anywhere else.

## Still unverified

None of the thirteen clusters. See below for what that does and does not mean.

## What this does not establish

Verification here is bibliographic: these works exist, say what the draft says they say, and in the
one case checked in full, do not do the thing the draft says they do not do. **No result of theirs
has been reproduced**, and a citation that is real can still be misread. §13(c) — running this
programme's instrument against one of these systems — remains the experiment that would settle
whether the channel is theirs and not only ours.
