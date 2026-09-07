# References for the deletion-certificates draft — verification status

**Verified 2026-09-07 with live network access from the drafting environment.** Until this date the
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

---

## Still unverified

These remain leads. They are **not** cited as fact anywhere the draft's argument turns on them, and
§0 continues to disclaim them.

* **Masked-value training and corpus isolation for the copy bound.** §10 already declines to claim
  "provably" for this and attributes the argument to prior work; the specific works are unchecked.
* **Attack-based unlearning benchmarks and their published critiques.** The draft's framing of "the
  standard" rests on this cluster being real, so it is the most important remaining gap.

## What this does not establish

Verification here is bibliographic: these works exist, say what the draft says they say, and in the
one case checked in full, do not do the thing the draft says they do not do. **No result of theirs
has been reproduced**, and a citation that is real can still be misread. §13(c) — running this
programme's instrument against one of these systems — remains the experiment that would settle
whether the channel is theirs and not only ours.
