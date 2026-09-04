# Deletion capacity, allocation, and the pod — what is claimed, at the size it survives

## The claim

**Superposition buys representation capacity and does not buy deletion capacity.** A
*d*-dimensional residual stream can *represent* exponentially many features with small interference —
Johnson–Lindenstrauss gives exponentially many almost-orthogonal directions, and superposition is the
observation that models use them. But a **clean deletion** of one fact — an orthogonal projection
removing a minimal subspace `A_i` such that no access path of fact *i* still yields the object and
every access path of every other fact still does — requires `A_i` to be orthogonal to every other
fact's readout subspace `V_j`. Mutually orthogonal subspaces satisfy `Σ dim A_i ≤ d`, so with each of
size *s*:

> **n ≤ d / s.  Clean-deletion capacity is linear in the dimension; representation capacity is not.**

**And the honest caveat, which strengthens rather than weakens what follows.** That bound holds for
*exactly* zero damage. If collateral is measured by whether the other fact's argmax flips, then `A_i`
need only be orthogonal to `V_j` to within each fact's logit margin, and almost-orthogonal sets in
`R^d` are exponentially large again. So the sharp bound is a limiting case and **not** a practical
limit — which means an observed failure to delete cleanly *cannot* be blamed on capacity at any
plausible scale.

**Measured (E-000043, frozen GPT-2 small, d = 768, 17 capital facts × 8 phrasings, six
pre-registered criteria, all PASS):**

| | |
|---|---|
| facts a subspace of their own basis silences | 12 of 17 (0.7059), to 0.0312 |
| directions demanded in total | **58 of 768** — `pressure` 0.0755, bound 159 facts |
| headroom left unused | **0.9245** |
| mean pairwise overlap of the deletion subspaces | 0.5566, against a **matched null of 0.1448** |
| **excess over the null** | **+0.4118** |
| bystander facts under the same ablation | 0.3897 from 1.0000 |

> **The failure is allocation, not capacity.** The model had 92% of its dimension budget free and
> still gave twelve facts deletion subspaces overlapping 0.41 more than random states put through the
> identical construction. Allocation is a training objective; capacity would have been a law of
> dimension.

## And the sharing is in the addressing, not the content

Splitting each fact's basis into its **content** direction (row 0 — what all its phrasings share) and
its **addressing** rows (the phrasing spread), against the same matched null:

| subspace | overlap | matched null | excess |
|---|---|---|---|
| content direction only | 0.2232 | 0.0638 | **+0.1594** |
| addressing rows only | 0.5954 | 0.1930 | **+0.4024** |
| **addressing minus content** | | | **+0.2430** |

**A fact's own content direction is nearly private. What is shared is the machinery that says which
phrasing asked for it.** That is the symlink result stated in activation space: *what a store keeps in
separate records — the object, and the keys that reach it — a representation keeps in one subspace, so
a deletion aimed at the content pays its collateral to the addressing.*

And it closes a loop with the reading side. E-000039-A measures that **88.6%** of the held-out
paraphrase gap is closed by forcing the address (`routing_share` 0.8861, worst seed 0.8818, against a
rule fixed in advance at 0.7). **Reading a fact through a new phrasing fails at addressing. Deleting a
fact damages bystanders through addressing.** One structure, measured from both ends.

**And the store is the limit case, which is where the symlink comes in.** An addressable memory's
addresses are *records*, not directions in a fixed-dimensional space, so it has no dimension bound at
all: clean-deletion capacity equals the record count. That is the precise, quantitative sense in which
giving a model an inode is a capacity property and not a convenience. Measured on the store side:
fact closure 1 for a canonical pod against *k* for *k* copies, `proved optimal` 1.00 in every arm
(E-000032), the store-side statistic `(closure − 1)/k` predicting the neural reader with error
**0.0000**, and the law `U = 1 + copies, T = k` holding in **105 of 105** cells (E-000041).

**The pod is the objective that closes the gap.** Within one fact, its access paths should share a
core — that is a symlink in activation space, and it makes the closure 1. Across facts, cores should
be disjoint — that is privacy, and it makes the collateral 0. E-000043 says there is room for both and
that GPT-2 took neither.

## An instrument bug worth recording, because it reversed the answer

E-000043's first run measured `rank(union) / Σ dim A_i` and called it efficiency. That is **linear
independence**; the theorem needs **orthogonality**. It reported **1.0000** — twelve subspaces
totalling 58 directions with rank exactly 58, a direct sum, "perfectly allocated" — in the same run
whose pairwise principal cosines were 0.5566 and 0.8559. Quoting that number alone would have
concluded the opposite of the truth. `σ_min` of the stacked bases is now primary; the rank is kept
beside it; and a test reproduces the failure with two subspaces at cosine 0.9 that rank scores 1.00.

---

# The earlier claim, and what became of it

This document exists to state a claim narrowly enough to be worth making, and to hand back the parts
of it that turn out to belong to other people. The research pass behind it found more prior art than
the claim was originally sized for, including one theorem that contradicts something I asserted
earlier in this programme.

## 0. The correction I owe first

I wrote, in framing this work, that LEACE gives a *sufficiency* guarantee and not a lower bound. That
is wrong, and it is wrong about the load-bearing part.

**LEACE Theorem 4.1** (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman, NeurIPS 2023,
arXiv:2306.03819) is an *if and only if*:

> "given some affine function r(x) = Px + b, the modified random vector r(X) linearly guards Z **if and
> only if** the columns of the cross-covariance matrix Σ_XZ are contained in the null space of P."

The "only if" half binds every affine erasure map: `dim ker(P) ≥ rank(Σ_XZ)`. No affine erasure
annihilating fewer than `ℓ = rank(Σ_XZ)` dimensions can achieve linear guardedness. And because LEACE
*attains* exactly `ℓ`, the interval is **closed**, not merely bounded below. Theorems 4.2 and 4.3 are
the sufficiency and minimal-perturbation results; 4.1 is the necessity one. The paper never advertises
it as a lower bound — it uses it to certify that SAL, Mean Projection and Fair PCA suffice — but a
certified lower bound on erasure cost is what it is.

So "the first certified lower bound on erasure in a representation" is **not available as a claim**,
and was never mine to make.

## 1. What else is owned, before anything is claimed

**The formal object, and its complexity, are already in print.** Adolfi, Vilas and Wareham (ICLR 2025,
arXiv:2410.08025) define *k-robustness*: "M is k-robust relative to H for I if for each subset
H′ ⊆ H, |H′| ≤ k, M(I) = (M/H′)(I)" — which is exactly "no set of k or fewer components erases the
behaviour", i.e. a certified lower bound of k+1 on a deletion closure over internal components. Their
Lemma 12 states the duality between minimum ablation and maximum robustness. Their Problem 8, Bounded
Global Necessary Circuit, is literally a hitting set over neurons: "a subset S of neurons in M of size
|S| ≤ k, such that S ∩ C ≠ ∅ for every circuit C in M that is sufficient" — NP-hard, in Σ₂ᵖ, and
inapproximable. Theorem 72: if the minimal-circuit problem is polynomial then P = NP. Theorem 73:
coW[1]-hard. Theorems 74–75 give the tractable route — fixed-parameter tractability, including in the
size of the candidate set.

**The disjoint-packing dual has been run, with sound certificates.** Bassan and Katz (arXiv:2210.13915)
build small contrastive sets, count disjoint singletons and take a minimum weighted vertex cover over
pairs, and report the ratio of upper to lower bound as a certified approximation factor — an interval
with a proof on the low side rather than a greedy number. Their must-hit sets come from a complete DNN
verifier, and their units are **input features**, not internal directions.

**The store-side argument is Freire, Gatterbauer, Immerman and Meliou's** (resilience and the minimum
contingency set), and the anomaly it exploits is Codd's.

**The J-lens is Anthropic's.** Gurnee, Sofroniew, … Lindsey, *Verbalizable Representations Form a
Global Workspace in Language Models* (Transformer Circuits, 6 July 2026): `J_ℓ = E[∂h_final,t′/∂h_ℓ,t]`,
"the rows of `W_U J_ℓ`" are the J-lens vectors, the J-space is the set of sparse nonnegative
combinations of them found by gradient pursuit, and its global ablation "does not ablate any tokens
that appear in the top-10 tokens of a clean forward pass".

**Detecting that an erasure happened, from one snapshot, is done.** Chen, Pal, Zhang, Qu and Liu
(arXiv:2506.14003, ICLR 2026) detect unlearning from intermediate activations at >90%. Youssef et al.
(arXiv:2505.20819) identify *which* fact an edit changed, from a post-edit snapshot alone, at up to
99%. Naor and Teague (STOC 2001) own the store-side form as weak history independence; DELF (USENIX
Security 2020) runs dangling-reference scans at Meta scale.

**Allocating a private carrier so deletion is cheap is MemSinks** (Ghosal, Maini, Raghunathan, ICML
2025, arXiv:2507.09937); **tying across access paths with a proof is Backpack** (Hewitt, Thickstun,
Manning, Liang, ACL 2023); **training for later editability is Sinitsin et al.** (ICLR 2020).

## 2. What is left

Three things, and they are smaller than the framing that produced them.

**(a) The disjoint-packing dual over internal directions, with must-hit sets from a decomposition
rather than from a verifier.** Adolfi et al. prove the object is coW[1]-hard in general and tractable
in the size of the candidate set, but produce no technique yielding a nonzero certified bound on a real
model. Bassan et al. produce the bound but over input features via complete verification. The J-lens
support is a candidate must-hit set that is cheap (one vector-Jacobian product per token — `so/jlens.py`),
principled (it is the direction the token's final logit is a function of), and small enough that the
fixed-parameter route is usable. `so/support.py` is that instrument.

**(b) The pod reading of the disjoint family.** For a fact with access paths Q and J-lens supports
{S_q}: the **pod core** is ⋂_q S_q and the **denormalisation degree** is the largest pairwise-disjoint
subfamily. A non-empty core means several keys reaching one object — a symlink detected in activation
space — and the degree is the number of independent copies of the fact, which is the store's
denormalisation degree read off a model. A bound of 1 is then not a weak result but the correct answer
for a fact stored as a pod. I found no prior statement of this correspondence.

**(c) A closure is not a deletion unless a refitted probe says so.** Nobody in the unlearning
literature reports a minimum removal-set size, so nobody has had to state this. A parallel measurement
on this repository's own checkpoint found a closure of 1.00 at collateral 0.0044 whose ablated states
still yielded the object to a freshly fitted linear probe at **0.9300** held out. Reporting a
(closure, collateral) pair without that probe reports a readout-path removal. E-000042 makes
`probe_after` a pre-registered criterion.

## 3. What the evidence says so far, including where it refuses

**E-000042, first run: VOID, and instructive.** The pool was rows of `W_U` — the logit lens, the J = I
special case. Eight directions removed, not one fact silenced.

**The check that appeared to fix it, and did not.** Removing the eight logit-lens rows of the eight
candidate capitals took the answer to 0.0000. Those rows *are* the readout of the candidate set, so
removing them makes the restricted argmax noise. That blinds the readout; it deletes nothing. The
paper's guard against ablating clean top-k tokens exists for this reason and is now carried over.

**E-000040.** The closure removes the fact (answer 0.9444 → 0.0000) at collateral 0.8611 from 1.0000,
using 3.33 directions. Only 0.0784 of a fact's basis is shared with other facts — the carrier is
mostly private — and yet only 0.5098 of facts can be silenced using nothing but their own directions.
**Privacy of the carrier is not sufficient for a clean deletion.** The traceless gap is +0.1236 on
average but positive for only 0.3611 of facts, and `hole_detectable` **FAILS** its pre-registered bar
at 0.6667 against 0.75. So E-000041's `T > U` does **not** carry into a representation on this
evidence, and the honest reading is that the store's law is a store's law.

**E-000039-A**, now with the oracle arm it was missing: `heldout/routing_share` **0.8861** mean,
0.8818 worst seed, against a rule fixed in advance at ≥ 0.7 → *train the address arm alone*. The
paraphrase gap is addressing, not transport.

## 4. Scope, stated because the numbers invite over-reading

A support size is a property of **the dictionary**, not of the model. arXiv:2608.10566 shows erasure
counts are not affine-invariant — "both quantities can change under an information-preserving
invertible reparameterization, so neither is intrinsically a concept dimension" — and names the
minimum guarding rank as the invariant instead. The certificate here licenses a statement about
ablations drawn from *this pool*, exactly as `certify_store_absence` licenses one about payloads
inside the domain it sweeps. Both are worth having and neither is a statement about the model as such.

And the must-hit property is **measured, not assumed**: `certify_must_hit` enumerates the subsets of
the complement, names its counterexample when one silences the query, refuses a query the model never
answered, and flags as vacuous a support that fills the pool and therefore had nothing to test against.
