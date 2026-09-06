# LOD-001 prior-art update: the detection floor is published, the dose ladder is published, and the framing narrows

Date: 2026-09-06
Status: **decisive narrowing, written before the recorded run produced a number.** The
pre-registration (`docs/novelty/lod001-preregister.md`) is unchanged and no bar in it moves. What
changes is the size of any sentence that may be read from it.

Every source below was **fetched and read in this session**, and the quotations were checked against
the paper's own HTML rather than against an abstract or a summary. Where an abstract-level read
disagreed with a full-text read, the full text is what is recorded, and the disagreement is noted.

## The two collisions

### 1. A detection floor for an internal-activation probe audit is already published

Florian Braun, *Excess Separability: Nuisance-Controlled Residual-Stream Probing for Benchmark
Contamination Detection*, arXiv:2608.12652 (12 Aug 2026, rev. 16 Aug 2026).

Verbatim, §3.8:

> The *detection floor* m∗ is the smallest m at which the test rejects at α=0.05 with power 0.8 at
> the given n; it states what the instrument can and cannot see.

And §L3:

> The detection floor m∗ should always be reported alongside a null so the null can be read at the
> right strength.

So the idea this experiment was built around — *an audit's null is uninterpretable without the
smallest effect it could have seen* — is **in print, for an internal-activation probe, and is already
prescribed as practice.** LOD-001 may not claim it. Executing a published recommendation is not a
novelty, and this document exists so that nothing downstream forgets it.

**An abstract-level read said otherwise and was wrong.** A first fetch of the abstract answered "No"
to whether the paper defines a detection floor. The full text answers yes, in §3.8. The abstract was
the wrong instrument for the question — which is, uncomfortably, this experiment's own subject.

### 2. The dose ladder as an instrument is already published

Emilio Ferrara, *Open-Weight Masked Introspection: Measuring What Language Models Can Report About
Their Own Computation*, arXiv:2608.20569 (20 Aug 2026).

Verbatim: the scaling operator "multiplies the activation by the strength parameter"; §3.6 "We
therefore order the ladder by perturbation magnitude rather than by the strength value"; the estimand
is "the detection margin as measured output divergence approaches zero",
`m0 = lim(D->0+)[Pr(y=1|int,D) - Pr(y=1|sham)]`; and §6, on sensitivity:

> Two sensitivity anchors tell us whether a null is informative. On the two dose-calibrated models, a
> linear probe trained on held-out activations measures how much intervention information is linearly
> recoverable at each site.

So a graded multiplicative dose on an activation at a named site, ordered by measured perturbation
rather than by the nominal knob, with an impact-matched random control and a **per-site linear-probe
sensitivity anchor**, is an existing instrument. `so/llm_adapter.py::set_inject_scale` is a
re-implementation of a known operator and is described as one from here on. The same operator is
older still in the steering literature — ActAdd (arXiv:2308.10248) and CAA (arXiv:2312.06681) own the
additive-injection-with-scalar-coefficient formulation and the layer × multiplier sweep — and the
J-lens paper this programme's audit rests on (Gurnee et al., transformer-circuits.pub/2026/workspace/)
already scales its own lens-coordinate swaps by a factor α.

## What this removes from the target, permanently

- **NO** — *An audit's verdict is uninterpretable without a detection limit.* Braun §3.8 and §L3.
- **NO** — *A dose ladder over an activation at a named site.* Ferrara §3.6; ActAdd; CAA; the workspace paper.
- **NO** — *Per-site sensitivity anchoring of a null.* Ferrara §6.
- **NO** — *Detection limits, calibration curves, power analysis* as ideas. A century old, and the
  pre-registration already disclaimed them.
- **NO** — *Per-item minimum detectable effect for a language-model audit*, behaviourally: Shihab et al.,
  arXiv:2608.07914 (reported by a hunter in this session; **not** independently verified here, and it
  is not load-bearing for anything below).

## What survives, checked against those same two papers

Three differences, each verified in the source that would kill it:

1. **Braun's ladder costs one trained model per rung; this one costs one forward pass.** Verbatim,
   his calibration is "On a family of calibration models trained with known duplication counts
   m ∈ {0,1,2,4,8,16,32}" — the dose is a *training* duplication count, so every rung is a separate
   checkpoint. That is why the floor is reported "per condition" and not per item, and why §3.9 says
   "Individually these are noisy … and we do not propose per-item verdicts." On an external store the
   dose is applied at inference to one item, over identical frozen weights and an identical prompt,
   and the zero rung is a row that was never written. **The obstacle Braun names is a property of the
   substrate, not of the audit.**
2. **Neither paper reports a floor as a function of the audit's READ SITE.** Braun's is per condition
   (§5.2: "the detection floor m∗ per condition"). Ferrara measures per-site probe *accuracy* as an
   anchor for whether a null is informative — not the smallest detectable dose at each site. A curve
   over sites is what turns a floor into a statement about siting.
3. **Neither is an accessibility or deletion audit.** Braun audits benchmark contamination; Ferrara
   audits introspective report. J-Access (arXiv:2608.11408) is the accessibility audit, and it states
   no sensitivity, no power and no minimum detectable residual anywhere. (Its item-level AUROC figure
   was reported by a hunter this session from the body; **only the abstract was independently fetched
   here**, and it does not carry the number, so that figure is not relied on below.)

## The target, narrowed to what those three support

> On a frozen model reading an external memory, the smallest residue an accessibility audit can see
> is a function of **where the audit is read**, and on this substrate it is measurable **per item at
> the cost of a forward pass** rather than a trained checkpoint per rung — which is the reason the
> published floor is per condition and declines per-item verdicts. Measured on the same weights, the
> same prompt and the same deletion, the floor at the site a composed deletion certificate chose is
> **above full retention**, so that certificate's PASS is purchasable by siting alone.

That is smaller than what the pre-registration's prose implies and it is what the sources leave. Any
claim document must lead with Braun and Ferrara, not with the floor.

## Disclosure: a plumbing cell was seen before the recorded run

The pre-registration was committed (69f6687) before the substrate finished training. A 6-pod,
alias-only, seed-0 **plumbing run** was then executed to prove the code path, at 1 thread, in 80
seconds, and its four printed lines were seen:

| site | `jspace` live − never | `LOD_alpha` |
|---|---|---|
| site8 | +0.000 | none of the ten rungs detected |
| site9 | −0.024 | none of the ten rungs detected |
| site10 | +0.786 | 0.375 |
| final | +0.833 | 0.375 |

Six pods, one seed, one address mode, and it is not the record: `so/results/lod001/` is. It is
disclosed because it was seen, it is consistent with WSC-001's recorded `alpha = 1` endpoints as the
pre-registration said those would be, and **the site10/final rungs are new information** — the
pre-registration's `L2` and `L3` were written without them. No bar was changed after seeing it, and
this table is the evidence for that: the bars are in the commit that precedes the run.
