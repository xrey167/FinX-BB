# Correction to `audit-siting-claim.md`: two of its three "genuinely new" points are refuted, and the third narrows

Date: 2026-09-06
Status: **correction of a claim published in this repository yesterday**, on evidence fetched and read
in this session. This is a retraction of two numbered points in
`docs/novelty/audit-siting-claim.md`, written the same day that document was.

The measurements in `audit-siting-claim.md` are untouched: no number in it moves. What is withdrawn
is what it says those numbers are new for.

## What that document claimed was new

> 1. **A siting rule with its mediator.** For an external memory, an accessibility audit must be read
>    at or after the site that carries the content …
> 2. **A demonstrated vacuity mode.** A composed store-and-audit certificate can return its deletion
>    verdict while its instrument has never seen the object live …
> 3. **The counterfactual that makes it measurable.** J-Access reports item-level AUROC at chance
>    because parametric unlearning has no per-item "never knew it" control. A pod store has one …

## Point 3 is REFUTED, by a paper six weeks older

Sen Yang and Yuen-Hei Yeung, *Unlearning as Distribution Restoration: A Controlled Counterfactual
Study, a Validated Selective Screen, and the Limits of Oracle-Free Certification*, arXiv:2607.19442
(21 Jul 2026). Its abstract, fetched this session, scores candidates against a **never-learned level**
per fact — a per-fact control built by replaying training without the facts, on a nonce-fact testbed
with a matched retraining reference.

So *"parametric unlearning has no per-item never-knew-it control"* is **false as written**. It has one;
it costs a training run per control, and the control has different weights. That is a difference in
price and in exactness, not in existence, and point 3 claimed existence.

The privacy-auditing literature closes the same door from the other side, and earlier. LiRA
(arXiv:2112.03570) builds per-example IN/OUT distributions in which the OUT models never saw the
example; U-LiRA (Hayes et al., arXiv:2403.01218) runs exactly the three states this store cycles —
trained, unlearned, and never-contained — at 128 shadow models per example, and argues per-example
over aggregate. These were reported by a prior-art hunter in this session; the U-LiRA and LiRA
constructions are well enough established that the point falls on 2607.19442 alone, which was
verified here.

**What survives of point 3, and it must be stated this narrowly or it dies:** every retrieved
construction of a per-item null either changes the weights (LiRA, U-LiRA, Yang & Yeung's replay) or
changes the input (retrieval-disabled arms; canary-in-prompt ICL audits such as ContextLeak,
arXiv:2512.16059, reported this session and not independently verified). The pod store's null changes
**neither**: identical frozen weights, a byte-identical token sequence, and one absent row. Whether a
null that costs one `deepcopy` and no retraining is worth a sentence is a question for the claim
document, not for this one.

## Point 1 is REFUTED as a proposition, by the primary source it is built on

Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*,
transformer-circuits.pub/2026/workspace/ (6 Jul 2026) — the paper that introduces the Jacobian lens
this audit uses. Fetched this session, verbatim:

> Note that in roughly the first third of the model, the readouts are noisy and largely
> uninterpretable

and, stating the exact ambiguity WSC-001's mediator was built to break:

> the absence of meaningful J-lens-accessible content in the first third of the model could indicate
> that either (1) the J-lens is degenerate at these depths and fails to resolve content that is in
> fact present, or (2) the early-layer residual stream genuinely carries no linearly accessible and
> causally relevant verbalizable content.

They also **run an experiment to disambiguate those two** — the ambiguous-input experiment, in which
mixed concept embeddings are injected and the layer at which the model commits to one interpretation
is measured.

So *"a null lens readout does not certify absence, and where you read it matters"* is not new: it is
stated, with its two-way ambiguity, by the lens's own authors, together with a disambiguating
experiment. `audit-siting-claim.md` point 1 is withdrawn as a proposition.

**What survives, and it is a different statement.** Gurnee's rule is about **depth in the model** —
the first third is uninterpretable, the workspace has an onset layer. For an **external** memory the
binding constraint is not depth. It is the site of the **write**, which is a configuration parameter
(`AdapterConfig.read_layers`, `so/llm_adapter.py:74`) and can be moved without touching the model. On
this substrate the audit's blind block is 8 of 12 — two thirds of the way down, well past the first
third — and it is blind not because the lens is degenerate at that depth but because the memory has
not been written yet. Those are different reasons, they license different rules, and only the second
one is a rule about auditing an external memory. The mediator is what distinguishes them, and its
specific form — the ratio of the **injected write's own displacement** between two candidate sites —
is not stated in any source retrieved this session.

## Point 2 narrows, and the near miss should be on the record

Yang & Yeung (2607.19442) again: their abstract records that "the injected model, which retains the
retain set by construction, fails the fixed retain threshold in 41/45 cells" — a screen behaving
wrongly on a model whose knowledge is intact by construction, validated against a blind challenge
panel. A hunter reported further, from the body and not verified here, that the paper uses the word
*vacuous* with a count and states that "A certificate worth the name must be able to answer
'uncertified'."

That is the same genre as WSC-001's point 2, six weeks earlier. What is not obviously in it — and
what `audit-siting-claim.md` should have led with — is the **mechanism of the vacuity**: not a
threshold set wrongly, but an instrument sited where the audited content has not arrived, with the
failed validity row printed underneath the passing headline on every seed. Whether that distinction
carries a sentence is again for the claim document.

## Consequence for LOD-001

LOD-001's pre-registration is unaffected: it registers a measurement, and none of the above touches a
bar. Its prior-art section is superseded by this document and by
`docs/novelty/lod001-prior-art-update.md`, which together mean any claim written from its record must
lead with Braun (arXiv:2608.12652), Ferrara (arXiv:2608.20569), Gurnee et al. and Yang & Yeung, and
must not restate points 1 or 3 above in any form.

## Standing

Twelve-plus retractions on this ledger, and this is the next. It was found the day after the claim was
published, by eight prior-art hunters each required to fetch its sources in-session, and every
load-bearing quotation above was then re-fetched and checked against the paper's own text rather than
against a summary of it.

---

# Addendum, same day: `A0` itself is withdrawn — the surviving sentence does not survive its own bar

The body of this document withdrew points 1 and 3 of `audit-siting-claim.md` and left point 2 and the
`A0` siting branch standing. **`A0` is now withdrawn too**, on a defect found by a hostile reviewer in
this round's sweep and verified here twice: once in the recorded JSON, and once against a fresh
three-seed training that had nothing to do with WSC-001.

## The defect

`so/experiments/wsc001_workspace_share.py:468-472`:

```python
for l in (8, 10):
    out[f"A/write{l}_shred_moves"] = max(g(f"partA/alias/write{l}/shred_moves", 0.0),
                                         g(f"partA/direct/write{l}/shred_moves", 0.0))
denom = max(out.get("A/write10_shred_moves", 0.0), 1e-9)
out["A/write_site_ratio"] = out.get("A/write8_shred_moves", 0.0) / denom
```

The `max` over address modes is taken **independently for each site**, and the ratio is then formed
from the two winners. Nothing requires them to come from the same mode, and on the recorded data they
do not.

## The numbers, from `so/results/wsc001_workspace_share.json`

| seed | write8 alias | write8 direct | write10 alias | write10 direct | as recorded | **within alias** | within direct |
|---|---|---|---|---|---|---|---|
| 0 | **4.390** | 0.357 | 82.108 | **89.952** | 0.0488 | **0.0535** | 0.0040 |
| 1 | 1.553 | 0.083 | 93.834 | 98.814 | 0.0157 | 0.0166 | 0.0008 |
| 2 | 2.480 | 0.467 | 76.865 | 80.328 | 0.0309 | 0.0323 | 0.0058 |

On seed 0 the numerator is the **alias**-mode write and the denominator is the **direct**-mode write.
Worst of three seeds, as recorded: **0.0488**, and the bar is ≤ 0.05 — PASS, by 0.0012. Computed
within the mode that supplies the numerator: **0.0535** — **FAIL**, and the `A0` branch
(`site8 ≤ 0.05` AND `ratio ≤ 0.05`, `wsc001_workspace_share.py:539-541`) does not fire.

## Independent corroboration, from a run that was not looking for this

LOD-001 retrained the same recipe from scratch on three seeds and recorded the same two quantities
within alias mode only. First-read-site over second-read-site write magnitude, arm A:
**0.061 / 0.0074 / 0.0323**. Worst seed **0.061**, against the same 0.05 bar — **FAIL**, on three
fresh trainings, by a different experiment, with no knowledge of the defect.

## And the margin was never large enough to carry a sentence

Across the three independent training draws on record the recorded statistic is 0.0488 / 0.0157 /
0.0309: an across-seed SD of **0.0135** and a worst-of-three margin to the bar of **+0.0012**, which
is **0.09 of one standard deviation of the statistic being thresholded**. The same disease is already
on the record one level down: `finalprobe(SHRED − NEVER)` moved 0.0536 (FAIL) → 0.0000 (PASS) between
two runs of the identical protocol on identical seeds (`so/results/e000063_retrain/README.md`) — a
swing the size of the bar it thresholds.

## What is withdrawn, and what is not

**Withdrawn:** the `A0` branch and the sentence built on it — *"an accessibility audit placed at the
memory's first read site certifies nothing, because the pod's content is not there"* — as a result
carried by the pre-registered ratio. The ratio does not clear its bar within a single address mode and
its margin is inside its own replicate noise.

**Not withdrawn, because they do not depend on the ratio:**

- E-000063's composed certificate returns its headline verdict underneath its own failed validity row,
  on three recorded seeds and three fresh ones (`so/results/e000063_retrain/`). That is read off
  `jprobe(ACTIVE − NEVER)` directly and no ratio enters it.
- LOD-001's saturation table: at the first read layer and the one after it, the probe reads the
  never-written arm at 0.214 to 0.996 against a chance of 0.062 and the live arm at the same level.
  That is an absolute accuracy, not a ratio, and it is the better instrument for the same question.
- LOD-002's `W1`: block 8 blind on one arm and sighted on the other, three seeds, identical model and
  depth. It is a contrast between two trained arms, not a threshold on a ratio.

**The lesson the programme should take, and it is not new to it:** no bar on this substrate — `A0`'s,
E-000063's ten, LOD-001's six, LOD-002's — has ever been calibrated against the across-training-run
dispersion of the statistic it thresholds. Two bars have now been shown to sit inside that dispersion.
That is a defect in how this programme sets bars, not in any one experiment, and stating it is not a
contribution: reporting variance before thresholding is textbook, and Braun (arXiv:2608.12652)
propagates baseline variance and withdraws his own significant result on exactly this basis.

**Count:** this is the third withdrawal from `audit-siting-claim.md` in twenty-four hours, and the one
that removes its last standing sentence.
