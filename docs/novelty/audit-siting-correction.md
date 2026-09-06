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
