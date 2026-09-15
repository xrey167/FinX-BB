# NOV-005 — the novelty search, calibrated and then run

Date: 2026-09-07
Record: `so/results/nov005/nov005_novelty_claim_audit.json`
Run: `make novelty`
Status: **floor met; two claims withdrawn, two narrowed, three survive; five citations added**

## The gap this closes

Every load-bearing novelty statement in this repository is a **null**. "Nothing I verified covers
this." "I found no work that demonstrates …" "The combination is unclaimed." "The paired arms in one
reader I could not find."

Those nulls came out of a 41-agent literature workflow whose own provenance note disclaims its
external citations, in an environment three successive drafts asserted had no network — an assertion
never tested and false (§31.57). On 2026-09-07 the situation was:

- the **positives** had been checked. `docs/paper/references.md` verifies eight clusters against
  their sources, two of them by reading the full text.
- the **negatives** had not been checked at all. Nothing had ever tested whether the search that
  produced them could find anything.

**§31.15 for the fifth time, now on the literature review itself: an instrument that cannot fail is
not evidence.** NOV-001 calibrated the reduction screen behind twenty-three kills. Nobody had
calibrated the screen behind the novelty verdicts, and the novelty verdicts are what decides whether
any of this is worth submitting.

## The calibration

Three propositions of known standing, phrased the way the novelty statement phrases its own claims,
run through the same instrument. All three returned their prior art:

| control | must find | returned |
|---|---|---|
| one operation on a shared record reaches every path | Codd | *A Relational Model of Data for Large Shared Data Banks*, CACM 13(6), 1970 |
| a learned predicate can be made to fail adversarially | arXiv:1608.04644 | Carlini and Wagner, IEEE S&P 2017 |
| the minimum tuple set falsifying a query is a named quantity | arXiv:1907.01129 | resilience / minimum contingency set, with the self-join-free dichotomy |

And the half usually left out. A **negative control** — a fabricated construct ("deletion certificates
for memory stores addressed by prime-indexed Gödel numbering with tri-symplectic revocation
manifolds") — returned topically adjacent noise and no match. A searcher that manufactures a hit for
anything would be as useless as one that finds nothing, and only the second failure is normally
tested.

`validity_floor_met = true`. That is the only reason anything below counts.

## The verdicts

| claim | verdict | what did it |
|---|---|---|
| N1 payload-derived index channel | **NARROWED** | Yao et al., arXiv:2609.04875, 4 Sep 2026 |
| N2 swept geometry of a learned gate | SURVIVES | — |
| N3 margin + floor + held-out seeds | **SUPERSEDED** | Yang & Yeung, arXiv:2607.19442, 21 Jul 2026 |
| N5 paired sharing and duplication arms | SURVIVES | — |
| N6 erasure–disclosure duality | SURVIVES | — |
| N7 composition with a record-level certificate | **SUPERSEDED** | Garg, Goldwasser & Vasudevan, Eurocrypt 2020 |
| N8 F1: the attack standard is not a guarantee | **NARROWED** | Yang & Yeung, arXiv:2607.19442 |

### N3 — withdrawn outright

Yang and Yeung have all three components, at 45 model-seed cells over five architecture families:

- **the margin.** A tolerance δ_equiv = 0.93 nats built from three independent retraining redraws per
  family, with survivors tested for TOST-equivalence against it — *"only 28.0% of survivors (CI
  [22.6%,34.0%]) are TOST-equivalent to never-learned at that tolerance"*.
- **the floor.** A sealed challenge panel of known-label models — *"The screen rejects Minj in 45/45
  cells and accepts the reference in 44/45"*. That is **two-sided** where E-000019's probe floor is
  one-sided: it shows the screen both catches a model that should fail and clears one that should
  pass.
- **the held-out design.** The analysis rule *"fixed before evaluating held-out families"*.

E-000019 is three seeds on one synthetic model. This was the weakest of the three §1 claims and it is
the one that fell first.

### N7 — withdrawn, and it had already been withdrawn once

The paper claimed the composition of a store-side guarantee with a record-level certificate over a
learned reader. Garg, Goldwasser and Vasudevan build exactly that and prove it. Read in the source:
the data collector of their Fig. 5 *"maintains a dataset as a history-independent dictionary Dict"*,
accepts *any* learning algorithm with a deletion operation, and on a request *"updates model to be
the output of delete(Dict, model, key)"*. **Theorem 3.4** bounds its 1-representative
deletion-compliance error at `1/λ + poly(λ)/2^λ`. Cohen, Smith, Swanberg and Vasudevan (CCS 2023)
then fold store-side and model-side definitions into one.

**The programme already knew.** `docs/so-novelty-2026-09-04.md` §6 withdrew this claim on the day it
was written. The withdrawal reached that document and never reached the paper, which went on
asserting it for three days — §31.62's drift, in the opposite direction and with a real consequence:
a submission would have claimed a 2020 Eurocrypt result.

### N8 — F1 is corroborated, not owned

The same Yang and Yeung paper has this paper's central thesis, as a section heading: *"the adversarial
boundary: forward-only certification is not sound"*. And it demonstrates it — a fixed-magnitude
logit-suppression penalty *"lands the forget-answer NLL within family tolerance, and the entire
forward battery accepts a suppressed model"* in **12 of 45 cells**, on a model whose knowledge is
intact.

That is F1, reached independently two months earlier, across five architecture families, with a
positive control this repository cannot match. It is **stronger evidence for F1 than anything here**,
and it is prior art rather than agreement.

What it leaves open is exactly this paper's other half. It declines the constructive step in its own
words — its result is *"an empirical selective test for methods-as-produced, not an adversarially
sound certificate"* — and proposes nothing to replace what it falsifies. F2 is that replacement. And
§2's channel is a mechanism a forward battery cannot reach in principle, because recovery runs
through a term that never holds the payload, so no elicitation of the payload touches it.

### N1 — narrowed to a measurement

Yao et al. publish the general shape three days before this audit: deployed stacks *"offer only a
forgetting affordance that operates on plaintext at a single layer"*, and deleting the persistent
memory record *"leaves leakage exactly unchanged from doing nothing"*, with behavioural extraction in
80–100% of episodes at zero string matches.

So **"enumerate every quantity derived from a payload and gate all of them" is no longer ours.** Two
gaps remain and both are narrow:

1. **Their artifacts carry the content.** Summaries and plans paraphrase the target; KV tensors
   encode it. The finding is that a copy survived. `k_rev(LN(o + r))` holds no payload and names
   none, and discriminates anyway — which is why every value-channel attack is at chance while
   recovery is exact.
2. **They measure presence and behaviour, not a candidate-set posterior.** Their binary reading would
   score PDX-001's tombstone row exactly as a top-1 audit does: 0.0000, with the search space cut
   88×. The false negative F1 closes twice on is invisible to their instrument too.

### What survives, and how weakly

N2, N5 and N6 returned nothing. Read that as *"these queries did not find it"*, not as *"it does not
exist"* — the queries are in the record so the next reader can beat them. N6 is the strongest of the
three: the adjacent work is there and stops short. MERIT (arXiv:2607.29173) studies precisely the
structure §8 exploits — after a vertex is deleted the index *"does not know which other vertices
contain outgoing edges to it"* — and treats it purely as a cost, stale edges that *"consume search
capacity"*, with no disclosure claim anywhere in it.

## The enforcement

A verdict in a document nobody reads, against a claim in one everybody does, is how §31.62 happened.
So the audit is not a document — it is a check. Any claim marked SUPERSEDED or NARROWED must leave a
**visible, marked withdrawal**: the paragraph carrying the assertion has to carry the string
`NOV-005`. Deleting the sentence instead is caught separately as `ASSERTION_NOT_FOUND`, because a
withdrawal that leaves no trace is not a withdrawal. `main()` exits non-zero on either.

It fired four times on first run, which is why the corrections in `docs/so-novelty-2026-09-04.md` and
in §6 and §12 of the draft exist.

## Two defects this experiment found in itself

- **A registry phrase that could not match.** N6's assertion phrase spanned a line wrap in the source
  document, so its enforcement silently checked nothing. Caught by the test that requires every
  phrase to be findable — the test that exists because a registry which fails to resolve passes a
  clean run.
- **Two exclusion rules that could never fire.** Adding the cited external figures to
  `so/paper_numbers.py` used `%\b` — and `%` is not a word character, so that boundary never matches.
  Both rules were dead on arrival and the coverage pass caught them. The same class as everything
  above, in the instrument written to police it.

## What this does not establish

No cited result has been reproduced. Every verdict rests on reading the source, and a source read
correctly can still be applied wrongly. **SUPERSEDED is a claim about priority, not about quality**:
N3 and N7 are done better elsewhere, which is the finding, and nothing here says the work behind them
was not worth doing. This is also not a systematic review — it is the queries one person ran in one
afternoon, recorded so that they can be attacked.

## What it changes for the paper

Three of the seven claims examined move, and the two that move furthest are the two the draft leaned
on hardest — F1's thesis and the composition. The paper is not weaker for it; it is narrower and
correctly attributed, and it now cites the two 2026 papers a reviewer in this area would have named
in the first round. The contribution that survives is what §6 of the novelty statement said it was
two days ago: **the audit and the measurement**, specifically the candidate-set posterior that the
published instruments — theirs and this repository's own first attack — cannot see.
