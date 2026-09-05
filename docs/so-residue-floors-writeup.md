# Not a claim: the residue floors, written up, with the external clearance they never had

*2026-09-05. This document was drafted as a novelty claim and is retained as a WRITE-UP. The finding
is not mine and is not new to this repository: ledger §31.41 and §31.45 state it, with these numbers,
dated before this file. What is added here is an external prior-art clearance the ledger did not do,
a framing of the same numbers, and one experiment (E-000058) that tests a sentence §31.41 asserts and
never measured. Every number is read from a record in this repository; every external citation was
fetched at the source.*

## Attribution first, because the claim died on it

§31.41 already names both floors, gives the mechanism, states the inversion, and draws the
methodological conclusion — verbatim:

> "The reader's off-pod outputs carry the number of active rows — every row sits in every routing
> softmax and the dereference pass-through bias is `log(n_cells)` … BLANK's 0.817 and DANGLE's 0.869
> sit *below* the two-fresh-rows floor … a paper reporting such an AUC without the two floors would be
> reporting the reader's arithmetic."

and it already cites the adversary shape as owned by Chen et al., arXiv:2506.14003. §31.45 carries the
GPT-2 half under the title *"both floors rise, the residue rises with them"*. E-000056's own lesson (iv)
— *check the repository's ledger before claiming, it has usually already said it* — fires here exactly
as it fired on C2. **This is a restatement, not a discovery, and it is recorded as one.**

What is genuinely added, and it is modest:

1. an external prior-art clearance the ledger did not run — Raeesi & Roed (arXiv:2607.00605) fetched
   and read at the source, its arms established as `FULL`/`DEL-ON`/`DEL-OFF` with no record-count arm,
   no insertion-order arm and no null distribution, which clears the nearest neighbour;
2. the per-seed reading below, which shows the score is uncorrelated with behavioural change;
3. **E-000058**, which measures the one sentence §31.41 asserts without measuring: that the row-count
   floor is *"a property of dense routing and not of history"*. That attribution has never been tested,
   and it is the only unowned thing in this document.

## The measurement

E-000051 put a query-only adversary against a frozen GPT-2 reading a canonical-pod store, three
seeds, 100 pods per seed, training nothing. It compared lifecycle states of a pod and scored a
five-fold cross-validated Mann-Whitney AUC on queries **that are not about the pod** — class (ii)
bystanders, class (iii) generic text. The null band at n = 200 is about 0.42–0.58.

Per seed, so that the AUC and the behavioural change can be read against each other:

| arm (positive vs reference) | what actually changed | AUC (ii), seeds 0/1/2 | max KL (ii), nats | top-1 agree (ii) |
|---|---|---|---|---:|
| `perm` vs `perm2` | **nothing** — summation order only | 0.455 / 0.493 / 0.497 | 2.8e-11 … 1.3e-10 | 1.000 |
| `cascade` vs `never` | a **complete deletion** | 0.888 / 0.883 / 0.836 | 7.3e-06 / 8.1e-07 / 1.2e-04 | **1.000** |
| `blank` vs `cascade` | SET NULL on an alias, 2 residue rows | 0.773 / 0.829 / 0.770 | 2.59 / 4.49 / 1.63 | 0.999 |
| `dangle` vs `cascade` | object evicted, 2 tombstone rows | 0.843 / 0.911 / 0.875 | 1.77 / 3.96 / 1.95 | 0.999 |
| **`add2` vs `perm`** | **two fresh live rows. NO deletion.** | 0.961 / **1.000** / 0.971 | 9.7e-04 / 3.5e-03 / 0.31 | 1.000 / 1.000 / 0.999 |

**The probe is calibrated.** On `perm` the two banks differ only by summation order — max KL ~1e-10,
numerically the same distribution — and the same cross-validated probe sits at 0.455–0.497. It does
not manufacture separation. Every number below is therefore signal, not overfitting.

**A complete deletion is separable at 0.836–0.888 with no behavioural change to read.** The largest KL
over all bystander queries is between **8.1e-07 and 1.2e-04 nats** and top-1 agreement is **exactly
1.000** on every seed. The detector is operating three to five orders of magnitude below this
programme's own locality bar of 0.05 nats. There is no behavioural trace of the deleted fact; the
detector fires anyway.

**An intervention that deletes nothing fires harder.** `add2` adds two live rows that touch no query,
and is separable at 0.961–1.000. The programme's own pre-registered criterion `gpt2/add2/auc_ii <= 0.6`
is recorded **FAIL at 1.0000**, and `add2/auc_iii <= 0.6` **FAIL at 0.9651**.

**And across arms the score is not a function of behavioural change at all:**

| arm | max KL (nats) | AUC |
|---|---:|---:|
| `perm` | ~1e-10 | 0.47 |
| `cascade` (a complete deletion) | ~1e-05 | **0.87** |
| `add2` (no deletion) | ~1e-03 | **0.96** |
| `blank` (leaks entities) | **2.6** | 0.77 |

The arm that moves the model most (`blank`, up to 4.5 nats) scores below the arm that moves it least
of the three real interventions (`cascade`, 1e-5 nats). A statistic that ranks interventions in an
order uncorrelated with their effect on the model is not measuring the model.

## The law

An off-target residue detector over an externalised memory separates conditions by the **structural
footprint of the store**, not by residual knowledge. Two channels carry it, and they are not the same
kind of thing:

- **F1 · CARDINALITY.** The number of live rows sits in every routing softmax; the dereference
  pass-through bias is `log(n_cells)`. This is a property of dense routing, not of history. It is what
  `add2` measures, and it does **not** close: under content-derived markers `add2` stays at 0.94–1.00.
- **F2 · PROVENANCE / WRITE ORDER.** The marker draw position is a write-order channel. It is what
  `cascade` vs `never` measures at 0.836–0.888 with a max KL of 1e-04 nats or less. It **does** close:
  E-000053 re-signs markers
  from an HMAC of row content and `cascade` vs `never` falls to **exactly 0.500 on all three classes**,
  with the frozen reader accepting the re-signed bank without retraining (gate acceptance 1.000,
  KL ≤ 1e-4, top-1 agreement 1.000, accuracy moved by ≤ 0.01).

So one nuisance channel is a bug with a free fix, and the other is a floor that no fix removes and
that every reported AUC must be quoted against.

## The consequence: the detector is anti-informative

On this substrate the ordering is inverted. The lifecycle primitive that **actually fails** is `BLANK`:
a blanked alias reads UNKNOWN in only **0.825** of cases and names some entity in **0.175**, against a
pre-registered rule requiring ≥ 0.90 and ≤ 0.05. It is the one broken deletion primitive of the three.

It also has the **lowest** off-target AUC of the three lifecycle arms — 0.791, below `dangle` at 0.876
and below a complete `cascade` deletion at 0.869.

**The residue detector ranks the broken primitive as the safest.** A practitioner using off-target
detectability to choose a deletion primitive would choose the one that leaks entities.

## Why it is not owned

Checked at the source, not from memory:

- **Chen, Pal, Zhang, Qu & Liu, "Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from
  Model Outputs", ICLR 2026, arXiv:2506.14003.** Detects whether a model was unlearned "with over 90%
  accuracy even under forget-irrelevant inputs", from prediction logits, textual outputs and
  intermediate activations. Forget-irrelevant inputs are exactly the off-target regime of the table
  above. The retrieved abstract and page report **no control arm in which the model was perturbed
  structurally without unlearning, and no floor or chance baseline for the detector.**
- **Raeesi & Roed, "Auditing Forgetting in Limited Memory Language Models", arXiv:2607.00605.** The
  nearest neighbour: the model is fixed and the **database state is varied at inference**, exactly this
  design. Fetched and read: its arms are `FULL` (fact present, retrieval on), `DEL-ON` (fact deleted,
  retrieval on) and `DEL-OFF` (fact deleted, retrieval off), decomposing behaviour into parametric
  leakage, retrieval-mediated correctness and a retrieval artifact rate. It holds **no arm matched on
  the number of records**, uses **no filler/unrelated-record control**, addresses **write order,
  insertion order and provenance nowhere**, and reports **no null distribution or chance level** for a
  deleted-vs-never detector. This was the single biggest threat to the claim and it does not land.
- **Hayes, Shumailov et al., "Inexact Unlearning Needs More Careful Evaluations", arXiv:2403.01218.**
  The standard calibration paper, but its calibration is per-example difficulty under U-LiRA over
  weight-space unlearning and needs shadow models. It is not a structure-matched null and does not
  apply to a frozen reader with no training stochasticity.

What is therefore not claimed as new: that output-level forgetting is insufficient (Illusion of
Erasure, arXiv:2606.23276; J-Access, arXiv:2608.11408; the mechanistic-unlearning line); that
unlearning leaves detectable traces (Chen et al. above); that evaluations need controls at all; AUC,
Mann-Whitney, cross-validation, or nuisance-floor subtraction as statistical technique.

What is claimed as new is narrow: **on a substrate where deletion ground truth is exact, the
off-target detectability of a memory lifecycle operation is carried by the store's cardinality and
write order rather than by residual knowledge — demonstrated by a complete deletion that is separable
at 0.836–0.888 while the largest output divergence over the scored queries is 1.2e-04 nats and top-1
agreement is exactly 1.000, and by a non-deletion control that is separable at up to 1.0000 — and one
of the two channels closes to exactly chance under content-derived markers while the other does not.**

## Why this substrate can say it and the field's cannot

The claim needs a system where "the model does not know the fact" is decidable. This one is
copy-bound: E-000002 recorded that with the memory layer masked the model answers **0%** and leaks
**0%** after REVOKE, on a world re-sampled every step so facts cannot enter the weights. Knowledge
presence is therefore exactly store presence, a NEVER-memory arm is constructible, and a nuisance
channel can be separated from a residue. In weight-space unlearning none of that holds, which is why
the detectors above cannot tell the two apart — not a fault of their execution, a property of their
substrate.

## What is not claimed

- Not that Chen et al.'s number is wrong. Their substrate is weight-space unlearning of a monolithic
  model; ours is an externalised pod store, and no measurement here transfers to theirs. The claim
  against that literature is a **proposed test**, stated below, not a result.
- Not that the two floors are the only nuisance channels. They are the two this substrate exposes.
- Not anything about an adversary who can choose the marker, or who holds the bank or the checkpoint.
- Not that `add2`'s channel is unfixable in general — only that content-derived markers, which close
  F2 to exactly 0.500, leave it at 0.94–1.00.

## The protocol (the contribution a reader can use)

Any reported deletion/unlearning residue statistic over an externalised or retrieval-backed memory
should be accompanied by two structure-matched arms, and is uninterpretable without them:

1. **A cardinality-matched null** — the same statistic under a perturbation that changes the number of
   live records by the same amount and deletes nothing. Here: `add2`, which reads 0.977 mean and
   1.0000 worst seed.
2. **An order/provenance-matched null** — the same statistic with record identifiers derived from
   content rather than from write position. Here: it moves `cascade` vs `never` from 0.869 to 0.500.

A residue AUC below its own cardinality null is not evidence of residue.

## Status, and what would kill it

**Status: NOT a novelty claim. A write-up of ledger §31.41 and §31.45, killed as a claim by those
sections on the day it was drafted, and retained because the write-up and the external clearance are
still useful and because E-000058 came out of it.** The GPT-2
numbers are E-000051's record; the content-marker repair is E-000053's, and E-000053 ran on the
**synthetic** reader only.

Registered as **E-000057**, not yet run: port the content-marker repair to the frozen GPT-2 reader
(E-000051's `gpt2` arm under `MVCCStore(content_markers=True)`), three seeds. Both outcomes are
informative — if `cascade` vs `never` falls to chance there too, the two-floor protocol is validated on
a real frozen LM; if it does not, a channel other than content and marker reaches the real reader and
must be named, which is the stronger finding. Its prerequisite is the `e000020_gpt2_seed{n}.pt`
checkpoints, which are gitignored and absent and cost ~90 min per seed to retrain.

This claim is withdrawn if any of the following holds:

1. a retrieved prior work defines a cardinality-matched or order-matched null for a deletion-residue
   statistic (Raeesi & Roed was the closest and does not);
2. the `add2` channel turns out to be an artefact of this adapter's dense routing that vanishes under a
   sparse or top-k reader — that would localise it rather than kill the protocol, and must be measured;
3. E-000057 shows the GPT-2 `cascade` channel does not move under content-derived markers **and** the
   cause is a defect in the port rather than a real channel;
4. the anti-informative ordering (`blank` lowest while `blank` is the failing primitive) does not
   reproduce on a second backbone.
