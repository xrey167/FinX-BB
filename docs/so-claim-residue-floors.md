# The claim: a deletion-residue detector reads the store's shape, not the model's memory

*2026-09-05. One claim, its evidence, and the prior art it has to survive. Every number below is
read from a record in this repository; every external citation was fetched at the source and the
sentence relied on is quoted.*

## The measurement

E-000051 put a query-only adversary against a frozen GPT-2 reading a canonical-pod store, three
seeds, 100 pods per seed, training nothing. It compared lifecycle states of a pod and scored a
five-fold cross-validated Mann-Whitney AUC on queries **that are not about the pod** — class (ii)
bystanders, class (iii) generic text. The null band at n = 200 is about 0.42–0.58.

| arm (positive vs reference) | what actually changed | AUC (ii) | max KL (ii) | top-1 agree (ii) |
|---|---|---:|---:|---:|
| `cascade` vs `never` | a **complete deletion** | **0.869** | **0.000** | **1.000** |
| `blank` vs `cascade` | SET NULL on an alias, 2 residue rows | 0.791 | 4.486 | 0.999 |
| `dangle` vs `cascade` | object evicted, 2 tombstone rows | 0.876 | 3.962 | 0.999 |
| `blank` vs `never` | as above | 0.890 | 4.486 | 0.999 |
| `dangle` vs `never` | as above | 0.925 | 3.961 | 0.999 |
| `perm` vs `perm2` | nothing (summation order) — the float floor | 0.482 | 0.000 | 1.000 |
| **`add2` vs `perm`** | **two fresh live rows. NO deletion.** | **0.977** | 0.310 | 0.999 |

Read the first row and the last row together.

**A complete deletion is separable at 0.869 while the reader's output distribution is bit-identical**
— max KL exactly 0.000, top-1 agreement exactly 1.000. Nothing about the model's behaviour changed.
The detector is not reading a trace of the deleted fact; there is no behavioural trace to read.

**An intervention that deletes nothing is separable at 0.977, and at 1.0000 on the worst seed.** The
programme's own pre-registered criterion `gpt2/add2/auc_ii <= 0.6` is recorded **FAIL at 1.0000**, and
`add2/auc_iii <= 0.6` **FAIL at 0.9651**. Adding two live rows that touch nothing, and that leave
top-1 agreement at 0.999, is more detectable than deleting the fact.

## The law

An off-target residue detector over an externalised memory separates conditions by the **structural
footprint of the store**, not by residual knowledge. Two channels carry it, and they are not the same
kind of thing:

- **F1 · CARDINALITY.** The number of live rows sits in every routing softmax; the dereference
  pass-through bias is `log(n_cells)`. This is a property of dense routing, not of history. It is what
  `add2` measures, and it does **not** close: under content-derived markers `add2` stays at 0.94–1.00.
- **F2 · PROVENANCE / WRITE ORDER.** The marker draw position is a write-order channel. It is what
  `cascade` vs `never` measures at 0.869 with KL 0.000. It **does** close: E-000053 re-signs markers
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
at 0.869 with a bit-identical output distribution, and by a non-deletion control that is separable at
1.0000 — and one of the two channels closes to exactly chance under content-derived markers while the
other does not.**

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

**Status: a candidate claim resting on recorded measurements, not yet a novelty claim.** The GPT-2
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
