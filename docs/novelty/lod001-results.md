# LOD-001 recorded: what the audit could have seen, and on which axis the question changes its answer

Date: 2026-09-06. Record: `so/results/lod001/lod001_detection_limit.{json,md}`. Three seeds, 16 pods
per seed, both address modes, both ladders, 369 seconds on 4 CPU cores. Pre-registration:
`docs/novelty/lod001-preregister.md`, committed before the substrate finished training. Prior-art
boundary: `docs/novelty/lod001-prior-art-update.md` and `docs/novelty/audit-siting-correction.md`,
both of which must be read first — **the detection floor and the dose ladder are published
instruments and are not claimed here.**

Worst seed throughout. For a floor, "worst" means LEAST SENSITIVE, and a seed on which nothing was
detected dominates.

## Validity: all six rows pass, so the pre-registered rows can be read

| row | worst seed | bar | |
|---|---|---|---|
| `V1` `alias/answer/a1` | 0.9688 | ≥ 0.80 | PASS |
| `V2` `alias/final/raw` live − never | 0.8125 | ≥ 0.30 | PASS |
| `V3a` `alias/answer` zero-dose − never | +0.0179 | ≤ 0.05 | PASS |
| `V3b` \|`alias/final/raw` zero-dose − never\| | 0.0625 | ≤ 0.10 | PASS |
| `V4a` gate at chord 0 | 0.9988 | ≥ 0.90 | PASS |
| `V4b` gate at chord 1.0 | 0.0101 | ≤ 0.10 | PASS |

## `L1` fires, and the saturation table is why it is worth more than WSC-001's version

At blocks 8 and 9 — **all five readout families, both address modes, all three seeds, both ladders** —
**no rung is detected.** The floor is above full retention: the audit could not have seen the pod with
all of it present. Worst-cell maxima at site8 are +0.022 (alias) and +0.027 (direct); at site9,
−0.004 and +0.000.

WSC-001 established the same endpoint and separated *the audit is blind here* from *nothing is here*
with a mediator that is a norm, naming that as its weakness. The absolute per-arm accuracies settle it
without a norm, and they say something the norm could not:

| mode | site | family | never | zero-dose | live | chance |
|---|---|---|---|---|---|---|
| alias | site8 | `jspace` | 0.214 | 0.134 | 0.219 | 0.062 |
| alias | site8 | `raw` | 0.536 | 0.152 | 0.540 | 0.062 |
| alias | site9 | `jspace` | 0.701 | 0.147 | 0.665 | 0.062 |
| alias | site9 | `raw` | **0.996** | 0.174 | **0.987** | 0.062 |
| alias | site10 | `jspace` | 0.076 | 0.054 | 0.978 | 0.062 |
| alias | final | `jspace` | 0.040 | 0.054 | 0.973 | 0.062 |

**The probe at blocks 8 and 9 is not blind. It is saturated.** It classifies the never-written arm at
0.214 to 0.996 against a chance of 0.062, and it classifies the live arm at the same level — at site9
the full residual reads 0.996 with no memory and 0.987 with all of it. Everything it reads is the
alias name in the prompt. An audit sited there and reported as an accuracy rather than as a contrast
would look **excellent**, and would be measuring the question.

That is a concrete false-confidence mode, and it is the reason a floor and a never-written control
have to be reported together: either alone is misleading here in opposite directions.

## `L2` holds

At block 10 and the final state the audit resolves: `jspace` alpha-floor **0.375** on every seed at
site10 (0.375 / 0.375 / 0.25 at the final state), and chord-floor 0.75 (0.8 / 0.75 / 0.75). Both
address modes agree.

## `L3` is answered **differently by the two axes**, and that is the result

Worst seed, alias mode; the direct mode gives the same ordering.

| ladder | audit (`jspace`, site10 / final) | output | in retained-payload units |
|---|---|---|---|
| write-side `alpha` | 0.375 | 0.375 (per seed 0.375 / 0.25 / 0.25) | audit 0.375, output 0.375 |
| store-side chord | 0.75 | 0.5 | **audit gate 0.4721, output gate 0.9714** |

- **On the write-side ladder: `L3c`.** The two floors are equal at the worst seed and the output is
  *better* on two of three seeds. No sentence is licensed, and none is written.
- **On the store-side ladder: `L3a`, on every seed and in both address modes.** The audit resolves a
  residue the output does not, by two rungs.

**The two axes disagree because they are different perturbations, and the mechanism is in the code.**
`alpha` scales the write toward zero, so a shrunken write still points at the right token and the
answer tracks it. The marker chord attenuates the payload **toward `' unknown'`** through the gate
(`values = payload * g + unk * (1 - g)`, `so/llm_adapter.py`), so a partially unsigned pod pushes the
answer to UNKNOWN — *which is what a successful deletion looks like* — while the payload is still
there.

Read operationally, and this is the one sentence the record supports on its own:

> On the store's own lifecycle dose, a pod whose marker is partly invalid answers UNKNOWN while
> **47%** of its payload is still reaching the model and is linearly decodable one block downstream. An
> output-based deletion check is blind across the whole range from 97% down to 47% retained payload.
> An internal audit is not — **provided it is sited past the write**, because at the two blocks before
> that its floor is above 100%.

## `L4`: the write-side ladder is not an artefact of its own arbitrariness — for the audit

Converting both floors into the gate's units: the audit's chord floor is gate **0.4721** and its alpha
floor is **0.375**, a difference of **0.097**, inside the pre-registered 0.10 agreement bar. Two
semantically different doses put the audit's floor in the same place.

They do not for the output: gate 0.9714 against alpha 0.375, a difference of **0.60**. So the
axis-invariance holds for the internal readout and fails for the behavioural one. The pre-registration
anticipated disagreement between the ladders and said the chord ladder would then carry the sentence;
what it did not anticipate is that the disagreement would be *between readouts* rather than between
axes, and that is recorded as an unregistered observation rather than as a licensed finding.

## `K2` fires in part, and it constrains every sentence above

`random` places the floor at the same rung as `jspace` at site10 on the alpha ladder (both 0.375), and
**`raw` beats `jspace`** at the final state on every seed (0.1875 against 0.375). The audit's own basis
is therefore not the best readout available at the site where it works, and **no sentence in this
document credits the Jacobian lens specifically**: every one of them is about *a readout at this site*.

## `L5`, and an unregistered asymmetry that is larger than the row it was written for

The zero-dose arm against never-written, worst seed: site8 −0.107, **site9 −0.656**, site10 −0.045,
final +0.054. As a leak row these pass — they are negative — but the magnitude at site9 is not noise
and it is not what the row was written for. A row that routes but injects nothing reads **worse** than
a store from which the row is absent, because the two banks differ in size and the reader's
dereference bias is `deref_pass_bias + log(n_cells)` (ledger §31.41, §31.45). **The zero-dose arm and
the never-written arm are not the same floor**, and any floor measured against one is not measured
against the other. `V3` checked only the final state and passed there (0.0625); it would have failed
at site9. That is a defect in the row as written, it is recorded here rather than patched, and the
floor used throughout this document is the never-written arm.

## `K3`, `K4`

`K3`: every limit is within one rung across the three seeds. `K4`: the behavioural alpha curve is
monotone on all three seeds (seed 0: 0.000, 0.000, 0.000, 0.036, 0.098, 0.210, 0.661, 0.839, 0.942,
0.969), so the rung ordering is a dose ordering and the write-side rows are read.

## What is NOT claimed

- **Not the detection floor.** Braun, arXiv:2608.12652 §3.8, publishes it for an internal-activation
  probe audit and §L3 already prescribes reporting it alongside a null.
- **Not the dose ladder.** Ferrara, arXiv:2608.20569 §3.6; ActAdd; CAA; and the workspace paper's own
  strength-scaled swaps.
- **Not the siting rule.** Gurnee et al. state it, with its two-way ambiguity, and run their own
  disambiguating experiment. `docs/novelty/audit-siting-correction.md` withdrew this programme's claim.
- **Not the per-item never-written control's existence.** Yang & Yeung, arXiv:2607.19442, build one by
  replaying training without the facts; LiRA and U-LiRA build per-example nulls by retraining.
- **Not the J-lens basis**, per `K2` above.
- Not a deletion guarantee, a certificate, a mechanism, or a novelty claim of any kind. Whether
  anything here supports one is decided in the claim document, after LOD-002 and the round's judge.
- Nothing about models above 124M, multi-token entities, free text, or any backbone but GPT-2 small.

## What a hostile reader should attack first

1. Sixteen pods per seed and three synthetic worlds. The resampling unit is the pod; the worlds are
   the replicates; a binomial interval at the row count is anti-conservative.
2. `delta = 0.30` is one bar. The floors are rungs of a ten-point ladder, so a floor is quantised and
   "0.375 against 0.375" hides everything finer than the gap between rungs.
3. The gate values that convert chord rungs into retained-payload fractions are means over pods; the
   conversion is not exact per pod.
4. The saturation finding is the strongest row here and it is the one that most obviously generalises
   beyond this substrate — which also means it is the one most likely to be already known in a form
   this session did not retrieve.
