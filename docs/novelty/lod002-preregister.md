# LOD-002 — does the audit's blind region follow the model's depth, or the memory's write site?

Date: 2026-09-06
Status: **pre-registered before this experiment has produced a number.** It exists to decide a
sentence this repository is currently asserting on one adapter's worth of evidence, and it can refute
that sentence.
Classification: a controlled substitution. **No novelty credit** for the lens, lens-based auditing,
probing, dose ladders, detection floors, siting rules, external memory, pods or pointer aliases — see
`docs/novelty/lod001-prior-art-update.md` and `docs/novelty/audit-siting-correction.md` for the full
boundary, both of which must be read before this file.

## The sentence on trial

`docs/novelty/audit-siting-correction.md` withdrew this programme's claim to the siting *rule*,
because the lens's own authors state it. Verbatim, Gurnee et al.:

> Note that in roughly the first third of the model, the readouts are noisy and largely
> uninterpretable

> the absence of meaningful J-lens-accessible content in the first third of the model could indicate
> that either (1) the J-lens is degenerate at these depths and fails to resolve content that is in
> fact present, or (2) the early-layer residual stream genuinely carries no linearly accessible and
> causally relevant verbalizable content.

That is a rule about **depth in the model**, published, with its own disambiguating experiment. The
one sentence the correction kept is different:

> For an external memory the binding constraint is not depth but the site of the **write**, which is a
> configuration parameter and can be moved without touching the model.

It is currently an **inference from a single adapter**, and this experiment is the measurement that
decides it. WSC-001's blind block is 8 of GPT-2 small's 12 — two thirds of the way down, well outside
Gurnee's first third — which is suggestive and is not proof: nothing so far has moved the write while
holding the model fixed.

## The design: one field changed

The identical frozen GPT-2 small, the identical recipe (`E20.train_or_load`, 3000 steps, BOS on every
prompt), the identical world and store construction, the identical ladder, the identical five readout
families, the identical probes trained on the live arm only. One field of `AdapterConfig` differs:

| arm | `read_layers` | capture sites | record |
|---|---|---|---|
| LOD-001 (the comparison) | `(8, 10)` | 8, 9, 10, final | `so/results/lod001/` |
| **LOD-002** | `(5, 7)` | 5, 6, 7, 8, 9, 10, final | `so/results/lod002/` |

Block 8 is the discriminator. On the comparison arm it is the FIRST read layer, where WSC-001 measured
the write to be small. On this arm it is DOWNSTREAM of both read layers. Model depth at block 8 is
identical, to the weight.

## The two outcomes, written before the run

- `W1` **the write.** On the (5, 7) arm, `alias/site8/jspace/LOD_alpha.detected == true` — the audit is
  SIGHTED at the same block where the comparison arm is blind → *the blind region follows where the
  memory is written, not the model's depth, and it is therefore a property the system's designer
  chooses.*
- `W2` **depth.** On the (5, 7) arm, `alias/site8` is undetected at every rung by every family, with
  the validity row passing → *the blind region is a property of the model at that depth. The sentence
  in `audit-siting-correction.md` is WITHDRAWN, not softened, and this programme's siting result
  reduces entirely to Gurnee et al.*
- `W3` **mixed** — some families detected at site8 and others not → no sentence; the table, and the
  per-family split reported as the finding.

Reported beside them, with no bar: the first site at which the floor becomes finite on each arm,
against that arm's read layers. The comparison arm's answer is `site10`, its SECOND read layer, so the
prediction on this arm is `site7` and not `site5` — the analogue, not the first read site. A result of
`site5` would be as interesting as `site7` and is not scored either way, because nothing in the record
establishes which of the two read sites carries the content on a moved pair.

## Validity rows — a failure VOIDS the experiment, and this is the likely failure

- `V1` `alias/answer/a1 >= 0.80` on the (5, 7) arm. **A shallower reader may simply fail to place a
  readable memory**, and a reader that cannot read is not an audit that cannot see. If this fails the
  experiment is VOID and reports only that the recipe does not transfer to these read layers.
- `V2` `alias/final/raw/a1_minus_never >= 0.30` — some readout somewhere sees the live memory.
- `V3` the zero-dose anchor, as in LOD-001: `alias/answer/a0 − alias/answer/never <= 0.05`.
- `V4` the store-side ladder moves the gate: `gate_by_chord[c=0] >= 0.90`, `gate_by_chord[c=1.0] <= 0.10`.

## By construction — declared, never scored

- Block 8 is downstream of read layers 5 and 7 **by arithmetic**, not by measurement. What is measured
  is whether being downstream is what makes the audit sighted there.
- The two arms are different trained adapters, so their absolute reading accuracies will differ; every
  comparison below is between each arm's own live and never controls, never between arms' raw numbers.
- `site5` being small is the analogue of WSC-001's first-read-site finding and is expected; it is not
  evidence for anything on its own, which is why the scored cell is `site8` and not `site5`.

## Kill conditions

1. `V1` fails → VOID. The recipe does not transfer; nothing is read from this arm.
2. `W2` fires → the kept sentence of `audit-siting-correction.md` is withdrawn in place, and this file
   is cited at the site of the withdrawal.
3. Single seed → **no sentence carries beyond "on one seed"** until seeds 1 and 2 exist. The record
   states the seed count in its first line.
4. If the comparison arm's own `site8` result does not reproduce on the recorded three-seed LOD-001 run,
   the discriminator has no baseline and this experiment is uninterpretable.

## What is NOT claimed

- Not the siting rule. Gurnee et al. own it; the correction document withdraws this programme's claim.
- Not that read layers (5, 7) are better or worse than (8, 10) for anything.
- Not a mechanism, an instrument, a certificate or a deletion result.
- Not anything about models above 124M, multi-token entities, free text, or any backbone but GPT-2
  small.

## Reproduce

```bash
SO_BOS=1 SO_CKPT_SUFFIX=_bos_L57 python -m so.experiments.lod002_write_site --seeds 0 --steps 3000
```
