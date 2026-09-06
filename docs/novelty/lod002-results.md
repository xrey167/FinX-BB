# LOD-002 recorded (seed 0): the audit's blind region follows the memory's write site, not the model's depth

Date: 2026-09-06. Record: `so/results/lod002/lod002_write_site-seed0only.{json,md}`.
**One seed**, 16 pods, both address modes, both ladders, 2339 seconds including training.
Pre-registration: `docs/novelty/lod002-preregister.md`, committed before the arm was trained.
Kill condition 3 of that file applies and is honoured throughout: **no sentence here carries beyond
"on one seed"** until seeds 1 and 2 exist. They are running.

## `W1` fires

All six validity rows pass, and the one that could have voided the experiment passes comfortably:
`V1 alias/answer/a1 = 0.9643` — the shallow reader reads the memory. So a blind audit on this arm
would have been a blind audit, not a reader that cannot read.

`jspace` alpha-floor per site, the two arms side by side. Same frozen GPT-2 small, same recipe, same
world, same ladder, same five families, same probes. One field differs: `read_layers`.

| site | arm A: `read_layers=(8,10)` (LOD-001, 3 seeds) | arm B: `read_layers=(5,7)` (this, 1 seed) |
|---|---|---|
| `site5` | — | **> 1** — blind |
| `site6` | — | **> 1** — blind |
| `site7` | — | 0.375 |
| **`site8`** | **> 1 — blind** | **0.375 — sighted** |
| `site9` | **> 1** — blind | 0.375 |
| `site10` | 0.375 | 0.375 |
| `final` | 0.375 | 0.375 |

**At the identical block 8 of the identical frozen model, the audit is blind on one arm and sighted on
the other.** Depth is held fixed to the weight. The only thing that moved is where the memory is
written.

So `W2` — *the blind region is a property of the model at that depth* — is refuted on this seed, and
the sentence kept in `docs/novelty/audit-siting-correction.md` stands rather than being withdrawn:

> Gurnee et al.'s rule is about depth in the model. For an external memory the binding constraint is
> the site of the **write**, which is a configuration parameter, so the audit's blind region is chosen
> by whoever configures the memory — not by the model.

## The structure does not merely move, it translates

Every feature of arm A's blind region reappears on arm B, shifted by exactly the read-layer offset.

| | arm A `(8,10)` | arm B `(5,7)` |
|---|---|---|
| blind blocks | 8, 9 | 5, 6 |
| first sighted block | 10 = second read layer | 7 = second read layer |
| write magnitude at the FIRST read site | **3.746** | **3.745** |
| write magnitude at the SECOND read site | 76.865 | 27.496 |
| probe at the blind blocks, never-written | 0.214 / 0.701 | 0.286 / 0.696 |
| probe at the blind blocks, live | 0.219 / 0.665 | 0.272 / 0.670 |
| probe at the first sighted block, never / live | 0.076 / 0.978 | 0.049 / 0.969 |

Two things in that table are worth more than the rest.

**The first read site's write magnitude is 3.746 on arm A and 3.745 on arm B.** It is a property of
*being the first read site*, not of block 8 or block 5. The adapter puts almost nothing pod-specific
into the residual at its first read layer wherever that layer is, and resolves the content at its
second.

**The saturation signature translates too.** At its blind blocks each arm's probe classifies the
never-written arm at 0.21–0.70 against a chance of 0.062 and classifies the live arm at the same
level. Both arms are saturated by prompt identity exactly where they are blind, and both drop to near
chance on the never arm (0.049–0.076) exactly where they start reading the memory. This is the
LOD-001 finding reproduced on an independently trained adapter with different read layers.

## The store-side ladder agrees, and more strongly than on arm A

`jspace` chord-floor: **0.825** at sites 7, 8, 9 and 10, and 0.75 at the final state, against the
answer's **0.7**. The audit again resolves a residue the output does not, and by more rungs than on
arm A. The alpha-ladder comparison again ties at 0.375 against 0.375. Both axes reproduce their arm-A
behaviour, including their disagreement with each other.

## What this licenses, stated at the size one seed allows

On one seed, with every validity row passing:

> An accessibility audit of an external memory has a blind region, the blind region sits at and just
> after the memory's **first write site** wherever that site is placed, and moving the write moves it —
> in the same frozen model, at the same depth, with the same weights. The audit's floor there is above
> full retention, and the probe is not blind but **saturated by the prompt**, so an audit reported as
> an accuracy rather than as a contrast against a never-written control would read 0.21 to 0.70 there
> and mean nothing.

The operational consequence, which is the reason this was worth measuring: **where a deletion audit
can see is a configuration choice of the memory system, not a property of the model.** A composed
store-and-audit certificate can therefore be made to return "no trace" by siting alone, with the
model, the store, the deletion and the behaviour all unchanged — and the thing that detects it is the
floor together with the never-written contrast, neither alone.

## What is NOT claimed

- **Not the siting rule**, and not the observation that early lens readouts are uninterpretable:
  Gurnee et al. own both, verbatim, together with a disambiguating experiment of their own. What is
  measured here is that for an external memory the constraint is not depth.
- **Not the detection floor** (Braun, arXiv:2608.12652 §3.8) or **the dose ladder** (Ferrara,
  arXiv:2608.20569 §3.6; ActAdd; CAA).
- **Not the J-lens basis:** `K2` of LOD-001 fires in part and the same caution applies here; every
  sentence above is about *a readout at a site*.
- Not a deletion guarantee, a certificate, or a mechanism.
- **Not three seeds.** One. Kill condition 3 binds until seeds 1 and 2 land.
- Nothing about models above 124M, multi-token entities, free text, or any backbone but GPT-2 small.

## How it could be wrong

1. One seed, one world, 16 pods. Arm A is three seeds; the comparison is therefore 3-against-1.
2. The two arms are different trained adapters. Every comparison above is within-arm (live against
   never-written on the same arm); no raw number is compared across arms except the write magnitudes,
   which are the mediator and are reported as such.
3. `(5, 7)` is one alternative placement. Two points make a line only if you already believe it is one;
   a third placement would test the translation properly and has not been run.
4. Direct-mode `site8` on arm B reads 0.5 rather than 0.375 — one rung worse than alias mode, still
   detected. Reported, not explained.
