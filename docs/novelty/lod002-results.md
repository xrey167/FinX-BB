# LOD-002 recorded (seed 0): the audit's blind region follows the memory's write site, not the model's depth

Date: 2026-09-06. Records: `so/results/lod002/lod002_write_site.{json,md}` (**three seeds**, 5193 s
including training) and `so/results/lod002/lod002_write_site-seed0only.{json,md}` (the seed-0 run this
document first reported, kept so the earlier version can be checked).
16 pods per seed, both address modes, both ladders.
Pre-registration: `docs/novelty/lod002-preregister.md`, committed before the arm was trained.

**Kill condition 3 is discharged: seeds 1 and 2 landed and `W1` holds on all three, in both address
modes.** One sentence written from the seed-0 run is **corrected below** by the other two seeds.

## `W1` fires

All six validity rows pass, and the one that could have voided the experiment passes comfortably:
`V1 alias/answer/a1 = 0.9643` — the shallow reader reads the memory. So a blind audit on this arm
would have been a blind audit, not a reader that cannot read.

`jspace` alpha-floor per site, the two arms side by side. Same frozen GPT-2 small, same recipe, same
world, same ladder, same five families, same probes. One field differs: `read_layers`.

| site | arm A: `read_layers=(8,10)` (LOD-001, 3 seeds) | arm B: `read_layers=(5,7)` (3 seeds) |
|---|---|---|
| `site5` | — | **> 1** — blind, every seed |
| `site6` | — | **> 1** — blind, every seed |
| `site7` | — | 0.375 / 0.375 / 0.375 |
| **`site8`** | **> 1 — blind, every seed** | **0.375 / 0.375 / 0.375 — sighted** |
| `site9` | **> 1** — blind, every seed | 0.375 / 0.375 / 0.375 |
| `site10` | 0.375 | 0.375 / 0.375 / 0.375 |
| `final` | 0.375 | 0.375 / 0.375 / 0.375 |

Every cell above is the same on all three seeds of each arm and in both address modes, with two
exceptions, both in direct mode and both still SIGHTED: arm B's `site8` reads 0.5 on seed 0 and its
`final` reads 0.5 on seed 1. Validity, worst of three seeds: `V1 alias/answer/a1 = 0.9196`.

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
| write magnitude at the FIRST read site, per seed | 3.746 / 0.479 / 2.480 | 3.745 / 2.403 / 4.489 |
| write magnitude at the SECOND read site, per seed | 61.4 / 64.9 / 76.9 | 27.5 / 34.7 / 31.3 |
| **first / second, per seed** | **0.061 / 0.007 / 0.032** | **0.136 / 0.069 / 0.143** |
| probe at the blind blocks, never-written | 0.214 / 0.701 | 0.29, 0.23, 0.24 / 0.70, 0.54, 0.64 |
| probe at the blind blocks, live | 0.219 / 0.665 | 0.27, 0.21, 0.24 / 0.67, 0.50, 0.63 |
| probe at the first sighted block, never / live | 0.076 / 0.978 | 0.05, 0.06, 0.08 / 0.97, 0.95, 0.96 |

**A sentence from the seed-0 run is CORRECTED here.** That version read: *"The first read site's write
magnitude is 3.746 on arm A and 3.745 on arm B — a property of being the first read site."* With three
seeds per arm that is **a coincidence of seed 0 and is withdrawn.** The per-seed magnitudes are
0.479 to 3.746 on arm A and 2.403 to 4.489 on arm B; they overlap, they are not equal, and no sentence
should have been built on two numbers agreeing to three decimal places. What survives is the weaker
and still sufficient statement: **on both arms the first read site carries a small fraction of the
second's write — 0.007 to 0.061 on arm A, 0.069 to 0.143 on arm B — and on both arms the audit is
blind exactly there.** Arm B's fraction is the larger of the two by roughly 2 to 10×, which is
reported and not explained.

**The saturation signature translates, on every seed.** At its blind blocks each arm's probe classifies the
never-written arm at 0.21–0.70 against a chance of 0.062 and classifies the live arm at the same level
— on all three seeds of arm B (site5 never/live 0.286/0.272, 0.228/0.205, 0.241/0.241; site6
0.696/0.670, 0.540/0.496, 0.638/0.634). Both arms are saturated by prompt identity exactly where they
are blind, and both drop to near chance on the never arm (0.049–0.080) exactly where they start
reading the memory (live 0.946–0.969). This is the LOD-001 finding reproduced on three independently
trained adapters with different read layers.

## The store-side ladder agrees, and more strongly than on arm A

`jspace` chord-floor: **0.825** at sites 7, 8, 9 and 10, and 0.75 at the final state, against the
answer's **0.7**. The audit again resolves a residue the output does not, and by more rungs than on
arm A. The alpha-ladder comparison again ties at 0.375 against 0.375. Both axes reproduce their arm-A
behaviour, including their disagreement with each other.

## What this licenses

On three seeds per arm, both address modes, with every validity row passing:

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
- Nothing about models above 124M, multi-token entities, free text, or any backbone but GPT-2 small.

## How it could be wrong

1. Three seeds per arm, one synthetic world per seed, 16 pods. The resampling unit is the pod and the
   worlds are the replicates.
2. The two arms are different trained adapters. Every comparison above is within-arm (live against
   never-written on the same arm); no raw number is compared across arms except the write magnitudes,
   which are the mediator and are reported as such.
3. `(5, 7)` is one alternative placement. Two points make a line only if you already believe it is one;
   a third placement would test the translation properly and has not been run.
4. Two direct-mode cells on arm B read 0.5 rather than 0.375 — one rung worse, still detected.
   Reported, not explained.
5. The seed-0 version of this document asserted an equality of write magnitudes that two further seeds
   withdrew. It is corrected in place above rather than quietly edited, and it is a reminder that a
   two-number coincidence is not a structure.
