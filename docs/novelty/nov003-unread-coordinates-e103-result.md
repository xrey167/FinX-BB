# NOV-003 result — E-000103's verdict was underdetermined, and the screen can be unfair in the other direction too

Date: 2026-09-06
Decision: **CROSSOVER_AGGREGATE_VERDICT_IS_NOT_A_VERDICT**
Major-invention claim: **NO** — and this one does not promote a mechanism either

## Registration

`frontier-update-nov002-2026-09-06.md` names this under "Where the remaining uncertainty actually
is":

> **Unread coordinates.** NOV-001's M1 and M2 defects are untested on this pair: neither E-000104 nor
> E-000105 instruments memory traffic or sequential depth, so a candidate could still hold an
> advantage in a coordinate no file in this programme counts. Instrumenting one of the twenty-one
> remaining kills for those coordinates is the only cheap way to find out.

E-000103 was chosen because its arms differ *visibly* on both: two sequential rank-1
Sherman-Morrison updates against one rank-2 Woodbury block, and a candidate that carries `z_del`,
`z_add` and two denominators between events.

## Two findings, and the first is about screens rather than about E-000103

### 1. The baseline as written is not a baseline

`woodbury_rankk_solution` rebuilds `inv_u` and `middle_inv` on **every one of the 24 sessions**,
though neither depends on that session's right-hand side `b`. Against a baseline that redoes
loop-invariant work every iteration, the candidate wins by 3.5×–4.2× on arithmetic without the
mechanism doing anything at all:

| n | candidate | generic hoisted | generic **as written** |
|---:|---:|---:|---:|
| 4 | 568 | 616 | **1,960** |
| 6 | 924 | 944 | **3,440** |
| 8 | 1,344 | 1,320 | **5,352** |
| 10 | 1,828 | 1,744 | **7,696** |

This is the **mirror of the representation subsidy**. There, the screen hands the baseline the
candidate's representation for free. Here, it denies the baseline an optimisation the candidate is
already using — the candidate caches `z` and `denom` across sessions precisely so it does not repeat
that work. A screen can be unfair in both directions, and this programme had only ever looked for
one of them.

Hoisting costs the baseline no extra arithmetic at all: `woodbury_rankk_inverse` already computes
`inv_u` and `middle_inv` internally (e000103 lines 121–125), so a hoisted implementation retains them
rather than rebuilding. It buys resident state, not multiplies.

### 2. Against a fair baseline there is no single verdict

| n | candidate | generic hoisted | verdict | improved / regressed |
|---:|---:|---:|---|---|
| 4 | 568 | 616 | **PROMOTE** | multiplies / sequential_rounds |
| 6 | 924 | 944 | **PROMOTE** | multiplies / sequential_rounds |
| 8 | 1,344 | 1,320 | **KILL** | — / multiplies, sequential_rounds |
| 10 | 1,828 | 1,744 | **KILL** | — / multiplies, sequential_rounds |

The candidate costs `8n² + 98n + 48` against the hoisted baseline's `6n² + 104n + 104`: a worse
quadratic term bought against a better linear one, so it wins while the sessions dominate and loses
once the dimension does. **The crossover sits between n = 6 and n = 8, inside E-000103's own
registered domain of (4, 6, 8, 10).**

The aggregate over all four dimensions returns KILL, and that number should not be read as the
answer. It is an artefact of which dimensions the domain happens to contain — a domain weighted
toward small n would have returned PROMOTE with equal authority. **E-000103's kill was not wrong so
much as underdetermined: it reported one verdict for a comparison that does not have one.**

## A defect this experiment found in its own instrument

The first run of this file screened `slow_memory_words` and returned PROMOTE at every dimension —
including n = 8 and n = 10, where the candidate spends *more* arithmetic. The cause: every arm's
resident state is `n² + 2n + O(1)` (the candidate keeps the inverse plus `z_del`, `z_add` and two
denominators; the hoisted baseline keeps the inverse plus `inv_u` and `middle_inv`), and the two
differ by a **constant 2 words**. `so.screen` promotes on any strict improvement, so a 2-word gap
outranked a 484-multiply regression.

A coordinate a model cannot resolve must be **declared unread, not scored**, or the screen
manufactures an advantage out of its own rounding — which is the class of defect NOV-001 was about,
committed by the file auditing it. `slow_memory_words` is therefore recorded in the result and
excluded from the screen, and it appears in `unread_coordinates` on every verdict so the omission is
visible rather than silent. This is the first use of that field and it is what it was added for.

The general lesson is one `so/screen.py` does not yet enforce: **a strict improvement is not the same
as a material one.** The screen has no notion of margin, and until it does, "improved on at least one
coordinate" can be satisfied by noise.

## What this settles

The unread coordinates **do** change what an experiment says — that half of the frontier update's
open question is answered, and answered yes. But not by revealing a hidden advantage. What they
revealed here is that E-000103's comparison has a crossover its single verdict could not express, and
that its baseline was carrying avoidable work.

**No mechanism is promoted.** A candidate that wins at small n, loses at large n, and costs roughly
twice the sequential depth throughout is not an invention, and nothing here suggests otherwise.

## Limits

One of the twenty-one kills NOV-002 did not cover; nothing here licenses a claim about the other
twenty. Operation counts are analytic in n and k, derived from the source of each routine in
e000103 rather than from instrumented execution — coarser than the exact per-cell counters NOV-002
validated against the originals, and the reason the `slow_memory_words` constant could not be
trusted. `sequential_rounds` counts dependent steps, not wall-clock, and takes no position on whether
the depth difference matters at these sizes.
