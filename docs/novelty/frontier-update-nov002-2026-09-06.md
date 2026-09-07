# Frontier update after NOV-002

Date: 2026-09-06
Status: **the instrument objection is closed on its two best candidates; no invention promoted**

This supersedes the open item in `frontier-update-nov001-pdx001-2026-09-06.md`, which said the next
step was not a new candidate mechanism but a charged re-run of E-000104 and E-000105. That re-run is
done.

## Result

Both kills survive charging, for different reasons.

| | build ratio cand/gen | charged total cand | charged total gen | verdict |
|---|---:|---:|---:|---|
| E-000104 | **1.00** | 503,040 | 503,040 | KILL (exact tie) |
| E-000105 | **3.04** | 937,728 | 329,472 | KILL (candidate strictly worse) |

E-000104's arms consume the same cached environments and do arithmetic of identical shape, so there
was no asymmetry for the subsidy to hide. E-000105's arms build the sensitivity matrix by different
algorithms, and the candidate's matrix propagation costs width³ per layer against the generic's
width² per layer per pod axis — so charging the build exposes a regression the original accounting
was concealing *in the candidate's favour*.

Counters were validated against the originals operation for operation (0 disagreements), and the arms
still reach identical exact states, so this changes the accounting and not the arithmetic.

## What the last three updates add up to

1. **The screen is uncalibrated** (NOV-001). False kill on an unread coordinate, false promote on a
   strict regression, and a representation subsidy that flips a verdict. Plus a work term that
   compares a pure function with itself — now confirmed in E-000104 as well as E-000105, so both of
   those kills rested on exact-state equality alone.
2. **The subsidy did not produce the two kills that mattered most** (NOV-002). The dichotomy stands
   and now rests on measured cost on both arms.
3. **The finding that travels is the audit, and it has an open experiment** (PDX-001). The
   payload-derived index channel reproduces across three store shapes, and the tombstone case leaks
   88× while scoring 0.0000 on the top-1 metric E-000028 reported.

Taken together these say the programme's reading of its own record has been right since 2026-09-04
and the twenty-three kills are, on the evidence available, real. The screen needs fixing on its own
account — an instrument that can false-promote a regression will eventually promote one — but fixing
it is maintenance, not a route to an invention.

## Where the remaining uncertainty actually is

Not in the subsidy. Two places:

**Unread coordinates.** NOV-001's M1 and M2 defects are untested on this pair: neither E-000104 nor
E-000105 instruments memory traffic or sequential depth, so a candidate could still hold an advantage
in a coordinate no file in this programme counts. Instrumenting one of the twenty-one remaining kills
for those coordinates is the only cheap way to find out, and it is a smaller job than it sounds
because the arithmetic does not need to change — only the counters.

**The other twenty-one.** NOV-002 covers the two nominated as most informative. Nothing here licenses
a claim about the rest.

## Status

**No major invention is promoted, and the case for looking for one in this direction is weaker than
it was this morning.** The next actions, in order:

1. **Run §5 of the 2026-09-04 novelty statement.** PDX-001 made it configuration rather than a
   rewrite: supply a `Store` whose `observe()` reads a real index with its own reported deletion
   metric. Both outcomes publish. This is now the highest-value unrun experiment in the programme and
   has been since it was written.
2. **Fix the screen.** Test improvement rather than equality of counts, read every coordinate the
   mechanism moves, and charge the representation. Cheap, and it stops the next twenty-three
   verdicts inheriting a predicate that can pass a strict regression.
3. **Write the measurement-and-audit paper.** Unchanged from 2026-09-04, and now with the objection
   "your kills came from a broken screen" answered on its two strongest cases.
