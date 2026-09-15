# NOV-001 result — the screen is uncalibrated, and it is uncalibrated in three separate ways

Date: 2026-09-06
Decision: **SCREEN_UNCALIBRATED**
Major-invention claim: **NO** — this is an instrument result, not a mechanism

## What was run

`so/experiments/nov001_screen_calibration.py`, stdlib only, exact rational arithmetic, no seeds that
matter and no sampling. Focused regressions: `so/tests/test_nov001_screen_calibration.py`, 12 tests.
Reproduce with `make calibrate`.

## The table

| id | mechanism | advantage in | programme | complete | improved / regressed |
|---|---|---|---|---|---|
| M1 | tiled_streaming_accumulation | slow-memory traffic | **KILL** | PROMOTE | `slow_memory_words` / — |
| M2 | draft_and_verify | sequential depth | PROMOTE | PROMOTE | `sequential_rounds` / `multiplies` |
| M3 | strassen_2x2 | multiplies | PROMOTE | PROMOTE | `multiplies` / — |
| M4 | prefix_index_range_sums | the representation | **KILL** | KILL | — / `multiplies` |
| M5 | wasteful_recompute | nothing | **PROMOTE** | KILL | — / `multiplies` |

Every mechanism reaches states **exactly identical** to its baseline; `exhaustive_state_mismatches`
is 0 for all five, and M2's is checked over its whole registered domain of 189 sequences rather than
sampled. So in every row the screen's state test passes and the verdict turns entirely on cost.

## Three defects, each with a witness

**1. False kill — the screen refuses a mechanism that wins on a coordinate it does not read.**

M1 computes an identical result and ties on multiplies at 72 = 72. It moves **36 slow-memory words
against the baseline's 60**. The programme screen sees two equal multiply counts and identical states
and returns KILL. The advantage is real, exact, and invisible.

**2. False promote — the predicate tests equality of counts, not improvement.**

M5 computes the identical value with **32 multiplies against the baseline's 16**, and is better at
nothing on any coordinate. The programme screen returns **PROMOTE**, because `total_candidate_mult ==
total_generic_mult` is false. A mechanism that is strictly worse and useful for nothing escapes the
kill on the same rule that refuses M1.

This is why M2 passes too, and it is worth separating: M2's PROMOTE is the right verdict reached for
the wrong reason. It offers a genuine trade — 1,000 sequential rounds against 1,134, bought with
9,564 multiplies against 4,536 — but the screen promotes it because the multiply counts differ, not
because the rounds improved. Flip the sign of that trade and the screen would promote it just the
same.

**3. Representation subsidy — the accounting decides the verdict, not the mechanism.**

M4's entire advantage is a precomputed difference table. Same states, same 22 queries, two
accountings:

| baseline | candidate cost | baseline cost | verdict |
|---|---|---|---|
| handed the table for free (the registered convention) | 16 build multiplies | 0 | **KILL** |
| charged for its own scan | 16 build multiplies | 121 multiplies | **PROMOTE** |

`subsidy_flips_verdict` is `true`. Nothing about the mechanism changed between those rows. Under the
registered convention, any mechanism whose advantage *is* its representation ties by construction —
and E-000102 through E-000105 are all mechanisms of exactly that kind.

## The E-000105 work terms cannot fail

Separately from the five mechanisms, the kill predicate in E-000105 was audited directly. Lines
176–178 of `so/experiments/e000105_stable_gate_cohort_transport_reduction.py`:

```python
candidate_work = mutation_multiplies(net)
generic_work = mutation_multiplies(net)
work_mismatch += int(candidate_work != generic_work)
```

`mutation_multiplies` is `net["out_dim"] * net["pod_dim"]` — a pure function of `net`, called twice
on the same `net`. Over 512 constructed inputs, `observed_mismatches` is **0** and
`mismatch_is_reachable` is **false**. `total_candidate_mult` and `total_generic_mult` are then both
`event_count * codes * <that same value>`, so they are equal for the same reason.

Both work terms in that kill predicate are tautologies. E-000105's kill rests entirely on exact-state
equality, and the frontier update's summary of it — "obtains exactly the same states **and exactly
the same mutation work**" — reports as a measurement something the code could not have measured. This
is the fourth time §31.15's rule has been paid for, and the first time on the screen rather than on
an audit instrument.

## What this means for the twenty-three kills

A KILL from this screen means: *no advantage in exact states or multiply count, against a baseline
holding my own representation.* It does not mean "not novel."

That is a narrow finding and it should be read narrowly. The screen is **not unpassable** — it
promotes M3, a pure arithmetic schedule with no representation to subsidise. Mechanisms of that
shape can pass. But every candidate from E-000102 onward is representation-based, which is precisely
the class the subsidy convention ties by construction, and E-000104 and E-000105 were both killed
after reporting large arithmetic reductions in their own registered domains (3.0x–9.8x and
89.78x–541.33x).

**No prior kill is claimed to reverse.** NOV-001 does not re-run any of them. What it establishes is
that the twenty-three verdicts were produced by an instrument that has never had a positive control,
that refuses on one coordinate of a cost model with at least three, that passes strict regressions,
and that in at least one case rested on a comparison of a function with itself.

## What would settle it

Re-run E-000104 and E-000105 — the two that reported the largest arithmetic reductions before being
killed — with the full cost vector recorded on both arms and the representation charged to whoever
builds it. Those two are chosen because they are the cases where the gap between "reported a large
reduction" and "was killed anyway" is widest, so they carry the most information about whether the
subsidy is doing the work. If they still kill under the complete screen, the dichotomy in the
frontier update is real and NOV-001 has only cleaned an instrument. If either survives, twenty-three
verdicts need re-reading and the programme has been refusing its own results.

## Limits

Three cost coordinates is not a full cost model; communication, cache behaviour, energy and numerical
conditioning are absent, so this can only ever find *more* false kills, never fewer. The five
mechanisms are toys chosen for exactness, not implementations. The resemblances named in the code are
a naming convenience and carry no claim about the standing of any published system. And the
registration for this experiment was written alongside the code rather than sealed before it, with
the two mid-flight changes recorded in `nov001-screen-calibration.md` rather than absorbed.
