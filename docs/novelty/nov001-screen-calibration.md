# NOV-001 — calibrate the programme's own novelty screen

Date: 2026-09-06
Status: **registration + instrument description**

## Why this exists

E-000086 through E-000108 return the same verdict twenty-three times: kill, no invention promoted.
The frontier update of 2026-09-06 states the boundary as a dichotomy — "addressable/factorized state
can be updated exactly, but generic algebra usually inherits the same advantage" — and asks the next
candidate to break it.

Nobody has asked the prior question. The verdicts all come from one screen, written the same way in
each experiment. In `so/experiments/e000105_stable_gate_cohort_transport_reduction.py`:

```python
kill = (
    validity and candidate_fresh_mismatch == 0 and generic_fresh_mismatch == 0
    and candidate_generic_mismatch == 0 and work_mismatch == 0 and aba_mismatch == 0
    and total_candidate_mult == total_generic_mult
)
```

A candidate is refused when a generic baseline, **handed the candidate's own representation**,
reaches the same exact states with the same multiply count.

This programme has a rule about instruments and has paid for it three times (ledger §31.15): *an
instrument that cannot fail is not evidence.* NOV-001 asks the mirror question, which has never been
asked here: **can this screen pass, and what is it measuring when it refuses?**

The method is the one E-000019 already established for attacks and no experiment has applied to the
screen. E-000019 reports that its probe reads live cells at 0.893–0.927, so "at chance after SHRED"
is not the artefact of a probe that never worked. That is a positive control. The screen has never
had one.

## What is registered

Five mechanisms, all exact, each run against its own generic baseline. Every number is produced by
`so/experiments/nov001_screen_calibration.py` — nothing is quoted from the literature, and the
mechanisms are named only by the shape they resemble.

| id | mechanism | advantage lives in | resembles |
|---|---|---|---|
| M1 | `tiled_streaming_accumulation` | slow-memory traffic | IO-aware / tiled kernels |
| M2 | `draft_and_verify` | sequential depth | speculative execution |
| M3 | `strassen_2x2` | multiplies | fast matrix multiplication |
| M4 | `prefix_index_range_sums` | the representation itself | E-000102 … E-000105 |
| M5 | `wasteful_recompute` | **nothing** | the null control |

Two screens are applied to each:

- **programme** — the screen as written in E-000102 … E-000106: exact states plus the multiply count,
  with the baseline handed the candidate's representation for free.
- **complete** — states must still match exactly, but the candidate must strictly *improve* at least
  one coordinate of the full cost vector `(multiplies, slow_memory_words, sequential_rounds)`, with
  the cost of building a representation charged to whoever builds it.

## Registered predictions and kill rules

1. **M3 must pass the programme screen.** If it does not, the screen is vacuous rather than
   restricted, and the finding is a different and larger one.
2. **M5 must be refused by any screen worth having.** It computes an identical result with strictly
   more arithmetic and an advantage nowhere.
3. The instruments must not be the same function. If the two screens agree on all five mechanisms,
   NOV-001 has found nothing and the programme's kills stand as written.

## Cost-model scope, stated up front

The complete screen reads three coordinates. That is not the full cost model of any real system —
communication volume, cache behaviour, energy, and numerical conditioning are all absent. The claim
NOV-001 can support is therefore strictly one-directional: a coordinate the screen does not read is a
verdict it cannot justify. Adding coordinates can only find more false kills, never fewer.

## Honest provenance of this registration

This document was written alongside the code in one session, not sealed before it. Recording that is
the point of §31.15. Two things changed once the first numbers were read, and both are recorded
rather than absorbed:

- **M2 did not behave as predicted.** It was registered expecting a false kill — identical states,
  advantage in sequential depth. It instead *passes* the programme screen, because it spends more
  multiplies than its baseline and the predicate tests equality of counts, not improvement. That is a
  defect in the opposite direction from the one expected, and it is the reason M5 was added: a clean
  witness was needed for "the screen passes things it should refuse".
- **The complete screen was rewritten once.** Its first form killed on any cost difference, which
  would have refused M2's genuine trade (fewer rounds, more multiplies). It now promotes on a strict
  improvement in at least one coordinate.

Neither change was made after seeing a verdict the author preferred; both were made because the first
draft could not express the distinction the data forced. The result document reports what the current
code produces, and the code is in the diff.

## What this cannot show

NOV-001 does not re-run E-000086 … E-000108 and makes **no claim that any specific prior kill
reverses.** Doing that requires re-instrumenting each experiment with its own cost vector, which is
the follow-on work, not this. It also makes no claim about the standing of any published mechanism
the five resemble; the resemblance is a naming convenience and every verdict here comes from this
file's own counters.
