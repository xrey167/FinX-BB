# Frontier update after NOV-001 / PDX-001

Date: 2026-09-06
Status: **instrument correction + a recovered open experiment; no invention promoted**

This updates `frontier-update-e104-e105-2026-09-06.md`, which closed with:

> **No major invention is promoted.** The increasingly consistent boundary is: addressable/factorized
> state can be updated exactly, but generic algebra usually inherits the same advantage […] The next
> invention candidate must break that dichotomy rather than rename one side of it.

Two results say that conclusion was reached with an instrument that had never been calibrated, and
that the programme's most transferable finding has an open experiment attached to it that nobody ran.

## 1. The screen that produced twenty-three kills is uncalibrated

NOV-001 ran mechanisms of known standing through the reduction screen as written in E-000102 …
E-000106. Three defects, each with an exact witness:

- **False kill.** A mechanism with identical exact states, an identical multiply count (72 = 72) and
  strictly less memory traffic (36 words against 60) is refused, because the coordinate it wins on is
  not read.
- **False promote.** A mechanism computing the identical result with strictly more arithmetic (32
  against 16) and an advantage nowhere is passed, because the predicate tests whether the counts
  *differ*, not whether the candidate is *better*.
- **Representation subsidy.** Handed its own precomputed table, a baseline ties and the candidate is
  killed; charged for building it, the same candidate on the same states and the same queries is
  promoted (16 against 121). `subsidy_flips_verdict = true`.

And directly in the E-000105 source, lines 176–178: `candidate_work` and `generic_work` are both
`mutation_multiplies(net)` on the same `net`. Over 512 inputs the mismatch is 0 and unreachable. Both
work terms in that kill predicate are tautologies; the kill rests on exact-state equality alone, and
this update's predecessor reported "exactly the same mutation work" as a measured quantity.

**§31.15's rule, for the fourth time, now applied to the screen rather than to an audit instrument:
an instrument that cannot fail is not evidence.**

### What this does and does not change

It does **not** reverse any kill. NOV-001 does not re-run E-000086 … E-000108, and the dichotomy above
may well survive. The screen is genuinely passable — it promotes a pure arithmetic schedule.

It does mean the twenty-three verdicts were produced by an instrument that reads one coordinate of a
cost model with at least three, that passes strict regressions, that subsidises the baseline with the
candidate's own representation, and that in at least one case compared a function with itself. Every
candidate from E-000102 onward is representation-based — exactly the class the subsidy ties by
construction.

**The next step is not a new candidate mechanism.** It is to re-run E-000104 and E-000105 with the
full cost vector recorded on both arms and the representation charged to whoever builds it. Those two
are chosen because each reported a large arithmetic reduction in its own registered domain (3.0x–9.8x
and 89.78x–541.33x) and was killed anyway, so they carry the most information about whether the
subsidy is doing the work. Either the dichotomy holds and the instrument is merely cleaner, or it
does not and the programme has been refusing its own results.

## 2. The open experiment on the finding that actually travels

`docs/so-novelty-2026-09-04.md` is unambiguous about what this programme has that the literature does
not, and it is not the architecture. It is the audit — E-000028's payload-derived index channel,
E-000029's specification-versus-learned-boundary gap, E-000035's deletion oracle. §5 names one small
experiment that would make the first of these matter outside the project: port it to a store someone
else built. No training, inference hours.

Thirty experiments were run after that sentence was written. None of them was that one.

PDX-001 does not finish it either — there are no checkpoints, no torch and no network here — but it
removes what was blocking it. The attack is no longer welded to `so/model.py`; it takes a store as an
argument, carries E-000019's validity floor and a vacuity guard, and reproduces across three
unrelated store shapes at top-1 1.0000, 1.0000 and 0.0000.

That third number is the result worth carrying forward. A soft-deleted vector-index node keeping the
adjacency list built from its own embedding **never names the payload** — and still cuts the
adversary's search space from 256 candidates to 2.92, a posterior of 0.3917 against chance 0.0039.
**E-000028's own top-1 headline would have called that deletion clean.** Any future run of this audit
against a real index must report the candidate-set posterior, or it will report a false negative.

## 3. Revised major-break status

Unchanged in substance: **no major invention is promoted**, and nothing here is one.

Changed in where the programme should look. The frontier update this replaces framed the task as
finding a mechanism that breaks a dichotomy. Two prior tasks now sit in front of that:

1. **Re-run E-000104 and E-000105 under the complete screen.** Until that is done, "generic algebra
   inherits the same advantage" is a claim about an instrument as much as about algebra.
2. **Run §5.** Supply a `Store` whose `observe()` reads a real index with its own reported deletion
   metric, and run PDX-001 against it. Both outcomes publish, as §5 already argued: either a recovery
   channel in a system people cite, or a null that localises the defect.

The programme's surviving results are negative results and audits. That is what the novelty statement
concluded on 2026-09-04 — "not as an architecture paper […] as a short measurement-and-audit paper" —
and thirty experiments later it is still the accurate reading of the record.
