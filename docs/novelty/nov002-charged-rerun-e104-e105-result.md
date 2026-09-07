# NOV-002 result — both kills survive the charge, and the subsidy was not what produced them

Date: 2026-09-06
Decision: **BOTH_KILLS_SURVIVE_CHARGING**
Major-invention claim: **NO** — and this result argues against the one NOV-001 left open

## Registration

This experiment was specified in advance, in `nov001-screen-calibration-result.md` §"What would
settle it":

> Re-run E-000104 and E-000105 — the two that reported the largest arithmetic reductions before being
> killed — with the full cost vector recorded on both arms and the representation charged to whoever
> builds it. […] If they still kill under the complete screen, the dichotomy in the frontier update
> is real and NOV-001 has only cleaned an instrument. If either survives, twenty-three verdicts need
> re-reading and the programme has been refusing its own results.

They still kill. NOV-001 has only cleaned an instrument.

## What was run

`so/experiments/nov002_charged_rerun_e104_e105.py`, stdlib only, exact `Fraction` arithmetic, over
each experiment's own registered workload — E-000104's 240 cells and E-000105's 144. Focused
regressions: 11 tests. Reproduce with `make charged`.

**Counter soundness.** The arithmetic is re-executed through counted variants that mirror each
original operation for operation, and every counted variant is checked against the original function
on the same inputs. `counter_disagreements_with_original` is **0** in both experiments, so the
counters describe the code that produced the kills rather than a model of it. The two arms still
reach identical exact states in every cell (`exact_state_mismatches` = 0), so charging changes the
accounting and not the arithmetic.

## The numbers

| | apply | build | **charged total** | full replay | as screened | charged |
|---|---:|---:|---:|---:|---|---|
| **E-000104** candidate | 243,200 | 259,840 | **503,040** | 1,542,400 | KILL | **KILL** |
| **E-000104** generic | 243,200 | 259,840 | **503,040** | | | |
| **E-000105** candidate | 31,104 | 906,624 | **937,728** | 8,008,704 | KILL | **KILL** |
| **E-000105** generic | 31,104 | 298,368 | **329,472** | | | |

## They survive for different reasons, which is why they had to be run separately

**E-000104 — the tie is real.** Build ratio candidate/generic is exactly **1.0**. Both arms consume
the same cached environments, built once by `candidate_environments`, and then do arithmetic of
identical shape: `candidate_local_update` and `generic_cached_environment_update` are two spellings
of (1×R)(R×R) followed by (1×R)(R×1). Charging the build charges both arms the same 259,840
multiplies. There is no subsidy to remove here, because there is no asymmetry to subsidise. The
complete screen reports no improved coordinate and no regressed one — an exact tie.

**E-000105 — charging makes the candidate worse, not better.** Build ratio **3.04**. The two arms
build the sensitivity matrix by different algorithms: `candidate_sensitivity` propagates a matrix and
pays `mm(diag(mask), Ws)` at width³ per layer, while `generic_region_sensitivity` propagates one
column at a time at width² per layer per pod axis. Checked directly on one net rather than through
the aggregate:

```
candidate build = depth·(w³ + w²·pd) + od·w·pd
generic   build = pd·(depth·w² + od·w)
```

So the representation whose free provision NOV-001 showed the screen was assuming is one the
candidate spends **more** to build. Removing the subsidy does not rescue this candidate; it exposes a
regression the original accounting was hiding in the candidate's favour. The complete screen reports
`regressed: multiplies 937,728 vs 329,472` and nothing improved.

## The work-term defect is in both originals

NOV-001 established this for E-000105. It holds identically for E-000104:

```
e000104:224   c_work = local_update_multiplies(rank)
e000104:225   g_work = local_update_multiplies(rank)
e000105:176   candidate_work = mutation_multiplies(net)
e000105:177   generic_work   = mutation_multiplies(net)
```

64 constructed cases, 0 mismatches in either, `either_reachable` false. Both kill predicates compare
a pure function with itself, so neither kill's work term is a measurement and both rest on
exact-state equality alone. That remains true and remains worth fixing — it is simply not what
produced these two verdicts.

## What this settles, and what it does not

**Settles.** The representation subsidy is a genuine defect of the screen and was *not* the cause of
these two kills. The dichotomy in the frontier update — addressable/factorized state can be updated
exactly, but generic algebra usually inherits the same advantage — stands, and now rests on measured
cost on both arms rather than on a work term that could not fail. NOV-001's three instrument findings
also stand; they were findings about the instrument, and this is what it looks like when a cleaned
instrument returns the same answer.

**Does not settle.** This covers two of twenty-three kills, chosen because NOV-001 nominated them as
carrying the most information. It says nothing about the other twenty-one. It does not show that no
representation-based mechanism can win under a charged accounting — only that these two do not. And
cost here is exact multiplies over each experiment's own registered workload; memory traffic and
sequential depth are not instrumented in either original, so NOV-001's M1 and M2 defects are
untested on this pair and could still be hiding an advantage in a coordinate neither file counts.

Note also what the kills do *not* say: against **full replay** both mechanisms remain far cheaper —
1,542,400 and 8,008,704 multiplies against 503,040 and 329,472. The verdict is "no advantage over the
generic arm", never "no advantage over recomputation".

## Consequence for the programme

The honest reading of the record is unchanged by this and was already correct on 2026-09-04: the
surviving results are negative results and audits, not an invention. What has changed is that the
strongest remaining objection to that reading — *maybe the kills were an artefact of a broken
screen* — has been tested on its two best candidates and does not hold.
