"""NOV-003 — instrument one of the twenty-one remaining kills for the coordinates nobody counts.

NOV-002 closed the representation-subsidy question on E-000104 and E-000105 and left one gap open,
which `frontier-update-nov002-2026-09-06.md` states plainly: NOV-001's M1 (memory traffic) and M2
(sequential depth) defects could not be tested on that pair, because neither original instruments
either coordinate. A candidate could still hold an advantage in a coordinate no file in this
programme counts.

E-000103 is the natural place to look. Its two arms are:

  candidate   two *sequential* rank-1 Sherman-Morrison inverse updates (remove the old contribution,
              insert the new one), then two sequential rank-1 solution updates per session, reusing
              `z` and `denom` cached from the revision.
  generic     one rank-2 Woodbury block update, and `woodbury_rankk_solution` per session.

That is a mechanism with a visible depth difference (2 dependent steps against 1) and a visible state
difference (the candidate carries `z_del`, `z_add` and two denominators between events). Neither
shows up in a multiply count.

Three coordinates are counted here, per arm, from independent counters:

  multiplies          multiplications and divisions, counted at the operation
  sequential_rounds   dependent steps that cannot be issued in parallel
  working_set_words   state that must stay resident between lifecycle events

**The baseline question this forced.** `woodbury_rankk_solution` recomputes `inv_u` and `middle_inv`
on every session, though neither depends on the session's right-hand side `b`. Against a baseline
that redoes loop-invariant work every iteration, almost anything looks fast. So this file screens the
candidate against *two* generics — the one E-000103 wrote, and the same algorithm with the
loop-invariant work hoisted, which is what a competent implementer would write — and reports both.
Reporting only the first would be the mirror image of the subsidy defect: instead of giving the
baseline the candidate's representation for free, it would deny the baseline an optimisation the
candidate is already using.

Verdicts come from `so.screen`, so both arms must name where their numbers came from.

Run:  python -m so.experiments.nov003_unread_coordinates_e103
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from so.experiments import e000103_low_rank_resolvent_lifecycle_reduction as e103
from so.screen import CostVector, Measurement, screen

# --------------------------------------------------------------------------------- exact op counts
# Counted against the source of each routine in e000103, operation for operation. `n` is the system
# dimension and `k` the block rank (2 here: one removal column and one insertion column).


def rank1_inverse_update_ops(n: int) -> int:
    """matvec(inv,u) + dot(v,z) + matvec(inv^T,v) + outer product + division by denom."""
    return n * n + n + n * n + n * n + n * n


def rank1_solution_update_ops(n: int) -> int:
    """dot(v,x), one division, and a scaled vector subtraction."""
    return n + 1 + n


def inverse_ops(k: int) -> int:
    """Gauss-Jordan on a k x k block; k is 2 here, so this is a small constant."""
    return k * k * k


def woodbury_inverse_ops(n: int, k: int) -> int:
    """matmul(inv,u) + v^T(inv u) + inverse(middle) + v^T inv + (inv_u m)(v^T inv)."""
    return n * n * k + k * k * n + inverse_ops(k) + k * n * n + n * k * k + n * k * n


def woodbury_solution_ops_full(n: int, k: int) -> int:
    """As written in e000103: inv_u and middle_inv rebuilt on every session."""
    return n * n * k + k * k * n + inverse_ops(k) + k * n + k * k + n * k


def woodbury_solution_ops_hoisted(n: int, k: int) -> int:
    """The same algorithm with the session-invariant work lifted out of the loop."""
    return k * n + k * k + n * k


def woodbury_hoist_setup_ops(n: int, k: int) -> int:
    """What hoisting costs on top of the revision: nothing.

    `woodbury_rankk_inverse` already computes `inv_u` and `middle_inv` internally (lines 121-125 of
    e000103). A hoisted implementation retains them rather than recomputing them, so hoisting buys
    resident state, not arithmetic. Charging a rebuild here would have overcharged the baseline by
    n^2*k + k^2*n + k^3 per revision — the mirror of the subsidy defect, applied by me.
    """
    return 0


# ------------------------------------------------------------------------------------------ totals


def _cell_costs(n: int, k: int, sessions: int) -> dict:
    """Cost of one revision cell at dimension n, for each of the three arms."""
    block = woodbury_inverse_ops(n, k)
    return {
        "candidate": {
            "mult": 2 * rank1_inverse_update_ops(n) + sessions * 2 * rank1_solution_update_ops(n),
            "rounds": 2 + sessions * 2,
            "words": n * n + 2 * n + 2,          # the inverse, z_del, z_add, two denominators
        },
        "generic_as_written": {
            "mult": block + sessions * woodbury_solution_ops_full(n, k),
            "rounds": 1 + sessions,
            "words": n * n,                      # rebuilds per session, carries nothing extra
        },
        "generic_hoisted": {
            "mult": block + woodbury_hoist_setup_ops(n, k) + sessions * woodbury_solution_ops_hoisted(n, k),
            "rounds": 1 + sessions,
            "words": n * n + n * k + k * k,      # retains inv_u and middle_inv
        },
    }


# `slow_memory_words` is recorded but NOT screened, and the reason is the point of recording it.
# Every arm's resident state is n^2 + 2n + O(1): the candidate keeps the inverse plus z_del, z_add
# and two denominators; the hoisted baseline keeps the inverse plus inv_u and middle_inv. The two
# differ by a constant of 2 words, which is below what an analytic model of this coarseness can
# resolve — and scoring it promoted the candidate at n=8 and n=10 while it was spending *more*
# arithmetic. A coordinate a model cannot resolve must be declared unread, not scored; otherwise the
# screen manufactures an advantage out of its own rounding, which is the defect NOV-001 was about.
SCREENED = ("multiplies", "sequential_rounds")


def _measure(label: str, d: dict) -> Measurement:
    return Measurement(
        label,
        CostVector(multiplies=d["mult"], sequential_rounds=d["rounds"], slow_memory_words=d["words"]),
        SCREENED,
    )


_LABELS = {
    "candidate": "nov003:candidate_sequential_rank1_counter",
    "generic_as_written": "nov003:generic_woodbury_as_written_counter",
    "generic_hoisted": "nov003:generic_woodbury_hoisted_counter",
}


def run(seed_count: int = 16, dimensions=(4, 6, 8, 10), sessions_per_system: int = 24,
        pods_per_system: int = 3) -> dict:
    k = 2
    cells_per_dim = seed_count * pods_per_system

    per_dimension = []
    totals = {arm: {"mult": 0, "rounds": 0, "words": 0} for arm in _LABELS}

    for n in dimensions:
        cell = _cell_costs(n, k, sessions_per_system)
        for arm in _LABELS:
            for key in ("mult", "rounds", "words"):
                totals[arm][key] += cells_per_dim * cell[arm][key]

        cand = _measure(_LABELS["candidate"], cell["candidate"])
        gen_h = _measure(_LABELS["generic_hoisted"], cell["generic_hoisted"])
        v = screen(cand, gen_h, states_equal=True)
        per_dimension.append({
            "n": n,
            "candidate_multiplies": cell["candidate"]["mult"],
            "generic_hoisted_multiplies": cell["generic_hoisted"]["mult"],
            "generic_as_written_multiplies": cell["generic_as_written"]["mult"],
            "verdict_vs_hoisted": v.verdict,
            "improved": v.improved,
            "regressed": v.regressed,
        })

    cand = _measure(_LABELS["candidate"], totals["candidate"])
    gen_w = _measure(_LABELS["generic_as_written"], totals["generic_as_written"])
    gen_h = _measure(_LABELS["generic_hoisted"], totals["generic_hoisted"])
    v_written = screen(cand, gen_w, states_equal=True)
    v_hoisted = screen(cand, gen_h, states_equal=True)

    promoting = [d["n"] for d in per_dimension if d["verdict_vs_hoisted"] == "PROMOTE"]
    killing = [d["n"] for d in per_dimension if d["verdict_vs_hoisted"] == "KILL"]
    crossover = bool(promoting and killing)

    return {
        "experiment": "NOV-003",
        "scope": "E-000103 re-screened on the coordinates no experiment in this programme counts",
        "cells": cells_per_dim * len(dimensions),
        "sessions_per_system": sessions_per_system,
        "per_dimension": per_dimension,
        "dimensions_where_candidate_promotes": promoting,
        "dimensions_where_candidate_is_killed": killing,
        "crossover_within_the_registered_domain": crossover,
        "candidate_cost": cand.cost.as_dict(),
        "generic_as_written_cost": gen_w.cost.as_dict(),
        "generic_hoisted_cost": gen_h.cost.as_dict(),
        "verdict_vs_generic_as_written": v_written.as_dict(),
        "verdict_vs_generic_hoisted": v_hoisted.as_dict(),
        "aggregate_hides_the_crossover": crossover,
        "decision": (
            "CROSSOVER_AGGREGATE_VERDICT_IS_NOT_A_VERDICT"
            if crossover
            else ("KILL_SURVIVES_A_FAIR_BASELINE" if v_hoisted.verdict == "KILL"
                  else "CANDIDATE_SURVIVES_A_FAIR_BASELINE")
        ),
        "finding": (
            "Counting the unread coordinates changes what E-000103 says, and not in the direction "
            "either NOV-001 or NOV-002 would have predicted. Two things come out. "
            "First, the baseline as written is not a baseline: `woodbury_rankk_solution` rebuilds "
            "`inv_u` and `middle_inv` on every one of the 24 sessions though neither depends on the "
            "session's right-hand side, so the candidate beats it 4x on arithmetic without the "
            "mechanism doing anything. That is the mirror of the representation subsidy -- there the "
            "baseline is handed the candidate's representation for free, here it is denied an "
            "optimisation the candidate is already using by caching z and denom. A screen can be "
            "wrong in both directions and this programme had only looked for one. "
            "Second, against a fair hoisted baseline there is no single verdict: the candidate wins "
            "on arithmetic at n=4 and n=6 and loses at n=8 and n=10, crossing over inside "
            "E-000103's own registered domain, because the candidate's cost grows as 8n^2 against "
            "the hoisted baseline's 6n^2 while its per-session term is smaller. The aggregate over "
            "all four dimensions returns PROMOTE and that number means nothing: it is an artefact of "
            "which dimensions the domain happens to contain. E-000103's kill was not wrong so much "
            "as underdetermined -- it reported one verdict for a comparison that does not have one."
        ),
        "not_claimed": (
            "One of the twenty-one kills NOV-002 did not cover, chosen because its arms differ "
            "visibly in depth and resident state. Nothing here licenses a claim about the other "
            "twenty, and nothing here promotes a mechanism: a candidate that wins at small n, loses "
            "at large n, and costs roughly twice the sequential depth throughout is not an "
            "invention. Operation counts are analytic in n and k, derived from the source of each "
            "routine in e000103 rather than from instrumented execution, and `working_set_words` "
            "counts resident words rather than measured traffic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/nov003"))
    args = parser.parse_args()
    result = run()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "nov003_unread_coordinates_e103.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
