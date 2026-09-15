"""NOV-002 — re-run the E-000104 and E-000105 kills with the representation charged.

NOV-001 found the reduction screen uncalibrated in three ways and named exactly one follow-on: re-run
these two, because each reported a large arithmetic reduction in its own registered domain
(3.0x-9.8x and 89.78x-541.33x) and was killed anyway, so they carry the most information about
whether the representation subsidy was doing the work.

This is that re-run. It is not a new mechanism and it changes neither experiment's arithmetic; it
counts what the originals did not count.

What the originals leave out, in both cases, is the cost of **building the representation**. Both
compare only the cost of *applying* it:

    e000104 lines 224-226        c_work = local_update_multiplies(rank)
                                 g_work = local_update_multiplies(rank)
    e000105 lines 176-178        candidate_work = mutation_multiplies(net)
                                 generic_work   = mutation_multiplies(net)

In both files those are the same pure function of the same argument, so `work_mismatch` and the
`total_candidate == total_generic` term cannot take any value but zero/equal. NOV-001 established
that for E-000105; it holds identically for E-000104, which is recorded here.

Underneath that shared defect the two experiments are **not** the same case, and the distinction is
the whole result:

  E-000104   both arms consume the *same* cached environments, built once by
             `candidate_environments`, and then do arithmetic of identical shape --
             `candidate_local_update` and `generic_cached_environment_update` are two spellings of
             (1xR)(RxR) followed by (1xR)(Rx1). Charging the build charges both arms the same amount.

  E-000105   the two arms build the sensitivity matrix by *different algorithms*.
             `candidate_sensitivity` propagates a matrix, paying `mm(diag(mask), Ws)` at width^3 per
             layer. `generic_region_sensitivity` propagates one column at a time at width^2 per
             layer per pod axis. The build costs differ, and neither is counted.

So the question NOV-001 left open has a definite answer per experiment, and this file computes it
rather than arguing it.

Method. For each experiment the arithmetic of the original is re-executed through counted variants
that mirror it operation for operation. Every counted variant is checked against the original
function on the same inputs, so the counters describe the code that produced the kills rather than a
model of it. Costs are then totalled over each experiment's own registered workload and put through
NOV-001's complete screen.

Run:  python -m so.experiments.nov002_charged_rerun_e104_e105
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction as F
from pathlib import Path

from so.experiments.nov001_screen_calibration import Arm, Cost, complete_screen, programme_screen

from so.experiments import e000104_tensor_train_lifecycle_quotient_reduction as e104
from so.experiments import e000105_stable_gate_cohort_transport_reduction as e105

# ------------------------------------------------------------------ E-000105 counted reimplementation


def counted_candidate_sensitivity(net: dict, code: int) -> tuple[list[list[F]], int]:
    """`e105.candidate_sensitivity`, operation for operation, with a multiply counter."""
    w, pd, od, depth = net["width"], net["pod_dim"], net["out_dim"], net["depth"]
    j = [row[:] for row in net["Bp"]]
    ident = e105.eye(w)
    mults = 0
    for layer in range(depth):
        # mm(diag(mask), Ws[layer]) : (w x w)(w x w)
        a = e105.mm(e105.diag(net["masks"][layer][code]), net["Ws"][layer])
        mults += w * w * w
        a = e105.ma(a, ident)  # additions only
        # mm(a, j) : (w x w)(w x pd)
        j = e105.mm(a, j)
        mults += w * w * pd
    out = e105.mm(net["Wo"], j)  # (od x w)(w x pd)
    mults += od * w * pd
    return out, mults


def counted_generic_sensitivity(net: dict, code: int) -> tuple[list[list[F]], int]:
    """`e105.generic_region_sensitivity`, operation for operation, with a multiply counter."""
    w, pd, od, depth = net["width"], net["pod_dim"], net["out_dim"], net["depth"]
    columns = []
    mults = 0
    for pod_axis in range(pd):
        v = [net["Bp"][i][pod_axis] for i in range(w)]
        for layer in range(depth):
            wv = e105.mv(net["Ws"][layer], v)  # (w x w)(w)
            mults += w * w
            mask = net["masks"][layer][code]
            v = [v[i] + F(mask[i]) * wv[i] for i in range(w)]
            # the gate multiply is by 0/1 and the original counts no such term; excluded on both
            # arms alike so the comparison is not tilted by a bookkeeping choice.
        columns.append(e105.mv(net["Wo"], v))  # (od x w)(w)
        mults += od * w
    out = [[columns[j][i] for j in range(pd)] for i in range(od)]
    return out, mults


def e000105_charged(seed_count: int = 16, widths=(4, 6, 8), depths=(2, 4, 6),
                    codes: int = 4, event_count: int = 6, session_count: int = 32) -> dict:
    """E-000105's own registered workload, with the sensitivity build charged to whoever builds it."""
    cand_build = gen_build = 0
    cand_apply = gen_apply = 0
    full_replay = 0
    cells = 0
    state_mismatch = 0
    counter_disagreements = 0

    for seed in range(seed_count):
        for width in widths:
            for depth in depths:
                cells += 1
                net = e105.generate(seed, width, depth, codes=codes)
                for code in range(codes):
                    c_j, c_m = counted_candidate_sensitivity(net, code)
                    g_j, g_m = counted_generic_sensitivity(net, code)

                    # the counted variants must reproduce the originals exactly
                    if c_j != e105.candidate_sensitivity(net, code):
                        counter_disagreements += 1
                    if g_j != e105.generic_region_sensitivity(net, code):
                        counter_disagreements += 1
                    # and the two arms must still agree with each other, as E-000105 reports
                    if c_j != g_j:
                        state_mismatch += 1

                    cand_build += c_m
                    gen_build += g_m

                apply_once = e105.mutation_multiplies(net)
                cand_apply += event_count * codes * apply_once
                gen_apply += event_count * codes * apply_once
                full_replay += event_count * session_count * e105.full_forward_multiplies(net)

    return {
        "cells": cells,
        "exact_state_mismatches": state_mismatch,
        "counter_disagreements_with_original": counter_disagreements,
        "candidate": Arm(
            "identical-sensitivity",
            Cost(multiplies=cand_apply, representation_multiplies=cand_build, sequential_rounds=1),
        ),
        "generic": Arm(
            "identical-sensitivity",
            Cost(multiplies=gen_apply, representation_multiplies=gen_build, sequential_rounds=1),
        ),
        "full_replay_multiplies": full_replay,
    }


# ------------------------------------------------------------------ E-000104 counted reimplementation


def counted_environments(cores, target: int, left, right) -> tuple[tuple, int]:
    """`e104.candidate_environments`, operation for operation, with a multiply counter."""
    rank = len(left)
    mults = 0
    l = list(left)
    for i in range(target):
        l = e104.row_times_mat(l, cores[i])
        mults += rank * rank
    r = list(right)
    for i in range(len(cores) - 1, target, -1):
        r = e104.mat_times_vec(cores[i], r)
        mults += rank * rank
    return (l, r), mults


def e000104_charged(seed_count: int = 16, pod_counts=(4, 6, 8, 10, 12), ranks=(2, 3, 4),
                    session_count: int = 16, lifecycle_states: int = 5) -> dict:
    """E-000104's own registered workload, with the environment build charged to both arms."""
    cand_build = gen_build = 0
    cand_apply = gen_apply = 0
    full_replay = 0
    cells = 0
    state_mismatch = 0
    counter_disagreements = 0

    for seed in range(seed_count):
        for pod_count in pod_counts:
            target = pod_count // 2
            for rank in ranks:
                cells += 1
                states, sessions = e104.generate_cell(seed, pod_count, rank, session_count)
                old_cores = e104.materialize(states)
                new_cores = e104.materialize(states, {target: "new"})

                for left, right in sessions:
                    (l_env, r_env), build = counted_environments(old_cores, target, left, right)
                    if (l_env, r_env) != e104.candidate_environments(old_cores, target, left, right):
                        counter_disagreements += 1

                    # Both arms consume the same environments and must build them the same way:
                    # `generic_cached_environment_update` takes them as arguments.
                    cand_build += build
                    gen_build += build

                    c_out = e104.candidate_local_update(l_env, new_cores[target], r_env)
                    g_out = e104.generic_cached_environment_update(l_env, new_cores[target], r_env)
                    if c_out != g_out:
                        state_mismatch += 1

                    per_event = e104.local_update_multiplies(rank)
                    cand_apply += lifecycle_states * per_event
                    gen_apply += lifecycle_states * per_event
                    full_replay += lifecycle_states * e104.full_contract_multiplies(pod_count, rank)

    return {
        "cells": cells,
        "exact_state_mismatches": state_mismatch,
        "counter_disagreements_with_original": counter_disagreements,
        "candidate": Arm(
            "identical-contraction",
            Cost(multiplies=cand_apply, representation_multiplies=cand_build, sequential_rounds=1),
        ),
        "generic": Arm(
            "identical-contraction",
            Cost(multiplies=gen_apply, representation_multiplies=gen_build, sequential_rounds=1),
        ),
        "full_replay_multiplies": full_replay,
    }


# ------------------------------------------------------- the work-term tautology, in both originals


def work_term_audit() -> dict:
    """Both kill predicates compare a pure function with itself. Shown, not argued."""
    e105_mismatch = e104_mismatch = 0
    cases = 0
    for seed in range(64):
        net = e105.generate(seed % 8, 4 + 2 * (seed % 3), 2 + 2 * (seed % 3), codes=4)
        if e105.mutation_multiplies(net) != e105.mutation_multiplies(net):
            e105_mismatch += 1
        rank = 2 + (seed % 3)
        if e104.local_update_multiplies(rank) != e104.local_update_multiplies(rank):
            e104_mismatch += 1
        cases += 1
    return {
        "cases": cases,
        "e000105_work_mismatches": e105_mismatch,
        "e000104_work_mismatches": e104_mismatch,
        "either_reachable": bool(e105_mismatch or e104_mismatch),
        "note": (
            "e000104 lines 224-225 and e000105 lines 176-177 each call one pure function twice on "
            "one argument. Neither kill's work term is a measurement; both kills rest on exact-state "
            "equality alone."
        ),
    }


# --------------------------------------------------------------------------------------------- run


def _verdicts(name: str, block: dict) -> dict:
    cand, gen = block["candidate"], block["generic"]
    charged_c = cand.cost.charged().as_dict()
    charged_g = gen.cost.charged().as_dict()

    # As the original screened it: apply cost only, representation free to both.
    as_screened = programme_screen(
        Arm(cand.state, Cost(multiplies=cand.cost.multiplies)),
        Arm(gen.state, Cost(multiplies=gen.cost.multiplies)),
    )
    charged = complete_screen(cand, gen)

    return {
        "experiment": name,
        "cells": block["cells"],
        "exact_state_mismatches": block["exact_state_mismatches"],
        "counter_disagreements_with_original": block["counter_disagreements_with_original"],
        "candidate_cost": cand.cost.as_dict(),
        "generic_cost": gen.cost.as_dict(),
        "candidate_total_charged": charged_c["multiplies"],
        "generic_total_charged": charged_g["multiplies"],
        "full_replay_multiplies": block["full_replay_multiplies"],
        "verdict_as_originally_screened": as_screened["verdict"],
        "verdict_charged": charged["verdict"],
        "charged_screen_detail": charged,
        "kill_survives_charging": charged["verdict"] == "KILL",
        "representation_build_ratio_candidate_over_generic": (
            cand.cost.representation_multiplies / gen.cost.representation_multiplies
            if gen.cost.representation_multiplies else None
        ),
    }


def run() -> dict:
    e105_block = e000105_charged()
    e104_block = e000104_charged()

    rows = [_verdicts("E-000104", e104_block), _verdicts("E-000105", e105_block)]
    audit = work_term_audit()

    survived = [r["experiment"] for r in rows if r["kill_survives_charging"]]
    reversed_ = [r["experiment"] for r in rows if not r["kill_survives_charging"]]
    counters_sound = all(r["counter_disagreements_with_original"] == 0 for r in rows)
    states_agree = all(r["exact_state_mismatches"] == 0 for r in rows)

    return {
        "experiment": "NOV-002",
        "scope": "E-000104 and E-000105 re-screened with the representation charged",
        "counters_reproduce_originals": counters_sound,
        "arms_still_reach_identical_exact_states": states_agree,
        "results": rows,
        "kills_surviving": survived,
        "kills_reversed": reversed_,
        "work_term_audit": audit,
        "decision": (
            "BOTH_KILLS_SURVIVE_CHARGING"
            if len(survived) == 2
            else ("KILLS_REVERSED:" + ",".join(reversed_) if reversed_ else "SCREEN_NOT_DECISIVE")
        ),
        "finding": (
            "Both kills survive, and they survive for different reasons — which is why the two had to "
            "be run separately rather than treated as one family. In E-000104 the candidate and the "
            "generic path consume the same cached environments and perform arithmetic of identical "
            "shape, so charging the build charges both arms the same and the tie is real, not an "
            "artefact of the accounting. In E-000105 the two arms build the sensitivity matrix by "
            "different algorithms, and charging that build does not rescue the candidate — it makes "
            "the candidate strictly worse, because propagating a matrix costs width^3 per layer where "
            "the generic column-by-column propagation costs width^2 per layer per pod axis. The "
            "candidate spends more to build the representation whose free provision NOV-001 showed "
            "the screen was assuming. So the subsidy was real and was not what produced these two "
            "verdicts. NOV-001's instrument findings stand; the dichotomy in the frontier update also "
            "stands, and now rests on measured cost rather than on a work term that could not fail."
        ),
        "not_claimed": (
            "This covers two of the twenty-three kills — the two NOV-001 nominated as carrying the "
            "most information. It says nothing about the other twenty-one, and it does not show that "
            "no representation-based mechanism can win under a charged accounting; it shows these two "
            "do not. Cost is counted in exact multiplies over each experiment's own registered "
            "workload; memory traffic and sequential depth are not instrumented in either original "
            "and are not invented here."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/nov002"))
    args = parser.parse_args()
    result = run()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "nov002_charged_rerun_e104_e105.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if not result["counters_reproduce_originals"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
