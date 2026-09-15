"""E-000113: task-essential sparse incremental accumulator reduction.

Exact integer structural assay.  The candidate keeps the real first hidden
accumulator used by the nonlinear task suffix and updates it by removing and
adding sparse Pod feature columns.  An independent generic sparse-feature
accumulator receives the identical state and columns.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence

DECISION = "KILL_TASK_ESSENTIAL_SPARSE_INCREMENTAL_ACCUMULATOR_AS_STANDALONE_LIFECYCLE_NOVELTY"

LIFECYCLE = (
    (0, 1, "UPDATE"),
    (1, None, "DELETE"),
    (None, 1, "RESTORE"),
    (1, 0, "ABA"),
)


def piecewise_nonlinearity(x: int) -> int:
    """Injective nonlinear piecewise-linear map over the integers."""
    return x if x >= 0 else 2 * x


def dense_suffix(
    accumulator: Sequence[int],
    w1: Sequence[Sequence[int]],
    b1: Sequence[int],
    w2: Sequence[Sequence[int]],
    b2: Sequence[int],
) -> tuple[int, ...]:
    hidden: list[int] = []
    for row, bias in zip(w1, b1):
        value = bias
        for weight, source in zip(row, accumulator):
            value += weight * source
        hidden.append(piecewise_nonlinearity(value))

    output: list[int] = []
    for row, bias in zip(w2, b2):
        value = bias
        for weight, source in zip(row, hidden):
            value += weight * source
        output.append(piecewise_nonlinearity(value))
    return tuple(output)


def pod_feature_column(arm: str, bucket: int, state: int, width: int) -> tuple[int, ...]:
    """Deterministic exact Pod feature column.

    In the global arm the bucket is ignored.  In the contextual arm the bucket
    interacts with Pod state, so even UPDATE/ABA deltas genuinely vary by
    context bucket instead of cancelling a common offset.
    """
    effective_bucket = 0 if arm == "global" else bucket
    return tuple(
        (state + 1) * 17
        + effective_bucket * 13 * (state + 1)
        + ((coordinate + 1) * (state + 2) % 11)
        for coordinate in range(width)
    )


def fresh_accumulator(
    bias: Sequence[int],
    static_columns: Sequence[Sequence[int]],
    active_static_ids: Iterable[int],
    arm: str,
    bucket: int,
    pod_state: int | None,
) -> tuple[int, ...]:
    value = list(bias)
    for feature_id in active_static_ids:
        column = static_columns[feature_id]
        for coordinate, element in enumerate(column):
            value[coordinate] += element
    if pod_state is not None:
        column = pod_feature_column(arm, bucket, pod_state, len(value))
        for coordinate, element in enumerate(column):
            value[coordinate] += element
    return tuple(value)


def candidate_transport(
    accumulator: Sequence[int],
    arm: str,
    bucket: int,
    old_state: int | None,
    new_state: int | None,
) -> tuple[tuple[int, ...], int]:
    """Lifecycle-labelled candidate implementation."""
    updated = list(accumulator)
    work = 0
    if old_state is not None:
        old_column = pod_feature_column(arm, bucket, old_state, len(updated))
        for coordinate, element in enumerate(old_column):
            updated[coordinate] -= element
            work += 1
    if new_state is not None:
        new_column = pod_feature_column(arm, bucket, new_state, len(updated))
        for coordinate, element in enumerate(new_column):
            updated[coordinate] += element
            work += 1
    return tuple(updated), work


def generic_sparse_feature_diff(
    accumulator: Sequence[int],
    removed_columns: Sequence[Sequence[int]],
    added_columns: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], int]:
    """Independent generic sparse-feature accumulator baseline."""
    result = list(accumulator)
    work = 0
    for column in removed_columns:
        for coordinate in range(len(result)):
            result[coordinate] = result[coordinate] - column[coordinate]
            work += 1
    for column in added_columns:
        for coordinate in range(len(result)):
            result[coordinate] = result[coordinate] + column[coordinate]
            work += 1
    return tuple(result), work


def shared_delta(arm: str, bucket: int, old_state: int | None, new_state: int | None, width: int) -> tuple[int, ...]:
    old_column = (0,) * width if old_state is None else pod_feature_column(arm, bucket, old_state, width)
    new_column = (0,) * width if new_state is None else pod_feature_column(arm, bucket, new_state, width)
    return tuple(new - old for old, new in zip(old_column, new_column))


def make_network(seed: int, width: int, arm: str) -> dict[str, object]:
    rng = random.Random((seed + 1) * 100_003 + width * 101 + (0 if arm == "global" else 1))
    static_columns = [tuple(rng.randint(-9, 9) for _ in range(width)) for _ in range(128)]
    bias = tuple(rng.randint(-5, 5) for _ in range(width))
    w1 = [[rng.randint(-3, 3) for _ in range(width)] for _ in range(width)]
    w2 = [[rng.randint(-3, 3) for _ in range(width)] for _ in range(width)]
    for coordinate in range(width):
        w1[coordinate][coordinate] += 5
        w2[coordinate][coordinate] += 5
    b1 = [rng.randint(-3, 3) for _ in range(width)]
    b2 = [rng.randint(-3, 3) for _ in range(width)]
    return {
        "rng": rng,
        "static_columns": static_columns,
        "bias": bias,
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
    }


def run_cell(seed: int, width: int, arm: str, sessions: int = 64, buckets: int = 8, static_count: int = 16) -> dict[str, object]:
    network = make_network(seed, width, arm)
    rng = network["rng"]
    static_columns = network["static_columns"]
    bias = network["bias"]
    w1 = network["w1"]
    w2 = network["w2"]
    b1 = network["b1"]
    b2 = network["b2"]

    metrics: dict[str, int] = {
        "lifecycle_cases": 0,
        "material_final_changes": 0,
        "candidate_fresh_accumulator_mismatches": 0,
        "generic_fresh_accumulator_mismatches": 0,
        "candidate_generic_accumulator_mismatches": 0,
        "candidate_fresh_final_mismatches": 0,
        "generic_fresh_final_mismatches": 0,
        "candidate_mutation_adds": 0,
        "generic_mutation_adds": 0,
        "fresh_first_layer_adds": 0,
        "nonlinear_suffix_multiplies": 0,
        "wrong_single_global_receipt_failures": 0,
    }
    unique_deltas: dict[str, set[tuple[int, ...]]] = {name: set() for _, _, name in LIFECYCLE}

    for session in range(sessions):
        bucket = session % buckets
        active_static_ids = rng.sample(range(len(static_columns)), static_count)

        def fresh(state: int | None) -> tuple[int, ...]:
            return fresh_accumulator(bias, static_columns, active_static_ids, arm, bucket, state)

        current_state: int | None = 0
        candidate_acc = fresh(current_state)
        generic_acc = candidate_acc
        previous_final = dense_suffix(candidate_acc, w1, b1, w2, b2)

        for old_state, new_state, lifecycle_name in LIFECYCLE:
            if current_state != old_state:
                raise AssertionError("registered lifecycle sequence drifted")

            candidate_acc, candidate_work = candidate_transport(
                candidate_acc, arm, bucket, old_state, new_state
            )
            removed = [] if old_state is None else [pod_feature_column(arm, bucket, old_state, width)]
            added = [] if new_state is None else [pod_feature_column(arm, bucket, new_state, width)]
            generic_acc, generic_work = generic_sparse_feature_diff(generic_acc, removed, added)
            fresh_acc = fresh(new_state)

            metrics["candidate_mutation_adds"] += candidate_work
            metrics["generic_mutation_adds"] += generic_work
            metrics["fresh_first_layer_adds"] += (static_count + (1 if new_state is not None else 0)) * width
            metrics["nonlinear_suffix_multiplies"] += 2 * width * width

            if candidate_acc != fresh_acc:
                metrics["candidate_fresh_accumulator_mismatches"] += 1
            if generic_acc != fresh_acc:
                metrics["generic_fresh_accumulator_mismatches"] += 1
            if candidate_acc != generic_acc:
                metrics["candidate_generic_accumulator_mismatches"] += 1

            fresh_final = dense_suffix(fresh_acc, w1, b1, w2, b2)
            candidate_final = dense_suffix(candidate_acc, w1, b1, w2, b2)
            generic_final = dense_suffix(generic_acc, w1, b1, w2, b2)
            if candidate_final != fresh_final:
                metrics["candidate_fresh_final_mismatches"] += 1
            if generic_final != fresh_final:
                metrics["generic_fresh_final_mismatches"] += 1
            if fresh_final != previous_final:
                metrics["material_final_changes"] += 1

            if arm == "contextual":
                wrong = list(fresh(old_state))
                wrong_delta = shared_delta(arm, 0, old_state, new_state, width)
                for coordinate, element in enumerate(wrong_delta):
                    wrong[coordinate] += element
                if tuple(wrong) != fresh_acc:
                    metrics["wrong_single_global_receipt_failures"] += 1

            unique_deltas[lifecycle_name].add(shared_delta(arm, bucket, old_state, new_state, width))
            metrics["lifecycle_cases"] += 1
            previous_final = fresh_final
            current_state = new_state

    return {
        "seed": seed,
        "width": width,
        "arm": arm,
        **metrics,
        "unique_exact_deltas_by_transition": {name: len(values) for name, values in unique_deltas.items()},
    }


def run_registered() -> dict[str, object]:
    cells = [
        run_cell(seed, width, arm)
        for seed in range(16)
        for width in (8, 16, 32)
        for arm in ("global", "contextual")
    ]

    count_keys = (
        "lifecycle_cases",
        "material_final_changes",
        "candidate_fresh_accumulator_mismatches",
        "generic_fresh_accumulator_mismatches",
        "candidate_generic_accumulator_mismatches",
        "candidate_fresh_final_mismatches",
        "generic_fresh_final_mismatches",
        "candidate_mutation_adds",
        "generic_mutation_adds",
        "fresh_first_layer_adds",
        "nonlinear_suffix_multiplies",
        "wrong_single_global_receipt_failures",
    )
    totals = {key: sum(int(cell[key]) for cell in cells) for key in count_keys}

    global_delta_max = max(
        max(cell["unique_exact_deltas_by_transition"].values())
        for cell in cells
        if cell["arm"] == "global"
    )
    contextual_delta_min = min(
        min(cell["unique_exact_deltas_by_transition"].values())
        for cell in cells
        if cell["arm"] == "contextual"
    )

    candidate_ready_work = totals["candidate_mutation_adds"] + totals["nonlinear_suffix_multiplies"]
    generic_ready_work = totals["generic_mutation_adds"] + totals["nonlinear_suffix_multiplies"]
    fresh_ready_work = totals["fresh_first_layer_adds"] + totals["nonlinear_suffix_multiplies"]

    passed = (
        totals["material_final_changes"] == totals["lifecycle_cases"]
        and totals["candidate_fresh_accumulator_mismatches"] == 0
        and totals["generic_fresh_accumulator_mismatches"] == 0
        and totals["candidate_generic_accumulator_mismatches"] == 0
        and totals["candidate_fresh_final_mismatches"] == 0
        and totals["generic_fresh_final_mismatches"] == 0
        and totals["candidate_mutation_adds"] == totals["generic_mutation_adds"]
        and candidate_ready_work == generic_ready_work
        and global_delta_max == 1
        and contextual_delta_min == 8
        and totals["wrong_single_global_receipt_failures"] > 0
    )

    return {
        "experiment": "E-000113",
        "decision": DECISION if passed else "FAIL_REGISTERED_KILL_SCREEN",
        "passed": passed,
        "configuration": {
            "seeds": 16,
            "widths": [8, 16, 32],
            "arms": ["global", "contextual"],
            "sessions_per_cell": 64,
            "context_buckets": 8,
            "static_active_features": 16,
            "static_feature_pool": 128,
            "lifecycle": [name for _, _, name in LIFECYCLE],
            "arithmetic": "exact integers",
        },
        "metrics": {
            "cells": len(cells),
            **totals,
            "global_arm_max_unique_delta_count": global_delta_max,
            "contextual_arm_min_unique_delta_count": contextual_delta_min,
            "candidate_ready_work": candidate_ready_work,
            "generic_ready_work": generic_ready_work,
            "fresh_ready_work": fresh_ready_work,
            "first_layer_refresh_over_incremental_ratio": totals["fresh_first_layer_adds"] / totals["candidate_mutation_adds"],
            "fresh_ready_over_incremental_ready_ratio": fresh_ready_work / candidate_ready_work,
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results/e000113"))
    args = parser.parse_args()
    result = run_registered()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / "e000113-task-essential-incremental-accumulator-reduction.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "metrics": result["metrics"]}, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
