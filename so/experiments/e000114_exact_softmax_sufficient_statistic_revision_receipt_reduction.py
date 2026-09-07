"""E-000114: exact softmax sufficient-statistic revision receipt reduction.

Structural exact-arithmetic assay. A canonical mutable Pod is one record inside a
task-essential softmax attention read. Each cached session retains its query,
the attention normalizer, and the already-materialized attention output. A Pod
UPDATE/DELETE/RESTORE/ABA can therefore patch the post-softmax read exactly by
subtracting the old weighted record and adding the new one.

An independently implemented generic dynamic normalized-weighted-average engine
receives the identical cached state and record mutation. Both then pay the same
required nonlinear suffix recomputation. The assay therefore tests whether this
apparently attractive post-nonlinear receipt exceeds generic exact aggregation
plus the strong "exact read-site patch + minimal suffix recompute" baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path
from typing import Sequence

DECISION = "KILL_EXACT_SOFTMAX_SUFFICIENT_STATISTIC_RECEIPT_AS_STANDALONE_LIFECYCLE_NOVELTY"

LIFECYCLE = (
    ("UPDATE", "old", "new"),
    ("DELETE", "new", None),
    ("RESTORE", None, "new"),
    ("ABA", "new", "old"),
)


def power_two(exponent: int) -> Fraction:
    """Exact exp-family weight: 2**score == exp(log(2) * score)."""
    if exponent >= 0:
        return Fraction(2**exponent, 1)
    return Fraction(1, 2 ** (-exponent))


def contribution(
    query: int,
    record: tuple[int, Sequence[int]],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    key, value = record
    weight = power_two(query * key)
    return weight, tuple(weight * element for element in value)


def fresh_attention(
    query: int,
    records: Sequence[tuple[int, Sequence[int]]],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Exact softmax attention for logits log(2) * query * key."""
    if not records:
        raise ValueError("registered attention must have at least one record")
    width = len(records[0][1])
    normalizer = Fraction(0)
    numerator = [Fraction(0) for _ in range(width)]
    for record in records:
        weight, weighted_value = contribution(query, record)
        normalizer += weight
        for coordinate, element in enumerate(weighted_value):
            numerator[coordinate] += element
    output = tuple(element / normalizer for element in numerator)
    return normalizer, output


def candidate_revision_receipt(
    query: int,
    output: Sequence[Fraction],
    normalizer: Fraction,
    old_record: tuple[int, Sequence[int]] | None,
    new_record: tuple[int, Sequence[int]] | None,
) -> tuple[Fraction, tuple[Fraction, ...], int]:
    """Lifecycle-labelled exact post-softmax patch.

    The actual attention output plus its scalar normalizer are sufficient to
    reconstruct the unnormalized numerator. The canonical Pod record is then
    removed/added exactly. `work` counts scalar multiply/add/divide operations
    on the cached sufficient state; exponent lookup/evaluation is counted
    symmetrically for candidate and generic baseline and is not needed for the
    decisive equality result.
    """
    numerator = [element * normalizer for element in output]
    work = len(numerator)

    if old_record is not None:
        old_weight, old_value = contribution(query, old_record)
        normalizer -= old_weight
        work += 1
        for coordinate, element in enumerate(old_value):
            numerator[coordinate] -= element
            work += 1

    if new_record is not None:
        new_weight, new_value = contribution(query, new_record)
        normalizer += new_weight
        work += 1
        for coordinate, element in enumerate(new_value):
            numerator[coordinate] += element
            work += 1

    revised = tuple(element / normalizer for element in numerator)
    work += len(revised)
    return normalizer, revised, work


def generic_dynamic_normalized_aggregate(
    query: int,
    aggregate_output: Sequence[Fraction],
    normalizer: Fraction,
    removed_records: Sequence[tuple[int, Sequence[int]]],
    added_records: Sequence[tuple[int, Sequence[int]]],
) -> tuple[Fraction, tuple[Fraction, ...], int]:
    """Independent generic exact dynamic weighted-average baseline.

    No Symlink, Pod, lifecycle, memory, or neural semantics are used here.
    """
    numerator = [
        aggregate_output[coordinate] * normalizer
        for coordinate in range(len(aggregate_output))
    ]
    work = len(numerator)

    for key, value in removed_records:
        weight = power_two(query * key)
        normalizer = normalizer - weight
        work += 1
        for coordinate in range(len(numerator)):
            numerator[coordinate] = numerator[coordinate] - weight * value[coordinate]
            work += 1

    for key, value in added_records:
        weight = power_two(query * key)
        normalizer = normalizer + weight
        work += 1
        for coordinate in range(len(numerator)):
            numerator[coordinate] = numerator[coordinate] + weight * value[coordinate]
            work += 1

    result = tuple(element / normalizer for element in numerator)
    work += len(result)
    return normalizer, result, work


def piecewise_nonlinearity(value: Fraction) -> Fraction:
    """Injective nonlinear piecewise-affine map over exact rationals."""
    return value if value >= 0 else 2 * value


def dense_suffix(
    attention_output: Sequence[Fraction],
    w1: Sequence[Sequence[int]],
    b1: Sequence[int],
    w2: Sequence[Sequence[int]],
    b2: Sequence[int],
) -> tuple[Fraction, ...]:
    """Two real nonlinear task layers after the last memory-read site."""
    hidden: list[Fraction] = []
    for row, bias in zip(w1, b1):
        value = Fraction(bias)
        for weight, source in zip(row, attention_output):
            value += weight * source
        hidden.append(piecewise_nonlinearity(value))

    output: list[Fraction] = []
    for row, bias in zip(w2, b2):
        value = Fraction(bias)
        for weight, source in zip(row, hidden):
            value += weight * source
        output.append(piecewise_nonlinearity(value))
    return tuple(output)


def make_cell(
    seed: int,
    width: int,
    memory_records: int,
) -> dict[str, object]:
    rng = random.Random((seed + 1) * 100_003 + width * 101 + memory_records * 7)

    static_records = [
        (
            rng.randint(-2, 2),
            tuple(rng.randint(-7, 7) for _ in range(width)),
        )
        for _ in range(memory_records)
    ]

    # Deliberately strong and distinct mutable records so every registered
    # lifecycle mutation is materially visible downstream.
    old_record = (0, tuple(10 + (coordinate % 5) for coordinate in range(width)))
    new_record = (
        2,
        tuple(-11 - ((coordinate * 3) % 7) for coordinate in range(width)),
    )

    w1 = [[rng.randint(-3, 3) for _ in range(width)] for _ in range(width)]
    w2 = [[rng.randint(-3, 3) for _ in range(width)] for _ in range(width)]
    for coordinate in range(width):
        w1[coordinate][coordinate] += 5
        w2[coordinate][coordinate] += 5

    return {
        "static_records": static_records,
        "old_record": old_record,
        "new_record": new_record,
        "w1": w1,
        "w2": w2,
        "b1": [rng.randint(-2, 2) for _ in range(width)],
        "b2": [rng.randint(-2, 2) for _ in range(width)],
    }


def run_cell(
    seed: int,
    width: int,
    *,
    memory_records: int = 64,
    sessions: int = 32,
    query_buckets: int = 8,
) -> dict[str, object]:
    cell = make_cell(seed, width, memory_records)
    base_static = cell["static_records"]
    old_record = cell["old_record"]
    new_record = cell["new_record"]
    w1 = cell["w1"]
    w2 = cell["w2"]
    b1 = cell["b1"]
    b2 = cell["b2"]

    record_by_label = {
        "old": old_record,
        "new": new_record,
        None: None,
    }

    metrics = {
        "lifecycle_cases": 0,
        "material_final_changes": 0,
        "candidate_fresh_attention_mismatches": 0,
        "generic_fresh_attention_mismatches": 0,
        "candidate_generic_attention_mismatches": 0,
        "candidate_fresh_final_mismatches": 0,
        "generic_fresh_final_mismatches": 0,
        "candidate_attention_update_work": 0,
        "generic_attention_update_work": 0,
        "fresh_attention_work": 0,
        "required_suffix_recompute_work": 0,
        "wrong_one_global_output_delta_failures": 0,
    }

    unique_output_deltas: dict[str, set[tuple[Fraction, ...]]] = {
        name: set() for name, _, _ in LIFECYCLE
    }
    first_delta_by_transition: dict[str, tuple[Fraction, ...]] = {}

    for session in range(sessions):
        session_rng = random.Random(
            (seed + 1) * 99_991 + (session + 1) * 97 + width * 13
        )
        # Distinct cached contexts while the canonical Pod record is shared.
        static_records = [
            (
                key,
                tuple(
                    element + session_rng.randint(-2, 2)
                    for element in value
                ),
            )
            for key, value in base_static
        ]
        query = 1 + (session % query_buckets)

        normalizer, attention_output = fresh_attention(
            query,
            [*static_records, old_record],
        )
        previous_final = dense_suffix(attention_output, w1, b1, w2, b2)

        for lifecycle_name, old_label, new_label in LIFECYCLE:
            old = record_by_label[old_label]
            new = record_by_label[new_label]

            candidate_z, candidate_output, candidate_work = candidate_revision_receipt(
                query,
                attention_output,
                normalizer,
                old,
                new,
            )

            removed = [] if old is None else [old]
            added = [] if new is None else [new]
            generic_z, generic_output, generic_work = generic_dynamic_normalized_aggregate(
                query,
                attention_output,
                normalizer,
                removed,
                added,
            )

            fresh_records = [*static_records]
            if new is not None:
                fresh_records.append(new)
            fresh_z, fresh_output = fresh_attention(query, fresh_records)

            candidate_final = dense_suffix(candidate_output, w1, b1, w2, b2)
            generic_final = dense_suffix(generic_output, w1, b1, w2, b2)
            fresh_final = dense_suffix(fresh_output, w1, b1, w2, b2)

            metrics["lifecycle_cases"] += 1
            if fresh_final != previous_final:
                metrics["material_final_changes"] += 1

            if candidate_z != fresh_z or candidate_output != fresh_output:
                metrics["candidate_fresh_attention_mismatches"] += 1
            if generic_z != fresh_z or generic_output != fresh_output:
                metrics["generic_fresh_attention_mismatches"] += 1
            if candidate_z != generic_z or candidate_output != generic_output:
                metrics["candidate_generic_attention_mismatches"] += 1
            if candidate_final != fresh_final:
                metrics["candidate_fresh_final_mismatches"] += 1
            if generic_final != fresh_final:
                metrics["generic_fresh_final_mismatches"] += 1

            metrics["candidate_attention_update_work"] += candidate_work
            metrics["generic_attention_update_work"] += generic_work

            # Exact fresh attention work: d+1 accumulator updates per record,
            # plus d divisions to materialize the post-softmax output.
            fresh_record_count = memory_records + (0 if new is None else 1)
            metrics["fresh_attention_work"] += (
                fresh_record_count * (width + 1) + width
            )

            # The guarantee-matched exact-ready path still has to recompute the
            # two downstream dense nonlinear task layers.
            metrics["required_suffix_recompute_work"] += 2 * width * width

            output_delta = tuple(
                fresh - stale
                for fresh, stale in zip(fresh_output, attention_output)
            )
            unique_output_deltas[lifecycle_name].add(output_delta)
            if lifecycle_name not in first_delta_by_transition:
                first_delta_by_transition[lifecycle_name] = output_delta
            elif output_delta != first_delta_by_transition[lifecycle_name]:
                metrics["wrong_one_global_output_delta_failures"] += 1

            normalizer = candidate_z
            attention_output = candidate_output
            previous_final = fresh_final

    return {
        "seed": seed,
        "width": width,
        **metrics,
        "unique_exact_output_deltas_by_transition": {
            name: len(values)
            for name, values in unique_output_deltas.items()
        },
    }


def run_registered() -> dict[str, object]:
    cells = [
        run_cell(seed, width)
        for seed in range(16)
        for width in (4, 8, 16)
    ]

    count_keys = (
        "lifecycle_cases",
        "material_final_changes",
        "candidate_fresh_attention_mismatches",
        "generic_fresh_attention_mismatches",
        "candidate_generic_attention_mismatches",
        "candidate_fresh_final_mismatches",
        "generic_fresh_final_mismatches",
        "candidate_attention_update_work",
        "generic_attention_update_work",
        "fresh_attention_work",
        "required_suffix_recompute_work",
        "wrong_one_global_output_delta_failures",
    )
    totals = {
        key: sum(int(cell[key]) for cell in cells)
        for key in count_keys
    }

    minimum_unique_deltas = min(
        min(cell["unique_exact_output_deltas_by_transition"].values())
        for cell in cells
    )

    candidate_ready_work = (
        totals["candidate_attention_update_work"]
        + totals["required_suffix_recompute_work"]
    )
    generic_ready_work = (
        totals["generic_attention_update_work"]
        + totals["required_suffix_recompute_work"]
    )
    fresh_ready_work = (
        totals["fresh_attention_work"]
        + totals["required_suffix_recompute_work"]
    )
    oracle_patch_suffix_work = totals["required_suffix_recompute_work"]

    passed = (
        totals["material_final_changes"] == totals["lifecycle_cases"]
        and totals["candidate_fresh_attention_mismatches"] == 0
        and totals["generic_fresh_attention_mismatches"] == 0
        and totals["candidate_generic_attention_mismatches"] == 0
        and totals["candidate_fresh_final_mismatches"] == 0
        and totals["generic_fresh_final_mismatches"] == 0
        and totals["candidate_attention_update_work"]
        == totals["generic_attention_update_work"]
        and candidate_ready_work == generic_ready_work
        and minimum_unique_deltas == 32
        and totals["wrong_one_global_output_delta_failures"] > 0
        and candidate_ready_work > oracle_patch_suffix_work
    )

    return {
        "experiment": "E-000114",
        "decision": DECISION if passed else "FAIL_REGISTERED_KILL_SCREEN",
        "passed": passed,
        "configuration": {
            "seeds": 16,
            "widths": [4, 8, 16],
            "sessions_per_cell": 32,
            "static_memory_records": 64,
            "query_buckets": 8,
            "lifecycle": [name for name, _, _ in LIFECYCLE],
            "attention": "exact softmax with logits log(2) * integer score",
            "arithmetic": "fractions.Fraction exact rationals",
            "cached_candidate_state_per_session": "query + scalar normalizer + materialized attention output",
        },
        "metrics": {
            "cells": len(cells),
            **totals,
            "minimum_unique_exact_output_deltas_per_transition": minimum_unique_deltas,
            "candidate_ready_work": candidate_ready_work,
            "generic_ready_work": generic_ready_work,
            "fresh_ready_work": fresh_ready_work,
            "oracle_exact_read_patch_plus_suffix_work": oracle_patch_suffix_work,
            "fresh_ready_over_candidate_ready_ratio": (
                fresh_ready_work / candidate_ready_work
            ),
            "candidate_ready_over_oracle_patch_suffix_ratio": (
                candidate_ready_work / oracle_patch_suffix_work
            ),
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/e000114"),
    )
    args = parser.parse_args()

    result = run_registered()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.results_dir
        / "e000114-exact-softmax-sufficient-statistic-revision-receipt-reduction.json"
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "metrics": result["metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
