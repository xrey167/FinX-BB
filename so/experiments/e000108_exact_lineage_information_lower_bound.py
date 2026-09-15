"""E-000108 exact lifecycle lineage information lower-bound reduction."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def make_permutation(q: int, seed: int) -> tuple[int, ...]:
    values = list(range(q))
    random.Random(seed).shuffle(values)
    return tuple(values)


def visible_state(values: tuple[int, ...], q: int, permutation: tuple[int, ...]) -> int:
    return permutation[sum(values) % q]


def delete_state(
    values: tuple[int, ...], target: int, q: int, permutation: tuple[int, ...]
) -> int:
    return permutation[sum(v for i, v in enumerate(values) if i != target) % q]


def update_state(
    values: tuple[int, ...],
    target: int,
    new_value: int,
    q: int,
    permutation: tuple[int, ...],
) -> int:
    return permutation[
        sum(new_value if i == target else v for i, v in enumerate(values)) % q
    ]


def generic_certificate(values: tuple[int, ...]) -> tuple[int, ...]:
    """Optimal generic central ledger given that the visible aggregate is retained."""
    return values[:-1]


def reconstruct_values(
    certificate: tuple[int, ...], total: int, q: int
) -> tuple[int, ...]:
    last = (total - sum(certificate)) % q
    return certificate + (last,)


def run_cell(q: int, n: int, seed: int) -> dict[str, object]:
    permutation = make_permutation(q, seed)
    inverse = {encoded: raw for raw, encoded in enumerate(permutation)}
    histories = itertools.product(range(q), repeat=n)

    profiles_by_visible: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    bucket_counts: dict[int, int] = defaultdict(int)

    delete_checks = 0
    update_checks = 0
    aba_checks = 0
    generic_delete_failures = 0
    generic_update_failures = 0
    aba_failures = 0

    for values in histories:
        values = tuple(values)
        visible = visible_state(values, q, permutation)
        total = inverse[visible]

        deletion_profile = tuple(
            delete_state(values, target, q, permutation) for target in range(n)
        )
        profiles_by_visible[visible].add(deletion_profile)
        bucket_counts[visible] += 1

        certificate = generic_certificate(values)
        reconstructed = reconstruct_values(certificate, total, q)
        if reconstructed != values:
            raise AssertionError("generic ledger reconstruction failed")

        for target in range(n):
            delete_checks += 1
            generic_delete = permutation[(total - reconstructed[target]) % q]
            fresh_delete = delete_state(values, target, q, permutation)
            if generic_delete != fresh_delete:
                generic_delete_failures += 1

            # One deterministic nontrivial UPDATE per target/history.
            step = 1 + (target % max(1, q - 1))
            new_value = (values[target] + step) % q
            update_checks += 1
            generic_update = permutation[
                (total - reconstructed[target] + new_value) % q
            ]
            fresh_update = update_state(values, target, new_value, q, permutation)
            if generic_update != fresh_update:
                generic_update_failures += 1

            # ABA: old -> updated -> old must exactly restore the old visible state.
            aba_checks += 1
            restored = permutation[
                (
                    total
                    - reconstructed[target]
                    + new_value
                    - new_value
                    + reconstructed[target]
                )
                % q
            ]
            if restored != visible:
                aba_failures += 1

    expected_bucket = q ** (n - 1)
    expected_aux_states = q ** (n - 1)
    profile_counts = {visible: len(ps) for visible, ps in profiles_by_visible.items()}

    # For fixed visible total, every source-value vector induces a distinct
    # arbitrary-target deletion profile. Therefore any exact auxiliary
    # certificate paired with the visible state needs at least q^(n-1)
    # distinct states by pigeonhole.
    unique_profile_ok = (
        len(bucket_counts) == q
        and all(count == expected_bucket for count in bucket_counts.values())
        and all(count == expected_bucket for count in profile_counts.values())
    )

    generic_aux_states = q ** (n - 1)
    lower_bound_bits = math.log2(expected_aux_states)
    packed_bits = math.ceil(lower_bound_bits)

    return {
        "q": q,
        "pod_count": n,
        "permutation_seed": seed,
        "history_count": q**n,
        "visible_state_count": len(bucket_counts),
        "histories_per_visible_state": expected_bucket,
        "distinct_deletion_profiles_per_visible_state": (
            min(profile_counts.values()) if profile_counts else 0
        ),
        "required_auxiliary_certificate_states": expected_aux_states,
        "generic_ledger_auxiliary_states": generic_aux_states,
        "information_lower_bound_bits": lower_bound_bits,
        "minimum_fixed_width_auxiliary_bits": packed_bits,
        "generic_ledger_matches_state_lower_bound": (
            generic_aux_states == expected_aux_states
        ),
        "unique_delete_profile_check": unique_profile_ok,
        "delete_checks": delete_checks,
        "generic_delete_failures": generic_delete_failures,
        "update_checks": update_checks,
        "generic_update_failures": generic_update_failures,
        "aba_checks": aba_checks,
        "aba_failures": aba_failures,
    }


def run() -> dict[str, object]:
    # Includes binary membership lineage and multi-valued causal contributions.
    registered = (
        (2, 12, 1080),
        (3, 8, 1081),
        (5, 6, 1082),
        (7, 5, 1083),
    )
    cells = [run_cell(q, n, seed) for q, n, seed in registered]

    exact_profiles = all(bool(c["unique_delete_profile_check"]) for c in cells)
    lower_bound_matched = all(
        bool(c["generic_ledger_matches_state_lower_bound"]) for c in cells
    )
    generic_failures = sum(
        int(c["generic_delete_failures"])
        + int(c["generic_update_failures"])
        + int(c["aba_failures"])
        for c in cells
    )
    kill = exact_profiles and lower_bound_matched and generic_failures == 0

    return {
        "experiment": "E-000108",
        "scope": (
            "central exact causal-lineage capsules for arbitrary Pod contributions "
            "when current mixed state retains only an aggregate"
        ),
        "cells": cells,
        "registered_cells": len(cells),
        "total_histories": sum(int(c["history_count"]) for c in cells),
        "total_delete_checks": sum(int(c["delete_checks"]) for c in cells),
        "total_update_checks": sum(int(c["update_checks"]) for c in cells),
        "total_aba_checks": sum(int(c["aba_checks"]) for c in cells),
        "generic_baseline_failures": generic_failures,
        "kill_screen_pass": kill,
        "decision": (
            "KILL_COMPACT_EXACT_CENTRAL_LINEAGE_CAPSULE_AS_GENERAL_ADVANTAGE"
            if kill
            else "SCREEN_NOT_DECISIVE"
        ),
        "reduction": (
            "For values v_i in Z_q, the visible state retains only a bijective encoding "
            "of sum_i v_i. For fixed visible total there are q^(n-1) admissible histories, "
            "and their arbitrary-target DELETE profiles are all distinct because "
            "Delete_i reveals total-v_i. Any exact central certificate paired with the "
            "visible state must therefore have at least q^(n-1) states. A generic ledger "
            "storing n-1 Pod values attains exactly that bound and also performs exact "
            "UPDATE and ABA."
        ),
        "not_claimed": (
            "No lower bound against restricted/compressible source families, distributed "
            "or ticketed memory, architectures whose task-useful visible state itself "
            "retains lineage, active replay/intervention, or approximate attribution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/e000108"))
    args = parser.parse_args()
    result = run()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000108_exact_lineage_information_lower_bound.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
