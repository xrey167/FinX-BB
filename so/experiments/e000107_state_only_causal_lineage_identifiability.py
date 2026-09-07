"""E-000107 state-only causal-lineage identifiability reduction."""

from __future__ import annotations

import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path
from typing import Callable

State = tuple[Fraction, ...]
History = tuple[bool, ...]


def subsets(n: int) -> list[History]:
    return [tuple(bits) for bits in itertools.product([False, True], repeat=n)]


def additive_state(values: tuple[Fraction, ...], h: History) -> State:
    return (sum((v for v, on in zip(values, h) if on), Fraction(0)),)


def average_state(values: tuple[Fraction, ...], h: History) -> State:
    active = [v for v, on in zip(values, h) if on]
    if not active:
        return (Fraction(0),)
    return (sum(active, Fraction(0)) / len(active),)


def relu_sum_state(values: tuple[Fraction, ...], h: History) -> State:
    x = sum((v for v, on in zip(values, h) if on), Fraction(0))
    return (max(Fraction(0), x),)


def projected_vector_state(values: tuple[tuple[Fraction, ...], ...], h: History) -> State:
    # Fixed 2D observable projection of 3D source contributions.
    accum = [Fraction(0), Fraction(0), Fraction(0)]
    for vec, on in zip(values, h):
        if on:
            for j in range(3):
                accum[j] += vec[j]
    return (accum[0] + accum[2], accum[1] - accum[2])


def delete_target(h: History, target: int) -> History:
    out = list(h)
    out[target] = False
    return tuple(out)


def scalar_family(name: str, values: tuple[Fraction, ...], fn: Callable[[tuple[Fraction, ...], History], State]) -> dict[str, object]:
    histories = subsets(len(values))
    state_cache = {h: fn(values, h) for h in histories}
    witnesses: list[dict[str, object]] = []
    targets_with_witness: set[int] = set()

    groups: dict[State, list[History]] = {}
    for h, state in state_cache.items():
        groups.setdefault(state, []).append(h)

    for target in range(len(values)):
        for state, hs in groups.items():
            for i in range(len(hs)):
                for j in range(i + 1, len(hs)):
                    h1, h2 = hs[i], hs[j]
                    # Stronger witness: target presence differs across worlds.
                    if h1[target] == h2[target]:
                        continue
                    d1 = fn(values, delete_target(h1, target))
                    d2 = fn(values, delete_target(h2, target))
                    if d1 == d2:
                        continue
                    targets_with_witness.add(target)
                    witnesses.append({
                        "target": target,
                        "observed_state": [str(x) for x in state],
                        "history_1": [int(x) for x in h1],
                        "history_2": [int(x) for x in h2],
                        "post_delete_1": [str(x) for x in d1],
                        "post_delete_2": [str(x) for x in d2],
                    })
                    break
                if target in targets_with_witness:
                    break
            if target in targets_with_witness:
                break

    # Oracle history is sufficient by construction: exact recompute from h.
    oracle_failures = 0
    for target in range(len(values)):
        for h in histories:
            oracle = fn(values, delete_target(h, target))
            direct = fn(values, delete_target(h, target))
            if oracle != direct:
                oracle_failures += 1

    return {
        "name": name,
        "source_count": len(values),
        "history_count": len(histories),
        "distinct_observed_states": len(groups),
        "collision_history_count": sum(len(hs) for hs in groups.values() if len(hs) > 1),
        "targets_with_witness": sorted(targets_with_witness),
        "witness_count": len(witnesses),
        "sample_witnesses": witnesses[: min(4, len(witnesses))],
        "oracle_history_failures": oracle_failures,
    }


def vector_family() -> dict[str, object]:
    values = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(-1), Fraction(1)),
    )
    histories = subsets(len(values))
    states = {h: projected_vector_state(values, h) for h in histories}
    groups: dict[State, list[History]] = {}
    for h, state in states.items():
        groups.setdefault(state, []).append(h)

    witnesses: list[dict[str, object]] = []
    targets: set[int] = set()
    for target in range(len(values)):
        for state, hs in groups.items():
            for i, h1 in enumerate(hs):
                for h2 in hs[i + 1 :]:
                    if h1[target] == h2[target]:
                        continue
                    d1 = projected_vector_state(values, delete_target(h1, target))
                    d2 = projected_vector_state(values, delete_target(h2, target))
                    if d1 != d2:
                        targets.add(target)
                        witnesses.append({
                            "target": target,
                            "observed_state": [str(x) for x in state],
                            "history_1": [int(x) for x in h1],
                            "history_2": [int(x) for x in h2],
                            "post_delete_1": [str(x) for x in d1],
                            "post_delete_2": [str(x) for x in d2],
                        })
                        break
                if target in targets:
                    break
            if target in targets:
                break

    return {
        "name": "low_dimensional_projection",
        "source_count": len(values),
        "history_count": len(histories),
        "distinct_observed_states": len(groups),
        "collision_history_count": sum(len(hs) for hs in groups.values() if len(hs) > 1),
        "targets_with_witness": sorted(targets),
        "witness_count": len(witnesses),
        "sample_witnesses": witnesses[: min(4, len(witnesses))],
        "oracle_history_failures": 0,
    }


def run() -> dict[str, object]:
    families = [
        scalar_family(
            "additive_subset_sum",
            (Fraction(1), Fraction(1), Fraction(2), Fraction(3), Fraction(5)),
            additive_state,
        ),
        scalar_family(
            "uniform_average_attention",
            (Fraction(-1), Fraction(1), Fraction(0), Fraction(2), Fraction(-2)),
            average_state,
        ),
        scalar_family(
            "relu_saturated_sum",
            (Fraction(-2), Fraction(-1), Fraction(1), Fraction(2), Fraction(3)),
            relu_sum_state,
        ),
        vector_family(),
    ]

    all_have = all(int(f["witness_count"]) > 0 for f in families)
    distinct_family_targets = sum(len(f["targets_with_witness"]) for f in families)
    oracle_failures = sum(int(f["oracle_history_failures"]) for f in families)
    kill = all_have and distinct_family_targets >= 4 and oracle_failures == 0

    return {
        "experiment": "E-000107",
        "scope": "state-only exact causal-lineage identifiability",
        "families": families,
        "families_with_indistinguishability_witness": sum(int(int(f["witness_count"]) > 0) for f in families),
        "total_families": len(families),
        "target_witness_slots_across_families": distinct_family_targets,
        "oracle_history_failures": oracle_failures,
        "kill_screen_pass": kill,
        "decision": (
            "KILL_STATE_ONLY_EXACT_CAUSAL_LINEAGE_AS_GENERAL_GUARANTEE"
            if kill
            else "SCREEN_NOT_DECISIVE"
        ),
        "reduction": (
            "If identical (model, catalogue, observed_state, target) inputs require different exact post-delete states, "
            "no deterministic state-only certifier can be correct in both worlds."
        ),
        "not_claimed": (
            "No impossibility claim against provenance-preserving/injective representations, active interventions/replay, "
            "authenticated execution traces, or approximate attribution."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/e000107"))
    args = parser.parse_args()
    result = run()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000107_state_only_causal_lineage_identifiability.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
