from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable

State = tuple[int, int]


def perm_inverse(p: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(p)
    for i, v in enumerate(p):
        out[v] = i
    return tuple(out)


def perm_compose(p: tuple[int, ...], q: tuple[int, ...]) -> tuple[int, ...]:
    """Return p after q."""
    return tuple(p[q[i]] for i in range(len(p)))


def full_cycle(n: int) -> tuple[int, ...]:
    return tuple(list(range(1, n)) + [0])


def conjugate_transport(F: tuple[int, ...], tau: tuple[int, ...]) -> tuple[int, ...]:
    return perm_compose(perm_compose(F, tau), perm_inverse(F))


def exhaustive_conjugacy_cell(n: int) -> dict[str, int | float | bool]:
    tau = full_cycle(n)
    transports: set[tuple[int, ...]] = set()
    total = 0
    for F_raw in permutations(range(n)):
        F = tuple(F_raw)
        transports.add(conjugate_transport(F, tau))
        total += 1
    expected = math.factorial(n - 1)
    return {
        "n": n,
        "suffix_bijections_enumerated": total,
        "distinct_exact_transport_maps": len(transports),
        "expected_n_cycle_conjugacy_class": expected,
        "class_match": len(transports) == expected,
        "receipt_lower_bound_bits_if_transport_materialized_without_suffix_access": math.log2(expected),
    }


@dataclass(frozen=True)
class Shear:
    target: int
    alpha: int
    beta: int
    gamma: int

    def forward(self, state: State, p: int) -> State:
        a, b = state
        if self.target == 1:
            delta = (self.alpha * a * a + self.beta * a + self.gamma) % p
            return a, (b + delta) % p
        delta = (self.alpha * b * b + self.beta * b + self.gamma) % p
        return (a + delta) % p, b

    def inverse(self, state: State, p: int) -> State:
        a, b = state
        if self.target == 1:
            delta = (self.alpha * a * a + self.beta * a + self.gamma) % p
            return a, (b - delta) % p
        delta = (self.alpha * b * b + self.beta * b + self.gamma) % p
        return (a - delta) % p, b


@dataclass(frozen=True)
class ReversibleSuffix:
    p: int
    layers: tuple[Shear, ...]

    def forward(self, state: State) -> State:
        for layer in self.layers:
            state = layer.forward(state, self.p)
        return state

    def inverse(self, state: State) -> State:
        for layer in reversed(self.layers):
            state = layer.inverse(state, self.p)
        return state


def make_suffix(seed: int, p: int, depth: int) -> ReversibleSuffix:
    rng = random.Random((seed + 1) * 1_000_003 + p * 1009 + depth)
    layers = []
    for i in range(depth):
        target = i % 2
        alpha = rng.randrange(1, p)
        beta = rng.randrange(0, p)
        gamma = rng.randrange(0, p)
        layers.append(Shear(target=target, alpha=alpha, beta=beta, gamma=gamma))
    return ReversibleSuffix(p=p, layers=tuple(layers))


def pod_edit(state: State, delta: int, p: int) -> State:
    return ((state[0] + delta) % p, state[1])


def candidate_state_only_transport(
    suffix: ReversibleSuffix, old_final: State, delta: int
) -> State:
    old_read_state = suffix.inverse(old_final)
    return suffix.forward(pod_edit(old_read_state, delta, suffix.p))


def baseline_suffix_recompute(
    suffix: ReversibleSuffix, old_read_state: State, delta: int
) -> State:
    return suffix.forward(pod_edit(old_read_state, delta, suffix.p))


def independently_compile_generic_transport_table(
    suffix: ReversibleSuffix, delta: int
) -> dict[State, State]:
    """Construct F(x)->F(tau(x)) without calling candidate inverse transport."""
    table: dict[State, State] = {}
    p = suffix.p
    for a in range(p):
        for b in range(p):
            x = (a, b)
            old_y = suffix.forward(x)
            new_y = suffix.forward(pod_edit(x, delta, p))
            if old_y in table and table[old_y] != new_y:
                raise AssertionError("suffix is not bijective")
            table[old_y] = new_y
    return table


def candidate_conjugate_table(
    suffix: ReversibleSuffix, delta: int
) -> dict[State, State]:
    """Materialize F o tau_delta o F^-1 over the complete final-state domain."""
    p = suffix.p
    table: dict[State, State] = {}
    for a in range(p):
        for b in range(p):
            y = (a, b)
            table[y] = candidate_state_only_transport(suffix, y, delta)
    return table


def registered_reversible_cells(
    seeds: Iterable[int] = range(16),
    primes: Iterable[int] = (11, 13, 17),
    depths: Iterable[int] = (2, 4, 8, 12),
    deltas: Iterable[int] = (1, 2, -1),
) -> dict:
    seeds = tuple(seeds)
    primes = tuple(primes)
    depths = tuple(depths)
    deltas = tuple(deltas)

    total_state_cases = 0
    candidate_fresh_mismatches = 0
    generic_fresh_mismatches = 0
    candidate_generic_table_mismatches = 0
    aba_failures = 0
    material_cases = 0
    cells = []

    candidate_layer_evals = 0
    baseline_layer_evals = 0
    table_entries = 0

    for seed in seeds:
        for p in primes:
            for depth in depths:
                suffix = make_suffix(seed, p, depth)

                inverse_failures = 0
                for a in range(p):
                    for b in range(p):
                        x = (a, b)
                        if suffix.inverse(suffix.forward(x)) != x:
                            inverse_failures += 1
                if inverse_failures:
                    raise AssertionError(
                        f"invertibility failure: seed={seed} p={p} depth={depth}"
                    )

                cell_cases = 0
                cell_mismatches = 0
                for delta in deltas:
                    candidate_table = candidate_conjugate_table(suffix, delta)
                    generic_table = independently_compile_generic_transport_table(
                        suffix, delta
                    )
                    table_entries += len(candidate_table)
                    if candidate_table != generic_table:
                        candidate_generic_table_mismatches += sum(
                            candidate_table[k] != generic_table[k]
                            for k in candidate_table
                        )

                    for a in range(p):
                        for b in range(p):
                            x = (a, b)
                            old_y = suffix.forward(x)
                            fresh_y = suffix.forward(pod_edit(x, delta, p))
                            cand_y = candidate_state_only_transport(
                                suffix, old_y, delta
                            )
                            gen_y = generic_table[old_y]

                            total_state_cases += 1
                            cell_cases += 1
                            if cand_y != fresh_y:
                                candidate_fresh_mismatches += 1
                                cell_mismatches += 1
                            if gen_y != fresh_y:
                                generic_fresh_mismatches += 1
                            if fresh_y != old_y:
                                material_cases += 1

                            restored = candidate_state_only_transport(
                                suffix, cand_y, -delta
                            )
                            if restored != old_y:
                                aba_failures += 1

                            candidate_layer_evals += 2 * depth
                            baseline_layer_evals += depth

                cells.append(
                    {
                        "seed": seed,
                        "p": p,
                        "depth": depth,
                        "states": p * p,
                        "deltas": len(deltas),
                        "cases": cell_cases,
                        "candidate_fresh_mismatches": cell_mismatches,
                        "state_only_candidate_layer_evals_per_case": 2 * depth,
                        "last_read_suffix_recompute_layer_evals_per_case": depth,
                        "candidate_to_baseline_layer_eval_ratio": 2.0,
                        "conjugate_table_entries_per_edit": p * p,
                    }
                )

    return {
        "cells": cells,
        "totals": {
            "registered_cells": len(cells),
            "exact_state_transport_cases": total_state_cases,
            "candidate_fresh_mismatches": candidate_fresh_mismatches,
            "generic_fresh_mismatches": generic_fresh_mismatches,
            "candidate_generic_table_mismatches": candidate_generic_table_mismatches,
            "aba_failures": aba_failures,
            "material_cases": material_cases,
            "material_fraction": material_cases / total_state_cases,
            "candidate_layer_evals": candidate_layer_evals,
            "baseline_layer_evals": baseline_layer_evals,
            "candidate_to_baseline_layer_eval_ratio": (
                candidate_layer_evals / baseline_layer_evals
            ),
            "materialized_transport_table_entries": table_entries,
        },
    }


def run_registered() -> dict:
    conjugacy = [exhaustive_conjugacy_cell(n) for n in (4, 5, 6, 7, 8)]
    reversible = registered_reversible_cells()
    totals = reversible["totals"]
    kill = (
        all(c["class_match"] for c in conjugacy)
        and totals["candidate_fresh_mismatches"] == 0
        and totals["generic_fresh_mismatches"] == 0
        and totals["candidate_generic_table_mismatches"] == 0
        and totals["aba_failures"] == 0
        and totals["material_fraction"] == 1.0
        and totals["candidate_to_baseline_layer_eval_ratio"] >= 2.0
        and all(
            c["candidate_to_baseline_layer_eval_ratio"] >= 2.0
            for c in reversible["cells"]
        )
    )
    return {
        "experiment": "E-000109",
        "title": "Reversible Suffix Conjugacy Exact Transport Reduction",
        "kill_screen_pass": kill,
        "decision": (
            "KILL_REVERSIBILITY_AND_CONJUGACY_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE"
            if kill
            else "DO_NOT_KILL"
        ),
        "major_invention": False,
        "conjugacy_enumeration": conjugacy,
        "reversible_suffix": reversible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ci-e109"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result = run_registered()
    out = args.results_dir / "e000109-result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["reversible_suffix"]["totals"], indent=2, sort_keys=True))
    print(json.dumps(result["conjugacy_enumeration"], indent=2, sort_keys=True))
    print(result["decision"])


if __name__ == "__main__":
    main()
