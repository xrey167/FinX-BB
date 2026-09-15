"""E-000104: exact tensor-train / MPS lifecycle quotient reduction.

Pure Python + Fraction structural kill screen. No model download is required.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as F
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Vec = List[F]
Mat = List[List[F]]


def eye(n: int) -> Mat:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def mat_add(a: Mat, b: Mat) -> Mat:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_scale(c: F, a: Mat) -> Mat:
    return [[c * x for x in row] for row in a]


def row_times_mat(row: Sequence[F], m: Mat) -> Vec:
    r = len(row)
    assert len(m) == r and len(m[0]) == r
    return [sum((row[i] * m[i][j] for i in range(r)), F(0)) for j in range(r)]


def mat_times_vec(m: Mat, x: Sequence[F]) -> Vec:
    r = len(x)
    assert len(m) == r and len(m[0]) == r
    return [sum((m[i][j] * x[j] for j in range(r)), F(0)) for i in range(r)]


def dot(a: Sequence[F], b: Sequence[F]) -> F:
    return sum((x * y for x, y in zip(a, b)), F(0))


def generic_matmul(a: Mat, b: Mat) -> Mat:
    """Independent generic matrix multiplication used by the baseline path."""
    m, k, n = len(a), len(a[0]), len(b[0])
    assert len(b) == k
    return [
        [sum((a[i][t] * b[t][j] for t in range(k)), F(0)) for j in range(n)]
        for i in range(m)
    ]


def full_contract(cores: Sequence[Mat], left: Sequence[F], right: Sequence[F]) -> F:
    row = list(left)
    for core in cores:
        row = row_times_mat(row, core)
    return dot(row, right)


def candidate_environments(
    cores: Sequence[Mat], target: int, left: Sequence[F], right: Sequence[F]
) -> Tuple[Vec, Vec]:
    """Lifecycle-labelled cached quotient around one mutable Pod core."""
    l = list(left)
    for i in range(target):
        l = row_times_mat(l, cores[i])
    r = list(right)
    for i in range(len(cores) - 1, target, -1):
        r = mat_times_vec(cores[i], r)
    return l, r


def candidate_local_update(left_env: Sequence[F], new_core: Mat, right_env: Sequence[F]) -> F:
    """Specialized lifecycle update implementation."""
    rank = len(left_env)
    tmp = [F(0) for _ in range(rank)]
    for j in range(rank):
        acc = F(0)
        for i in range(rank):
            acc += left_env[i] * new_core[i][j]
        tmp[j] = acc
    out = F(0)
    for j in range(rank):
        out += tmp[j] * right_env[j]
    return out


def generic_cached_environment_update(
    left_env: Sequence[F], new_core: Mat, right_env: Sequence[F]
) -> F:
    """Ordinary tensor-network contraction using the same cached environments.

    This path deliberately does not call candidate_local_update.
    """
    a = [list(left_env)]
    b = [[x] for x in right_env]
    return generic_matmul(generic_matmul(a, new_core), b)[0][0]


def local_update_multiplies(rank: int) -> int:
    # (1 x R)(R x R) plus (1 x R)(R x 1)
    return rank * rank + rank


def full_contract_multiplies(pod_count: int, rank: int) -> int:
    return pod_count * rank * rank + rank


def _ones(rank: int) -> Mat:
    return [[F(1) for _ in range(rank)] for _ in range(rank)]


def _old_core(rng: random.Random, rank: int, pod_index: int) -> Mat:
    m: Mat = []
    for i in range(rank):
        row: Vec = []
        for j in range(rank):
            if i == j:
                row.append(F(2 + ((pod_index + i + rng.randint(0, 2)) % 3)))
            else:
                row.append(F(rng.randint(0, 1)))
        m.append(row)
    return m


def generate_cell(seed: int, pod_count: int, rank: int, session_count: int) -> Tuple[List[Dict[str, Mat]], List[Tuple[Vec, Vec]]]:
    rng = random.Random((seed + 1) * 1_000_003 + pod_count * 10_007 + rank * 997)
    ones = _ones(rank)
    states: List[Dict[str, Mat]] = []
    for p in range(pod_count):
        old = _old_core(rng, rank, p)
        new_step = F(1 + ((seed + p) % 2))
        restore_step = F(3 + ((seed + 2 * p) % 3))
        states.append(
            {
                "old": old,
                "new": mat_add(old, mat_scale(new_step, ones)),
                "absent": eye(rank),
                "restore": mat_add(old, mat_scale(restore_step, ones)),
            }
        )
    sessions: List[Tuple[Vec, Vec]] = []
    for s in range(session_count):
        left = [F(1 + rng.randint(0, 3)) for _ in range(rank)]
        right = [F(1 + rng.randint(0, 3)) for _ in range(rank)]
        # Make boundary pairs distinct even if the RNG happens to collide.
        left[s % rank] += F(s + 1)
        sessions.append((left, right))
    return states, sessions


def materialize(states: Sequence[Dict[str, Mat]], overrides: Dict[int, str] | None = None) -> List[Mat]:
    overrides = overrides or {}
    return [states[i][overrides.get(i, "old")] for i in range(len(states))]


def third_order_interaction(
    states: Sequence[Dict[str, Mat]], selected: Sequence[int], left: Sequence[F], right: Sequence[F]
) -> F:
    assert len(selected) == 3 and len(set(selected)) == 3
    total = F(0)
    for mask in range(8):
        overrides: Dict[int, str] = {}
        ones = 0
        for bit, pod in enumerate(selected):
            if mask & (1 << bit):
                overrides[pod] = "new"
                ones += 1
        sign = F(-1 if ((3 - ones) % 2) else 1)
        total += sign * full_contract(materialize(states, overrides), left, right)
    return total


def run(
    seed_count: int = 16,
    pod_counts: Sequence[int] = (4, 6, 8, 10, 12),
    ranks: Sequence[int] = (2, 3, 4),
    session_count: int = 16,
) -> dict:
    cells = 0
    session_cells = 0
    lifecycle_cases = 0
    candidate_fresh_mismatch = 0
    generic_fresh_mismatch = 0
    candidate_generic_mismatch = 0
    work_mismatch = 0
    material_update_cases = 0
    interaction_checks = 0
    interaction_material = 0
    aba_cases = 0
    aba_mismatch = 0
    max_integer_bits = 0
    total_candidate_multiplies = 0
    total_generic_multiplies = 0
    total_full_gold_multiplies = 0
    best_full_to_local_ratio = 0.0
    worst_full_to_local_ratio = float("inf")

    lifecycle_states = (
        ("UPDATE", "new"),
        ("DELETE", "absent"),
        ("RESTORE", "restore"),
        ("ROLLBACK", "old"),
        ("ABA", "old"),
    )

    def observe(x: F) -> None:
        nonlocal max_integer_bits
        max_integer_bits = max(max_integer_bits, abs(x.numerator).bit_length(), x.denominator.bit_length())

    for seed in range(seed_count):
        for pod_count in pod_counts:
            if pod_count < 4:
                raise ValueError("registered interaction control requires pod_count >= 4")
            target = pod_count // 2
            triple = (target - 1, target, target + 1)
            for rank in ranks:
                cells += 1
                states, sessions = generate_cell(seed, pod_count, rank, session_count)
                old_cores = materialize(states)
                c_work = local_update_multiplies(rank)
                g_work = local_update_multiplies(rank)
                f_work = full_contract_multiplies(pod_count, rank)
                ratio = f_work / c_work
                best_full_to_local_ratio = max(best_full_to_local_ratio, ratio)
                worst_full_to_local_ratio = min(worst_full_to_local_ratio, ratio)

                for left, right in sessions:
                    session_cells += 1
                    old_gold = full_contract(old_cores, left, right)
                    left_env, right_env = candidate_environments(old_cores, target, left, right)

                    new_gold = full_contract(materialize(states, {target: "new"}), left, right)
                    if new_gold != old_gold:
                        material_update_cases += 1

                    interaction_checks += 1
                    interaction = third_order_interaction(states, triple, left, right)
                    observe(interaction)
                    if interaction != 0:
                        interaction_material += 1

                    # UPDATE, DELETE, RESTORE, ROLLBACK and ABA all reuse the
                    # same unaffected environments; only the target core changes.
                    for label, state_name in lifecycle_states:
                        lifecycle_cases += 1
                        fresh = full_contract(materialize(states, {target: state_name}), left, right)
                        candidate = candidate_local_update(left_env, states[target][state_name], right_env)
                        generic = generic_cached_environment_update(left_env, states[target][state_name], right_env)
                        observe(fresh)
                        observe(candidate)
                        observe(generic)
                        if candidate != fresh:
                            candidate_fresh_mismatch += 1
                        if generic != fresh:
                            generic_fresh_mismatch += 1
                        if candidate != generic:
                            candidate_generic_mismatch += 1
                        if c_work != g_work:
                            work_mismatch += 1
                        total_candidate_multiplies += c_work
                        total_generic_multiplies += g_work
                        total_full_gold_multiplies += f_work

                        if label == "ABA":
                            aba_cases += 1
                            # Explicit sequence witness: old -> new -> old.
                            mid = candidate_local_update(left_env, states[target]["new"], right_env)
                            back = candidate_local_update(left_env, states[target]["old"], right_env)
                            if mid == old_gold or back != old_gold:
                                aba_mismatch += 1

    kill_screen_pass = (
        candidate_fresh_mismatch == 0
        and generic_fresh_mismatch == 0
        and candidate_generic_mismatch == 0
        and work_mismatch == 0
        and material_update_cases == session_cells
        and interaction_material == interaction_checks
        and aba_mismatch == 0
    )
    return {
        "experiment": "E-000104",
        "scope": "exact tensor-train/MPS lifecycle quotient reduction",
        "seed_count": seed_count,
        "pod_counts": list(pod_counts),
        "ranks": list(ranks),
        "session_count_per_cell": session_count,
        "cells": cells,
        "session_cells": session_cells,
        "lifecycle_cases": lifecycle_cases,
        "candidate_fresh_mismatch": candidate_fresh_mismatch,
        "generic_fresh_mismatch": generic_fresh_mismatch,
        "candidate_generic_mismatch": candidate_generic_mismatch,
        "work_mismatch": work_mismatch,
        "material_update_cases": material_update_cases,
        "interaction_checks": interaction_checks,
        "interaction_material": interaction_material,
        "aba_cases": aba_cases,
        "aba_mismatch": aba_mismatch,
        "total_candidate_mutation_multiplies": total_candidate_multiplies,
        "total_generic_mutation_multiplies": total_generic_multiplies,
        "total_full_gold_multiplies": total_full_gold_multiplies,
        "worst_full_to_local_multiply_ratio": worst_full_to_local_ratio,
        "best_full_to_local_multiply_ratio": best_full_to_local_ratio,
        "max_exact_integer_bit_length": max_integer_bits,
        "kill_screen_pass": kill_screen_pass,
        "decision": (
            "KILL_TENSOR_TRAIN_QUOTIENT_AS_STANDALONE_NOVELTY_SEAM"
            if kill_screen_pass
            else "DO_NOT_INTERPRET_REDUCTION"
        ),
        "not_claimed": (
            "No impossibility theorem for lifecycle-native neural architectures; no real-reader, "
            "LLM deletion, matched-memory, inference-overhead, or patent-clearance claim."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--results-dir", type=Path, default=Path("so/results/e000104"))
    args = ap.parse_args()
    result = run(seed_count=args.seed_count, session_count=args.sessions)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000104_tensor_train_lifecycle_quotient_reduction.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
