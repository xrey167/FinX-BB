"""E-000103: exact low-rank resolvent lifecycle transport reduction.

Pure-Python / Fraction structural kill screen. No model download is required.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Vec = List[F]
Mat = List[List[F]]


def _f(x: int | F) -> F:
    return x if isinstance(x, F) else F(x)


def eye(n: int) -> Mat:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def mat_copy(a: Mat) -> Mat:
    return [row[:] for row in a]


def mat_add(a: Mat, b: Mat) -> Mat:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def mat_sub(a: Mat, b: Mat) -> Mat:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matmul(a: Mat, b: Mat) -> Mat:
    m, k, n = len(a), len(a[0]), len(b[0])
    assert k == len(b)
    return [[sum((a[i][t] * b[t][j] for t in range(k)), F(0)) for j in range(n)] for i in range(m)]


def matvec(a: Mat, x: Sequence[F]) -> Vec:
    return [sum((aij * xj for aij, xj in zip(row, x)), F(0)) for row in a]


def transpose(a: Mat) -> Mat:
    return [list(col) for col in zip(*a)]


def outer(u: Sequence[F], v: Sequence[F]) -> Mat:
    return [[ui * vj for vj in v] for ui in u]


def dot(a: Sequence[F], b: Sequence[F]) -> F:
    return sum((x * y for x, y in zip(a, b)), F(0))


def vec_add(a: Sequence[F], b: Sequence[F]) -> Vec:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: Sequence[F], b: Sequence[F]) -> Vec:
    return [x - y for x, y in zip(a, b)]


def vec_scale(c: F, x: Sequence[F]) -> Vec:
    return [c * xi for xi in x]


def inverse(a: Mat) -> Mat:
    n = len(a)
    assert n and all(len(row) == n for row in a)
    aug = [a[i][:] + eye(n)[i] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ZeroDivisionError("singular matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        p = aug[col][col]
        aug[col] = [x / p for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            c = aug[r][col]
            if c:
                aug[r] = [x - c * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def solve(a: Mat, b: Sequence[F]) -> Vec:
    return matvec(inverse(a), b)


def rank1_inverse_update(inv: Mat, u: Sequence[F], v: Sequence[F]) -> Tuple[Mat, Vec, F]:
    """Return inverse of M + u v^T from inv(M), plus reusable z and denominator."""
    z = matvec(inv, u)
    denom = F(1) + dot(v, z)
    if denom == 0:
        raise ZeroDivisionError("Sherman-Morrison singular update")
    w = matvec(transpose(inv), v)  # inv^T v, so w^T = v^T inv
    correction = [[z[i] * w[j] / denom for j in range(len(w))] for i in range(len(z))]
    return mat_sub(inv, correction), z, denom


def rank1_solution_update(x: Sequence[F], z: Sequence[F], denom: F, v: Sequence[F]) -> Vec:
    return vec_sub(x, vec_scale(dot(v, x) / denom, z))


def _columns(cols: Sequence[Sequence[F]]) -> Mat:
    return [list(row) for row in zip(*cols)]


def woodbury_rankk_inverse(inv: Mat, u_cols: Sequence[Sequence[F]], v_cols: Sequence[Sequence[F]]) -> Mat:
    """Direct Woodbury for M + U V^T, independently from sequential rank-1 update."""
    u = _columns(u_cols)  # n x k
    v = _columns(v_cols)  # n x k
    inv_u = matmul(inv, u)  # n x k
    vt_inv_u = matmul(transpose(v), inv_u)  # k x k
    kdim = len(u_cols)
    middle = mat_add(eye(kdim), vt_inv_u)
    middle_inv = inverse(middle)
    vt_inv = matmul(transpose(v), inv)  # k x n
    correction = matmul(matmul(inv_u, middle_inv), vt_inv)
    return mat_sub(inv, correction)


def woodbury_rankk_solution(
    x: Sequence[F], inv: Mat, u_cols: Sequence[Sequence[F]], v_cols: Sequence[Sequence[F]]
) -> Vec:
    u = _columns(u_cols)
    v = _columns(v_cols)
    inv_u = matmul(inv, u)
    middle = mat_add(eye(len(u_cols)), matmul(transpose(v), inv_u))
    middle_inv = inverse(middle)
    vt_x = matvec(transpose(v), x)
    coeff = matvec(middle_inv, vt_x)
    return vec_sub(x, matvec(inv_u, coeff))


@dataclass(frozen=True)
class PodRank1:
    old_u: Tuple[F, ...]
    old_v: Tuple[F, ...]
    new_u: Tuple[F, ...]
    new_v: Tuple[F, ...]


def add_rank1(m: Mat, u: Sequence[F], v: Sequence[F], sign: int = 1) -> Mat:
    o = outer(u, v)
    if sign == -1:
        o = [[-x for x in row] for row in o]
    return mat_add(m, o)


def operator(base: Mat, pods: Sequence[PodRank1], overrides: dict[int, str] | None = None) -> Mat:
    overrides = overrides or {}
    m = mat_copy(base)
    for i, p in enumerate(pods):
        state = overrides.get(i, "old")
        if state == "absent":
            continue
        if state == "old":
            m = add_rank1(m, p.old_u, p.old_v)
        elif state == "new":
            m = add_rank1(m, p.new_u, p.new_v)
        else:
            raise ValueError(state)
    return m


def _rand_vec(rng: random.Random, n: int, lo: int = -2, hi: int = 2) -> Tuple[F, ...]:
    while True:
        x = tuple(F(rng.randint(lo, hi)) for _ in range(n))
        if any(xi != 0 for xi in x):
            return x


def generate_system(seed: int, n: int) -> Tuple[Mat, List[PodRank1], List[Vec]]:
    """Generate a deterministic, well-conditioned-in-practice exact rational system.

    We still verify exact invertibility of every lifecycle operator and retry deterministically.
    """
    for attempt in range(200):
        rng = random.Random((seed + 1) * 1_000_003 + n * 9_973 + attempt * 97)
        base = [[F(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    base[i][j] = F(80 + 3 * i + rng.randint(0, 5))
                else:
                    base[i][j] = F(rng.choice([-1, 0, 0, 0, 1]))
        pods = [
            PodRank1(
                old_u=_rand_vec(rng, n),
                old_v=_rand_vec(rng, n),
                new_u=_rand_vec(rng, n),
                new_v=_rand_vec(rng, n),
            )
            for _ in range(4)
        ]
        try:
            inverse(operator(base, pods))
            for p in range(4):
                inverse(operator(base, pods, {p: "absent"}))
                inverse(operator(base, pods, {p: "new"}))
                q = (p + 1) % 4
                inverse(operator(base, pods, {q: "absent"}))
                inverse(operator(base, pods, {p: "new", q: "absent"}))
        except ZeroDivisionError:
            continue
        sessions: List[Vec] = []
        for _ in range(24):
            b = [F(rng.randint(-6, 6)) for _ in range(n)]
            if not any(b):
                b[0] = F(1)
            sessions.append(b)
        return base, pods, sessions
    raise RuntimeError(f"failed to generate invertible system seed={seed} n={n}")


def _eq_mat(a: Mat, b: Mat) -> bool:
    return a == b


def _eq_vec(a: Sequence[F], b: Sequence[F]) -> bool:
    return list(a) == list(b)


def run(seed_count: int = 16, dimensions: Sequence[int] = (4, 6, 8, 10), sessions_per_system: int = 24) -> dict:
    if sessions_per_system != 24:
        raise ValueError("registered assay fixes sessions_per_system=24")

    revision_cells = 0
    session_cases = 0
    candidate_fresh_mismatch = 0
    generic_fresh_mismatch = 0
    candidate_generic_inverse_mismatch = 0
    delete_mismatch = 0
    restore_mismatch = 0
    aba_mismatch = 0
    material_changed = 0
    interaction_cells = 0
    interaction_cells_material = 0
    max_num_bits = 0
    max_den_bits = 0

    def observe_fraction(x: F) -> None:
        nonlocal max_num_bits, max_den_bits
        max_num_bits = max(max_num_bits, abs(x.numerator).bit_length())
        max_den_bits = max(max_den_bits, x.denominator.bit_length())

    for seed in range(seed_count):
        for n in dimensions:
            base, pods, sessions = generate_system(seed, n)
            old_m = operator(base, pods)
            old_inv = inverse(old_m)
            for pidx, pod in enumerate(pods):
                revision_cells += 1
                # Candidate: REMOVE old contribution, then INSERT new contribution.
                neg_old_u = tuple(-x for x in pod.old_u)
                cand_del_inv, z_del, den_del = rank1_inverse_update(old_inv, neg_old_u, pod.old_v)
                cand_new_inv, z_add, den_add = rank1_inverse_update(cand_del_inv, pod.new_u, pod.new_v)

                fresh_del_m = operator(base, pods, {pidx: "absent"})
                fresh_del_inv = inverse(fresh_del_m)
                fresh_new_m = operator(base, pods, {pidx: "new"})
                fresh_new_inv = inverse(fresh_new_m)

                # Independent generic baseline: one rank-2 Woodbury update.
                generic_new_inv = woodbury_rankk_inverse(
                    old_inv,
                    [neg_old_u, pod.new_u],
                    [pod.old_v, pod.new_v],
                )
                if not _eq_mat(cand_del_inv, fresh_del_inv):
                    delete_mismatch += 1
                if not _eq_mat(cand_new_inv, fresh_new_inv):
                    restore_mismatch += 1
                if not _eq_mat(cand_new_inv, generic_new_inv):
                    candidate_generic_inverse_mismatch += 1

                # ABA: revised -> remove new -> restore exact old contribution.
                neg_new_u = tuple(-x for x in pod.new_u)
                aba_del_inv, _, _ = rank1_inverse_update(cand_new_inv, neg_new_u, pod.new_v)
                aba_old_inv, _, _ = rank1_inverse_update(aba_del_inv, pod.old_u, pod.old_v)
                if not _eq_mat(aba_old_inv, old_inv):
                    aba_mismatch += 1

                for b in sessions:
                    session_cases += 1
                    old_x = matvec(old_inv, b)
                    fresh_del_x = matvec(fresh_del_inv, b)
                    fresh_new_x = matvec(fresh_new_inv, b)

                    cand_del_x = rank1_solution_update(old_x, z_del, den_del, pod.old_v)
                    cand_new_x = rank1_solution_update(cand_del_x, z_add, den_add, pod.new_v)
                    generic_new_x = woodbury_rankk_solution(
                        old_x,
                        old_inv,
                        [neg_old_u, pod.new_u],
                        [pod.old_v, pod.new_v],
                    )

                    if not _eq_vec(cand_del_x, fresh_del_x):
                        delete_mismatch += 1
                    if not _eq_vec(cand_new_x, fresh_new_x):
                        candidate_fresh_mismatch += 1
                    if not _eq_vec(generic_new_x, fresh_new_x):
                        generic_fresh_mismatch += 1
                    if cand_new_x != old_x:
                        material_changed += 1
                    for val in cand_new_x:
                        observe_fraction(val)

                    # Full ABA state comparison, using the returned old inverse to avoid
                    # reusing the forward candidate solution update path.
                    aba_x = matvec(aba_old_inv, b)
                    if aba_x != old_x:
                        aba_mismatch += 1

                # Cross-Pod interaction witness: target response changes when q is absent.
                qidx = (pidx + 1) % len(pods)
                m_old_q_abs = operator(base, pods, {qidx: "absent"})
                m_new_q_abs = operator(base, pods, {pidx: "new", qidx: "absent"})
                inv_old_q_abs = inverse(m_old_q_abs)
                inv_new_q_abs = inverse(m_new_q_abs)
                interaction_cells += 1
                # Require at least one registered session to witness changed target response.
                witnessed = False
                for b in sessions:
                    resp_all = vec_sub(matvec(fresh_new_inv, b), matvec(old_inv, b))
                    resp_q_abs = vec_sub(matvec(inv_new_q_abs, b), matvec(inv_old_q_abs, b))
                    if resp_all != resp_q_abs:
                        witnessed = True
                        break
                if witnessed:
                    interaction_cells_material += 1

    material_rate = material_changed / session_cases if session_cases else 0.0
    interaction_rate = interaction_cells_material / interaction_cells if interaction_cells else 0.0
    exact_pass = (
        candidate_fresh_mismatch == 0
        and generic_fresh_mismatch == 0
        and candidate_generic_inverse_mismatch == 0
        and delete_mismatch == 0
        and restore_mismatch == 0
        and aba_mismatch == 0
        and material_rate >= 0.95
        and interaction_rate >= 0.95
    )

    return {
        "experiment": "E-000103",
        "scope": "low-rank resolvent lifecycle transport / generic Woodbury reduction",
        "seed_count": seed_count,
        "dimensions": list(dimensions),
        "pods_per_system": 4,
        "sessions_per_system": sessions_per_system,
        "revision_cells": revision_cells,
        "session_cases": session_cases,
        "candidate_fresh_state_mismatches": candidate_fresh_mismatch,
        "generic_woodbury_fresh_state_mismatches": generic_fresh_mismatch,
        "candidate_generic_inverse_mismatches": candidate_generic_inverse_mismatch,
        "delete_mismatches": delete_mismatch,
        "restore_mismatches": restore_mismatch,
        "aba_mismatches": aba_mismatch,
        "material_changed_sessions": material_changed,
        "material_edit_rate": material_rate,
        "interaction_cells": interaction_cells,
        "interaction_cells_material": interaction_cells_material,
        "interaction_rate": interaction_rate,
        "max_fraction_numerator_bits": max_num_bits,
        "max_fraction_denominator_bits": max_den_bits,
        "kill_screen_pass": exact_pass,
        "decision": (
            "KILL_RESOLVENT_TRANSPORT_AS_STANDALONE_NOVELTY_SEAM"
            if exact_pass
            else "DO_NOT_INTERPRET_KILL_SCREEN"
        ),
        "interpretation": (
            "Exact fleet transport exists in the registered architecture, but an independent generic "
            "Woodbury baseline reproduces the same inverse and session states. Any speedup belongs to "
            "classical dynamic low-rank linear algebra, not to Pod/neural naming."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-count", type=int, default=16)
    ap.add_argument("--results-dir", type=Path, default=Path("so/results"))
    args = ap.parse_args()
    result = run(seed_count=args.seed_count)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000103_low_rank_resolvent_lifecycle_reduction.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
