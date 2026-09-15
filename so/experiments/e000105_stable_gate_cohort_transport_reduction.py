"""E-000105: exact stable-gate cohort transport reduction.

Pure Python + Fraction structural kill screen. No model download required.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction as F
from pathlib import Path
from typing import List, Sequence, Tuple

Vec = List[F]
Mat = List[List[F]]


def eye(n: int) -> Mat:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def mv(a: Mat, x: Sequence[F]) -> Vec:
    return [sum((v * z for v, z in zip(row, x)), F(0)) for row in a]


def va(a: Sequence[F], b: Sequence[F]) -> Vec:
    return [x + y for x, y in zip(a, b)]


def vsub(a: Sequence[F], b: Sequence[F]) -> Vec:
    return [x - y for x, y in zip(a, b)]


def mm(a: Mat, b: Mat) -> Mat:
    return [
        [sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def ma(a: Mat, b: Mat) -> Mat:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def diag(mask: Sequence[int]) -> Mat:
    n = len(mask)
    return [[F(mask[i] if i == j else 0) for j in range(n)] for i in range(n)]


def _mat(rng: random.Random, m: int, n: int) -> Mat:
    return [[F(rng.randint(1, 3)) for _ in range(n)] for __ in range(m)]


def _vec(rng: random.Random, n: int) -> Vec:
    return [F(rng.randint(1, 3)) for _ in range(n)]


def generate(seed: int, width: int, depth: int, codes: int = 4, pod_dim: int = 3,
             ctx_dim: int = 3, out_dim: int = 3) -> dict:
    rng = random.Random((seed + 1) * 1_000_003 + width * 10_007 + depth * 997)
    net = {
        "width": width, "depth": depth, "codes": codes, "pod_dim": pod_dim,
        "ctx_dim": ctx_dim, "out_dim": out_dim,
        "Bc": _mat(rng, width, ctx_dim), "Bp": _mat(rng, width, pod_dim),
        "b0": _vec(rng, width), "Ws": [], "Cs": [], "bs": [], "masks": [],
    }
    for layer in range(depth):
        net["Ws"].append(_mat(rng, width, width))
        net["Cs"].append(_mat(rng, width, ctx_dim))
        net["bs"].append(_vec(rng, width))
        layer_masks = []
        for code in range(codes):
            mask = [0 if ((i + 2 * layer) % codes) == code else 1 for i in range(width)]
            if not any(mask):
                mask[(code + 1) % width] = 1
            layer_masks.append(mask)
        net["masks"].append(layer_masks)
    net["Wo"] = _mat(rng, out_dim, width)
    net["Co"] = _mat(rng, out_dim, ctx_dim)
    net["bo"] = _vec(rng, out_dim)
    return net


def fresh_forward(net: dict, context: Sequence[F], pod: Sequence[F], code: int) -> Vec:
    h = va(va(mv(net["Bc"], context), mv(net["Bp"], pod)), net["b0"])
    for layer in range(net["depth"]):
        z = va(va(mv(net["Ws"][layer], h), mv(net["Cs"][layer], context)), net["bs"][layer])
        gated = [F(m) * x for m, x in zip(net["masks"][layer][code], z)]
        h = va(h, gated)
    return va(va(mv(net["Wo"], h), mv(net["Co"], context)), net["bo"])


def candidate_sensitivity(net: dict, code: int) -> Mat:
    """Lifecycle-labelled exact Pod sensitivity, propagated as a matrix."""
    j = [row[:] for row in net["Bp"]]
    ident = eye(net["width"])
    for layer in range(net["depth"]):
        a = ma(mm(diag(net["masks"][layer][code]), net["Ws"][layer]), ident)
        j = mm(a, j)
    return mm(net["Wo"], j)


def generic_region_sensitivity(net: dict, code: int) -> Mat:
    """Independent ordinary fixed-region forward-mode propagation, column by column."""
    columns: List[Vec] = []
    for pod_axis in range(net["pod_dim"]):
        v = [net["Bp"][i][pod_axis] for i in range(net["width"])]
        for layer in range(net["depth"]):
            wv = mv(net["Ws"][layer], v)
            mask = net["masks"][layer][code]
            v = [v[i] + F(mask[i]) * wv[i] for i in range(net["width"])]
        columns.append(mv(net["Wo"], v))
    return [[columns[j][i] for j in range(net["pod_dim"])] for i in range(net["out_dim"])]


def apply(old_output: Sequence[F], sensitivity: Mat, delta_pod: Sequence[F]) -> Vec:
    return va(old_output, mv(sensitivity, delta_pod))


def full_forward_multiplies(net: dict) -> int:
    w, cd, pd, od, depth = net["width"], net["ctx_dim"], net["pod_dim"], net["out_dim"], net["depth"]
    return w * cd + w * pd + depth * (w * w + w * cd) + od * w + od * cd


def mutation_multiplies(net: dict) -> int:
    return net["out_dim"] * net["pod_dim"]


def sessions_for(seed: int, width: int, depth: int, count: int, codes: int) -> List[Tuple[Vec, int]]:
    rng = random.Random((seed + 17) * 999_983 + width * 313 + depth * 37)
    sessions: List[Tuple[Vec, int]] = []
    for s in range(count):
        context = [F(1 + rng.randint(0, 4)) for _ in range(3)]
        context[s % 3] += F(s + 1)
        sessions.append((context, s % codes))
    return sessions


def run(seed_count: int = 16, widths: Sequence[int] = (4, 6, 8),
        depths: Sequence[int] = (2, 4, 6), session_count: int = 32, codes: int = 4) -> dict:
    old = [F(1), F(2), F(3)]
    new = [F(2), F(4), F(6)]
    absent = [F(0), F(0), F(0)]
    restore = [F(4), F(5), F(7)]

    network_cells = session_cells = lifecycle_cases = 0
    candidate_fresh_mismatch = generic_fresh_mismatch = candidate_generic_mismatch = 0
    sensitivity_mismatch = material_update_cases = 0
    interaction_checks = interaction_material = 0
    aba_cases = aba_mismatch = work_mismatch = 0
    total_candidate_mult = total_generic_mult = total_full_mult = 0
    best_ratio, worst_ratio, max_bits = 0.0, float("inf"), 0

    def observe(values: Sequence[F]) -> None:
        nonlocal max_bits
        for x in values:
            max_bits = max(max_bits, abs(x.numerator).bit_length(), x.denominator.bit_length())

    for seed in range(seed_count):
        for width in widths:
            for depth in depths:
                network_cells += 1
                net = generate(seed, width, depth, codes=codes)
                candidate_js = [candidate_sensitivity(net, z) for z in range(codes)]
                generic_js = [generic_region_sensitivity(net, z) for z in range(codes)]
                sensitivity_mismatch += sum(int(a != b) for a, b in zip(candidate_js, generic_js))

                update_delta = vsub(new, old)
                cohort_corrections = [mv(j, update_delta) for j in candidate_js]
                for a in range(codes):
                    for b in range(a + 1, codes):
                        interaction_checks += 1
                        interaction_material += int(cohort_corrections[a] != cohort_corrections[b])

                sessions = sessions_for(seed, width, depth, session_count, codes)
                candidate_work = mutation_multiplies(net)
                generic_work = mutation_multiplies(net)
                work_mismatch += int(candidate_work != generic_work)
                # UPDATE, DELETE, RESTORE, ROLLBACK, ABA-forward, ABA-back.
                event_count = 6
                total_candidate_mult += event_count * codes * candidate_work
                total_generic_mult += event_count * codes * generic_work
                total_full_mult += event_count * session_count * full_forward_multiplies(net)
                ratio = (session_count * full_forward_multiplies(net)) / (codes * candidate_work)
                best_ratio, worst_ratio = max(best_ratio, ratio), min(worst_ratio, ratio)

                for context, code in sessions:
                    session_cells += 1
                    jc, jg = candidate_js[code], generic_js[code]
                    y_old = fresh_forward(net, context, old, code)
                    y_new = fresh_forward(net, context, new, code)
                    observe(y_old); observe(y_new)
                    material_update_cases += int(y_new != y_old)

                    for target in (new, absent, restore):
                        lifecycle_cases += 1
                        fresh = fresh_forward(net, context, target, code)
                        candidate = apply(y_old, jc, vsub(target, old))
                        generic = apply(y_old, jg, vsub(target, old))
                        observe(fresh); observe(candidate); observe(generic)
                        candidate_fresh_mismatch += int(candidate != fresh)
                        generic_fresh_mismatch += int(generic != fresh)
                        candidate_generic_mismatch += int(candidate != generic)

                    # ROLLBACK begins from genuinely new cached mixed state.
                    lifecycle_cases += 1
                    fresh_old = fresh_forward(net, context, old, code)
                    candidate_rb = apply(y_new, jc, vsub(old, new))
                    generic_rb = apply(y_new, jg, vsub(old, new))
                    candidate_fresh_mismatch += int(candidate_rb != fresh_old)
                    generic_fresh_mismatch += int(generic_rb != fresh_old)
                    candidate_generic_mismatch += int(candidate_rb != generic_rb)

                    # ABA old -> new -> old.
                    lifecycle_cases += 1
                    aba_cases += 1
                    candidate_up = apply(y_old, jc, vsub(new, old))
                    candidate_back = apply(candidate_up, jc, vsub(old, new))
                    generic_up = apply(y_old, jg, vsub(new, old))
                    generic_back = apply(generic_up, jg, vsub(old, new))
                    aba_mismatch += int(candidate_back != fresh_old or generic_back != fresh_old)
                    candidate_fresh_mismatch += int(candidate_back != fresh_old)
                    generic_fresh_mismatch += int(generic_back != fresh_old)
                    candidate_generic_mismatch += int(candidate_back != generic_back)

    interaction_fraction = interaction_material / interaction_checks if interaction_checks else 0.0
    validity = (
        material_update_cases == session_cells and sensitivity_mismatch == 0
        and interaction_fraction >= 0.95
    )
    kill = (
        validity and candidate_fresh_mismatch == 0 and generic_fresh_mismatch == 0
        and candidate_generic_mismatch == 0 and work_mismatch == 0 and aba_mismatch == 0
        and total_candidate_mult == total_generic_mult
    )
    return {
        "experiment": "E-000105",
        "scope": "stable-gate piecewise-affine cohort exact-transport reduction",
        "seed_count": seed_count, "widths": list(widths), "depths": list(depths), "codes": codes,
        "sessions_per_network": session_count, "network_cells": network_cells,
        "session_cells": session_cells, "lifecycle_cases": lifecycle_cases,
        "material_update_cases": material_update_cases, "interaction_checks": interaction_checks,
        "interaction_material": interaction_material, "interaction_fraction": interaction_fraction,
        "sensitivity_mismatch": sensitivity_mismatch,
        "candidate_fresh_mismatch": candidate_fresh_mismatch,
        "generic_fresh_mismatch": generic_fresh_mismatch,
        "candidate_generic_mismatch": candidate_generic_mismatch,
        "aba_cases": aba_cases, "aba_mismatch": aba_mismatch, "work_mismatch": work_mismatch,
        "total_candidate_mutation_multiplies": total_candidate_mult,
        "total_generic_mutation_multiplies": total_generic_mult,
        "total_full_replay_multiplies": total_full_mult,
        "best_full_to_cohort_multiply_ratio": best_ratio,
        "worst_full_to_cohort_multiply_ratio": worst_ratio,
        "max_exact_integer_bit_length": max_bits,
        "validity_pass": validity, "kill_screen_pass": kill,
        "decision": "KILL_STABLE_GATE_COHORT_TRANSPORT_AS_STANDALONE_NOVELTY_SEAM" if kill else "DO_NOT_KILL",
        "not_claimed": "Does not cover Pod-induced activation-region transitions; no real-reader or systems/patent claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    result = run(args.seed_count, session_count=args.sessions)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.results_dir:
        root = Path(args.results_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "e000105-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
