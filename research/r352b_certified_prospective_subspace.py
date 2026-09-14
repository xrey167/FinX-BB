from __future__ import annotations

import time

import numpy as np

from research import r352_certified_prospective_subspace as base


def compile_query(net: base.Network, query: np.ndarray) -> base.ResidualProgram:
    t0 = time.perf_counter()
    atoms: list[base.Atom] = []
    stable_pos = stable_neg = unstable = 0

    env_lower = [1.0] + [-1.0] * base.WORLD_DIM
    env_upper = [1.0] + [1.0] * base.WORLD_DIM
    base_dim = 1 + base.WORLD_DIM
    input_expr = np.zeros((base.Q_DIM + base.WORLD_DIM, base_dim), dtype=np.float64)
    input_expr[:base.Q_DIM, 0] = query
    input_expr[base.Q_DIM:, 1:] = np.eye(base.WORLD_DIM)
    current_expr = input_expr

    for w, b in zip(net.weights, net.biases):
        env_dim_before = len(env_lower)
        if current_expr.shape[1] < env_dim_before:
            current_expr = np.pad(current_expr, ((0, 0), (0, env_dim_before-current_expr.shape[1])))
        # All preactivations in one dense layer depend only on the previous layer,
        # so this matrix intentionally has the pre-layer environment width.
        pre = w @ current_expr
        pre[:, 0] += b
        next_rows: list[np.ndarray] = []

        for j in range(base.WIDTH):
            coeff = pre[j]
            # Atoms introduced by earlier neurons in this same layer cannot be
            # dependencies of sibling preactivations; extend them with exact zeros.
            if coeff.size < len(env_lower):
                coeff = np.pad(coeff, (0, len(env_lower)-coeff.size))
            lo, hi = base.expression_interval(
                coeff,
                np.asarray(env_lower, dtype=np.float64),
                np.asarray(env_upper, dtype=np.float64),
            )
            if lo >= 0.0:
                stable_pos += 1
                next_rows.append(coeff.copy())
            elif hi <= 0.0:
                stable_neg += 1
                next_rows.append(np.zeros(len(env_lower), dtype=np.float64))
            else:
                unstable += 1
                atoms.append(base.Atom(coeff.copy(), max(0.0, lo), max(0.0, hi)))
                env_lower.append(max(0.0, lo))
                env_upper.append(max(0.0, hi))
                next_rows = [np.pad(r, (0, 1)) for r in next_rows]
                row = np.zeros(len(env_lower), dtype=np.float64)
                row[-1] = 1.0
                next_rows.append(row)

        current_expr = np.vstack(next_rows)

    if current_expr.shape[1] < len(env_lower):
        current_expr = np.pad(current_expr, ((0,0),(0,len(env_lower)-current_expr.shape[1])))
    out_expr = net.out_w @ current_expr
    out_expr[:, 0] += net.out_b
    return base.ResidualProgram(
        atoms=atoms,
        out_expr=out_expr,
        stable_positive=stable_pos,
        stable_negative=stable_neg,
        unstable=unstable,
        original_relus=base.LAYERS * base.WIDTH,
        compile_seconds=time.perf_counter() - t0,
    )


base.compile_query = compile_query
base.STAGE = "R352B-CERTIFIED-PROSPECTIVE-RESIDUAL-CIRCUIT"

if __name__ == "__main__":
    raise SystemExit(base.main())
