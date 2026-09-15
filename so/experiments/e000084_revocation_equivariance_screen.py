"""E-000084 -- structural screen for revocation-equivariant neural state.

Hypothesis (NOT a novelty claim): if mutable knowledge is represented as a
commuting group action on a dedicated neural state and all downstream persistent
computation is equivariant to that action, removing one pod can be implemented
by the exact inverse action on the already-computed state.  This would be a
neural-specific way to avoid replaying the entire downstream computation.

This file is deliberately a numerical structural screen, not an LLM capability
experiment.  It does not satisfy the >=0.95 real-symlink prerequisite and must
not be used to promote CAVI/FinX-BB claims.

The test contrasts two networks receiving the same commuting per-pod rotations:

1. An equivariant complex-valued lane.  Downstream layers rescale each complex
   channel only by rotation-invariant magnitudes, so F(T_theta z) =
   T_theta F(z) by construction.  Deleting any pod after deep computation by
   T_-theta must match a clean recomputation without that pod.
2. A generic real MLP with the same input rotations.  Applying the inverse
   rotation at the output is expected to fail because ordinary nonlinear
   computation is not equivariant to this lifecycle action.

The important falsifier is exact counterfactual equality after arbitrary
single deletions, multi-deletions, and rollback.  Timing is reported only as a
screen; no speed claim is promoted from NumPy microbenchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _rotate(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return z * np.exp(1j * theta)


def _equivariant_deep(z: np.ndarray, alpha: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Channel-wise nonlinear radial dynamics; exactly phase-equivariant."""
    out = z.copy()
    for a, b in zip(alpha, bias):
        # abs(out)^2 is invariant to every channel phase rotation.  The positive
        # rescaling can be nonlinear while preserving equivariance exactly up to
        # floating point arithmetic.
        scale = np.exp(0.025 * np.tanh(a * (np.abs(out) ** 2) + b))
        out = out * scale
    return out


def _ordinary_deep(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    out = x.copy()
    for w, b in zip(weights, biases):
        out = np.tanh(w @ out + b)
    return out


def _complex_to_real(z: np.ndarray) -> np.ndarray:
    return np.concatenate([z.real, z.imag])


def _real_rotate(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c = x.shape[0] // 2
    re, im = x[:c], x[c:]
    ct, st = np.cos(theta), np.sin(theta)
    return np.concatenate([ct * re - st * im, st * re + ct * im])


def run_seed(seed: int, *, channels: int, depth: int, pods: int, delete_trials: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    z0 = z0 / np.sqrt(np.mean(np.abs(z0) ** 2))

    # Small rotations avoid trivial wraparound while still creating substantial
    # pod-specific transformations.  Actions commute because they are diagonal
    # per-channel phase rotations.
    pod_theta = rng.normal(scale=0.20, size=(pods, channels))
    active = np.ones(pods, dtype=bool)
    total_theta = pod_theta.sum(axis=0)

    alpha = rng.normal(scale=0.30, size=(depth, channels))
    bias = rng.normal(scale=0.20, size=(depth, channels))

    dim = 2 * channels
    ordinary_weights: list[np.ndarray] = []
    ordinary_biases: list[np.ndarray] = []
    for _ in range(depth):
        # Residual-ish dense map keeps the generic baseline numerically stable
        # but intentionally has no phase-equivariance constraint.
        w = np.eye(dim) + rng.normal(scale=0.04 / math.sqrt(dim), size=(dim, dim))
        ordinary_weights.append(w)
        ordinary_biases.append(rng.normal(scale=0.02, size=dim))

    z_all_in = _rotate(z0, total_theta)
    z_all_out = _equivariant_deep(z_all_in, alpha, bias)
    x_all_out = _ordinary_deep(_complex_to_real(z_all_in), ordinary_weights, ordinary_biases)

    equiv_identity_errors: list[float] = []
    repair_errors: list[float] = []
    ordinary_inverse_errors: list[float] = []
    multi_delete_errors: list[float] = []
    rollback_errors: list[float] = []

    trial_pods = rng.choice(pods, size=min(delete_trials, pods), replace=False)
    for p in trial_pods:
        theta_p = pod_theta[p]

        # Direct equivariance identity on the same base state.
        lhs = _equivariant_deep(_rotate(z0, total_theta), alpha, bias)
        rhs = _rotate(_equivariant_deep(z0, alpha, bias), total_theta)
        equiv_identity_errors.append(float(np.max(np.abs(lhs - rhs))))

        # Delete p *after* deep computation via the inverse lifecycle action.
        repaired = _rotate(z_all_out, -theta_p)
        clean = _equivariant_deep(_rotate(z0, total_theta - theta_p), alpha, bias)
        repair_errors.append(float(np.max(np.abs(repaired - clean))))

        # Same inverse action on an ordinary nonlinear network should not be a
        # valid repair rule.
        ordinary_repaired = _real_rotate(x_all_out, -theta_p)
        ordinary_clean = _ordinary_deep(
            _complex_to_real(_rotate(z0, total_theta - theta_p)),
            ordinary_weights,
            ordinary_biases,
        )
        ordinary_inverse_errors.append(float(np.max(np.abs(ordinary_repaired - ordinary_clean))))

        # Rollback/re-add: delete then restore the same pod and recover the old
        # state without replaying the depth stack.
        restored = _rotate(repaired, theta_p)
        rollback_errors.append(float(np.max(np.abs(restored - z_all_out))))

    # Multi-delete order independence and counterfactual equality.
    if pods >= 4:
        for _ in range(min(12, delete_trials)):
            idx = rng.choice(pods, size=3, replace=False)
            theta_del = pod_theta[idx].sum(axis=0)
            repaired = _rotate(z_all_out, -theta_del)
            clean = _equivariant_deep(_rotate(z0, total_theta - theta_del), alpha, bias)
            multi_delete_errors.append(float(np.max(np.abs(repaired - clean))))

    # Microbenchmark only: inverse lifecycle action versus full depth replay.
    reps_inverse = 5000
    reps_recompute = 100
    p = int(trial_pods[0])
    t0 = time.perf_counter_ns()
    tmp = z_all_out
    for _ in range(reps_inverse):
        tmp = _rotate(z_all_out, -pod_theta[p])
    inverse_ns = (time.perf_counter_ns() - t0) / reps_inverse

    t0 = time.perf_counter_ns()
    tmp2 = z_all_out
    for _ in range(reps_recompute):
        tmp2 = _equivariant_deep(_rotate(z0, total_theta - pod_theta[p]), alpha, bias)
    recompute_ns = (time.perf_counter_ns() - t0) / reps_recompute

    max_equiv = max(equiv_identity_errors, default=0.0)
    max_repair = max(repair_errors, default=0.0)
    max_multi = max(multi_delete_errors, default=0.0)
    max_rollback = max(rollback_errors, default=0.0)
    min_ordinary_failure = min(ordinary_inverse_errors, default=0.0)

    checks = {
        "equivariance_identity_le_1e_10": max_equiv <= 1e-10,
        "single_delete_inverse_matches_clean_le_1e_10": max_repair <= 1e-10,
        "multi_delete_inverse_matches_clean_le_1e_10": max_multi <= 1e-10,
        "rollback_restores_prior_state_le_1e_10": max_rollback <= 1e-10,
        "generic_nonlinear_inverse_is_not_a_valid_repair": min_ordinary_failure >= 1e-4,
    }
    return {
        "seed": seed,
        "channels": channels,
        "depth": depth,
        "pods": pods,
        "delete_trials": int(len(trial_pods)),
        "max_equivariance_identity_error": max_equiv,
        "max_single_delete_repair_error": max_repair,
        "max_multi_delete_repair_error": max_multi,
        "max_rollback_error": max_rollback,
        "min_generic_nonlinear_inverse_error": min_ordinary_failure,
        "median_generic_nonlinear_inverse_error": float(np.median(ordinary_inverse_errors)),
        "inverse_action_ns": float(inverse_ns),
        "full_recompute_ns": float(recompute_ns),
        "microbench_recompute_over_inverse": float(recompute_ns / max(inverse_ns, 1e-9)),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=24)
    ap.add_argument("--pods", type=int, default=32)
    ap.add_argument("--delete-trials", type=int, default=16)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [
        run_seed(
            seed,
            channels=a.channels,
            depth=a.depth,
            pods=a.pods,
            delete_trials=a.delete_trials,
        )
        for seed in a.seeds
    ]
    rec = {
        "experiment": "E-000084",
        "title": "revocation-equivariance structural falsification screen",
        "result": "structural_pass" if all(r["pass"] for r in rows) else "falsified",
        "all_pass": all(r["pass"] for r in rows),
        "rows": rows,
        "breakthrough": False,
        "interpretation": (
            "If all checks pass, exact post-computation removal is numerically possible when the downstream "
            "neural map is constrained to be equivariant to a commuting per-pod lifecycle action. This is only "
            "a structural mechanism screen; the practical research question is whether an LLM reader can retain "
            "real-symlink capability and useful cross-pod reasoning under such constraints with lower lifecycle "
            "repair cost than dense recomputation and fixed source isolation."
        ),
        "not_claimed": (
            "No novelty claim for equivariant networks, group actions, invertible/reversible networks, adapters, "
            "cache invalidation, unlearning, or this NumPy benchmark. No real LLM, symlink, BYPASS, UNKNOWN, "
            "REVOKE/SHRED, leakage, J-space, or generated-history guarantee is established here."
        ),
        "promotion_gate": (
            ">=0.95 fresh and held-out real-symlink capability across >=3 training seeds first; then lifecycle "
            "attacks, deleted-object leakage <=0.02, UNKNOWN >=0.90, exact/no-damage BYPASS, generic KL <=0.05, "
            "J-space audit, scaling/performance, >1 public backbone, and final prior-art/patent search."
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "e000084_revocation_equivariance.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
