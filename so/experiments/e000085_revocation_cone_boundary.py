"""E-000085 -- falsification screen for the revocation-equivariant candidate.

This is NOT a novelty or capability claim.  It tests a necessary consequence of
using an exact inverse group action to revoke already-computed neural state.

Let state be split into a durable lane b, on which the lifecycle group acts
trivially, and a revocable lane z, on which a pod acts by a complex phase
rotation.  If downstream computation F is equivariant, then the durable output
must be invariant to the pod action.  Therefore pod-dependent information may
remain useful in non-trivial equivariant representations, but it cannot be
written into a trivial/durable representation and later be removed by applying
only the inverse action to z.

The screen compares:
  1. a phase-equivariant stack whose durable updates depend only on invariants;
  2. the same stack with a pod-dependent phase-sensitive write into b.

For (1), post-computation inverse revocation must equal clean recomputation and
the durable lane must carry no signal about which group element was applied.
For (2), the durable lane acquires a measurable pod-dependent signal, and an
inverse action on z must fail to recover the clean counterfactual.

The result is an architecture constraint, not an invention: equivariant neural
networks, group representations and orthogonal/phase transforms are established
prior art.  A useful FinX-BB direction survives only if a real LLM can keep the
entire pod-dependent persistence cone revocable/equivariant while preserving the
strict symlink, lifecycle, locality, J-space and performance gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def rotate(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return z * np.exp(1j * theta)


def safe_stack(
    b: np.ndarray,
    z: np.ndarray,
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    durable_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Deep map equivariant to arbitrary channelwise phase rotations of z.

    The durable lane is a trivial group representation.  It may consume |z|^2,
    which is invariant, but never the phase that carries the lifecycle action.
    """
    b_out = b.copy()
    z_out = z.copy()
    for a, be, w in zip(alpha, beta, durable_w):
        inv = np.abs(z_out) ** 2
        b_out = np.tanh(b_out + 0.025 * (w @ inv))
        # Real scale depends only on invariants and durable state, so phase is
        # transported exactly through depth.
        scale = np.exp(0.02 * np.tanh(a * inv + be + 0.05 * b_out.mean()))
        z_out = z_out * scale
    return b_out, z_out


def leaky_stack(
    b: np.ndarray,
    z: np.ndarray,
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    durable_w: np.ndarray,
    leak_w: np.ndarray,
    leak_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same computation plus a phase-sensitive write into the durable lane."""
    b_out = b.copy()
    z_out = z.copy()
    for a, be, w, lw in zip(alpha, beta, durable_w, leak_w):
        inv = np.abs(z_out) ** 2
        phase_features = np.concatenate([z_out.real, z_out.imag])
        b_out = np.tanh(
            b_out
            + 0.025 * (w @ inv)
            + leak_strength * (lw @ phase_features)
        )
        scale = np.exp(0.02 * np.tanh(a * inv + be + 0.05 * b_out.mean()))
        z_out = z_out * scale
    return b_out, z_out


def decode(z: np.ndarray, probe: np.ndarray) -> float:
    """A phase-sensitive late readout: useful signal can live in revocable z."""
    return float(np.real(np.vdot(probe, z)))


def _maxabs(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))


def run_seed(
    seed: int,
    *,
    channels: int,
    durable_dim: int,
    depth: int,
    pods: int,
    delete_trials: int,
    leak_strengths: list[float],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    b0 = rng.normal(scale=0.4, size=durable_dim)
    z0 = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    z0 /= np.sqrt(np.mean(np.abs(z0) ** 2))

    pod_theta = rng.normal(scale=0.24, size=(pods, channels))
    total_theta = pod_theta.sum(axis=0)
    alpha = rng.normal(scale=0.25, size=(depth, channels))
    beta = rng.normal(scale=0.15, size=(depth, channels))
    durable_w = rng.normal(scale=1.0 / np.sqrt(channels), size=(depth, durable_dim, channels))
    leak_w = rng.normal(
        scale=1.0 / np.sqrt(2 * channels),
        size=(depth, durable_dim, 2 * channels),
    )

    # A fixed late readout makes sure the group-carrying lane is not merely a
    # numerically decorative symmetry: pod actions measurably alter a usable
    # scalar output before revocation.
    probe = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    probe /= np.linalg.norm(probe)

    b_all_safe, z_all_safe = safe_stack(
        b0, rotate(z0, total_theta), alpha=alpha, beta=beta, durable_w=durable_w
    )

    selected = rng.choice(pods, size=min(delete_trials, pods), replace=False)
    safe_repair_errors: list[float] = []
    safe_durable_signals: list[float] = []
    safe_decode_shifts: list[float] = []
    safe_decode_repair_errors: list[float] = []

    for p in selected:
        theta_p = pod_theta[p]
        b_clean, z_clean = safe_stack(
            b0,
            rotate(z0, total_theta - theta_p),
            alpha=alpha,
            beta=beta,
            durable_w=durable_w,
        )
        # In a truly equivariant stack, deletion after the depth stack requires
        # only the inverse action on the non-trivial representation.
        z_repaired = rotate(z_all_safe, -theta_p)
        safe_repair_errors.append(max(_maxabs(z_repaired, z_clean), _maxabs(b_all_safe, b_clean)))
        safe_durable_signals.append(_maxabs(b_all_safe, b_clean))
        before = decode(z_all_safe, probe)
        clean_decode = decode(z_clean, probe)
        safe_decode_shifts.append(abs(before - clean_decode))
        safe_decode_repair_errors.append(abs(decode(z_repaired, probe) - clean_decode))

    leak_rows: list[dict[str, float]] = []
    for leak_strength in leak_strengths:
        b_all, z_all = leaky_stack(
            b0,
            rotate(z0, total_theta),
            alpha=alpha,
            beta=beta,
            durable_w=durable_w,
            leak_w=leak_w,
            leak_strength=leak_strength,
        )
        repair_errors: list[float] = []
        durable_signals: list[float] = []
        for p in selected:
            theta_p = pod_theta[p]
            b_clean, z_clean = leaky_stack(
                b0,
                rotate(z0, total_theta - theta_p),
                alpha=alpha,
                beta=beta,
                durable_w=durable_w,
                leak_w=leak_w,
                leak_strength=leak_strength,
            )
            # Lifecycle inverse has no action on b.  Any phase-sensitive value
            # already written there is therefore outside the repairable orbit.
            z_repaired = rotate(z_all, -theta_p)
            repair_errors.append(max(_maxabs(z_repaired, z_clean), _maxabs(b_all, b_clean)))
            durable_signals.append(_maxabs(b_all, b_clean))
        leak_rows.append(
            {
                "leak_strength": float(leak_strength),
                "median_durable_pod_signal": float(np.median(durable_signals)),
                "min_durable_pod_signal": float(np.min(durable_signals)),
                "median_inverse_repair_error": float(np.median(repair_errors)),
                "min_inverse_repair_error": float(np.min(repair_errors)),
                "max_inverse_repair_error": float(np.max(repair_errors)),
            }
        )

    nonzero = [r for r in leak_rows if r["leak_strength"] > 0]
    checks = {
        "safe_inverse_matches_clean_le_1e_10": max(safe_repair_errors, default=0.0) <= 1e-10,
        "safe_trivial_lane_has_no_pod_signal_le_1e_10": max(safe_durable_signals, default=0.0) <= 1e-10,
        "safe_late_readout_carries_nontrivial_pod_signal": float(np.median(safe_decode_shifts)) >= 1e-3,
        "safe_late_readout_repairs_le_1e_10": max(safe_decode_repair_errors, default=0.0) <= 1e-10,
        "phase_sensitive_durable_write_breaks_inverse_repair": all(
            r["min_inverse_repair_error"] >= 1e-4 for r in nonzero
        ),
        "phase_sensitive_write_creates_durable_pod_signal": all(
            r["min_durable_pod_signal"] >= 1e-4 for r in nonzero
        ),
    }

    return {
        "seed": seed,
        "channels": channels,
        "durable_dim": durable_dim,
        "depth": depth,
        "pods": pods,
        "delete_trials": int(len(selected)),
        "max_safe_inverse_repair_error": max(safe_repair_errors, default=0.0),
        "max_safe_durable_pod_signal": max(safe_durable_signals, default=0.0),
        "median_safe_late_readout_pod_signal": float(np.median(safe_decode_shifts)),
        "max_safe_late_readout_repair_error": max(safe_decode_repair_errors, default=0.0),
        "leak_rows": leak_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--durable-dim", type=int, default=32)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--pods", type=int, default=32)
    ap.add_argument("--delete-trials", type=int, default=16)
    ap.add_argument("--leak-strengths", type=float, nargs="*", default=[0.0, 0.002, 0.01, 0.05])
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [
        run_seed(
            s,
            channels=a.channels,
            durable_dim=a.durable_dim,
            depth=a.depth,
            pods=a.pods,
            delete_trials=a.delete_trials,
            leak_strengths=list(a.leak_strengths),
        )
        for s in a.seeds
    ]
    all_pass = all(r["pass"] for r in rows)
    rec = {
        "experiment": "E-000085",
        "title": "revocation-cone / trivial-representation boundary",
        "result": "boundary_confirmed" if all_pass else "boundary_not_confirmed",
        "all_pass": all_pass,
        "breakthrough": False,
        "rows": rows,
        "interpretation": (
            "For an exact inverse-action lifecycle repair, pod-dependent information may be carried through deep "
            "computation in non-trivial equivariant state, but a phase-sensitive write into state on which the "
            "lifecycle group acts trivially leaves a dependency that the inverse cannot remove.  Therefore an "
            "O(1)/local inverse-revocation design must keep the full persistence cone of pod-dependent semantics "
            "inside revocable/equivariant representations, or explicitly transform/recompute every state that "
            "received such information."
        ),
        "not_claimed": (
            "No novelty for group equivariance, complex phases, orthogonal transformations, reversible networks, "
            "knowledge editing, or the mathematical representation argument.  This is a falsification boundary "
            "for E-000084, not a real-LLM capability result."
        ),
        "next_gate": (
            "A real-symlink reader must first reach >=0.95 fresh/held-out correctness on >=3 training seeds.  "
            "Only then compare an explicitly revocation-equivariant persistence cone against clean recomputation, "
            "late-write/source-isolated baselines and KVEraser-style approximate repair under the unchanged "
            "lifecycle, leakage, UNKNOWN, BYPASS, J-space, scaling and runtime gates."
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "e000085_revocation_cone_boundary.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
