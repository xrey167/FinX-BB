"""EQ-AUDIT-001: test E-000084 against its algebraically equivalent late bind.

A numerical falsification of a novelty/utility interpretation, NOT an LLM test.
The authoritative original must match its recorded Git blob before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
from typing import Any
import numpy as np

SOURCE_BLOB = "8b82f36128b33eab19cbfbb8f5b5021e3651833e"
TOLERANCE = 1e-10


def load_original(path: Path) -> Any:
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError(f"Unqualified original source: {blob}")
    spec = importlib.util.spec_from_file_location("e84_original", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def world(seed: int, channels: int, depth: int, pods: int):
    if min(channels, depth, pods) < 1:
        raise ValueError("positive dimensions required")
    # Exact original RNG order through the radial-network parameters.
    rng = np.random.default_rng(seed)
    z = rng.normal(size=channels) + 1j * rng.normal(size=channels)
    z /= np.sqrt(np.mean(np.abs(z) ** 2))
    theta = rng.normal(scale=0.20, size=(pods, channels))
    alpha = rng.normal(scale=0.30, size=(depth, channels))
    bias = rng.normal(scale=0.20, size=(depth, channels))
    return z, theta, alpha, bias


def radial_trace(z, alpha, bias):
    out = z.copy()
    states, gates = [out.copy()], []
    for a, b in zip(alpha, bias):
        scale = np.exp(0.025 * np.tanh(a * (np.abs(out) ** 2) + b))
        out = out * scale
        gates.append(scale)
        states.append(out.copy())
    return np.stack(states), np.stack(gates)


def maxabs(x) -> float:
    return float(np.max(np.abs(x)))


def numerical_cell(original, seed: int, depth: int, channels=64, pods=32,
                   subset_trials=64, edits=128) -> dict:
    z, theta, alpha, bias = world(seed, channels, depth, pods)
    rng = np.random.default_rng(seed + 4000 + depth)
    rotate, deep = original._rotate, original._equivariant_deep
    anchor = deep(z, alpha, bias)
    total = theta.sum(axis=0)
    early = deep(rotate(z, total), alpha, bias)
    plain, plain_gates = radial_trace(z, alpha, bias)
    conditioned, conditioned_gates = radial_trace(rotate(z, total), alpha, bias)
    # Check our trace really is the upstream computation before using its gates.
    trace_error = maxabs(conditioned[-1] - early)
    late_error = maxabs(early - rotate(anchor, total))
    gate_error = maxabs(conditioned_gates - plain_gates)
    all_layer_error = maxabs(conditioned - plain * np.exp(1j * total)[None, :])
    amplitude_error = maxabs(np.abs(conditioned) - np.abs(plain))
    subset_error = 0.0
    for _ in range(subset_trials):
        mask = rng.integers(0, 2, size=pods).astype(bool)
        th = theta[mask].sum(axis=0)
        subset_error = max(subset_error, maxabs(deep(rotate(z, th), alpha, bias) - rotate(anchor, th)))
    # Same state and parameters; alternate replacement, zeroing, and restoration.
    values, total_current, inverse_state = theta.copy(), total.copy(), early.copy()
    inverse_error, late_edit_error, intermethod_error = 0.0, 0.0, 0.0
    for i in range(edits):
        p = int(rng.integers(pods))
        replacement = (np.zeros(channels) if i % 3 == 0 else
                       theta[p] if i % 3 == 1 else rng.normal(0, 0.2, channels))
        delta = replacement - values[p]
        values[p] = replacement
        total_current = total_current + delta
        inverse_state = rotate(inverse_state, delta)
        late_state = rotate(anchor, total_current)
        # Fresh sum avoids sharing drift from the incremental aggregate.
        reference = deep(rotate(z, values.sum(axis=0)), alpha, bias)
        inverse_error = max(inverse_error, maxabs(inverse_state - reference))
        late_edit_error = max(late_edit_error, maxabs(late_state - reference))
        intermethod_error = max(intermethod_error, maxabs(inverse_state - late_state))
    # Finite-difference interactions exist, but are already reproduced by late bind.
    pair_early = (deep(rotate(z, theta[0] + theta[1]), alpha, bias)
                  - deep(rotate(z, theta[0]), alpha, bias)
                  - deep(rotate(z, theta[1]), alpha, bias) + anchor)
    pair_late = (rotate(anchor, theta[0] + theta[1]) - rotate(anchor, theta[0])
                 - rotate(anchor, theta[1]) + anchor)
    # Explicit SUM-encoding collision, not a general impossibility for symlink readers.
    altered = theta.copy()
    shift = rng.normal(0, 0.25, channels)
    altered[0] += shift
    altered[1] -= shift
    collision_error = maxabs(deep(rotate(z, altered.sum(axis=0)), alpha, bias) - early)
    metrics = {
        "source_trace_error": trace_error,
        "early_vs_late_error": late_error,
        "all_layer_phase_factorization_error": all_layer_error,
        "radial_gate_phase_intervention_error": gate_error,
        "radial_amplitude_phase_intervention_error": amplitude_error,
        "subset_early_vs_late_max_error": subset_error,
        "sequential_inverse_vs_fresh_max_error": inverse_error,
        "sequential_late_vs_fresh_max_error": late_edit_error,
        "sequential_inverse_vs_late_max_error": intermethod_error,
        "pair_interaction_early_vs_late_error": maxabs(pair_early - pair_late),
        "pair_interaction_magnitude": maxabs(pair_early),
        "pod_phase_has_material_output_effect": maxabs(early - anchor),
        "sum_encoding_collision_error": collision_error,
        "collision_per_source_change": maxabs(shift),
    }
    equality_names = [k for k in metrics if k.endswith("error")]
    return {"seed": seed, "depth": depth, "channels": channels, "pods": pods,
            "subset_trials": subset_trials, "sequential_edits": edits, "metrics": metrics,
            "pass": all(metrics[k] <= TOLERANCE for k in equality_names)
                    and metrics["pod_phase_has_material_output_effect"] > 1e-4
                    and metrics["pair_interaction_magnitude"] > 1e-6}


def benchmark_cell(original, seed: int, depth: int, channels=64, pods=32,
                   events=96, rounds=7) -> dict:
    """Paired CPU microbenchmark, same event stream, including setup in each round.

    No runtime authority, alias routing, LLM, generated history, or security claim.
    Early and ordinary implementations retain an anchor-sized vector plus total.
    Current emitted outputs are transient in both timing loops.
    """
    z, theta, alpha, bias = world(seed, channels, depth, pods)
    rng = np.random.default_rng(9000 + seed + depth)
    deltas = rng.normal(scale=0.02, size=(events, channels))
    start_total = theta.sum(axis=0)
    rotate, deep = original._rotate, original._equivariant_deep

    def apply(method):
        total = start_total.copy()
        if method == "eager_inverse":
            state = deep(rotate(z, total), alpha, bias)
            for delta in deltas:
                total = total + delta
                state = rotate(state, delta)
        elif method == "ordinary_late_bind":
            anchor = deep(z, alpha, bias)
            state = rotate(anchor, total)
            for delta in deltas:
                total = total + delta
                state = rotate(anchor, total)
        elif method == "full_recompute":
            state = deep(rotate(z, total), alpha, bias)
            for delta in deltas:
                total = total + delta
                state = deep(rotate(z, total), alpha, bias)
        else:
            raise ValueError(method)
        return state

    methods = ["eager_inverse", "ordinary_late_bind", "full_recompute"]
    outputs = {m: apply(m) for m in methods}
    equality = max(maxabs(v - outputs["full_recompute"]) for v in outputs.values())
    if equality > TOLERANCE:
        raise AssertionError(f"Timing invalid: unequal outputs {equality}")
    timings = {m: [] for m in methods}
    for r in range(rounds):
        for m in np.roll(methods, r % 3):
            t0 = time.perf_counter_ns()
            apply(m)
            timings[m].append(time.perf_counter_ns() - t0)
    medians = {m: float(np.median(v)) for m, v in timings.items()}
    return {"seed": seed, "depth": depth, "events": events, "rounds": rounds,
            "output_equivalence_max_error": equality,
            "total_sequence_including_setup_ns_raw": timings,
            "total_sequence_including_setup_ns_median": medians,
            "full_over_inverse": medians["full_recompute"] / medians["eager_inverse"],
            "full_over_ordinary_late": medians["full_recompute"] / medians["ordinary_late_bind"],
            "ordinary_late_over_inverse": medians["ordinary_late_bind"] / medians["eager_inverse"],
            "persistent_state_bytes_each_excluding_shared_model_and_pod_registry":
                int(z.nbytes + start_total.nbytes),
            "complex_phase_rotations_per_update_each": 1,
            "deep_layers_per_update_inverse": 0,
            "deep_layers_per_update_ordinary_late": 0,
            "deep_layers_per_update_full_recompute": depth}


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    source_default = root / "upstream/e000084_revocation_equivariance_screen.py"
    if not source_default.exists():
        source_default = root.parents[1] / "so/experiments/e000084_revocation_equivariance_screen.py"
    ap.add_argument("--source", type=Path, default=source_default)
    ap.add_argument("--out", type=Path, default=root / "results/equivariance_audit.json")
    args = ap.parse_args()
    original = load_original(args.source)
    original_rows = [original.run_seed(s, channels=64, depth=24, pods=32, delete_trials=16)
                     for s in range(5)]
    cells = [numerical_cell(original, s, d) for d in (1, 8, 24, 96) for s in range(5)]
    bench = [benchmark_cell(original, s, d) for d in (1, 8, 24, 96) for s in range(5)]
    record = {
        "experiment": "EQ-AUDIT-001", "upstream_run": 33969542375,
        "upstream_commit": "7689a2aabb6f551d0f3b757e10bd5bb02d93a3ce",
        "source_git_blob_verified": SOURCE_BLOB,
        "python": platform.python_version(), "numpy": np.__version__,
        "tolerance": TOLERANCE, "seed_type": "synthetic intervention/parameter seeds, NOT training seeds",
        "original_reproduction": original_rows, "cells": cells, "benchmarks": bench,
        "original_identity_preserved": all(r["pass"] for r in original_rows),
        "late_binding_collapse_confirmed": all(r["pass"] for r in cells),
        "breakthrough": False,
        "classification": "structural utility/novelty interpretation falsified for the pinned E84 screen",
        "full_CAVI_battery_run": False, "real_symlink_capability": "not measured",
        "claim_boundary": "No universal impossibility for lifecycle-aware neural computation; no LLM, security, or J-lens qualification.",
        "proof": "For fixed x and available phase action, F(T_g x)=T_g F(x). Cache H=F(x); materialize T_g H. After deletion, T_{g-p} H exactly equals fresh F(T_{g-p} x).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False))
    print(json.dumps({k: record[k] for k in ("experiment", "original_identity_preserved",
          "late_binding_collapse_confirmed", "breakthrough", "classification")}, indent=2))
    print("worst_late_error", max(c["metrics"]["early_vs_late_error"] for c in cells))
    print("worst_edit_late_error", max(c["metrics"]["sequential_late_vs_fresh_max_error"] for c in cells))
    print("worst_gate_error", max(c["metrics"]["radial_gate_phase_intervention_error"] for c in cells))
    for d in (1, 8, 24, 96):
        selected = [b for b in bench if b["depth"] == d]
        print("depth", d, "ordinary_over_inverse_median", np.median([b["ordinary_late_over_inverse"] for b in selected]),
              "full_over_inverse_median", np.median([b["full_over_inverse"] for b in selected]))
    if not record["original_identity_preserved"] or not record["late_binding_collapse_confirmed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
