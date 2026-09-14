from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

STAGE = "R342-CAUSAL-REFERENTIAL-POLYNOMIAL-TENSOR"
PODS = int(os.environ.get("SO_R342_PODS", "4096"))
QUERIES = int(os.environ.get("SO_R342_QUERIES", "2500"))
REFS = int(os.environ.get("SO_R342_REFS", "6"))
HIDDEN = int(os.environ.get("SO_R342_HIDDEN", "10"))
OUT = int(os.environ.get("SO_R342_OUT", "8"))
QUAD_TERMS = int(os.environ.get("SO_R342_QUAD_TERMS", "6"))
UPDATES = int(os.environ.get("SO_R342_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R342_SERVES", "100000"))
RACE_PROB = float(os.environ.get("SO_R342_RACE_PROB", "0.08"))
REPORT_PATH = Path(os.environ.get("SO_R342_REPORT", "r342_report.json"))
SEED = 3420914


def compile_quadratic(
    z0: np.ndarray,
    a: np.ndarray,
    linear: np.ndarray,
    bias: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile a quadratic neural region into a polynomial of live world refs.

    z = z0 + A^T x, with x being the current scalar values of selected refs.
    y = bias + z@linear + sum_m gamma[m] * (z·p[m]) * (z·q[m]).

    Returns constant, linear coefficients [R,OUT], quadratic [R,R,OUT].
    """
    const = bias + z0 @ linear
    lin = a @ linear  # [R,H] @ [H,OUT]
    quad = np.zeros((REFS, REFS, OUT), dtype=np.float64)

    for m in range(QUAD_TERMS):
        p0 = float(z0 @ p[m])
        q0 = float(z0 @ q[m])
        pa = a @ p[m]  # [R]
        qa = a @ q[m]
        const = const + gamma[m] * (p0 * q0)
        lin = lin + np.outer(p0 * qa + q0 * pa, gamma[m])
        # Keep ordered coefficients. Materialization sums i,j exactly as compiled.
        quad = quad + (pa[:, None, None] * qa[None, :, None]) * gamma[m][None, None, :]
    return const, lin, quad


def numeric_network(
    x: np.ndarray,
    z0: np.ndarray,
    a: np.ndarray,
    linear: np.ndarray,
    bias: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    z = z0 + x @ a
    y = bias + z @ linear
    for m in range(QUAD_TERMS):
        y = y + gamma[m] * float(z @ p[m]) * float(z @ q[m])
    return y


def materialize_polynomial(
    x: np.ndarray,
    const: np.ndarray,
    lin: np.ndarray,
    quad: np.ndarray,
) -> np.ndarray:
    y = const + x @ lin
    # einsum implements sum_{i,j} x_i x_j Q_{ij,d}.
    y = y + np.einsum("i,j,ijd->d", x, x, quad, optimize=True)
    return y


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)

    world = nrng.normal(size=PODS).astype(np.float64)
    generations = np.ones(PODS, dtype=np.int64)

    # Stable routing / references are compiled from B-plane state and are unaffected
    # by ordinary value writes.
    selected = np.empty((QUERIES, REFS), dtype=np.int32)
    for qid in range(QUERIES):
        selected[qid] = rng.sample(range(PODS), REFS)

    # Query-specific neural programs. A maps live scalar refs into hidden features;
    # the neural region then contains explicit multiplicative interactions.
    z0 = nrng.normal(scale=0.35, size=(QUERIES, HIDDEN)).astype(np.float64)
    a = nrng.normal(scale=0.28, size=(QUERIES, REFS, HIDDEN)).astype(np.float64)
    linear = nrng.normal(scale=0.22, size=(QUERIES, HIDDEN, OUT)).astype(np.float64)
    bias = nrng.normal(scale=0.05, size=(QUERIES, OUT)).astype(np.float64)
    p = nrng.normal(scale=0.30, size=(QUERIES, QUAD_TERMS, HIDDEN)).astype(np.float64)
    qvec = nrng.normal(scale=0.30, size=(QUERIES, QUAD_TERMS, HIDDEN)).astype(np.float64)
    gamma = nrng.normal(scale=0.12, size=(QUERIES, QUAD_TERMS, OUT)).astype(np.float64)

    t0 = time.perf_counter()
    c0 = np.empty((QUERIES, OUT), dtype=np.float64)
    c1 = np.empty((QUERIES, REFS, OUT), dtype=np.float64)
    c2 = np.empty((QUERIES, REFS, REFS, OUT), dtype=np.float64)
    for qid in range(QUERIES):
        c0[qid], c1[qid], c2[qid] = compile_quadratic(
            z0[qid], a[qid], linear[qid], bias[qid], p[qid], qvec[qid], gamma[qid]
        )
    compile_seconds = time.perf_counter() - t0

    reverse_counts = np.bincount(selected.reshape(-1), minlength=PODS)

    def write(pid: int, same: bool) -> None:
        generations[pid] += 1
        if not same:
            world[pid] = nrng.normal()

    def materialize(qid: int) -> tuple[np.ndarray, np.ndarray]:
        refs = selected[qid]
        seen = generations[refs].copy()
        x = world[refs]
        return materialize_polynomial(x, c0[qid], c1[qid], c2[qid]), seen

    def full(qid: int) -> np.ndarray:
        refs = selected[qid]
        return numeric_network(
            world[refs], z0[qid], a[qid], linear[qid], bias[qid], p[qid], qvec[qid], gamma[qid]
        )

    eager_numeric_invalidations = 0
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        same = rng.random() < 0.35
        same_value_updates += int(same)
        eager_numeric_invalidations += int(reverse_counts[pid])
        ts = time.perf_counter_ns(); write(pid, same); write_ns.append(time.perf_counter_ns() - ts)

    expected_conflicts = detected_conflicts = escaped_conflicts = retries = 0
    same_value_races = semantic_mismatches = 0
    max_error = 0.0

    for _ in range(SERVES):
        qid = rng.randrange(QUERIES)
        out, seen = materialize(qid)
        raced = rng.random() < RACE_PROB
        if raced:
            pid = int(selected[qid, rng.randrange(REFS)])
            same = rng.random() < 0.45
            expected_conflicts += 1
            same_value_races += int(same)
            write(pid, same)

        current = generations[selected[qid]]
        if not np.array_equal(seen, current):
            detected_conflicts += 1
            out, seen = materialize(qid)
            retries += 1
            current = generations[selected[qid]]
        elif raced:
            escaped_conflicts += 1

        if not np.array_equal(seen, current):
            escaped_conflicts += 1
            continue

        oracle = full(qid)
        err = float(np.max(np.abs(out - oracle)))
        max_error = max(max_error, err)
        semantic_mismatches += int(err > 1e-9)

    poly_ns = []
    full_ns = []
    for _ in range(5000):
        qid = rng.randrange(QUERIES)
        ts = time.perf_counter_ns(); materialize(qid); poly_ns.append(time.perf_counter_ns() - ts)
        ts = time.perf_counter_ns(); full(qid); full_ns.append(time.perf_counter_ns() - ts)

    # Demonstrate that the selected nonlinear network cannot in general collapse to
    # the affine R340 form. Fit the best affine predictor for query 0 over sampled
    # worlds and evaluate on held-out worlds.
    fit_n = 2500
    test_n = 1200
    x_fit = nrng.normal(size=(fit_n, REFS))
    y_fit = np.stack([
        numeric_network(x, z0[0], a[0], linear[0], bias[0], p[0], qvec[0], gamma[0])
        for x in x_fit
    ])
    design = np.concatenate([np.ones((fit_n, 1)), x_fit], axis=1)
    coeff, *_ = np.linalg.lstsq(design, y_fit, rcond=None)
    x_test = nrng.normal(size=(test_n, REFS))
    y_test = np.stack([
        numeric_network(x, z0[0], a[0], linear[0], bias[0], p[0], qvec[0], gamma[0])
        for x in x_test
    ])
    pred_affine = np.concatenate([np.ones((test_n, 1)), x_test], axis=1) @ coeff
    affine_rmse = float(np.sqrt(np.mean((pred_affine - y_test) ** 2)))
    affine_max_error = float(np.max(np.abs(pred_affine - y_test)))

    numeric_cache_bytes = QUERIES * OUT * 8
    polynomial_cache_bytes = c0.nbytes + c1.nbytes + c2.nbytes + selected.nbytes

    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Referential Polynomial Tensor (CRPT)",
        "pods": PODS,
        "queries": QUERIES,
        "live_refs_per_query": REFS,
        "hidden_dimension": HIDDEN,
        "output_dimension": OUT,
        "quadratic_neural_terms": QUAD_TERMS,
        "scheduled_updates": UPDATES,
        "same_value_updates": same_value_updates,
        "transactional_serves": SERVES,
        "expected_race_conflicts": expected_conflicts,
        "detected_race_conflicts": detected_conflicts,
        "escaped_race_conflicts": escaped_conflicts,
        "retries": retries,
        "same_value_race_conflicts": same_value_races,
        "semantic_mismatches_vs_full_current_nonlinear_network": semantic_mismatches,
        "max_abs_error_vs_full_current_nonlinear_network": max_error,
        "derived_cache_invalidations_or_patches_on_world_write": 0,
        "conventional_numeric_cache_invalidations_counterfactual": eager_numeric_invalidations,
        "write_fanout_elimination_fraction": 1.0 if eager_numeric_invalidations else 0.0,
        "compile_seconds": compile_seconds,
        "median_world_write_ns_python": statistics.median(write_ns),
        "median_polynomial_materialization_ns_python": statistics.median(poly_ns),
        "median_full_nonlinear_network_ns_python": statistics.median(full_ns),
        "full_over_polynomial_materialization_ratio": statistics.median(full_ns) / max(1.0, statistics.median(poly_ns)),
        "best_affine_control_heldout_rmse_query0": affine_rmse,
        "best_affine_control_heldout_max_error_query0": affine_max_error,
        "numeric_cache_bytes_counterfactual": numeric_cache_bytes,
        "referential_polynomial_cache_bytes": polynomial_cache_bytes,
        "referential_to_numeric_cache_byte_ratio": polynomial_cache_bytes / numeric_cache_bytes,
        "cached_state_contains_current_world_values": False,
        "cached_state_contains_last_seen_generations": False,
        "mechanism": (
            "CRPT lifts a nonlinear neural region into a polynomial whose variables are live canonical world references. Linear terms and "
            "explicit multiplicative neural interactions are compiled into constant/linear/quadratic coefficients. The retained activation "
            "contains no current value bytes; arbitrary world rewrites are observed only when the polynomial is materialized and generation-validated."
        ),
        "novelty_hypothesis": (
            "A neural activation may be a symbolic/factorized function over future mutable world cells rather than a numeric hidden vector. "
            "R342 extends the R340/R341 affine/reference-valued idea to genuine cross-value nonlinear interaction while preserving zero "
            "write-time derived-cache maintenance."
        ),
        "claim_boundary": (
            "Polynomial neural networks and symbolic polynomial evaluation are established. The open question is whether generation-safe live "
            "world references as the variables of a retained neural activation, evaluated under current authority at inference time, have "
            "direct prior art. Degree/storage growth remains a major limitation and must be solved before this is practical."
        ),
        "dod_status": "NOT_DOD; nonlinear referential-neural-state gate",
    }
    report["contract_pass"] = (
        semantic_mismatches == 0
        and max_error <= 1e-9
        and detected_conflicts == expected_conflicts
        and escaped_conflicts == 0
        and retries == expected_conflicts
        and affine_rmse > 1e-3
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Causal Referential Polynomial Tensor contract failed")


if __name__ == "__main__":
    main()
