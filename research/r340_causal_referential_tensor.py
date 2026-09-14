from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

STAGE = "R340-CAUSAL-REFERENTIAL-TENSOR"
PODS = int(os.environ.get("SO_R340_PODS", "4096"))
QUERIES = int(os.environ.get("SO_R340_QUERIES", "8000"))
DIM = int(os.environ.get("SO_R340_DIM", "16"))
TOPK = int(os.environ.get("SO_R340_TOPK", "6"))
LAYERS = int(os.environ.get("SO_R340_LAYERS", "6"))
UPDATES = int(os.environ.get("SO_R340_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R340_SERVES", "120000"))
RACE_PROB = float(os.environ.get("SO_R340_RACE_PROB", "0.08"))
REPORT_PATH = Path(os.environ.get("SO_R340_REPORT", "r340_report.json"))
SEED = 3400914


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compose_affine(gates: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Compile h_{l+1} = (h_l * gate_l) @ W_l + b_l into h@M + c."""
    m = np.eye(DIM, dtype=np.float64)
    c = np.zeros(DIM, dtype=np.float64)
    for l in range(LAYERS):
        # Row-vector convention: (h * g) @ W = h @ (diag(g) @ W)
        a = gates[l][:, None] * weights[l]
        m = m @ a
        c = c @ a + biases[l]
    return m, c


def full_region(x: np.ndarray, gates: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    h = x
    for l in range(LAYERS):
        h = (h * gates[l]) @ weights[l] + biases[l]
    return h


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)

    # Stable routing identity and mutable world payload are separate by construction.
    keys = nrng.normal(size=(PODS, DIM))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12
    query_vecs = nrng.normal(size=(QUERIES, DIM))
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-12
    values = nrng.normal(size=(PODS, DIM))
    generations = np.ones(PODS, dtype=np.int64)

    # Shared neural affine weights. Query-specific base gates are world-independent.
    weights = []
    biases = []
    for _ in range(LAYERS):
        w = nrng.normal(scale=0.18, size=(DIM, DIM))
        w += np.eye(DIM) * 0.72
        weights.append(w.astype(np.float64))
        biases.append(nrng.normal(scale=0.05, size=DIM).astype(np.float64))
    gates = 0.55 + 0.9 / (1.0 + np.exp(-nrng.normal(size=(QUERIES, LAYERS, DIM))))

    # Compile stable-key attention routes and query-specific affine neural closures.
    t0 = time.perf_counter()
    scores = query_vecs @ keys.T
    selected = np.argpartition(scores, -TOPK, axis=1)[:, -TOPK:]
    selected_scores = np.take_along_axis(scores, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    selected_scores = np.take_along_axis(selected_scores, order, axis=1)
    alphas = softmax(selected_scores)
    del scores, selected_scores, order

    closure_m = np.empty((QUERIES, DIM, DIM), dtype=np.float64)
    closure_c = np.empty((QUERIES, DIM), dtype=np.float64)
    for q in range(QUERIES):
        closure_m[q], closure_c[q] = compose_affine(gates[q], weights, biases)
    compile_seconds = time.perf_counter() - t0

    # Reverse fanout exists only to quantify what eager numeric-cache maintenance would cost.
    reverse_counts = np.bincount(selected.reshape(-1), minlength=PODS)

    def write(pid: int, same_value: bool = False) -> None:
        generations[pid] += 1
        if not same_value:
            values[pid] = nrng.normal(size=DIM)

    def materialize(q: int) -> tuple[np.ndarray, np.ndarray]:
        refs = selected[q]
        gens = generations[refs].copy()
        live = (alphas[q][:, None] * values[refs]).sum(axis=0)
        out = live @ closure_m[q] + closure_c[q]
        return out, gens

    def full_current(q: int) -> np.ndarray:
        refs = selected[q]
        live = (alphas[q][:, None] * values[refs]).sum(axis=0)
        return full_region(live, gates[q], weights, biases)

    # Scheduled canonical writes: CRT performs no derived closure invalidation or patching.
    theoretical_eager_numeric_patches = 0
    same_value_writes = 0
    canonical_write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        same = rng.random() < 0.35
        same_value_writes += int(same)
        theoretical_eager_numeric_patches += int(reverse_counts[pid])
        ts = time.perf_counter_ns()
        write(pid, same)
        canonical_write_ns.append(time.perf_counter_ns() - ts)

    # Transactional serving. The closure contains no copied mutable value/generation.
    expected_conflicts = 0
    detected_conflicts = 0
    escaped_conflicts = 0
    retries = 0
    same_value_race_conflicts = 0
    semantic_mismatches = 0
    max_error = 0.0
    materialize_ns = []
    full_ns = []

    for _ in range(SERVES):
        q = rng.randrange(QUERIES)
        refs = selected[q]
        ts = time.perf_counter_ns()
        out, seen_gens = materialize(q)
        materialize_ns.append(time.perf_counter_ns() - ts)

        raced = rng.random() < RACE_PROB
        if raced:
            pid = int(refs[rng.randrange(TOPK)])
            same = rng.random() < 0.45
            expected_conflicts += 1
            same_value_race_conflicts += int(same)
            write(pid, same)

        current_gens = generations[refs]
        if not np.array_equal(seen_gens, current_gens):
            detected_conflicts += 1
            out, seen_gens = materialize(q)
            retries += 1
            current_gens = generations[refs]
        elif raced:
            # A selected generation was always incremented for an injected race.
            escaped_conflicts += 1

        # Commit barrier oracle.
        if not np.array_equal(seen_gens, current_gens):
            escaped_conflicts += 1
            continue

        ts = time.perf_counter_ns()
        oracle = full_current(q)
        full_ns.append(time.perf_counter_ns() - ts)
        err = float(np.max(np.abs(out - oracle)))
        max_error = max(max_error, err)
        if err > 1e-10:
            semantic_mismatches += 1

    # Dedicated paired latency sample, avoiding race/update noise.
    paired_closure = []
    paired_full = []
    for _ in range(5000):
        q = rng.randrange(QUERIES)
        ts = time.perf_counter_ns(); _ = materialize(q); paired_closure.append(time.perf_counter_ns() - ts)
        ts = time.perf_counter_ns(); _ = full_current(q); paired_full.append(time.perf_counter_ns() - ts)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Referential Tensor (CRT) / Live Neural Closure",
        "pods": PODS,
        "queries": QUERIES,
        "topk_live_references_per_closure": TOPK,
        "value_dimension": DIM,
        "compiled_affine_neural_layers": LAYERS,
        "scheduled_world_updates": UPDATES,
        "same_value_scheduled_updates": same_value_writes,
        "transactional_serves": SERVES,
        "injected_race_conflicts_expected": expected_conflicts,
        "race_conflicts_detected": detected_conflicts,
        "race_conflicts_escaped": escaped_conflicts,
        "retries": retries,
        "same_value_race_conflicts": same_value_race_conflicts,
        "semantic_mismatches_vs_full_current_recompute": semantic_mismatches,
        "max_abs_error_vs_full_current_recompute": max_error,
        "closure_compile_seconds": compile_seconds,
        "derived_closure_invalidations_on_world_write": 0,
        "derived_closure_numeric_patches_on_world_write": 0,
        "theoretical_eager_numeric_cache_patches": theoretical_eager_numeric_patches,
        "write_fanout_reduction_vs_eager_numeric_cache": 1.0 if theoretical_eager_numeric_patches else 0.0,
        "median_canonical_world_write_ns_python": statistics.median(canonical_write_ns),
        "median_live_closure_materialization_ns_python": statistics.median(paired_closure),
        "median_full_affine_region_recompute_ns_python": statistics.median(paired_full),
        "median_full_over_closure_speedup": statistics.median(paired_full) / max(1.0, statistics.median(paired_closure)),
        "closure_contains_mutable_value_bytes": False,
        "closure_contains_last_seen_generation": False,
        "mechanism": (
            "A retained neural artifact is a world-parameterized affine closure X(W)=b+sum_j A_j*deref(ref_j), not a copied numeric output. "
            "Stable-key attention supplies canonical refs/weights and any world-independent affine/gated neural region is partially evaluated "
            "into closure coefficients. Ordinary value updates therefore mutate only the canonical world cell. On serve, current values and "
            "generations are dereferenced transactionally; a generation race triggers retry before publication."
        ),
        "novelty_hypothesis": (
            "Treat selected live world references as first-class neural activation operands rather than immediately materialized V vectors. "
            "The resulting hidden state is a reusable function of future world state. This converts a class of neural cache invalidation/repair "
            "problems into late dereference, while retaining exact current-world equivalence over the referential region."
        ),
        "claim_boundary": (
            "Partial evaluation, closures, symbolic execution, pointers, stable-key attention and affine algebra are established individually. "
            "R340 tests their composition as a first-class lifecycle-aware neural tensor semantics. Arbitrary value-dependent nonlinear neural "
            "regions are not solved by this affine gate and require richer referential operators."
        ),
        "dod_status": "NOT_DOD; first-class referential-neural-state mechanism gate",
    }
    report["contract_pass"] = (
        semantic_mismatches == 0
        and max_error <= 1e-10
        and detected_conflicts == expected_conflicts
        and escaped_conflicts == 0
        and retries == expected_conflicts
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Causal Referential Tensor contract failed")


if __name__ == "__main__":
    main()
