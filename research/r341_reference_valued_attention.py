from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

STAGE = "R341-REFERENCE-VALUED-ATTENTION"
PODS = int(os.environ.get("SO_R341_PODS", "4096"))
QUERIES = int(os.environ.get("SO_R341_QUERIES", "2400"))
WORLD_DIM = int(os.environ.get("SO_R341_WORLD_DIM", "8"))
HIDDEN = int(os.environ.get("SO_R341_HIDDEN", "16"))
HEADS = int(os.environ.get("SO_R341_HEADS", "4"))
TOPK = int(os.environ.get("SO_R341_TOPK", "3"))
BLOCKS = int(os.environ.get("SO_R341_BLOCKS", "8"))
UPDATES = int(os.environ.get("SO_R341_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R341_SERVES", "100000"))
RACE_PROB = float(os.environ.get("SO_R341_RACE_PROB", "0.08"))
REPORT_PATH = Path(os.environ.get("SO_R341_REPORT", "r341_report.json"))
SEED = 3410914

if HIDDEN % HEADS != 0:
    raise RuntimeError("HIDDEN must be divisible by HEADS")
HEAD_DIM = HIDDEN // HEADS


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def compile_post_attention_transform(
    gates: np.ndarray,
    shifts: np.ndarray,
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compile h <- h + gate_l * (h @ W_l) + shift_l into h@M+c.

    Gates/shifts are B-plane/query state only. Therefore this whole residual region
    can transform a reference-valued activation without dereferencing world values.
    """
    m = np.eye(HIDDEN, dtype=np.float64)
    c = np.zeros(HIDDEN, dtype=np.float64)
    for l in range(BLOCKS):
        # row-vector: gate * (h@W) == h @ (W * gate[None,:])
        a = np.eye(HIDDEN, dtype=np.float64) + weights[l] * gates[l][None, :]
        m = m @ a
        c = c @ a + shifts[l]
    return m, c


def apply_blocks(
    h: np.ndarray,
    gates: np.ndarray,
    shifts: np.ndarray,
    weights: list[np.ndarray],
) -> np.ndarray:
    x = h
    for l in range(BLOCKS):
        x = x + (x @ weights[l]) * gates[l] + shifts[l]
    return x


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)

    # Model-independent mutable world values and generations.
    world = nrng.normal(size=(PODS, WORLD_DIM)).astype(np.float64)
    generations = np.ones(PODS, dtype=np.int64)

    # Stable address/routing keys. Ordinary value writes cannot change them.
    key_dim = 12
    keys = nrng.normal(size=(HEADS, PODS, key_dim))
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True) + 1e-12
    queries = nrng.normal(size=(QUERIES, HEADS, key_dim))
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True) + 1e-12

    # Head value projections and output projection are neural weights.
    wv = nrng.normal(scale=0.25, size=(HEADS, WORLD_DIM, HEAD_DIM)).astype(np.float64)
    wo = nrng.normal(scale=0.22, size=(HIDDEN, HIDDEN)).astype(np.float64)

    # Eight B-conditioned reference-preserving residual blocks.
    block_w = []
    for _ in range(BLOCKS):
        w = nrng.normal(scale=0.10, size=(HIDDEN, HIDDEN))
        w += np.eye(HIDDEN) * 0.18
        block_w.append(w.astype(np.float64))
    b_gates = 0.35 + 0.9 / (1.0 + np.exp(-nrng.normal(size=(QUERIES, BLOCKS, HIDDEN))))
    b_shifts = nrng.normal(scale=0.03, size=(QUERIES, BLOCKS, HIDDEN)).astype(np.float64)

    # Compile reference-valued attention routes + post-attention neural transform.
    t0 = time.perf_counter()
    selected = np.empty((QUERIES, HEADS, TOPK), dtype=np.int32)
    alpha = np.empty((QUERIES, HEADS, TOPK), dtype=np.float64)
    for h in range(HEADS):
        scores = queries[:, h] @ keys[h].T
        ids = np.argpartition(scores, -TOPK, axis=1)[:, -TOPK:]
        s = np.take_along_axis(scores, ids, axis=1)
        order = np.argsort(-s, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        s = np.take_along_axis(s, order, axis=1)
        selected[:, h] = ids
        alpha[:, h] = softmax(s)

    post_m = np.empty((QUERIES, HIDDEN, HIDDEN), dtype=np.float64)
    post_c = np.empty((QUERIES, HIDDEN), dtype=np.float64)
    for q in range(QUERIES):
        post_m[q], post_c[q] = compile_post_attention_transform(
            b_gates[q], b_shifts[q], block_w
        )
    compile_seconds = time.perf_counter() - t0

    # Reverse fanout exists only to quantify maintenance of a conventional numeric
    # cache. Reference-valued attention performs no write-time cache patch/kill.
    reverse_counts = np.bincount(selected.reshape(-1), minlength=PODS)

    def write(pid: int, same: bool) -> None:
        generations[pid] += 1
        if not same:
            world[pid] = nrng.normal(size=WORLD_DIM)

    def numeric_attention(q: int) -> np.ndarray:
        heads = []
        for h in range(HEADS):
            refs = selected[q, h]
            projected = world[refs] @ wv[h]
            heads.append((alpha[q, h][:, None] * projected).sum(axis=0))
        concat = np.concatenate(heads)
        return concat @ wo

    def full_numeric(q: int) -> np.ndarray:
        return apply_blocks(numeric_attention(q), b_gates[q], b_shifts[q], block_w)

    def materialize_reference_stream(q: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        """Dereference only at the final materialization barrier.

        Before this function the cached state contains only refs, stable attention
        weights, and B-conditioned neural transforms. No current world V/generation
        has been copied into the residual stream.
        """
        seen = []
        head_numeric = []
        for h in range(HEADS):
            refs = selected[q, h]
            seen.append(generations[refs].copy())
            # This is the single late dereference of the head's live world values.
            projected = world[refs] @ wv[h]
            head_numeric.append((alpha[q, h][:, None] * projected).sum(axis=0))
        h0 = np.concatenate(head_numeric) @ wo
        return h0 @ post_m[q] + post_c[q], tuple(seen)

    # Stronger explicit coefficient oracle: derive the same final tensor directly as
    # a sum of live world references, without first materializing attention heads.
    # It validates that the residual stream is genuinely reference-valued, not merely
    # a late numeric cache hidden behind an API.
    head_insert = []
    for h in range(HEADS):
        # WORLD_DIM -> HEAD_DIM -> concat slice -> WO gives WORLD_DIM -> HIDDEN.
        tmp = np.zeros((WORLD_DIM, HIDDEN), dtype=np.float64)
        start = h * HEAD_DIM
        tmp[:, start:start + HEAD_DIM] = wv[h]
        head_insert.append(tmp @ wo)

    def materialize_explicit_reference_sum(q: int) -> np.ndarray:
        out = post_c[q].copy()
        for h in range(HEADS):
            transform = head_insert[h] @ post_m[q]
            refs = selected[q, h]
            for k, ref in enumerate(refs):
                out += alpha[q, h, k] * (world[int(ref)] @ transform)
        return out

    # Ordinary writes: no derived query cache operation at all.
    eager_numeric_invalidations = 0
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        same = rng.random() < 0.35
        same_value_updates += int(same)
        eager_numeric_invalidations += int(reverse_counts[pid])
        ts = time.perf_counter_ns()
        write(pid, same)
        write_ns.append(time.perf_counter_ns() - ts)

    expected_conflicts = 0
    detected_conflicts = 0
    escaped_conflicts = 0
    retries = 0
    same_value_races = 0
    semantic_mismatches = 0
    explicit_reference_mismatches = 0
    max_full_error = 0.0
    max_explicit_error = 0.0

    for _ in range(SERVES):
        q = rng.randrange(QUERIES)
        out, seen = materialize_reference_stream(q)

        raced = rng.random() < RACE_PROB
        if raced:
            h = rng.randrange(HEADS)
            k = rng.randrange(TOPK)
            pid = int(selected[q, h, k])
            same = rng.random() < 0.45
            expected_conflicts += 1
            same_value_races += int(same)
            write(pid, same)

        current = tuple(generations[selected[q, h]] for h in range(HEADS))
        if any(not np.array_equal(a, b) for a, b in zip(seen, current)):
            detected_conflicts += 1
            out, seen = materialize_reference_stream(q)
            retries += 1
            current = tuple(generations[selected[q, h]] for h in range(HEADS))
        elif raced:
            escaped_conflicts += 1

        if any(not np.array_equal(a, b) for a, b in zip(seen, current)):
            escaped_conflicts += 1
            continue

        oracle = full_numeric(q)
        explicit = materialize_explicit_reference_sum(q)
        e1 = float(np.max(np.abs(out - oracle)))
        e2 = float(np.max(np.abs(explicit - oracle)))
        max_full_error = max(max_full_error, e1)
        max_explicit_error = max(max_explicit_error, e2)
        semantic_mismatches += int(e1 > 1e-10)
        explicit_reference_mismatches += int(e2 > 1e-10)

    # Timing: late referential materialization versus all eight residual blocks.
    ref_ns = []
    full_ns = []
    explicit_ns = []
    for _ in range(5000):
        q = rng.randrange(QUERIES)
        ts = time.perf_counter_ns(); materialize_reference_stream(q); ref_ns.append(time.perf_counter_ns() - ts)
        ts = time.perf_counter_ns(); full_numeric(q); full_ns.append(time.perf_counter_ns() - ts)
        ts = time.perf_counter_ns(); materialize_explicit_reference_sum(q); explicit_ns.append(time.perf_counter_ns() - ts)

    # Cache/storage accounting. A numeric cache is small but stale; the referential
    # cache is larger and future-bindable. We report rather than hide this trade-off.
    numeric_cache_bytes = QUERIES * HIDDEN * 8
    referential_cache_bytes = (
        selected.nbytes + alpha.nbytes + post_m.nbytes + post_c.nbytes
    )

    report = {
        "stage": STAGE,
        "architecture_candidate": "Reference-Valued Attention + Referential Residual Stream (RVA-RRS)",
        "pods": PODS,
        "queries": QUERIES,
        "heads": HEADS,
        "topk_per_head": TOPK,
        "world_value_dimension": WORLD_DIM,
        "hidden_dimension": HIDDEN,
        "reference_preserving_residual_blocks": BLOCKS,
        "scheduled_world_updates": UPDATES,
        "same_value_updates": same_value_updates,
        "transactional_serves": SERVES,
        "expected_race_conflicts": expected_conflicts,
        "detected_race_conflicts": detected_conflicts,
        "escaped_race_conflicts": escaped_conflicts,
        "retries": retries,
        "same_value_race_conflicts": same_value_races,
        "semantic_mismatches_vs_full_numeric_network": semantic_mismatches,
        "explicit_reference_sum_mismatches": explicit_reference_mismatches,
        "max_abs_error_vs_full_numeric_network": max_full_error,
        "max_explicit_reference_sum_error": max_explicit_error,
        "derived_cache_invalidations_or_patches_on_world_write": 0,
        "conventional_numeric_cache_invalidations_counterfactual": eager_numeric_invalidations,
        "write_fanout_elimination_fraction": 1.0 if eager_numeric_invalidations else 0.0,
        "median_canonical_write_ns_python": statistics.median(write_ns),
        "median_reference_stream_materialization_ns_python": statistics.median(ref_ns),
        "median_full_numeric_network_ns_python": statistics.median(full_ns),
        "median_explicit_reference_sum_ns_python": statistics.median(explicit_ns),
        "full_numeric_over_reference_stream_speedup": statistics.median(full_ns) / max(1.0, statistics.median(ref_ns)),
        "numeric_cache_bytes_counterfactual": numeric_cache_bytes,
        "referential_cache_bytes": referential_cache_bytes,
        "referential_to_numeric_cache_byte_ratio": referential_cache_bytes / numeric_cache_bytes,
        "cached_state_contains_current_world_values": False,
        "cached_state_contains_last_seen_generations": False,
        "compile_seconds": compile_seconds,
        "mechanism": (
            "Attention does not immediately return a numeric world-derived hidden vector. It returns a sparse reference-valued object: "
            "canonical PodRefs plus stable attention coefficients. Eight B-conditioned residual blocks transform the referential stream by "
            "updating world-independent coefficient transforms. Current V pages are dereferenced only at a materialization barrier, followed "
            "by exact generation validation before publication."
        ),
        "novelty_hypothesis": (
            "Make references to live mutable world cells first-class attention values and hidden-state operands. A cached residual stream can "
            "therefore be a reusable neural function of future world state rather than a tensor tied to the world version in which it was "
            "created. This is a change to neural activation semantics, not only a cache-coherence wrapper."
        ),
        "claim_boundary": (
            "Sparse attention, pointers/references, partial evaluation, symbolic tensors and affine transformations have prior art. The open "
            "novelty question is the specific neural semantics in which attention V produces generation-safe live references that remain "
            "undereferenced across multiple residual blocks and can be cached across arbitrary payload rewrites."
        ),
        "dod_status": "NOT_DOD; reference-valued neural activation gate",
    }
    report["contract_pass"] = (
        semantic_mismatches == 0
        and explicit_reference_mismatches == 0
        and max_full_error <= 1e-10
        and max_explicit_error <= 1e-10
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
        raise SystemExit("Reference-Valued Attention contract failed")


if __name__ == "__main__":
    main()
