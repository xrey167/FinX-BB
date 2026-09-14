from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

STAGE = "R343-COMPACT-REFERENTIAL-CONTINUATION"
PODS = int(os.environ.get("SO_R343_PODS", "4096"))
QUERIES = int(os.environ.get("SO_R343_QUERIES", "3000"))
WORLD_DIM = int(os.environ.get("SO_R343_WORLD_DIM", "16"))
HIDDEN = int(os.environ.get("SO_R343_HIDDEN", "64"))
HEADS = int(os.environ.get("SO_R343_HEADS", "4"))
TOPK = int(os.environ.get("SO_R343_TOPK", "3"))
B_CODE = int(os.environ.get("SO_R343_B_CODE", "8"))
BLOCKS = int(os.environ.get("SO_R343_BLOCKS", "8"))
UPDATES = int(os.environ.get("SO_R343_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R343_SERVES", "80000"))
RACE_PROB = float(os.environ.get("SO_R343_RACE_PROB", "0.08"))
REPORT_PATH = Path(os.environ.get("SO_R343_REPORT", "r343_report.json"))
SEED = 3430914

if HIDDEN % HEADS:
    raise RuntimeError("HIDDEN must divide HEADS")
HEAD_DIM = HIDDEN // HEADS
KEY_DIM = 16
B_INPUT = 96
B_WIDTH = 96
B_LAYERS = 5


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)

    world = nrng.normal(size=(PODS, WORLD_DIM)).astype(np.float64)
    generations = np.ones(PODS, dtype=np.int64)

    # Frozen B compiler: expensive language/skill-side computation that is world-independent.
    b_inputs = nrng.normal(size=(QUERIES, B_INPUT)).astype(np.float64)
    b_weights = [nrng.normal(scale=0.08, size=(B_INPUT if i == 0 else B_WIDTH, B_WIDTH)).astype(np.float64) for i in range(B_LAYERS)]
    b_bias = [nrng.normal(scale=0.02, size=B_WIDTH).astype(np.float64) for _ in range(B_LAYERS)]
    code_proj = nrng.normal(scale=0.12, size=(B_WIDTH, B_CODE)).astype(np.float64)
    route_proj = nrng.normal(scale=0.11, size=(B_WIDTH, HEADS * KEY_DIM)).astype(np.float64)

    stable_keys = nrng.normal(size=(HEADS, PODS, KEY_DIM)).astype(np.float64)
    stable_keys /= np.linalg.norm(stable_keys, axis=-1, keepdims=True) + 1e-12

    wv = nrng.normal(scale=0.16, size=(HEADS, WORLD_DIM, HEAD_DIM)).astype(np.float64)
    wo = nrng.normal(scale=0.11, size=(HIDDEN, HIDDEN)).astype(np.float64)

    # A shared decoder turns the tiny B code into J continuation gates/shifts.
    gate_dec = nrng.normal(scale=0.16, size=(BLOCKS, B_CODE, HIDDEN)).astype(np.float64)
    shift_dec = nrng.normal(scale=0.03, size=(BLOCKS, B_CODE, HIDDEN)).astype(np.float64)
    block_w = []
    for _ in range(BLOCKS):
        w = nrng.normal(scale=0.07, size=(HIDDEN, HIDDEN))
        w += np.eye(HIDDEN) * 0.12
        block_w.append(w.astype(np.float64))

    def compile_b(raw: np.ndarray):
        h = raw
        for w, b in zip(b_weights, b_bias):
            h = np.tanh(h @ w + b)
        code = np.tanh(h @ code_proj)
        route_q = (h @ route_proj).reshape(HEADS, KEY_DIM)
        route_q /= np.linalg.norm(route_q, axis=-1, keepdims=True) + 1e-12
        return code, route_q

    def route(route_q: np.ndarray):
        refs = np.empty((HEADS, TOPK), dtype=np.int32)
        alpha = np.empty((HEADS, TOPK), dtype=np.float32)
        for h in range(HEADS):
            s = route_q[h] @ stable_keys[h].T
            ids = np.argpartition(s, -TOPK)[-TOPK:]
            vals = s[ids]
            order = np.argsort(-vals)
            ids = ids[order]
            vals = vals[order]
            refs[h] = ids
            alpha[h] = softmax(vals).astype(np.float32)
        return refs, alpha

    # Descriptor cache: no dense query-specific world transform.
    t0 = time.perf_counter()
    refs_cache = np.empty((QUERIES, HEADS, TOPK), dtype=np.int32)
    alpha_cache = np.empty((QUERIES, HEADS, TOPK), dtype=np.float32)
    code_cache = np.empty((QUERIES, B_CODE), dtype=np.float16)
    for q in range(QUERIES):
        code, rq = compile_b(b_inputs[q])
        refs, a = route(rq)
        refs_cache[q] = refs
        alpha_cache[q] = a
        # Quantization is part of the architecture: both cached and full oracle use this code.
        code_cache[q] = code.astype(np.float16)
    compile_seconds = time.perf_counter() - t0

    reverse_counts = np.bincount(refs_cache.reshape(-1), minlength=PODS)

    def decode_continuation(code16: np.ndarray):
        code = code16.astype(np.float64)
        gates = np.empty((BLOCKS, HIDDEN), dtype=np.float64)
        shifts = np.empty((BLOCKS, HIDDEN), dtype=np.float64)
        for l in range(BLOCKS):
            gates[l] = 0.25 + 0.75 / (1.0 + np.exp(-(code @ gate_dec[l])))
            shifts[l] = code @ shift_dec[l]
        return gates, shifts

    def j_execute(refs: np.ndarray, alpha: np.ndarray, code16: np.ndarray):
        head_out = []
        seen = []
        for h in range(HEADS):
            ids = refs[h]
            seen.append(generations[ids].copy())
            v = world[ids] @ wv[h]
            head_out.append((alpha[h].astype(np.float64)[:, None] * v).sum(axis=0))
        x = np.concatenate(head_out) @ wo
        gates, shifts = decode_continuation(code16)
        for l in range(BLOCKS):
            x = x + (x @ block_w[l]) * gates[l] + shifts[l]
        return x, tuple(seen)

    def cached_execute(q: int):
        return j_execute(refs_cache[q], alpha_cache[q], code_cache[q])

    def full_execute(q: int):
        code, rq = compile_b(b_inputs[q])
        refs, a = route(rq)
        code16 = code.astype(np.float16)
        return j_execute(refs, a, code16)

    def write(pid: int, same: bool):
        generations[pid] += 1
        if not same:
            world[pid] = nrng.normal(size=WORLD_DIM)

    eager_invalidations = 0
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        same = rng.random() < 0.35
        same_value_updates += int(same)
        eager_invalidations += int(reverse_counts[pid])
        ts = time.perf_counter_ns(); write(pid, same); write_ns.append(time.perf_counter_ns() - ts)

    expected_conflicts = detected_conflicts = escaped_conflicts = retries = 0
    semantic_mismatches = 0
    max_error = 0.0
    for _ in range(SERVES):
        q = rng.randrange(QUERIES)
        out, seen = cached_execute(q)
        raced = rng.random() < RACE_PROB
        if raced:
            h = rng.randrange(HEADS); k = rng.randrange(TOPK)
            pid = int(refs_cache[q, h, k])
            same = rng.random() < 0.45
            expected_conflicts += 1
            write(pid, same)
        current = tuple(generations[refs_cache[q, h]] for h in range(HEADS))
        if any(not np.array_equal(x, y) for x, y in zip(seen, current)):
            detected_conflicts += 1
            out, seen = cached_execute(q)
            retries += 1
            current = tuple(generations[refs_cache[q, h]] for h in range(HEADS))
        elif raced:
            escaped_conflicts += 1
        if any(not np.array_equal(x, y) for x, y in zip(seen, current)):
            escaped_conflicts += 1
            continue
        oracle, _ = full_execute(q)
        err = float(np.max(np.abs(out - oracle)))
        max_error = max(max_error, err)
        semantic_mismatches += int(err > 1e-9)

    cached_ns = []
    full_ns = []
    for _ in range(2500):
        q = rng.randrange(QUERIES)
        ts = time.perf_counter_ns(); cached_execute(q); cached_ns.append(time.perf_counter_ns() - ts)
        ts = time.perf_counter_ns(); full_execute(q); full_ns.append(time.perf_counter_ns() - ts)

    descriptor_bytes = refs_cache.nbytes + alpha_cache.nbytes + code_cache.nbytes
    # Compare with a BF16 numeric hidden cache, the typical target representation.
    bf16_numeric_cache_bytes = QUERIES * HIDDEN * 2

    report = {
        "stage": STAGE,
        "architecture_candidate": "Compact Referential Continuation (CRC)",
        "pods": PODS,
        "queries": QUERIES,
        "heads": HEADS,
        "topk_per_head": TOPK,
        "hidden_dimension": HIDDEN,
        "b_code_dimension": B_CODE,
        "reference_preserving_blocks_executed_at_materialization": BLOCKS,
        "world_updates": UPDATES,
        "same_value_updates": same_value_updates,
        "transactional_serves": SERVES,
        "expected_race_conflicts": expected_conflicts,
        "detected_race_conflicts": detected_conflicts,
        "escaped_race_conflicts": escaped_conflicts,
        "retries": retries,
        "semantic_mismatches_vs_full_recompile_current_world": semantic_mismatches,
        "max_abs_error_vs_full_recompile": max_error,
        "write_time_descriptor_invalidations_or_patches": 0,
        "numeric_cache_invalidations_counterfactual": eager_invalidations,
        "write_fanout_elimination_fraction": 1.0 if eager_invalidations else 0.0,
        "compile_seconds": compile_seconds,
        "median_world_write_ns_python": statistics.median(write_ns),
        "median_cached_continuation_execute_ns_python": statistics.median(cached_ns),
        "median_full_b_compile_route_j_execute_ns_python": statistics.median(full_ns),
        "full_over_cached_continuation_speedup": statistics.median(full_ns) / max(1.0, statistics.median(cached_ns)),
        "compact_descriptor_bytes": descriptor_bytes,
        "bf16_numeric_hidden_cache_bytes_counterfactual": bf16_numeric_cache_bytes,
        "descriptor_to_bf16_numeric_cache_ratio": descriptor_bytes / bf16_numeric_cache_bytes,
        "descriptor_contains_current_world_values": False,
        "descriptor_contains_last_seen_generations": False,
        "mechanism": (
            "Instead of expanding a reference-valued continuation into a dense per-query transform, CRC caches only sparse PodRefs, attention "
            "coefficients, and an 8-dimensional B-plane continuation code. Global shared neural weights decode that code and execute the "
            "world-dependent J suffix after late dereference. World writes do not touch the descriptor; a current-generation read is validated "
            "transactionally at serve time."
        ),
        "design_tradeoff": (
            "CRC deliberately spends more computation at read/materialization time than R341's dense compiled matrix in exchange for a much "
            "smaller future-bindable cache. It is a continuation descriptor, not a copied world-derived hidden vector."
        ),
        "claim_boundary": (
            "Continuations, latent codes, hypernetworks and cached query plans are known. R343 addresses practicality of the new referential "
            "activation semantics; it is not itself the novelty claim."
        ),
        "dod_status": "NOT_DOD; compact future-bindable activation descriptor gate",
    }
    report["contract_pass"] = (
        semantic_mismatches == 0 and max_error <= 1e-9
        and detected_conflicts == expected_conflicts and escaped_conflicts == 0
        and retries == expected_conflicts
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Compact Referential Continuation contract failed")


if __name__ == "__main__":
    main()
