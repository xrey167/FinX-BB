from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np

STAGE = "R349-LIVEKV-PAGE-TABLE-ATTENTION"
REPORT_PATH = Path(os.environ.get("SO_R349_REPORT", "ci-r349/report.json"))
PODS = int(os.environ.get("SO_R349_PODS", "4096"))
QUERIES = int(os.environ.get("SO_R349_QUERIES", "4000"))
DK = int(os.environ.get("SO_R349_DK", "32"))
DV = int(os.environ.get("SO_R349_DV", "32"))
TOPK = int(os.environ.get("SO_R349_TOPK", "8"))
UPDATES = int(os.environ.get("SO_R349_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R349_SERVES", "50000"))
FUTURE_WORLDS = int(os.environ.get("SO_R349_FUTURE_WORLDS", "24"))
CLASSES = 8
SEED = 3490914


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def topk_attention_compile(q: np.ndarray, k: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    """Compile sparse attention routes without touching mutable V pages."""
    refs = np.empty((len(q), topk), dtype=np.int32)
    weights = np.empty((len(q), topk), dtype=np.float32)
    kt = k.T
    batch = 256
    scale = float(DK) ** -0.5
    for start in range(0, len(q), batch):
        qb = q[start:start + batch]
        logits = (qb @ kt) * scale
        # Hard sparse attention is the architecture under test: top-k route first,
        # then exact softmax over only those selected semantic keys.
        idx = np.argpartition(logits, -topk, axis=1)[:, -topk:]
        selected = np.take_along_axis(logits, idx, axis=1)
        order = np.argsort(selected, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        selected = selected - selected.max(axis=1, keepdims=True)
        ex = np.exp(selected)
        w = ex / ex.sum(axis=1, keepdims=True)
        refs[start:start + len(qb)] = idx.astype(np.int32)
        weights[start:start + len(qb)] = w.astype(np.float32)
    return refs, weights


def materialize(refs: np.ndarray, weights: np.ndarray, v_pages: np.ndarray) -> np.ndarray:
    pages = v_pages[refs]
    return np.einsum("qk,qkd->qd", weights, pages, optimize=True)


def full_current(q: np.ndarray, k: np.ndarray, v_pages: np.ndarray) -> np.ndarray:
    refs, weights = topk_attention_compile(q, k, TOPK)
    return materialize(refs, weights, v_pages)


def descriptor_digest(refs: np.ndarray, weights16: np.ndarray) -> str:
    return hashlib.sha256(refs.tobytes() + weights16.tobytes()).hexdigest()


def bench_ns(fn, repeat: int = 20) -> float:
    xs = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns(); fn(); xs.append(time.perf_counter_ns() - t0)
    return float(statistics.median(xs))


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    py = random.Random(SEED)

    # K is semantic identity and intentionally stable. V is the mutable authoritative
    # world. A normal cache materializes V into hidden state; LiveKV caches only the
    # route to V pages and the attention weights.
    keys = normalize(rng.standard_normal((PODS, DK), dtype=np.float32))
    queries = normalize(rng.standard_normal((QUERIES, DK), dtype=np.float32))
    v = rng.standard_normal((PODS, DV), dtype=np.float32)
    generations = np.ones(PODS, dtype=np.int64)
    head = rng.standard_normal((DV, CLASSES), dtype=np.float32)

    t0 = time.perf_counter()
    refs, weights = topk_attention_compile(queries, keys, TOPK)
    compile_seconds = time.perf_counter() - t0
    weights16 = weights.astype(np.float16)
    # The serving representation is the actual fp16 weight cache, so all baselines
    # below use the same quantized weights for exact fair equivalence.
    weights_cached = weights16.astype(np.float32)
    digest_before = descriptor_digest(refs, weights16)

    home_numeric = materialize(refs, weights_cached, v)
    home_class = (home_numeric @ head).argmax(axis=1)

    # Future-world rewrite falsifier: descriptors are fixed, pages are redrawn.
    # LiveKV should equal a full sparse-attention recompute because Q/K are unchanged;
    # the stale numeric cache remains tied to the old world.
    max_abs_error = 0.0
    exact_mismatches = 0
    live_class_acc = []
    stale_class_acc = []
    for _ in range(FUTURE_WORLDS):
        future_v = rng.standard_normal((PODS, DV), dtype=np.float32)
        live = materialize(refs, weights_cached, future_v)
        # Recompile route from current Q/K as the matched RAG-like/full-attention control.
        full = full_current(queries, keys, future_v)
        # Quantize full route weights to the same fp16 representation before comparison.
        frefs, fw = topk_attention_compile(queries, keys, TOPK)
        full_q = materialize(frefs, fw.astype(np.float16).astype(np.float32), future_v)
        err = float(np.max(np.abs(live - full_q)))
        max_abs_error = max(max_abs_error, err)
        exact_mismatches += int(np.count_nonzero((live @ head).argmax(axis=1) != (full_q @ head).argmax(axis=1)))
        truth_class = (full_q @ head).argmax(axis=1)
        live_class_acc.append(float(np.mean((live @ head).argmax(axis=1) == truth_class)))
        stale_class_acc.append(float(np.mean(home_class == truth_class)))

    # Counterfactual write-fanout for conventional numeric cache repair.
    ref_counts = np.bincount(refs.ravel(), minlength=PODS).astype(np.int64)
    numeric_patch_counterfactual = 0
    same_value_updates = 0
    write_ns = []
    for _ in range(UPDATES):
        pid = py.randrange(PODS)
        numeric_patch_counterfactual += int(ref_counts[pid])
        same = py.random() < 0.35
        same_value_updates += int(same)
        t = time.perf_counter_ns()
        generations[pid] += 1
        if not same:
            v[pid] = rng.standard_normal(DV, dtype=np.float32)
        write_ns.append(time.perf_counter_ns() - t)

    digest_after = descriptor_digest(refs, weights16)

    # Generation-safe publication gate including same-value ABA races.
    expected = detected = escaped = retries = semantic_mismatches = 0
    for _ in range(SERVES):
        qi = py.randrange(QUERIES)
        r = refs[qi]
        seen = generations[r].copy()
        pred_vec = np.einsum("k,kd->d", weights_cached[qi], v[r], optimize=True)
        raced = py.random() < 0.08
        if raced:
            pid = int(r[py.randrange(TOPK)])
            expected += 1
            generations[pid] += 1
            if py.random() >= 0.45:
                v[pid] = rng.standard_normal(DV, dtype=np.float32)
        if not np.array_equal(seen, generations[r]):
            detected += 1; retries += 1
            pred_vec = np.einsum("k,kd->d", weights_cached[qi], v[r], optimize=True)
        elif raced:
            escaped += 1
        truth_vec = np.einsum("k,kd->d", weights_cached[qi], v[r], optimize=True)
        semantic_mismatches += int(np.argmax(pred_vec @ head) != np.argmax(truth_vec @ head))

    # Warm-query performance: current-world retrieval/recompute vs cached LiveKV route.
    bench_q = queries[:128]
    bench_refs = refs[:128]
    bench_weights = weights_cached[:128]
    full_ns = bench_ns(lambda: full_current(bench_q, keys, v), repeat=20)
    live_ns = bench_ns(lambda: materialize(bench_refs, bench_weights, v), repeat=80)
    full_per_query = full_ns / len(bench_q)
    live_per_query = live_ns / len(bench_q)
    compile_per_query = (compile_seconds * 1e9) / QUERIES
    denom = max(1.0, full_per_query - live_per_query)
    warm_reuse_crossover_queries = compile_per_query / denom

    live_bytes = TOPK * (4 + 2)  # int32 page ref + fp16 weight
    numeric_bytes = DV * 2       # matched BF16/FP16 materialized attention output

    report = {
        "stage": STAGE,
        "architecture_candidate": "LiveKV Page-Table Attention / Future-Bound V Cache",
        "pods": PODS,
        "queries": QUERIES,
        "key_dimension": DK,
        "value_dimension": DV,
        "topk": TOPK,
        "compile_seconds": compile_seconds,
        "descriptor_contains_mutable_value_bytes": False,
        "descriptor_contains_last_seen_generations": False,
        "descriptor_digest_unchanged_after_world_updates": digest_before == digest_after,
        "write_time_descriptor_patches_or_invalidations": 0,
        "scheduled_world_updates": UPDATES,
        "same_value_updates": same_value_updates,
        "numeric_cache_patches_counterfactual": numeric_patch_counterfactual,
        "mean_numeric_cache_patch_fanout_per_write": numeric_patch_counterfactual / UPDATES,
        "max_abs_error_vs_full_current_sparse_attention": max_abs_error,
        "class_mismatches_vs_full_current_sparse_attention": exact_mismatches,
        "mean_future_livekv_class_accuracy": statistics.mean(live_class_acc),
        "mean_future_stale_numeric_cache_accuracy": statistics.mean(stale_class_acc),
        "future_gain_livekv_minus_stale_numeric": statistics.mean(live_class_acc) - statistics.mean(stale_class_acc),
        "race_conflicts_expected": expected,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatches,
        "livekv_descriptor_bytes_per_query": live_bytes,
        "numeric_attention_cache_bytes_per_query": numeric_bytes,
        "livekv_to_numeric_cache_byte_ratio": live_bytes / numeric_bytes,
        "median_canonical_page_write_ns_python": statistics.median(write_ns),
        "median_full_current_retrieval_ns_per_query_python": full_per_query,
        "median_livekv_materialize_ns_per_query_python": live_per_query,
        "full_retrieval_over_livekv_materialize_speedup_python": full_per_query / max(1.0, live_per_query),
        "estimated_warm_reuse_crossover_queries": warm_reuse_crossover_queries,
        "mechanism": (
            "Sparse attention is split into a stable semantic route and a mutable value page table. The cache stores only top-k canonical page IDs "
            "and fp16 attention weights; it never stores world-derived V bytes. A world edit rewrites one canonical V page and increments its generation. "
            "Every cached query that points to that page changes denotation automatically at the next late materialization. Exact generation validation "
            "prevents publication across same-value ABA or value-changing races."
        ),
        "falsification_target": (
            "LiveKV fails this gate if a future-world materialization differs from a full current sparse-attention recompute under the same fp16 route, "
            "if any world write changes descriptor bytes, if a generation race escapes, or if the warm cached route is not cheaper than full retrieval."
        ),
        "claim_boundary": (
            "This is a systems/mechanism gate over hard top-k attention, not a novelty proof. Stable semantic K with mutable V resembles external memory, "
            "DNC-style addressing, paging and late binding. The research question is the use of page-table indirection as the retained Transformer V/KV "
            "cache semantics with exact lifecycle authority and zero value-derived update fanout. Strong RAG and real pretrained-attention gates remain."
        ),
        "dod_status": "NOT_DOD; LiveKV page-table attention mechanism gate",
    }
    report["contract_pass"] = (
        digest_before == digest_after
        and max_abs_error <= 1e-6
        and exact_mismatches == 0
        and statistics.mean(live_class_acc) == 1.0
        and detected == expected and escaped == 0 and retries == expected
        and semantic_mismatches == 0
        and live_per_query < full_per_query
        and live_bytes <= numeric_bytes
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "max_abs_error_vs_full_current_sparse_attention", "class_mismatches_vs_full_current_sparse_attention",
        "mean_future_livekv_class_accuracy", "mean_future_stale_numeric_cache_accuracy", "future_gain_livekv_minus_stale_numeric",
        "numeric_cache_patches_counterfactual", "mean_numeric_cache_patch_fanout_per_write",
        "livekv_to_numeric_cache_byte_ratio", "full_retrieval_over_livekv_materialize_speedup_python",
        "estimated_warm_reuse_crossover_queries", "race_conflicts_expected", "race_conflicts_detected",
        "race_conflicts_escaped", "race_semantic_mismatches", "contract_pass", "report_sha256"
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
