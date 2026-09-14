from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

STAGE = "R334-DELTA-WORLD-ATTENTION"
PODS = int(os.environ.get("SO_R334_PODS", "2048"))
QUERIES = int(os.environ.get("SO_R334_QUERIES", "1024"))
DK = int(os.environ.get("SO_R334_DK", "32"))
DV = int(os.environ.get("SO_R334_DV", "48"))
TOPK = int(os.environ.get("SO_R334_TOPK", "8"))
UPDATES = int(os.environ.get("SO_R334_UPDATES", "5000"))
ENTANGLED_PROBES = int(os.environ.get("SO_R334_ENTANGLED_PROBES", "4"))
ENTANGLE_ALPHA = float(os.environ.get("SO_R334_ALPHA", "3.0"))
REPORT_PATH = Path(os.environ.get("SO_R334_REPORT", "r334_report.json"))
SEED = 3340914


@dataclass
class EntangledEntry:
    selected: torch.Tensor
    generations: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor


def topk_attention_one(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
    scores = q @ keys.T
    score, idx = torch.topk(scores, TOPK, largest=True, sorted=True)
    w = torch.softmax(score, dim=0)
    out = w @ values[idx]
    return idx, w, out


def topk_attention_all(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
    scores = queries @ keys.T
    score, idx = torch.topk(scores, TOPK, dim=1, largest=True, sorted=True)
    w = torch.softmax(score, dim=1)
    selected_values = values[idx]
    out = (w.unsqueeze(-1) * selected_values).sum(dim=1)
    return idx, w, out


def entangled_keys(base_keys: torch.Tensor, values: torch.Tensor, proj: torch.Tensor):
    return F.normalize(base_keys + ENTANGLE_ALPHA * (values @ proj), dim=-1)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rng = random.Random(SEED)
    g = torch.Generator().manual_seed(SEED)
    dtype = torch.float64

    # Stable routing/address state.
    stable_keys = F.normalize(torch.randn(PODS, DK, generator=g, dtype=dtype), dim=-1)
    queries = F.normalize(torch.randn(QUERIES, DK, generator=g, dtype=dtype), dim=-1)
    values = torch.randn(PODS, DV, generator=g, dtype=dtype)
    generations = torch.ones(PODS, dtype=torch.long)

    # Control: ordinary mutable value payload is allowed to alter its key.
    value_to_key = torch.randn(DV, DK, generator=g, dtype=dtype) / (DV ** 0.5)
    current_entangled_keys = entangled_keys(stable_keys, values, value_to_key)

    # Initial stable attention cache.
    t0 = time.perf_counter()
    stable_idx, stable_w, stable_out = topk_attention_all(queries, stable_keys, values)
    stable_dep_gen = generations[stable_idx].clone()
    initial_stable_idx = stable_idx.clone()
    initial_stable_w = stable_w.clone()

    reverse: list[list[tuple[int, int]]] = [[] for _ in range(PODS)]
    for qid in range(QUERIES):
        for slot in range(TOPK):
            reverse[int(stable_idx[qid, slot])].append((qid, slot))

    ent_idx, ent_w, ent_out = topk_attention_all(queries, current_entangled_keys, values)
    ent_entries = [
        EntangledEntry(
            selected=ent_idx[qid].clone(),
            generations=generations[ent_idx[qid]].clone(),
            weights=ent_w[qid].clone(),
            output=ent_out[qid].clone(),
        )
        for qid in range(QUERIES)
    ]
    build_seconds = time.perf_counter() - t0

    local_patch_ns = []
    touched_queries = []
    same_value_updates = 0
    same_value_selected_lifetime_refreshes = 0
    same_value_selected_output_changes = 0
    stable_incremental_audit_max_error = 0.0
    stable_generation_mismatches = 0

    entangled_generation_valid_probes = 0
    entangled_hidden_route_false_accepts = 0
    entangled_hidden_output_false_accepts = 0
    entangled_generation_invalid_refreshes = 0

    for step in range(UPDATES):
        pid = rng.randrange(PODS)
        old_value = values[pid].clone()
        same = rng.random() < 0.35
        if same:
            new_value = old_value.clone()
            same_value_updates += 1
        else:
            new_value = torch.randn(DV, generator=g, dtype=dtype)
        delta_v = new_value - old_value
        generations[pid] += 1
        values[pid] = new_value

        # Stable split-K/V attention: K and therefore selection/softmax weights do
        # not change. Patch exactly the queries that selected this value page.
        t_patch = time.perf_counter_ns()
        refs = reverse[pid]
        for qid, slot in refs:
            before = stable_out[qid].clone() if same else None
            stable_out[qid] += stable_w[qid, slot] * delta_v
            stable_dep_gen[qid, slot] = generations[pid]
            if same:
                same_value_selected_lifetime_refreshes += 1
                if not torch.equal(before, stable_out[qid]):
                    same_value_selected_output_changes += 1
        local_patch_ns.append(time.perf_counter_ns() - t_patch)
        touched_queries.append(len(refs))

        # Cheap exact local audit against direct attention over the cached route.
        for qid, _ in refs[:8]:
            direct = (stable_w[qid].unsqueeze(-1) * values[stable_idx[qid]]).sum(dim=0)
            err = float((direct - stable_out[qid]).abs().max())
            stable_incremental_audit_max_error = max(stable_incremental_audit_max_error, err)
            stable_generation_mismatches += int(
                not torch.equal(stable_dep_gen[qid], generations[stable_idx[qid]])
            )

        # Update only the changed physical key row in the entangled control.
        current_entangled_keys[pid] = F.normalize(
            stable_keys[pid] + ENTANGLE_ALPHA * (values[pid] @ value_to_key), dim=0
        )

        # Selected-generation validation misses "newly enters top-k" dependencies.
        for _ in range(ENTANGLED_PROBES):
            qid = rng.randrange(QUERIES)
            entry = ent_entries[qid]
            valid = torch.equal(entry.generations, generations[entry.selected])
            fresh_idx, fresh_w, fresh_out = topk_attention_one(
                queries[qid], current_entangled_keys, values
            )
            if valid:
                entangled_generation_valid_probes += 1
                route_changed = not torch.equal(entry.selected, fresh_idx)
                output_changed = float((entry.output - fresh_out).abs().max()) > 1e-10
                entangled_hidden_route_false_accepts += int(route_changed)
                entangled_hidden_output_false_accepts += int(output_changed)
            else:
                entangled_generation_invalid_refreshes += 1

            # Refresh after observing the admission result, keeping the control
            # population realistic rather than allowing one stale entry to persist.
            if (not valid) or (not torch.equal(entry.selected, fresh_idx)) or float((entry.output - fresh_out).abs().max()) > 1e-10:
                ent_entries[qid] = EntangledEntry(
                    selected=fresh_idx.clone(),
                    generations=generations[fresh_idx].clone(),
                    weights=fresh_w.clone(),
                    output=fresh_out.clone(),
                )

    # Global exact oracle after all incremental stable-K patches.
    oracle_idx, oracle_w, oracle_out = topk_attention_all(queries, stable_keys, values)
    final_output_error = float((oracle_out - stable_out).abs().max())
    final_route_mismatches = int((oracle_idx != stable_idx).any(dim=1).sum())
    final_weight_error = float((oracle_w - stable_w).abs().max())
    final_generation_mismatches = int(
        (stable_dep_gen != generations[stable_idx]).any(dim=1).sum()
    )
    original_route_mismatches = int((oracle_idx != initial_stable_idx).any(dim=1).sum())
    original_weight_error = float((oracle_w - initial_stable_w).abs().max())

    # Performance counterfactual: full QK/top-k/V recomputation versus one local
    # generation/value patch. Full recompute result is intentionally discarded.
    full_times = []
    for _ in range(5):
        ts = time.perf_counter_ns()
        _ = topk_attention_all(queries, stable_keys, values)
        full_times.append(time.perf_counter_ns() - ts)

    median_patch = statistics.median(local_patch_ns)
    median_full = statistics.median(full_times)
    global_invalidation_counterfactual = QUERIES * UPDATES
    local_touched_total = sum(touched_queries)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Delta World Attention (DWA) / split stable-K mutable-V world attention",
        "pods": PODS,
        "queries": QUERIES,
        "key_dimension": DK,
        "value_dimension": DV,
        "topk": TOPK,
        "updates": UPDATES,
        "same_value_generation_updates": same_value_updates,
        "build_seconds": build_seconds,
        "stable_incremental_local_audit_max_error": stable_incremental_audit_max_error,
        "stable_final_full_recompute_output_max_error": final_output_error,
        "stable_final_route_mismatches": final_route_mismatches,
        "stable_final_weight_max_error": final_weight_error,
        "stable_route_mismatches_vs_initial": original_route_mismatches,
        "stable_weight_error_vs_initial": original_weight_error,
        "stable_generation_mismatches_local_audits": stable_generation_mismatches,
        "stable_generation_mismatches_final": final_generation_mismatches,
        "same_value_selected_lifetime_refreshes": same_value_selected_lifetime_refreshes,
        "same_value_selected_numeric_output_changes": same_value_selected_output_changes,
        "mean_queries_patched_per_value_update": statistics.mean(touched_queries),
        "p99_queries_patched_per_value_update": sorted(touched_queries)[int(0.99 * (len(touched_queries) - 1))],
        "local_query_patches_total": local_touched_total,
        "global_query_invalidations_counterfactual": global_invalidation_counterfactual,
        "local_invalidation_reduction_vs_global": 1.0 - local_touched_total / global_invalidation_counterfactual,
        "median_local_delta_patch_ns_python": median_patch,
        "median_full_attention_recompute_ns": median_full,
        "full_recompute_over_local_patch_speedup": median_full / max(median_patch, 1),
        "entangled_generation_valid_probes": entangled_generation_valid_probes,
        "entangled_generation_invalid_refreshes": entangled_generation_invalid_refreshes,
        "entangled_hidden_route_false_accepts": entangled_hidden_route_false_accepts,
        "entangled_hidden_output_false_accepts": entangled_hidden_output_false_accepts,
        "mechanism": (
            "Governed attention splits address routing from mutable payload: K is derived from lifecycle-stable semantic/address identity, while V is "
            "a generation-scoped current value page. For fixed Q and K, attention weights are invariant under ordinary value updates, so a sparse "
            "value rewrite has the exact update law O' = O + alpha_j * (V'_j - V_j) for every query that selected page j. The reverse index patches "
            "only those outputs and refreshes their generation lifetime; same-value rewrites refresh lifetime while leaving numeric output unchanged."
        ),
        "novel_architecture_hypothesis": (
            "A World Port need not rerun neural routing when only reality's payload changes. By enforcing stable-K/mutable-V separation, world attention "
            "becomes incrementally patchable exactly, while entangled K/V attention creates hidden dependencies where an unselected value update can "
            "change future routing without invalidating the generations of the previously selected pages."
        ),
        "claim_boundary": (
            "Attention, key/value separation, sparse top-k, reverse indexes and incremental linear updates are established individually. R334 tests "
            "their composition as an exact generation-aware update law for governed neural world attention. Novelty of the complete CKCA/WIT system "
            "still requires literature/patent audit and a real-model kernel implementation."
        ),
        "dod_status": "NOT_DOD; exact incremental world-attention mechanism gate",
    }
    report["contract_pass"] = (
        final_output_error < 1e-9
        and final_route_mismatches == 0
        and final_weight_error < 1e-12
        and final_generation_mismatches == 0
        and original_route_mismatches == 0
        and same_value_selected_numeric_output_changes == 0
        and entangled_hidden_route_false_accepts > 0
        and entangled_hidden_output_false_accepts > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Delta World Attention contract failed")


if __name__ == "__main__":
    main()
