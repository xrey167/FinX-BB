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

STAGE = "R334B-DELTA-WORLD-ATTENTION"
PODS = int(os.environ.get("SO_R334B_PODS", "2048"))
QUERIES = int(os.environ.get("SO_R334B_QUERIES", "1024"))
DK = int(os.environ.get("SO_R334B_DK", "32"))
DV = int(os.environ.get("SO_R334B_DV", "48"))
TOPK = int(os.environ.get("SO_R334B_TOPK", "8"))
UPDATES = int(os.environ.get("SO_R334B_UPDATES", "5000"))
CONTROL_PROBES = int(os.environ.get("SO_R334B_CONTROL_PROBES", "4"))
ALPHA = float(os.environ.get("SO_R334B_ALPHA", "3.0"))
REPORT_PATH = Path(os.environ.get("SO_R334B_REPORT", "r334b_report.json"))
SEED = 3340914


@dataclass
class Cache:
    ids: torch.Tensor
    generations: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor


def attend_all(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    score = q @ k.T
    top_score, ids = torch.topk(score, TOPK, dim=1, largest=True, sorted=True)
    w = torch.softmax(top_score, dim=1)
    out = (w.unsqueeze(-1) * v[ids]).sum(dim=1)
    return ids, w, out


def attend_one(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    score = q @ k.T
    top_score, ids = torch.topk(score, TOPK, largest=True, sorted=True)
    w = torch.softmax(top_score, dim=0)
    out = w @ v[ids]
    return ids, w, out


def make_entangled(base_k: torch.Tensor, v: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    return F.normalize(base_k + ALPHA * (v @ proj), dim=-1)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rng = random.Random(SEED)
    g = torch.Generator().manual_seed(SEED)
    dtype = torch.float64

    # Governed addresses route through immutable semantic keys. Ordinary world
    # writes can change V and generation, but not K.
    stable_k = F.normalize(torch.randn(PODS, DK, generator=g, dtype=dtype), dim=-1)
    queries = F.normalize(torch.randn(QUERIES, DK, generator=g, dtype=dtype), dim=-1)
    values = torch.randn(PODS, DV, generator=g, dtype=dtype)
    generations = torch.ones(PODS, dtype=torch.long)

    # Control architecture deliberately entangles mutable payload into K.
    value_to_key = torch.randn(DV, DK, generator=g, dtype=dtype) / (DV ** 0.5)
    ent_k = make_entangled(stable_k, values, value_to_key)

    t0 = time.perf_counter()
    stable_ids, stable_w, stable_out = attend_all(queries, stable_k, values)
    initial_ids = stable_ids.clone()
    initial_w = stable_w.clone()
    stable_seen_gen = generations[stable_ids].clone()

    reverse: list[list[tuple[int, int]]] = [[] for _ in range(PODS)]
    for qid in range(QUERIES):
        for slot in range(TOPK):
            reverse[int(stable_ids[qid, slot])].append((qid, slot))

    ent_ids, ent_w, ent_out = attend_all(queries, ent_k, values)
    ent_cache = [
        Cache(
            ids=ent_ids[q].clone(),
            generations=generations[ent_ids[q]].clone(),
            weights=ent_w[q].clone(),
            output=ent_out[q].clone(),
        )
        for q in range(QUERIES)
    ]
    build_seconds = time.perf_counter() - t0

    same_value_updates = 0
    same_value_selected_refreshes = 0
    same_value_numeric_changes = 0
    stable_local_patch_count = 0
    touched_per_update = []
    patch_ns = []
    local_audit_max_error = 0.0
    local_generation_mismatches = 0

    control_valid_probes = 0
    control_invalid_refreshes = 0
    control_hidden_route_false_accepts = 0
    control_hidden_output_false_accepts = 0

    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        old_v = values[pid].clone()
        same = rng.random() < 0.35
        new_v = old_v.clone() if same else torch.randn(DV, generator=g, dtype=dtype)
        if same:
            same_value_updates += 1
        delta = new_v - old_v
        generations[pid] += 1
        values[pid] = new_v

        # Exact DWA update law for fixed Q/K:
        # O_q' = O_q + alpha_qj * (V_j' - V_j)
        ts = time.perf_counter_ns()
        refs = reverse[pid]
        for qid, slot in refs:
            before = stable_out[qid].clone() if same else None
            stable_out[qid] += stable_w[qid, slot] * delta
            stable_seen_gen[qid, slot] = generations[pid]
            stable_local_patch_count += 1
            if same:
                same_value_selected_refreshes += 1
                if not torch.equal(before, stable_out[qid]):
                    same_value_numeric_changes += 1
        patch_ns.append(time.perf_counter_ns() - ts)
        touched_per_update.append(len(refs))

        for qid, _slot in refs[:8]:
            direct = (stable_w[qid].unsqueeze(-1) * values[stable_ids[qid]]).sum(dim=0)
            local_audit_max_error = max(
                local_audit_max_error,
                float((direct - stable_out[qid]).abs().max()),
            )
            local_generation_mismatches += int(
                not torch.equal(stable_seen_gen[qid], generations[stable_ids[qid]])
            )

        # The entangled control changes a physical routing key when the payload changes.
        ent_k[pid] = F.normalize(
            stable_k[pid] + ALPHA * (values[pid] @ value_to_key), dim=0
        )

        # A selected-generation cache check can miss an updated unselected page
        # that newly enters top-k because its K changed.
        for _probe in range(CONTROL_PROBES):
            qid = rng.randrange(QUERIES)
            old = ent_cache[qid]
            selected_generation_valid = torch.equal(
                old.generations, generations[old.ids]
            )
            fresh_ids, fresh_w, fresh_out = attend_one(
                queries[qid], ent_k, values
            )
            route_changed = not torch.equal(old.ids, fresh_ids)
            output_changed = float((old.output - fresh_out).abs().max()) > 1e-10

            if selected_generation_valid:
                control_valid_probes += 1
                control_hidden_route_false_accepts += int(route_changed)
                control_hidden_output_false_accepts += int(output_changed)
            else:
                control_invalid_refreshes += 1

            if (not selected_generation_valid) or route_changed or output_changed:
                ent_cache[qid] = Cache(
                    ids=fresh_ids.clone(),
                    generations=generations[fresh_ids].clone(),
                    weights=fresh_w.clone(),
                    output=fresh_out.clone(),
                )

    # Full exact current-world oracle for stable K.
    oracle_ids, oracle_w, oracle_out = attend_all(queries, stable_k, values)
    final_output_error = float((oracle_out - stable_out).abs().max())
    final_route_mismatches = int((oracle_ids != stable_ids).any(dim=1).sum())
    final_weight_error = float((oracle_w - stable_w).abs().max())
    final_generation_mismatches = int(
        (stable_seen_gen != generations[stable_ids]).any(dim=1).sum()
    )
    route_changes_from_initial = int((oracle_ids != initial_ids).any(dim=1).sum())
    weight_changes_from_initial = float((oracle_w - initial_w).abs().max())

    full_times = []
    for _ in range(7):
        ts = time.perf_counter_ns()
        _ = attend_all(queries, stable_k, values)
        full_times.append(time.perf_counter_ns() - ts)

    global_counterfactual = QUERIES * UPDATES
    median_patch = statistics.median(patch_ns)
    median_full = statistics.median(full_times)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Delta World Attention (DWA): stable-K, generation-scoped mutable-V",
        "pods": PODS,
        "queries": QUERIES,
        "topk": TOPK,
        "key_dimension": DK,
        "value_dimension": DV,
        "updates": UPDATES,
        "same_value_generation_updates": same_value_updates,
        "same_value_selected_lifetime_refreshes": same_value_selected_refreshes,
        "same_value_selected_numeric_output_changes": same_value_numeric_changes,
        "build_seconds": build_seconds,
        "stable_local_patch_count": stable_local_patch_count,
        "mean_queries_patched_per_update": statistics.mean(touched_per_update),
        "p99_queries_patched_per_update": sorted(touched_per_update)[int(0.99 * (len(touched_per_update) - 1))],
        "global_query_invalidations_counterfactual": global_counterfactual,
        "local_patch_reduction_vs_global": 1.0 - stable_local_patch_count / global_counterfactual,
        "stable_local_audit_max_error": local_audit_max_error,
        "stable_local_generation_mismatches": local_generation_mismatches,
        "stable_final_full_recompute_output_max_error": final_output_error,
        "stable_final_route_mismatches": final_route_mismatches,
        "stable_final_weight_max_error": final_weight_error,
        "stable_final_generation_mismatches": final_generation_mismatches,
        "stable_route_changes_from_initial": route_changes_from_initial,
        "stable_weight_changes_from_initial": weight_changes_from_initial,
        "median_local_delta_patch_ns_python": median_patch,
        "median_full_attention_recompute_ns": median_full,
        "full_recompute_over_local_patch_speedup": median_full / max(median_patch, 1),
        "control_generation_valid_probes": control_valid_probes,
        "control_generation_invalid_refreshes": control_invalid_refreshes,
        "control_hidden_route_false_accepts": control_hidden_route_false_accepts,
        "control_hidden_output_false_accepts": control_hidden_output_false_accepts,
        "mechanism": (
            "DWA makes governed address routing a stable-key operation and places ordinary mutable reality only in value pages. Because Q and K "
            "do not change on an ordinary value write, selected addresses and softmax weights are invariant. A changed selected page therefore "
            "updates every dependent attention output exactly with O'=O+alpha_j*(V'_j-V_j), while its generation lifetime is refreshed even for "
            "same-value ABA rewrites."
        ),
        "architectural_hypothesis": (
            "Separating routing lifetime from payload lifetime turns a class of neural world updates into exact sparse delta operations. When "
            "mutable payload is allowed to alter K, an unselected write can silently change future top-k and escape validation that checks only "
            "the generations of the previously selected pages. Stable-K removes that hidden dependency by construction."
        ),
        "claim_boundary": (
            "Attention, key/value separation, top-k routing, reverse indexes and incremental updates are established components. R334b validates "
            "the exact generation-aware update law and the hidden-dependency counterexample inside CKCA; novelty of the complete architecture is "
            "not yet claimed."
        ),
        "dod_status": "NOT_DOD; exact sparse world-attention update gate",
    }
    report["contract_pass"] = (
        final_output_error < 1e-9
        and final_route_mismatches == 0
        and final_weight_error < 1e-12
        and final_generation_mismatches == 0
        and route_changes_from_initial == 0
        and weight_changes_from_initial < 1e-12
        and same_value_numeric_changes == 0
        and control_hidden_route_false_accepts > 0
        and control_hidden_output_false_accepts > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("R334b Delta World Attention contract failed")


if __name__ == "__main__":
    main()
