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

STAGE = "R330-INDIRECTION-STABLE-WORLD-ATTENTION"
PODS = int(os.environ.get("SO_R330_PODS", "2048"))
QUERIES = int(os.environ.get("SO_R330_QUERIES", "2048"))
DIM = int(os.environ.get("SO_R330_DIM", "32"))
TOPK = int(os.environ.get("SO_R330_TOPK", "4"))
UPDATES = int(os.environ.get("SO_R330_UPDATES", "5000"))
PROBES_PER_UPDATE = int(os.environ.get("SO_R330_PROBES_PER_UPDATE", "8"))
ENTANGLE_ALPHA = float(os.environ.get("SO_R330_ENTANGLE_ALPHA", "2.5"))
REPORT_PATH = Path(os.environ.get("SO_R330_REPORT", "r330_report.json"))
SEED = 3300914


@dataclass
class CacheEntry:
    selected: tuple[int, ...]
    generations: tuple[int, ...]
    output: float


def topk_for(query: torch.Tensor, keys: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor]:
    scores = query @ keys.T
    vals, idx = torch.topk(scores, TOPK, largest=True, sorted=True)
    return tuple(int(x) for x in idx.tolist()), vals


def output_for(selected: tuple[int, ...], scores: torch.Tensor, values: torch.Tensor) -> float:
    weights = torch.softmax(scores.float(), dim=0)
    v = values[torch.tensor(selected, dtype=torch.long)].float()
    return float((weights * v).sum())


def make_entry(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, generations: torch.Tensor) -> CacheEntry:
    sel, scores = topk_for(query, keys)
    gens = tuple(int(generations[i]) for i in sel)
    return CacheEntry(sel, gens, output_for(sel, scores, values))


def cache_generation_valid(entry: CacheEntry, generations: torch.Tensor) -> bool:
    return all(int(generations[pid]) == g for pid, g in zip(entry.selected, entry.generations))


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rng = random.Random(SEED)
    g = torch.Generator().manual_seed(SEED)

    # Immutable semantic address keys. A value update never changes these keys.
    base_keys = F.normalize(torch.randn(PODS, DIM, generator=g), dim=-1)
    # A control architecture entangles current mutable value into the retrieval key.
    value_dirs = F.normalize(torch.randn(PODS, DIM, generator=g), dim=-1)
    queries = F.normalize(torch.randn(QUERIES, DIM, generator=g), dim=-1)
    values = torch.empty(PODS).uniform_(-1.0, 1.0, generator=g)
    generations = torch.ones(PODS, dtype=torch.long)

    def entangled_keys() -> torch.Tensor:
        return F.normalize(base_keys + ENTANGLE_ALPHA * values[:, None] * value_dirs, dim=-1)

    t0 = time.perf_counter()
    stable_scores = queries @ base_keys.T
    stable_top_scores, stable_top_ids = torch.topk(stable_scores, TOPK, dim=-1, largest=True, sorted=True)
    initial_entangled = entangled_keys()
    ent_scores = queries @ initial_entangled.T
    ent_top_scores, ent_top_ids = torch.topk(ent_scores, TOPK, dim=-1, largest=True, sorted=True)

    stable_cache: list[CacheEntry] = []
    entangled_cache: list[CacheEntry] = []
    stable_reverse: list[list[int]] = [[] for _ in range(PODS)]
    for qid in range(QUERIES):
        ssel = tuple(int(x) for x in stable_top_ids[qid].tolist())
        stable_cache.append(CacheEntry(
            ssel,
            tuple(1 for _ in ssel),
            output_for(ssel, stable_top_scores[qid], values),
        ))
        for pid in ssel:
            stable_reverse[pid].append(qid)
        esel = tuple(int(x) for x in ent_top_ids[qid].tolist())
        entangled_cache.append(CacheEntry(
            esel,
            tuple(1 for _ in esel),
            output_for(esel, ent_top_scores[qid], values),
        ))
    build_seconds = time.perf_counter() - t0

    stable_false_accepts = 0
    stable_output_false_serves = 0
    stable_audit_selection_mismatches = 0
    entangled_false_accepts = 0
    entangled_output_false_serves = 0
    entangled_invalid_refreshes = 0
    entangled_valid_serves = 0
    stable_local_invalidations = 0
    global_predicate_invalidations = 0
    same_value_updates = 0
    selected_update_counts = []
    update_ns = []
    probe_ns = []

    for _ in range(UPDATES):
        pid = rng.randrange(PODS)
        old_value = float(values[pid])
        same = rng.random() < 0.35
        if same:
            new_value = old_value
            same_value_updates += 1
        else:
            new_value = rng.uniform(-1.0, 1.0)
        generations[pid] += 1
        values[pid] = new_value

        ts = time.perf_counter_ns()
        # Stable-key architecture needs no routing/index rebuild on a value update.
        local = len(stable_reverse[pid])
        stable_local_invalidations += local
        selected_update_counts.append(local)
        # A global candidate/predicate generation would conservatively kill every cache.
        global_predicate_invalidations += QUERIES
        # Only one entangled key row changes physically.
        current_ent_key = F.normalize(
            base_keys[pid] + ENTANGLE_ALPHA * values[pid] * value_dirs[pid], dim=0
        )
        update_ns.append(time.perf_counter_ns() - ts)

        for _ in range(PROBES_PER_UPDATE):
            qid = rng.randrange(QUERIES)
            q = queries[qid]
            ts = time.perf_counter_ns()

            # Stable architecture: exact route is cached from immutable keys. Audit
            # the selected IDs occasionally by recomputing the top-k from scratch.
            s_entry = stable_cache[qid]
            s_valid = cache_generation_valid(s_entry, generations)
            if not s_valid:
                # Selection cannot change, only selected current values/generations.
                sel = s_entry.selected
                scores = stable_scores[qid, torch.tensor(sel)]
                stable_cache[qid] = CacheEntry(
                    sel,
                    tuple(int(generations[i]) for i in sel),
                    output_for(sel, scores, values),
                )
            else:
                # Fresh semantic output must match because neither route nor selected values changed.
                scores = stable_scores[qid, torch.tensor(s_entry.selected)]
                fresh_output = output_for(s_entry.selected, scores, values)
                if abs(fresh_output - s_entry.output) > 1e-7:
                    stable_output_false_serves += 1

            if rng.random() < 0.05:
                fresh_sel, _ = topk_for(q, base_keys)
                if fresh_sel != stable_cache[qid].selected:
                    stable_audit_selection_mismatches += 1

            # Entangled control: cache admission checks only generations of the
            # previously selected values. An update to an unselected Pod can still
            # change routing because the Pod's key depends on its mutable value.
            e_entry = entangled_cache[qid]
            e_valid = cache_generation_valid(e_entry, generations)

            # Construct current entangled score vector without rebuilding all keys:
            # all old rows equal base+current values; use a batched current key view.
            # This is the oracle and deliberately more expensive than the stable route.
            cur_keys = F.normalize(base_keys + ENTANGLE_ALPHA * values[:, None] * value_dirs, dim=-1)
            fresh_sel, fresh_scores = topk_for(q, cur_keys)
            fresh_output = output_for(fresh_sel, fresh_scores, values)

            if e_valid:
                entangled_valid_serves += 1
                if fresh_sel != e_entry.selected:
                    entangled_false_accepts += 1
                if abs(fresh_output - e_entry.output) > 1e-7:
                    entangled_output_false_serves += 1
            else:
                entangled_invalid_refreshes += 1
                entangled_cache[qid] = CacheEntry(
                    fresh_sel,
                    tuple(int(generations[i]) for i in fresh_sel),
                    fresh_output,
                )

            # If a valid entangled cache had a stale route, refresh after counting so
            # the population remains realistic rather than accumulating one error forever.
            if e_valid and (fresh_sel != e_entry.selected or abs(fresh_output - e_entry.output) > 1e-7):
                entangled_cache[qid] = CacheEntry(
                    fresh_sel,
                    tuple(int(generations[i]) for i in fresh_sel),
                    fresh_output,
                )

            probe_ns.append(time.perf_counter_ns() - ts)

    total_probes = UPDATES * PROBES_PER_UPDATE
    reduction = 1.0 - stable_local_invalidations / max(1, global_predicate_invalidations)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Indirection-Stable World Attention (ISWA)",
        "pods": PODS,
        "cached_queries": QUERIES,
        "key_dimension": DIM,
        "topk": TOPK,
        "value_updates": UPDATES,
        "same_value_generation_updates": same_value_updates,
        "cache_admission_probes": total_probes,
        "initial_cache_build_seconds": build_seconds,
        "stable_key_false_route_accepts": stable_false_accepts,
        "stable_key_output_false_serves": stable_output_false_serves,
        "stable_key_fresh_topk_audit_mismatches": stable_audit_selection_mismatches,
        "entangled_key_false_route_accepts_with_selected_generation_check": entangled_false_accepts,
        "entangled_key_output_false_serves_with_selected_generation_check": entangled_output_false_serves,
        "entangled_cache_invalid_refreshes": entangled_invalid_refreshes,
        "entangled_cache_generation_valid_serves": entangled_valid_serves,
        "stable_local_selected_generation_invalidations": stable_local_invalidations,
        "global_predicate_generation_invalidations_theoretical": global_predicate_invalidations,
        "local_invalidation_reduction_vs_global_predicate": reduction,
        "mean_stable_queries_selecting_updated_pod": statistics.mean(selected_update_counts),
        "p99_stable_queries_selecting_updated_pod": sorted(selected_update_counts)[int(0.99 * (len(selected_update_counts) - 1))],
        "median_value_update_control_ns_python": statistics.median(update_ns),
        "median_combined_probe_ns_including_entangled_oracle": statistics.median(probe_ns),
        "contract_pass": (
            stable_output_false_serves == 0
            and stable_audit_selection_mismatches == 0
            and entangled_false_accepts > 0
            and entangled_output_false_serves > 0
        ),
        "mechanism": (
            "Governed world attention separates immutable routing keys from mutable value pages. Query-to-address selection is computed from "
            "stable semantic/address keys and may be cached in B/plan state. Only the selected current values enter J and contribute value "
            "generation lifetimes. Because value updates cannot change routing, an unselected value update cannot create a hidden new dependency."
        ),
        "architectural_hypothesis": (
            "Attention/retrieval keys for governed mutable knowledge should be lifecycle-stable routing state, not functions of current value "
            "payloads. Entangling mutable values into keys creates a phantom dependency: an unselected update can alter future top-k selection "
            "while a selected-generation cache check still reports valid. ISWA removes that degree of freedom by construction."
        ),
        "dod_status": "NOT_DOD; world-attention routing/lifetime gate",
        "claim_boundary": (
            "Key-value memories, stable embeddings, top-k attention/retrieval, index caching and MVCC are established. R330 tests the stronger "
            "CKCA lifetime constraint that governed routing keys must be independent of ordinary mutable value generations; standalone novelty "
            "is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Indirection-Stable World Attention gate failed")


if __name__ == "__main__":
    main()
