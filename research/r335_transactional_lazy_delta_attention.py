from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch

STAGE = "R335-TRANSACTIONAL-LAZY-DELTA-ATTENTION"
PODS = int(os.environ.get("SO_R335_PODS", "10000"))
CACHES = int(os.environ.get("SO_R335_CACHES", "100000"))
TOPK = int(os.environ.get("SO_R335_TOPK", "4"))
DV = int(os.environ.get("SO_R335_DV", "8"))
SCHEDULED_WRITES = int(os.environ.get("SO_R335_WRITES", "50000"))
READS = int(os.environ.get("SO_R335_READS", "200000"))
RACE_PROB = float(os.environ.get("SO_R335_RACE_PROB", "0.08"))
REPORT_PATH = Path(os.environ.get("SO_R335_REPORT", "r335_report.json"))
SEED = 3350914


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rng = random.Random(SEED)
    g = torch.Generator().manual_seed(SEED)
    dtype = torch.float64

    world_values = torch.randn(PODS, DV, generator=g, dtype=dtype)
    generations = torch.ones(PODS, dtype=torch.long)

    # Stable-K route was already established by R334. Here each retained attention
    # view stores only its selected stable addresses, weights, last seen generations,
    # last seen values and output.
    selected = torch.randint(0, PODS, (CACHES, TOPK), generator=g)
    # Make duplicate addresses within a cache extremely unlikely and repair them if present.
    for slot in range(1, TOPK):
        dup = torch.zeros(CACHES, dtype=torch.bool)
        for prior in range(slot):
            dup |= selected[:, slot].eq(selected[:, prior])
        while bool(dup.any()):
            selected[dup, slot] = torch.randint(0, PODS, (int(dup.sum()),), generator=g)
            dup = torch.zeros(CACHES, dtype=torch.bool)
            for prior in range(slot):
                dup |= selected[:, slot].eq(selected[:, prior])

    raw_w = torch.rand(CACHES, TOPK, generator=g, dtype=dtype)
    weights = raw_w / raw_w.sum(dim=1, keepdim=True)
    seen_gen = generations[selected].clone()
    seen_values = world_values[selected].clone()
    outputs = (weights.unsqueeze(-1) * seen_values).sum(dim=1)

    reverse_count = torch.bincount(selected.reshape(-1), minlength=PODS)

    lazy_slot_patches = 0
    lazy_reads_with_patch = 0
    coalesced_generation_jumps = 0
    same_value_writes = 0
    same_value_lifetime_refreshes = 0
    write_count = 0
    theoretical_eager_cache_patches = 0
    write_ns = []
    read_ns = []
    commit_conflicts_expected = 0
    commit_conflicts_detected = 0
    commit_conflicts_escaped = 0
    retries = 0
    naive_race_lifetime_false_serves = 0
    semantic_mismatches = 0
    lifetime_mismatches = 0

    def mutate(pid: int, same: bool) -> None:
        nonlocal same_value_writes, write_count, theoretical_eager_cache_patches
        old = world_values[pid].clone()
        generations[pid] += 1
        if same:
            same_value_writes += 1
            world_values[pid] = old
        else:
            world_values[pid] = torch.randn(DV, generator=g, dtype=dtype)
        write_count += 1
        theoretical_eager_cache_patches += int(reverse_count[pid])

    def catch_up(cache_id: int) -> int:
        nonlocal lazy_slot_patches, coalesced_generation_jumps, same_value_lifetime_refreshes
        pids = selected[cache_id]
        current_g = generations[pids]
        stale = seen_gen[cache_id].ne(current_g)
        if not bool(stale.any()):
            return 0
        slots = torch.where(stale)[0]
        for slot_t in slots:
            slot = int(slot_t)
            pid = int(pids[slot])
            old_g = int(seen_gen[cache_id, slot])
            new_g = int(generations[pid])
            if new_g - old_g > 1:
                coalesced_generation_jumps += new_g - old_g - 1
            old_v = seen_values[cache_id, slot].clone()
            new_v = world_values[pid]
            delta = new_v - old_v
            before = outputs[cache_id].clone()
            outputs[cache_id] += weights[cache_id, slot] * delta
            if torch.equal(old_v, new_v):
                same_value_lifetime_refreshes += 1
                if not torch.equal(before, outputs[cache_id]):
                    raise AssertionError("same-value lifetime refresh changed numeric output")
            seen_values[cache_id, slot] = new_v
            seen_gen[cache_id, slot] = new_g
            lazy_slot_patches += 1
        return len(slots)

    # Interleave O(1) world writes and accesses to retained attention views.
    writes_done = 0
    reads_done = 0
    t0 = time.perf_counter()
    while writes_done < SCHEDULED_WRITES or reads_done < READS:
        do_write = writes_done < SCHEDULED_WRITES and (reads_done >= READS or rng.random() < 0.20)
        if do_write:
            pid = rng.randrange(PODS)
            ts = time.perf_counter_ns()
            mutate(pid, rng.random() < 0.40)
            write_ns.append(time.perf_counter_ns() - ts)
            writes_done += 1
            continue

        cid = rng.randrange(CACHES)
        ts = time.perf_counter_ns()
        patched = catch_up(cid)
        lazy_reads_with_patch += int(patched > 0)

        # Neural Commit Barrier: inject an update after catch-up but before serve.
        race = rng.random() < RACE_PROB
        if race:
            pids = selected[cid]
            pid = int(pids[rng.randrange(TOPK)])
            mutate(pid, rng.random() < 0.45)
            commit_conflicts_expected += 1
            naive_race_lifetime_false_serves += 1

        pids = selected[cid]
        commit_ok = torch.equal(seen_gen[cid], generations[pids])
        if not commit_ok:
            commit_conflicts_detected += 1
            retries += 1
            catch_up(cid)
            # No second injected race on the retry in this bounded gate.
            commit_ok = torch.equal(seen_gen[cid], generations[pids])
        if not commit_ok:
            commit_conflicts_escaped += 1

        # Current-world oracle checks both semantic bytes and temporal identity.
        oracle = (weights[cid].unsqueeze(-1) * world_values[pids]).sum(dim=0)
        semantic_mismatches += int(float((oracle - outputs[cid]).abs().max()) > 1e-10)
        lifetime_mismatches += int(not torch.equal(seen_gen[cid], generations[pids]))
        read_ns.append(time.perf_counter_ns() - ts)
        reads_done += 1

    workload_seconds = time.perf_counter() - t0

    # Full final admission audit. Every retained view is lazily brought to current,
    # then compared against a vectorized full-current recomputation oracle.
    audit_patches_before = lazy_slot_patches
    audit_t0 = time.perf_counter()
    for cid in range(CACHES):
        catch_up(cid)
    final_oracle = (weights.unsqueeze(-1) * world_values[selected]).sum(dim=1)
    final_semantic_error = float((final_oracle - outputs).abs().max())
    final_lifetime_mismatches = int(
        (seen_gen != generations[selected]).any(dim=1).sum()
    )
    audit_seconds = time.perf_counter() - audit_t0
    final_audit_patches = lazy_slot_patches - audit_patches_before

    report = {
        "stage": STAGE,
        "architecture_candidate": "Transactional Lazy Delta Attention (TLDA)",
        "pods": PODS,
        "retained_attention_views": CACHES,
        "topk": TOPK,
        "value_dimension": DV,
        "scheduled_writes": SCHEDULED_WRITES,
        "race_writes": write_count - SCHEDULED_WRITES,
        "total_writes": write_count,
        "reads": READS,
        "same_value_generation_writes": same_value_writes,
        "lazy_slot_patches_during_workload_and_audit": lazy_slot_patches,
        "lazy_reads_with_patch": lazy_reads_with_patch,
        "coalesced_intermediate_generations_skipped": coalesced_generation_jumps,
        "same_value_lifetime_refreshes": same_value_lifetime_refreshes,
        "theoretical_eager_cache_patches": theoretical_eager_cache_patches,
        "lazy_patch_reduction_vs_eager": 1.0 - lazy_slot_patches / max(1, theoretical_eager_cache_patches),
        "commit_conflicts_expected": commit_conflicts_expected,
        "commit_conflicts_detected": commit_conflicts_detected,
        "commit_conflicts_escaped": commit_conflicts_escaped,
        "retries": retries,
        "naive_no_commit_barrier_lifetime_false_serves": naive_race_lifetime_false_serves,
        "semantic_mismatches_during_serves": semantic_mismatches,
        "lifetime_mismatches_during_serves": lifetime_mismatches,
        "final_full_recompute_semantic_max_error": final_semantic_error,
        "final_lifetime_mismatched_views": final_lifetime_mismatches,
        "final_audit_lazy_patches": final_audit_patches,
        "workload_seconds": workload_seconds,
        "final_audit_seconds": audit_seconds,
        "median_o1_world_write_ns_python": statistics.median(write_ns),
        "median_transactional_view_read_ns_python": statistics.median(read_ns),
        "mechanism": (
            "R334's stable-K attention makes the selected addresses and weights reusable across ordinary value changes. TLDA removes write fanout "
            "entirely: a world write advances only the canonical value generation. Each retained attention view keeps its last-seen selected "
            "generations/values and, on actual access, patches O += alpha_j*(V_current-V_seen) only for selected pages whose generations advanced. "
            "Multiple unseen generations coalesce into one exact patch. A Neural Commit Barrier rechecks the selected generations immediately "
            "before serve and retries the local patch if a race occurred."
        ),
        "novel_architecture_hypothesis": (
            "For stable-route world attention, lifecycle coherence can be made pull-based instead of invalidation-fanout-based: writes are O(1), "
            "cold cached neural views absorb no maintenance work, hot views catch up by exact value deltas, and generation identity still detects "
            "same-value ABA transitions and in-flight races. This combines incremental neural attention with MVCC-like temporal admission without "
            "making every world mutation proportional to the number of derived neural artifacts."
        ),
        "claim_boundary": (
            "Lazy materialized views, delta propagation, optimistic validation and MVCC are established individually. R335 tests their exact "
            "composition with generation-scoped stable-route neural attention. Novelty and production value require real-model/paged-attention "
            "integration and external prior-art audit."
        ),
        "dod_status": "NOT_DOD; transactional lazy attention-lifecycle gate",
    }
    report["contract_pass"] = (
        commit_conflicts_detected == commit_conflicts_expected
        and commit_conflicts_escaped == 0
        and semantic_mismatches == 0
        and lifetime_mismatches == 0
        and final_semantic_error < 1e-9
        and final_lifetime_mismatches == 0
        and naive_race_lifetime_false_serves > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Transactional Lazy Delta Attention contract failed")


if __name__ == "__main__":
    main()
