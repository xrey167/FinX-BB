from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from research.ckca_v3_world_abi import (
    AuthorityStore,
    LifetimeFabric,
    LifetimeViolation,
    Plan,
    PlanCache,
    Sealed,
    WorldABI,
)

STAGE = "R326-CAUSAL-LIVE-SEMANTIC-VIEWS"
PODS = int(os.environ.get("SO_R326_PODS", "10000"))
VIEWS = int(os.environ.get("SO_R326_VIEWS", "50000"))
UPDATES = int(os.environ.get("SO_R326_UPDATES", "12000"))
READS = int(os.environ.get("SO_R326_READS", "180000"))
REPORT_PATH = Path(os.environ.get("SO_R326_REPORT", "r326_report.json"))
SEED = 3260914
OPS = ("add8", "xor8", "max", "select_even")


@dataclass
class LiveView:
    plan: Plan
    sealed: Sealed[int] | None = None
    recomputes: int = 0
    cache_hits: int = 0

    def _compute(self, abi: WorldABI) -> int:
        tx = abi.transaction(self.plan)
        borrowed = tx.execute()
        tx.commit()
        self.sealed = tx.seal(borrowed)
        self.recomputes += 1
        return borrowed.value

    def read(self, abi: WorldABI) -> int:
        if self.sealed is None:
            return self._compute(abi)
        try:
            value = abi.serve(self.sealed)
            self.cache_hits += 1
            return value
        except LifetimeViolation:
            return self._compute(abi)


def oracle(authority: AuthorityStore, plan: Plan) -> int:
    a = authority.pods[plan.left_pod]
    b = authority.pods[plan.right_pod]
    if not a.live or not b.live:
        # This experiment uses updates only, not revocation, to isolate lazy view
        # invalidation and same-value generation semantics.
        raise RuntimeError("revoked source in R326 oracle")
    if plan.opcode == "add8":
        return (a.value + b.value) & 0xFF
    if plan.opcode == "xor8":
        return (a.value ^ b.value) & 0xFF
    if plan.opcode == "max":
        return max(a.value, b.value)
    if plan.opcode == "select_even":
        return a.value if (a.value & 1) == 0 else b.value
    raise ValueError(plan.opcode)


def main() -> None:
    rng = random.Random(SEED)
    authority = AuthorityStore()
    lifetimes = LifetimeFabric()
    abi = WorldABI(authority, lifetimes)
    plans = PlanCache()

    for pid in range(PODS):
        authority.create(pid, rng.randrange(256))
        authority.bind(f"pod-{pid}", pid)

    # Build a persistent transcript-like population of semantic live views.
    views: list[LiveView] = []
    reverse_view_count = [0] * PODS
    for i in range(VIEWS):
        left = rng.randrange(PODS)
        right = rng.randrange(PODS)
        op = OPS[i % len(OPS)]
        p = plans.get_or_compile(op, left, right, authority)
        views.append(LiveView(p))
        reverse_view_count[left] += 1
        reverse_view_count[right] += 1

    # Warm a subset, as a long-running conversation/server would.
    warm_count = VIEWS // 2
    for i in range(warm_count):
        value = views[i].read(abi)
        if value != oracle(authority, views[i].plan):
            raise AssertionError("warm-up mismatch")

    same_value_updates = 0
    invalidation_nodes_touched = []
    theoretical_eager_view_rewrites = []
    eager_message_rewrites_performed = 0
    read_mismatches = 0
    stale_cache_reuse_escapes = 0
    read_ns = []
    update_ns = []
    recompute_before = sum(v.recomputes for v in views)
    hits_before = sum(v.cache_hits for v in views)

    # Interleave world mutations and view reads. Update performs no transcript/view
    # rewrite: it only advances the canonical generation and invalidates the leaf.
    updates_done = 0
    reads_done = 0
    t0 = time.perf_counter()
    while updates_done < UPDATES or reads_done < READS:
        do_update = updates_done < UPDATES and (reads_done >= READS or rng.random() < 0.075)
        if do_update:
            pid = rng.randrange(PODS)
            p = authority.pods[pid]
            old_value = p.value
            same = rng.random() < 0.40
            new_value = old_value if same else rng.randrange(256)
            if same:
                same_value_updates += 1
            old_dep = authority.update(pid, new_value)
            ts = time.perf_counter_ns()
            touched = abi.expire_generation(old_dep)
            update_ns.append(time.perf_counter_ns() - ts)
            invalidation_nodes_touched.append(touched)
            theoretical_eager_view_rewrites.append(reverse_view_count[pid])
            updates_done += 1
            continue

        idx = rng.randrange(VIEWS)
        v = views[idx]
        ts = time.perf_counter_ns()
        got = v.read(abi)
        read_ns.append(time.perf_counter_ns() - ts)
        expected = oracle(authority, v.plan)
        if got != expected:
            read_mismatches += 1
        # If a stale factor ever survives, abi.serve would have returned the old
        # value and oracle comparison would catch it. Keep a separate named metric.
        if got != expected and v.sealed is not None:
            stale_cache_reuse_escapes += 1
        reads_done += 1

    elapsed = time.perf_counter() - t0
    recompute_after = sum(v.recomputes for v in views)
    hits_after = sum(v.cache_hits for v in views)
    lazy_recomputes_during_workload = recompute_after - recompute_before
    cache_hits_during_workload = hits_after - hits_before

    # Full audit: every live view is read once and must converge to the fresh oracle.
    full_audit_mismatches = 0
    audit_recomputes_before = sum(v.recomputes for v in views)
    for v in views:
        if v.read(abi) != oracle(authority, v.plan):
            full_audit_mismatches += 1
    audit_recomputes = sum(v.recomputes for v in views) - audit_recomputes_before

    eager_total = sum(theoretical_eager_view_rewrites)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Live Semantic Views (CLSV) over CKCA World ABI",
        "pods": PODS,
        "views": VIEWS,
        "updates": UPDATES,
        "reads": READS,
        "same_value_generation_updates": same_value_updates,
        "initially_warmed_views": warm_count,
        "read_mismatches": read_mismatches,
        "stale_cache_reuse_escapes": stale_cache_reuse_escapes,
        "full_audit_mismatches": full_audit_mismatches,
        "lazy_recomputes_during_workload": lazy_recomputes_during_workload,
        "cache_hits_during_workload": cache_hits_during_workload,
        "lazy_recompute_fraction_of_reads": lazy_recomputes_during_workload / READS,
        "audit_recomputes_needed_to_make_all_views_current": audit_recomputes,
        "theoretical_eager_view_rewrites_for_updates": eager_total,
        "eager_message_rewrites_performed": eager_message_rewrites_performed,
        "deferred_work_fraction_vs_eager_during_workload": (
            1.0 - lazy_recomputes_during_workload / eager_total if eager_total else 0.0
        ),
        "mean_invalidated_factor_nodes_per_update": statistics.mean(invalidation_nodes_touched),
        "p99_invalidated_factor_nodes_per_update": sorted(invalidation_nodes_touched)[int(0.99 * (len(invalidation_nodes_touched) - 1))],
        "median_generation_invalidation_ns_python": statistics.median(update_ns),
        "median_live_view_read_ns_python": statistics.median(read_ns),
        "elapsed_workload_seconds": elapsed,
        "plan_recompiles_due_to_value_updates": plans.compiles - len({
            (v.plan.opcode, v.plan.left_pod, v.plan.right_pod, v.plan.schema_generation, v.plan.alias_generation)
            for v in views
        }),
        "contract_pass": read_mismatches == 0 and stale_cache_reuse_escapes == 0 and full_audit_mismatches == 0,
        "mechanism": (
            "A persisted answer/transcript may store a live semantic expression (plan + sealed cached result) rather than a flat factual "
            "string. World updates invalidate only generation lifetime factors; no message is eagerly rewritten. On the next actual access, "
            "a dead view lazily re-executes its immutable B plan against current capabilities, commits, reseals, and then renders."
        ),
        "architectural_hypothesis": (
            "A long-lived AI transcript should behave like a generation-aware live materialized view: language structure is persistent, "
            "governed factual content is revalidated/recomputed on demand. This removes stale transcript resurrection without turning every "
            "world update into a global conversation rewrite."
        ),
        "dod_status": "NOT_DOD; lazy semantic-view mechanism gate",
        "claim_boundary": (
            "Reactive dataflow, materialized views, memoization and lazy invalidation are established. R326 tests their composition with "
            "generation-scoped neural knowledge lifetimes and semantic transcripts; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("causal live semantic-view contract failed")


if __name__ == "__main__":
    main()
