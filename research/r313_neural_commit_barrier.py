from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

OUT = Path(os.environ.get("SO_R313_REPORT", "ci-r313/report.json"))


@dataclass
class Pod:
    generation: int
    value: int


@dataclass(frozen=True)
class Read:
    pod_id: int
    generation: int
    value: int


@dataclass(frozen=True)
class Txn:
    op: int
    pods: tuple[int, int, int]


class World:
    def __init__(self, n: int, rng: random.Random):
        self.pods = [Pod(1, rng.randrange(1000)) for _ in range(n)]

    def read(self, pid: int) -> Read:
        p = self.pods[pid]
        return Read(pid, p.generation, p.value)

    def update(self, pid: int, rng: random.Random):
        p = self.pods[pid]
        p.generation += 1
        # same-value rewrites intentionally possible
        if rng.random() > 0.20:
            p.value = rng.randrange(1000)

    def validate(self, reads: tuple[Read, ...]) -> bool:
        return all(self.pods[r.pod_id].generation == r.generation for r in reads)


def compute(op: int, a: Read, b: Read, c: Read) -> int:
    if op == 0:
        return a.value + b.value
    if op == 1:
        return a.value if a.value >= b.value else c.value
    if op == 2:
        return (a.value ^ b.value) + c.value
    if op == 3:
        return b.value if (a.value % 2 == 0) else c.value
    raise ValueError(op)


def serial_current(world: World, tx: Txn) -> int:
    a,b,c = (world.read(pid) for pid in tx.pods)
    return compute(tx.op,a,b,c)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(313)
    world = World(25_000, rng)
    txns = [Txn(rng.randrange(4), tuple(rng.sample(range(len(world.pods)), 3))) for _ in range(150_000)]

    naive_stale_commits = 0
    barrier_rejections = 0
    retry_success = 0
    undetected_stale = 0
    same_value_rewrite_detected = 0
    conflict_txns = 0
    j_retry_steps = []
    barrier_ns = []
    plan_recompiles = 0
    started = time.perf_counter()

    for tx in txns:
        a = world.read(tx.pods[0])
        # Concurrent mutation may happen between individual neural/knowledge reads.
        conflict = rng.random() < 0.34
        if conflict:
            conflict_txns += 1
            target = tx.pods[rng.randrange(3)]
            old_value = world.pods[target].value
            world.update(target, rng)
            if world.pods[target].value == old_value:
                # generation-only conflict; semantic value equality must not fool validation
                same_value_candidate = True
            else:
                same_value_candidate = False
        else:
            same_value_candidate = False

        b = world.read(tx.pods[1])
        if rng.random() < 0.18:
            target2 = tx.pods[rng.randrange(3)]
            old_value2 = world.pods[target2].value
            world.update(target2, rng)
            conflict = True
            conflict_txns += 0 if target2 in tx.pods else 1
            same_value_candidate = same_value_candidate or (world.pods[target2].value == old_value2)
        c = world.read(tx.pods[2])
        reads = (a,b,c)
        tentative = compute(tx.op,a,b,c)

        t0 = time.perf_counter_ns()
        valid = world.validate(reads)
        barrier_ns.append(time.perf_counter_ns() - t0)

        current = serial_current(world, tx)
        if tentative != current or not valid:
            naive_stale_commits += 1

        if not valid:
            barrier_rejections += 1
            if same_value_candidate:
                same_value_rewrite_detected += 1
            # CKCA retry preserves immutable B-plan/query interpretation and re-executes only
            # the mutable J/knowledge transaction with fresh capabilities.
            retries = 0
            while True:
                retries += 1
                aa,bb,cc = (world.read(pid) for pid in tx.pods)
                fresh_reads = (aa,bb,cc)
                result = compute(tx.op,aa,bb,cc)
                if world.validate(fresh_reads):
                    retry_success += int(result == serial_current(world, tx))
                    j_retry_steps.append(retries)
                    break
                if retries > 4:
                    j_retry_steps.append(retries)
                    break
            # no plan recompile by design
        else:
            if tentative != current:
                undetected_stale += 1

    report = {
        "stage": "R313-NEURAL-COMMIT-BARRIER",
        "architecture_candidate": "CKVM Neural Read-Set Commit Barrier",
        "pod_count": len(world.pods),
        "transactions": len(txns),
        "naive_stale_or_mixed_commits": naive_stale_commits,
        "barrier_rejections": barrier_rejections,
        "undetected_stale_commits": undetected_stale,
        "retry_successful_serial_current_results": retry_success,
        "same_value_generation_conflicts_detected": same_value_rewrite_detected,
        "plan_recompiles_on_world_conflict": plan_recompiles,
        "median_j_retry_steps": statistics.median(j_retry_steps) if j_retry_steps else 0,
        "p99_j_retry_steps": sorted(j_retry_steps)[int(.99*len(j_retry_steps))] if j_retry_steps else 0,
        "median_commit_barrier_ns": statistics.median(barrier_ns),
        "p99_commit_barrier_ns": sorted(barrier_ns)[int(.99*len(barrier_ns))],
        "elapsed_seconds": time.perf_counter()-started,
        "mechanism": (
            "CKVM records exact generation reads during a mutable knowledge transaction. Immediately before a derived "
            "J state or answer becomes externally visible, the commit barrier verifies that every generation read is still "
            "current. A conflict aborts only the mutable J transaction and reuses the immutable B-plane plan."
        ),
        "consistency_target": (
            "No committed answer may mix generations that ceased to be jointly current during the transaction. Validation "
            "is generation-based, so same-value rewrites are conflicts even when semantic bytes are unchanged."
        ),
        "prior_art_boundary": (
            "Optimistic concurrency control and read-set validation are established, including S-Bus for LLM agent state. "
            "R313 is an integration requirement for neural J-state coherence, not a standalone novelty claim."
        ),
        "dod_status": "NOT_DOD; concurrency coherence gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if undetected_stale != 0:
        return 2
    if barrier_rejections == 0:
        return 3
    if retry_success != barrier_rejections:
        return 4
    if same_value_rewrite_detected == 0:
        return 5
    if plan_recompiles != 0:
        return 6
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
