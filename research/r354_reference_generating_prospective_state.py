from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R354-REFERENCE-GENERATING-PROSPECTIVE-STATE"
REPORT_PATH = Path(os.environ.get("SO_R354_REPORT", "ci-r354/report.json"))
SEED = 3540914
PODS = int(os.environ.get("SO_R354_PODS", "8192"))
PROGRAMS = int(os.environ.get("SO_R354_PROGRAMS", "6000"))
HOPS = int(os.environ.get("SO_R354_HOPS", "6"))
UPDATES = int(os.environ.get("SO_R354_UPDATES", "60000"))
SERVES = int(os.environ.get("SO_R354_SERVES", "180000"))


@dataclass
class Cell:
    generation: int
    next_pod: int
    payload: int


@dataclass(frozen=True)
class Program:
    start: int
    hops: int
    digest: str


@dataclass(frozen=True)
class SnapshotChain:
    final_pod: int
    captured_chain: tuple[int, ...]
    generations: tuple[int, ...]


def make_program(start: int, hops: int) -> Program:
    raw = f"FOLLOW:{start}:{hops}".encode()
    return Program(start, hops, hashlib.sha256(raw).hexdigest())


def fresh_follow(p: Program, world: list[Cell]) -> tuple[int, list[tuple[int,int]]]:
    pid = p.start
    deps: list[tuple[int,int]] = []
    for _ in range(p.hops):
        c = world[pid]
        deps.append((pid, c.generation))
        pid = c.next_pod
    final = world[pid]
    deps.append((pid, final.generation))
    return final.payload, deps


def compile_snapshot(p: Program, world: list[Cell]) -> SnapshotChain:
    pid = p.start
    chain = []
    gens = []
    for _ in range(p.hops):
        c = world[pid]
        chain.append(pid); gens.append(c.generation)
        pid = c.next_pod
    chain.append(pid); gens.append(world[pid].generation)
    return SnapshotChain(pid, tuple(chain), tuple(gens))


def snapshot_serve(s: SnapshotChain, world: list[Cell]) -> int:
    return world[s.final_pod].payload


def valid(deps: list[tuple[int,int]], world: list[Cell]) -> bool:
    return all(world[pid].generation == g for pid, g in deps)


def descriptor_digest(programs: list[Program]) -> str:
    h = hashlib.sha256()
    for p in programs:
        h.update(p.digest.encode())
    return h.hexdigest()


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    world = [Cell(1, rng.randrange(PODS), rng.randrange(1_000_000)) for _ in range(PODS)]
    programs = [make_program(rng.randrange(PODS), HOPS) for _ in range(PROGRAMS)]
    snapshots = [compile_snapshot(p, world) for p in programs]
    before = descriptor_digest(programs)

    # Reverse map for the retrospective chain invalidation counterfactual.
    snapshot_reverse: list[list[int]] = [[] for _ in range(PODS)]
    for i, s in enumerate(snapshots):
        for pid in set(s.captured_chain):
            snapshot_reverse[pid].append(i)

    snapshot_false_serves = 0
    prospective_false_serves = 0
    expected = detected = escaped = retries = retry_mismatches = 0
    same_value_payload_updates = 0
    same_target_pointer_rewrites = 0
    pointer_updates = payload_updates = 0
    dynamic_readsets = []
    snapshot_dep_sizes = []
    snapshot_update_fanout = 0
    update_ns = []

    updates_done = serves_done = 0
    t0 = time.perf_counter()
    while updates_done < UPDATES or serves_done < SERVES:
        if updates_done < UPDATES and (serves_done >= SERVES or rng.random() < 0.25):
            pid = rng.randrange(PODS)
            c = world[pid]
            snapshot_update_fanout += len(snapshot_reverse[pid])
            ts = time.perf_counter_ns()
            c.generation += 1
            if rng.random() < 0.55:
                pointer_updates += 1
                if rng.random() < 0.30:
                    same_target_pointer_rewrites += 1
                else:
                    c.next_pod = rng.randrange(PODS)
            else:
                payload_updates += 1
                if rng.random() < 0.35:
                    same_value_payload_updates += 1
                else:
                    c.payload = rng.randrange(1_000_000)
            update_ns.append(time.perf_counter_ns() - ts)
            updates_done += 1
            continue

        i = rng.randrange(PROGRAMS)
        p = programs[i]
        oracle, deps = fresh_follow(p, world)
        dynamic_readsets.append(len(deps))
        snapshot_dep_sizes.append(len(set(snapshots[i].captured_chain)))
        snapshot_false_serves += int(snapshot_serve(snapshots[i], world) != oracle)

        candidate = oracle
        raced = rng.random() < 0.08
        if raced:
            # Change an actually traversed cell after the speculative path was read.
            pid, _ = deps[rng.randrange(len(deps))]
            c = world[pid]
            c.generation += 1
            if rng.random() < 0.50:
                c.next_pod = rng.randrange(PODS)
            elif rng.random() < 0.50:
                c.payload = rng.randrange(1_000_000)
            expected += 1

        if not valid(deps, world):
            detected += 1; retries += 1
            candidate, deps2 = fresh_follow(p, world)
            if not valid(deps2, world):
                raise AssertionError("unexpected retry race")
            oracle2, _ = fresh_follow(p, world)
            retry_mismatches += int(candidate != oracle2)
        elif raced:
            escaped += 1

        current, _ = fresh_follow(p, world)
        prospective_false_serves += int(candidate != current)
        serves_done += 1

    elapsed = time.perf_counter() - t0
    after = descriptor_digest(programs)

    final_audit = 0
    for p in programs:
        x, deps = fresh_follow(p, world)
        y, _ = fresh_follow(p, world)
        final_audit += int(x != y or not valid(deps, world))

    report = {
        "stage": STAGE,
        "architecture_candidate": "Reference-Generating Prospective Neural State (RG-PNS)",
        "world_cells": PODS,
        "programs": PROGRAMS,
        "pointer_hops": HOPS,
        "updates": UPDATES,
        "serves": SERVES,
        "pointer_updates": pointer_updates,
        "payload_updates": payload_updates,
        "same_target_pointer_generation_rewrites": same_target_pointer_rewrites,
        "same_value_payload_generation_rewrites": same_value_payload_updates,
        "descriptor_digest_unchanged_after_world_updates": before == after,
        "prospective_false_serves": prospective_false_serves,
        "snapshot_chain_false_serves": snapshot_false_serves,
        "generation_conflicts_expected": expected,
        "generation_conflicts_detected": detected,
        "generation_conflicts_escaped": escaped,
        "retries": retries,
        "retry_mismatches": retry_mismatches,
        "final_audit_mismatches": final_audit,
        "mean_dynamic_readset": statistics.mean(dynamic_readsets),
        "mean_snapshot_dependency_set": statistics.mean(snapshot_dep_sizes),
        "snapshot_update_invalidation_counterfactual": snapshot_update_fanout,
        "prospective_program_write_time_patches_or_invalidations": 0,
        "median_update_ns_python": statistics.median(update_ns),
        "elapsed_seconds": elapsed,
        "mechanism": (
            "The retained state stores a reference-generating operation FOLLOW(start,hops), not the final reference produced under the old world. "
            "Each dereference reads the current cell, whose value can itself produce the next canonical reference. The final knowledge address is "
            "therefore generated lazily by the current authoritative world. Every traversed generation enters the dynamic read-set and is checked at commit."
        ),
        "research_significance": (
            "This extends PNS beyond fixed references and guarded choices to mutable graph topology / pointer chasing. A future world revision can change "
            "the identity of the next knowledge cell at every hop while the retained program stays byte-identical. Snapshotting a historical reference "
            "chain is observably wrong under topology changes, whereas reference-generating PNS rematerializes the current chain on demand."
        ),
        "claim_boundary": (
            "Pointer chasing, neural random-access machines, graph traversal and transactional read-sets are established. R354 is an architecture-completeness "
            "gate for future-bindable state, not a standalone novelty claim."
        ),
        "dod_status": "NOT_DOD; dynamic-reference-generation prospective-state gate",
    }
    report["contract_pass"] = (
        before == after and prospective_false_serves == 0 and final_audit == 0
        and detected == expected and escaped == 0 and retry_mismatches == 0
        and snapshot_false_serves > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
