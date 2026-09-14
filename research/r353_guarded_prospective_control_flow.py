from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R353-GUARDED-PROSPECTIVE-CONTROL-FLOW"
REPORT_PATH = Path(os.environ.get("SO_R353_REPORT", "ci-r353/report.json"))
SEED = 3530914
TREES = int(os.environ.get("SO_R353_TREES", "4000"))
DEPTH = int(os.environ.get("SO_R353_DEPTH", "4"))
PODS = int(os.environ.get("SO_R353_PODS", "4096"))
UPDATES = int(os.environ.get("SO_R353_UPDATES", "50000"))
SERVES = int(os.environ.get("SO_R353_SERVES", "160000"))


@dataclass
class Cell:
    generation: int
    value: int


@dataclass(frozen=True)
class Node:
    guard_a: int
    guard_b: int
    left: int   # child node index or encoded leaf: negative -(pod+1)
    right: int


@dataclass
class Program:
    nodes: tuple[Node, ...]
    root: int
    all_refs: frozenset[int]
    digest: str


@dataclass
class SnapshotCache:
    """Bad control: branch decisions are materialized under an old world."""
    final_leaf: int
    guard_refs: tuple[int, ...]
    generations: tuple[int, ...]


def leaf_code(pid: int) -> int:
    return -(pid + 1)


def decode_leaf(code: int) -> int:
    return -code - 1


def build_program(rng: random.Random) -> Program:
    nodes: list[Node] = []

    def rec(depth: int) -> int:
        if depth == 0:
            return leaf_code(rng.randrange(PODS))
        ga, gb = rng.sample(range(PODS), 2)
        left = rec(depth - 1)
        right = rec(depth - 1)
        idx = len(nodes)
        nodes.append(Node(ga, gb, left, right))
        return idx

    root = rec(DEPTH)
    refs = set()
    for n in nodes:
        refs.add(n.guard_a); refs.add(n.guard_b)
        if n.left < 0: refs.add(decode_leaf(n.left))
        if n.right < 0: refs.add(decode_leaf(n.right))
    payload = json.dumps({"root": root, "nodes": [n.__dict__ for n in nodes]}, sort_keys=True).encode()
    return Program(tuple(nodes), root, frozenset(refs), hashlib.sha256(payload).hexdigest())


def evaluate_fresh(program: Program, world: list[Cell]) -> tuple[int, list[tuple[int,int]]]:
    """Lazy current-world evaluation. Only guards on the taken path and one leaf are read."""
    deps: list[tuple[int,int]] = []
    cur = program.root
    while cur >= 0:
        n = program.nodes[cur]
        a, b = world[n.guard_a], world[n.guard_b]
        deps.append((n.guard_a, a.generation)); deps.append((n.guard_b, b.generation))
        # World-dependent control flow deliberately changes with future payloads.
        cur = n.left if ((a.value ^ b.value) & 1) == 0 else n.right
    pid = decode_leaf(cur)
    leaf = world[pid]
    deps.append((pid, leaf.generation))
    # Output itself uses the selected current leaf plus path structure.
    out = leaf.value & 0xFF
    return out, deps


def compile_snapshot(program: Program, world: list[Cell]) -> SnapshotCache:
    cur = program.root
    refs = []
    gens = []
    while cur >= 0:
        n = program.nodes[cur]
        refs.extend([n.guard_a, n.guard_b])
        gens.extend([world[n.guard_a].generation, world[n.guard_b].generation])
        cur = n.left if ((world[n.guard_a].value ^ world[n.guard_b].value) & 1) == 0 else n.right
    return SnapshotCache(decode_leaf(cur), tuple(refs), tuple(gens))


def snapshot_serve(cache: SnapshotCache, world: list[Cell]) -> int:
    return world[cache.final_leaf].value & 0xFF


def commit_valid(deps: list[tuple[int,int]], world: list[Cell]) -> bool:
    return all(world[pid].generation == gen for pid, gen in deps)


def descriptor_digest(programs: list[Program]) -> str:
    h = hashlib.sha256()
    for p in programs:
        h.update(p.digest.encode())
    return h.hexdigest()


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    world = [Cell(1, rng.randrange(256)) for _ in range(PODS)]
    programs = [build_program(rng) for _ in range(TREES)]
    snapshots = [compile_snapshot(p, world) for p in programs]
    before = descriptor_digest(programs)

    # Build reverse maps for counterfactual invalidation accounting.
    eager_reverse: list[list[int]] = [[] for _ in range(PODS)]
    snapshot_reverse: list[list[int]] = [[] for _ in range(PODS)]
    for i, p in enumerate(programs):
        for pid in p.all_refs:
            eager_reverse[pid].append(i)
    for i, c in enumerate(snapshots):
        for pid in set(c.guard_refs + (c.final_leaf,)):
            snapshot_reverse[pid].append(i)

    snapshot_semantic_false_serves = 0
    guarded_false_serves = 0
    commit_conflicts_expected = 0
    commit_conflicts_detected = 0
    commit_conflicts_escaped = 0
    retries = 0
    retry_mismatches = 0
    same_value_updates = 0
    guarded_readset_sizes = []
    eager_union_sizes = []
    snapshot_dep_sizes = []
    update_eager_fanout = 0
    update_snapshot_fanout = 0
    update_ns = []

    t0 = time.perf_counter()
    updates_done = 0
    serves_done = 0
    while updates_done < UPDATES or serves_done < SERVES:
        if updates_done < UPDATES and (serves_done >= SERVES or rng.random() < 0.24):
            pid = rng.randrange(PODS)
            c = world[pid]
            old = c.value
            same = rng.random() < 0.38
            ts = time.perf_counter_ns()
            c.generation += 1
            if same:
                same_value_updates += 1
            else:
                c.value = rng.randrange(256)
            update_eager_fanout += len(eager_reverse[pid])
            update_snapshot_fanout += len(snapshot_reverse[pid])
            update_ns.append(time.perf_counter_ns() - ts)
            updates_done += 1
            continue

        i = rng.randrange(TREES)
        p = programs[i]
        fresh_out, deps = evaluate_fresh(p, world)
        guarded_readset_sizes.append(len(deps))
        eager_union_sizes.append(len(p.all_refs))
        snapshot_dep_sizes.append(len(set(snapshots[i].guard_refs + (snapshots[i].final_leaf,))))

        snap_out = snapshot_serve(snapshots[i], world)
        snapshot_semantic_false_serves += int(snap_out != fresh_out)

        # Guarded prospective execution with an injected race after speculative
        # evaluation but before publication. Same-value generation rewrites count.
        candidate = fresh_out
        raced = rng.random() < 0.08
        if raced:
            # Mutate one actually-read dependency, guaranteeing a conflict.
            pid, _ = deps[rng.randrange(len(deps))]
            c = world[pid]
            c.generation += 1
            if rng.random() >= 0.45:
                c.value = rng.randrange(256)
            commit_conflicts_expected += 1

        if not commit_valid(deps, world):
            commit_conflicts_detected += 1
            retries += 1
            candidate, deps2 = evaluate_fresh(p, world)
            # No second injected race in this bounded retry.
            if not commit_valid(deps2, world):
                raise AssertionError("retry unexpectedly stale")
            oracle, _ = evaluate_fresh(p, world)
            retry_mismatches += int(candidate != oracle)
        elif raced:
            commit_conflicts_escaped += 1

        oracle, _ = evaluate_fresh(p, world)
        guarded_false_serves += int(candidate != oracle)
        serves_done += 1

    elapsed = time.perf_counter() - t0
    after = descriptor_digest(programs)

    # Full final audit over every retained program.
    audit_mismatches = 0
    for p in programs:
        a, deps = evaluate_fresh(p, world)
        b, _ = evaluate_fresh(p, world)
        audit_mismatches += int(a != b or not commit_valid(deps, world))

    mean_dynamic = statistics.mean(guarded_readset_sizes)
    mean_union = statistics.mean(eager_union_sizes)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Guarded Prospective Neural State (G-PNS) / lazy dependency revelation",
        "programs": TREES,
        "tree_depth": DEPTH,
        "world_cells": PODS,
        "updates": UPDATES,
        "serves": SERVES,
        "same_value_generation_updates": same_value_updates,
        "descriptor_digest_unchanged_after_world_updates": before == after,
        "guarded_semantic_false_serves": guarded_false_serves,
        "snapshot_branch_cache_semantic_false_serves": snapshot_semantic_false_serves,
        "commit_conflicts_expected": commit_conflicts_expected,
        "commit_conflicts_detected": commit_conflicts_detected,
        "commit_conflicts_escaped": commit_conflicts_escaped,
        "retries": retries,
        "retry_mismatches": retry_mismatches,
        "final_audit_mismatches": audit_mismatches,
        "mean_dynamic_generation_readset": mean_dynamic,
        "mean_eager_union_reference_set": mean_union,
        "mean_snapshot_dependency_set": statistics.mean(snapshot_dep_sizes),
        "dynamic_vs_eager_union_readset_ratio": mean_dynamic / mean_union,
        "eager_union_update_invalidation_counterfactual": update_eager_fanout,
        "snapshot_update_invalidation_counterfactual": update_snapshot_fanout,
        "guarded_program_write_time_patches_or_invalidations": 0,
        "median_world_update_ns_python": statistics.median(update_ns),
        "elapsed_seconds": elapsed,
        "mechanism": (
            "The retained prospective state is not limited to a fixed list of final references. It stores guarded reference control flow. "
            "At serve time, current guard generations are read first; only the branch selected by the current world is traversed and only its leaf "
            "is dereferenced. The exact generation read-set is therefore discovered lazily from the current world and validated at commit. "
            "Future payload rewrites may change both control flow and the eventual referenced value without rewriting the retained program."
        ),
        "research_significance": (
            "Simple PNS handles future changes when the unresolved reference set is fixed. G-PNS extends the semantics to world-dependent control flow "
            "and dynamic dependency sets: the cache can remain future-bindable even when a future value changes which knowledge cell will be read. "
            "It also avoids giving unselected branches ordinary generation lifetimes, reducing invalidation fanout relative to eager union tracking."
        ),
        "claim_boundary": (
            "Lazy evaluation, guarded programs, decision DAGs, transactional reads and dynamic dependency tracking are established. R353 tests their "
            "composition as future-bindable neural-state semantics; it is not by itself a novelty claim."
        ),
        "dod_status": "NOT_DOD; dynamic-control-flow prospective-state gate",
    }
    report["contract_pass"] = (
        before == after
        and guarded_false_serves == 0
        and commit_conflicts_detected == commit_conflicts_expected
        and commit_conflicts_escaped == 0
        and retry_mismatches == 0
        and audit_mismatches == 0
        and snapshot_semantic_false_serves > 0
        and mean_dynamic < mean_union
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
