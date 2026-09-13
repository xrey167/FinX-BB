from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

OUT = Path(os.environ.get("SO_R297_REPORT", "ci-r297/report.json"))


@dataclass
class StateRecord:
    factor_id: int
    # Audit-only explicit dependency set. Production hot path does not inspect this.
    deps: tuple[tuple[int, int], ...]


class LifetimeFactorDAG:
    """Exact conjunctive lifetime algebra with O(1) read validation.

    Each Pod generation is an immutable leaf. A derived neural/J-Space state owns one
    factor node whose validity is the AND of its parents. Expiring a leaf propagates
    invalidity through reverse edges. A new Pod generation creates a fresh leaf, so an
    old factor can never become valid again (ABA-safe monotonic lifetime semantics).
    """

    def __init__(self, pod_count: int):
        self.valid: list[bool] = []
        self.children: list[list[int]] = []
        self.pod_generation = [1] * pod_count
        self.pod_live = [True] * pod_count
        self.pod_leaf = [-1] * pod_count
        for p in range(pod_count):
            self.pod_leaf[p] = self._new_node((), True)

    def _new_node(self, parents: tuple[int, ...], valid: bool | None = None) -> int:
        node = len(self.valid)
        if valid is None:
            valid = all(self.valid[x] for x in parents)
        self.valid.append(bool(valid))
        self.children.append([])
        for parent in parents:
            self.children[parent].append(node)
        return node

    def factor_from_pods(self, pods: tuple[int, ...]) -> tuple[int, tuple[tuple[int, int], ...]]:
        parents = tuple(self.pod_leaf[p] for p in pods)
        deps = tuple(sorted((p, self.pod_generation[p]) for p in pods))
        return self._new_node(parents), deps

    def compose(self, states: tuple[StateRecord, ...]) -> StateRecord:
        parents = tuple(s.factor_id for s in states)
        deps = tuple(sorted(set(d for s in states for d in s.deps)))
        return StateRecord(self._new_node(parents), deps)

    def invalidate_node(self, root: int) -> int:
        if not self.valid[root]:
            return 0
        stack = [root]
        touched = 0
        while stack:
            node = stack.pop()
            if not self.valid[node]:
                continue
            self.valid[node] = False
            touched += 1
            stack.extend(self.children[node])
        return touched

    def update_pod(self, pod: int) -> int:
        old_leaf = self.pod_leaf[pod]
        touched = self.invalidate_node(old_leaf)
        self.pod_generation[pod] += 1
        self.pod_live[pod] = True
        self.pod_leaf[pod] = self._new_node((), True)
        return touched

    def revoke_pod(self, pod: int) -> int:
        old_leaf = self.pod_leaf[pod]
        touched = self.invalidate_node(old_leaf)
        self.pod_generation[pod] += 1
        self.pod_live[pod] = False
        self.pod_leaf[pod] = -1
        return touched

    def revive_pod(self, pod: int) -> None:
        self.pod_generation[pod] += 1
        self.pod_live[pod] = True
        self.pod_leaf[pod] = self._new_node((), True)

    def factor_valid(self, factor_id: int) -> bool:
        return self.valid[factor_id]

    def naive_valid(self, deps: tuple[tuple[int, int], ...]) -> bool:
        return all(
            self.pod_live[p] and self.pod_generation[p] == generation
            for p, generation in deps
        )


def percentile(xs: list[int | float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return float(ys[min(len(ys) - 1, int((len(ys) - 1) * q))])


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seed = 297
    rng = random.Random(seed)

    pod_count = 50_000
    direct_states_n = 220_000
    composed_states_n = 60_000
    dag = LifetimeFactorDAG(pod_count)

    build_t0 = time.perf_counter()
    states: list[StateRecord] = []
    dependency_hist: dict[int, int] = {}

    for _ in range(direct_states_n):
        k = rng.choices((1, 2, 3, 4, 6, 8), weights=(18, 28, 24, 16, 9, 5), k=1)[0]
        pods = tuple(sorted(rng.sample(range(pod_count), k)))
        factor, deps = dag.factor_from_pods(pods)
        states.append(StateRecord(factor, deps))
        dependency_hist[k] = dependency_hist.get(k, 0) + 1

    # Build higher-order J-Space states from existing states. The factor graph shares
    # lifetime structure; audit deps are flattened only for the correctness oracle.
    for _ in range(composed_states_n):
        a = states[rng.randrange(len(states))]
        b = states[rng.randrange(len(states))]
        states.append(dag.compose((a, b)))

    build_seconds = time.perf_counter() - build_t0
    nodes_before_updates = len(dag.valid)
    edges = sum(len(x) for x in dag.children)

    # Hot-path validity benchmark before invalidations. Factor check is one bool read;
    # oracle scans every dependency generation.
    query_indices = [rng.randrange(len(states)) for _ in range(250_000)]
    t0 = time.perf_counter_ns()
    factor_true = 0
    for i in query_indices:
        factor_true += int(dag.factor_valid(states[i].factor_id))
    factor_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    naive_true = 0
    for i in query_indices:
        naive_true += int(dag.naive_valid(states[i].deps))
    naive_ns = time.perf_counter_ns() - t0
    if factor_true != naive_true:
        raise RuntimeError("pre-update factor/oracle mismatch")

    # Perform real lifecycle changes. Reverse propagation pays only when a source dies;
    # inference subsequently pays one validity bit read.
    update_touched = []
    update_ns = []
    updated_pods = []
    for _ in range(750):
        pod = rng.randrange(pod_count)
        updated_pods.append(pod)
        t0 = time.perf_counter_ns()
        touched = dag.update_pod(pod)
        update_ns.append(time.perf_counter_ns() - t0)
        update_touched.append(touched)

        # Generate a small amount of new work under the new generation to ensure that
        # the graph remains useful after repeated invalidation.
        for _new in range(3):
            other = rng.randrange(pod_count)
            if other == pod or dag.pod_leaf[other] < 0:
                continue
            factor, deps = dag.factor_from_pods(tuple(sorted((pod, other))))
            states.append(StateRecord(factor, deps))

    # Exactness audit after many generations.
    audit_indices = rng.sample(range(len(states)), 50_000)
    mismatches = 0
    for i in audit_indices:
        s = states[i]
        if dag.factor_valid(s.factor_id) != dag.naive_valid(s.deps):
            mismatches += 1

    # ABA / same-value-return style test: old capability stays dead after two updates;
    # a newly derived state is live. Generation identity, not semantic value equality,
    # controls lifetime.
    aba_pod = 1234
    old_factor, old_deps = dag.factor_from_pods((aba_pod,))
    dag.update_pod(aba_pod)
    middle_factor, middle_deps = dag.factor_from_pods((aba_pod,))
    dag.update_pod(aba_pod)
    newest_factor, newest_deps = dag.factor_from_pods((aba_pod,))
    aba_ok = (
        not dag.factor_valid(old_factor)
        and not dag.factor_valid(middle_factor)
        and dag.factor_valid(newest_factor)
        and not dag.naive_valid(old_deps)
        and not dag.naive_valid(middle_deps)
        and dag.naive_valid(newest_deps)
    )

    # Revocation / revive test: no prior factor may resurrect after the Pod becomes live
    # again with a fresh generation.
    rev_pod = 4321
    pre_rev_factor, pre_rev_deps = dag.factor_from_pods((rev_pod,))
    dag.revoke_pod(rev_pod)
    revoked_old_dead = not dag.factor_valid(pre_rev_factor) and not dag.naive_valid(pre_rev_deps)
    dag.revive_pod(rev_pod)
    post_rev_factor, post_rev_deps = dag.factor_from_pods((rev_pod,))
    revoke_revival_ok = (
        revoked_old_dead
        and not dag.factor_valid(pre_rev_factor)
        and dag.factor_valid(post_rev_factor)
        and dag.naive_valid(post_rev_deps)
    )

    # Locality: compare invalidation touch counts to the entire factor universe.
    touched_fraction = [x / max(nodes_before_updates, 1) for x in update_touched]
    report = {
        "stage": "R297-CAUSAL-LIFETIME-FACTOR-DAG",
        "architecture_candidate": "Causal Lifetime Factor DAG (CLFD)",
        "seed": seed,
        "pod_count": pod_count,
        "direct_states_initial": direct_states_n,
        "composed_states_initial": composed_states_n,
        "states_after_updates": len(states),
        "factor_nodes_before_updates": nodes_before_updates,
        "factor_edges_before_updates": edges,
        "dependency_hist_direct": dependency_hist,
        "build_seconds": build_seconds,
        "validity_queries": len(query_indices),
        "factor_validation_total_ns": factor_ns,
        "naive_validation_total_ns": naive_ns,
        "factor_validation_ns_per_query": factor_ns / len(query_indices),
        "naive_validation_ns_per_query": naive_ns / len(query_indices),
        "validation_speedup_vs_naive": naive_ns / factor_ns,
        "lifecycle_updates": len(update_touched),
        "median_update_ns": statistics.median(update_ns),
        "p99_update_ns": percentile(update_ns, 0.99),
        "mean_invalidated_nodes_per_update": statistics.mean(update_touched),
        "median_invalidated_nodes_per_update": statistics.median(update_touched),
        "p99_invalidated_nodes_per_update": percentile(update_touched, 0.99),
        "mean_fraction_factor_graph_touched_per_update": statistics.mean(touched_fraction),
        "p99_fraction_factor_graph_touched_per_update": percentile(touched_fraction, 0.99),
        "post_update_exactness_audit_cases": len(audit_indices),
        "post_update_factor_vs_explicit_dependency_mismatches": mismatches,
        "aba_generation_test_pass": aba_ok,
        "revoke_then_revive_no_resurrection_pass": revoke_revival_ok,
        "hot_path_complexity": "O(1) validity bit read per J/cache state",
        "update_complexity": "O(number of causally dependent factor nodes actually invalidated)",
        "mechanism": (
            "Every Pod generation is an immutable lifetime leaf. Derived J-Space/cache states reference "
            "conjunctive factor nodes. Reverse edges propagate invalidity once when a source generation "
            "expires, converting repeated read-time dependency scans into one O(1) validity-bit check. "
            "New generations allocate new leaves, so old dependent state is monotonic-dead and cannot "
            "resurrect under ABA, same-value rewrites, revocation or later revival."
        ),
        "claim_boundary": (
            "This is an exact runtime lifetime-data-structure experiment, not a neural-quality result and "
            "not a standalone novelty claim. It is intended to make transitive lifetime closure compatible "
            "with the <5% hot-path overhead target."
        ),
        "dod_status": "NOT_DOD; lifetime-factor scalability gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if mismatches != 0:
        return 2
    if not aba_ok or not revoke_revival_ok:
        return 3
    if report["validation_speedup_vs_naive"] <= 1.0:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
