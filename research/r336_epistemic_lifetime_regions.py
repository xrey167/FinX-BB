from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

STAGE = "R336-EPISTEMIC-LIFETIME-REGIONS"
PODS = int(os.environ.get("SO_R336_PODS", "50000"))
TRANSACTIONS = int(os.environ.get("SO_R336_TRANSACTIONS", "120000"))
OPS_PER_TX = int(os.environ.get("SO_R336_OPS_PER_TX", "48"))
MAX_READS = int(os.environ.get("SO_R336_MAX_READS", "8"))
RACE_PROB = float(os.environ.get("SO_R336_RACE_PROB", "0.10"))
REPORT_PATH = Path(os.environ.get("SO_R336_REPORT", "r336_report.json"))
SEED = 3360914


@dataclass
class Pod:
    generation: int = 1
    value: int = 0


@dataclass
class Region:
    region_id: int
    readset: dict[int, int] = field(default_factory=dict)
    committed: bool = False


@dataclass(frozen=True)
class Borrowed:
    value: int
    region_id: int


@dataclass(frozen=True)
class Sealed:
    value: int
    factor_id: int


class RegionRuntime:
    """Borrowed tensors carry one region ID, not a copied generation set.

    The region owns the exact dynamic read-set. Arbitrary J operations inside the
    region propagate only the region ID. Commit validates the region read-set once;
    retained artifacts then carry one sealed factor ID.
    """

    def __init__(self, pods: list[Pod]):
        self.pods = pods
        self.next_region = 1
        self.next_factor = 1
        self.factor_live: dict[int, bool] = {}
        self.factor_deps: dict[int, tuple[tuple[int, int], ...]] = {}
        self.reverse: dict[tuple[int, int], list[int]] = {}

    def begin(self) -> Region:
        r = Region(self.next_region)
        self.next_region += 1
        return r

    def read(self, r: Region, pid: int) -> Borrowed:
        p = self.pods[pid]
        r.readset[pid] = p.generation
        return Borrowed(p.value, r.region_id)

    @staticmethod
    def op(r: Region, a: Borrowed, b: Borrowed, opcode: int) -> Borrowed:
        if a.region_id != r.region_id or b.region_id != r.region_id:
            raise RuntimeError("cross-region borrow forbidden")
        if opcode == 0:
            v = (a.value + b.value) & 0xFFFFFFFF
        elif opcode == 1:
            v = a.value ^ b.value
        elif opcode == 2:
            v = (a.value * 1664525 + b.value * 1013904223) & 0xFFFFFFFF
        else:
            v = max(a.value, b.value)
        return Borrowed(v, r.region_id)

    def commit(self, r: Region) -> bool:
        for pid, generation in r.readset.items():
            if self.pods[pid].generation != generation:
                return False
        r.committed = True
        return True

    def seal(self, r: Region, x: Borrowed) -> Sealed:
        if not r.committed or x.region_id != r.region_id:
            raise RuntimeError("uncommitted/cross-region escape")
        fid = self.next_factor
        self.next_factor += 1
        deps = tuple(sorted(r.readset.items()))
        self.factor_live[fid] = True
        self.factor_deps[fid] = deps
        for dep in deps:
            self.reverse.setdefault(dep, []).append(fid)
        return Sealed(x.value, fid)

    def invalidate(self, pid: int, old_generation: int) -> int:
        touched = 0
        for fid in self.reverse.get((pid, old_generation), ()):
            if self.factor_live.get(fid, False):
                self.factor_live[fid] = False
                touched += 1
        return touched

    def serve(self, x: Sealed) -> bool:
        return self.factor_live.get(x.factor_id, False)


class NaiveLineage:
    """Control: every intermediate value materializes its full generation set."""

    @staticmethod
    def read(pid: int, pod: Pod) -> tuple[int, frozenset[tuple[int, int]]]:
        return pod.value, frozenset([(pid, pod.generation)])

    @staticmethod
    def op(a, b, opcode: int):
        av, adeps = a
        bv, bdeps = b
        if opcode == 0:
            v = (av + bv) & 0xFFFFFFFF
        elif opcode == 1:
            v = av ^ bv
        elif opcode == 2:
            v = (av * 1664525 + bv * 1013904223) & 0xFFFFFFFF
        else:
            v = max(av, bv)
        return v, adeps | bdeps


def run_region_tx(runtime: RegionRuntime, pids: list[int], opcodes: list[int]):
    r = runtime.begin()
    vals = [runtime.read(r, pid) for pid in pids]
    x = vals[0]
    for i, opcode in enumerate(opcodes):
        y = vals[(i + 1) % len(vals)]
        x = runtime.op(r, x, y, opcode)
    return r, x


def run_naive_tx(pods: list[Pod], pids: list[int], opcodes: list[int]):
    vals = [NaiveLineage.read(pid, pods[pid]) for pid in pids]
    x = vals[0]
    for i, opcode in enumerate(opcodes):
        x = NaiveLineage.op(x, vals[(i + 1) % len(vals)], opcode)
    return x


def main() -> None:
    rng = random.Random(SEED)
    pods = [Pod(1, rng.randrange(1 << 24)) for _ in range(PODS)]
    runtime = RegionRuntime(pods)

    # Pre-generate transaction programs so both metadata strategies execute the
    # exact same reads/operations and benchmark only representation overhead.
    programs = []
    for _ in range(TRANSACTIONS):
        nreads = rng.randint(2, MAX_READS)
        pids = rng.sample(range(PODS), nreads)
        opcodes = [rng.randrange(4) for _ in range(OPS_PER_TX)]
        programs.append((pids, opcodes))

    # Functional equivalence + timing without races.
    region_ns = []
    naive_ns = []
    value_mismatches = 0
    final_readset_mismatches = 0
    sampled = min(TRANSACTIONS, 40000)
    for i in range(sampled):
        pids, opcodes = programs[i]
        ts = time.perf_counter_ns()
        r, rx = run_region_tx(runtime, pids, opcodes)
        region_ns.append(time.perf_counter_ns() - ts)

        ts = time.perf_counter_ns()
        nx = run_naive_tx(pods, pids, opcodes)
        naive_ns.append(time.perf_counter_ns() - ts)

        value_mismatches += int(rx.value != nx[0])
        region_deps = frozenset(r.readset.items())
        final_readset_mismatches += int(region_deps != nx[1])

    # Lifecycle/race stress. Region ID propagates through J; exact read-set lives
    # once in the transaction object and is validated only at the commit boundary.
    conflicts_expected = 0
    conflicts_detected = 0
    conflicts_escaped = 0
    retries = 0
    stale_serves_rejected = 0
    stale_serves_escaped = 0
    same_value_races = 0
    factors_created = 0
    invalidation_touched = 0
    commit_ns = []
    seal_ns = []
    admission_ns = []

    start = sampled
    for i in range(start, TRANSACTIONS):
        pids, opcodes = programs[i]
        r, x = run_region_tx(runtime, pids, opcodes)

        race = rng.random() < RACE_PROB
        if race:
            pid = rng.choice(pids)
            pod = pods[pid]
            old_gen = pod.generation
            same = rng.random() < 0.45
            old_value = pod.value
            pod.generation += 1
            pod.value = old_value if same else rng.randrange(1 << 24)
            same_value_races += int(same)
            invalidation_touched += runtime.invalidate(pid, old_gen)
            conflicts_expected += 1

        ts = time.perf_counter_ns()
        ok = runtime.commit(r)
        commit_ns.append(time.perf_counter_ns() - ts)
        if not ok:
            conflicts_detected += 1
            retries += 1
            r, x = run_region_tx(runtime, pids, opcodes)
            if not runtime.commit(r):
                conflicts_escaped += 1
        elif race:
            conflicts_escaped += 1

        ts = time.perf_counter_ns()
        sealed = runtime.seal(r, x)
        seal_ns.append(time.perf_counter_ns() - ts)
        factors_created += 1

        # Post-commit source mutation for a subset, including same-value ABA.
        if rng.random() < 0.12:
            pid = rng.choice(pids)
            pod = pods[pid]
            old_gen = pod.generation
            old_value = pod.value
            pod.generation += 1
            pod.value = old_value if rng.random() < 0.45 else rng.randrange(1 << 24)
            invalidation_touched += runtime.invalidate(pid, old_gen)
            ts = time.perf_counter_ns()
            live = runtime.serve(sealed)
            admission_ns.append(time.perf_counter_ns() - ts)
            if live:
                stale_serves_escaped += 1
            else:
                stale_serves_rejected += 1
        else:
            ts = time.perf_counter_ns()
            _ = runtime.serve(sealed)
            admission_ns.append(time.perf_counter_ns() - ts)

    # Metadata footprint proxy: the naive representation creates/propagates a full
    # dependency set at every intermediate op. Region representation uses one int
    # region ID per intermediate plus one final read-set per transaction.
    avg_reads = statistics.mean(len(p[0]) for p in programs)
    naive_dependency_pair_materializations_est = TRANSACTIONS * OPS_PER_TX * avg_reads
    region_dependency_pair_storage_est = TRANSACTIONS * avg_reads
    region_handle_propagations = TRANSACTIONS * OPS_PER_TX

    report = {
        "stage": STAGE,
        "architecture_candidate": "Epistemic Lifetime Regions (ELR)",
        "pods": PODS,
        "transactions": TRANSACTIONS,
        "ops_per_transaction": OPS_PER_TX,
        "max_reads": MAX_READS,
        "mean_reads_per_transaction": avg_reads,
        "functional_sample_transactions": sampled,
        "value_mismatches_region_vs_naive": value_mismatches,
        "final_readset_mismatches_region_vs_naive": final_readset_mismatches,
        "median_region_j_metadata_compute_ns": statistics.median(region_ns),
        "median_naive_per_value_lineage_compute_ns": statistics.median(naive_ns),
        "region_over_naive_compute_ratio": statistics.median(region_ns) / statistics.median(naive_ns),
        "naive_dependency_pair_materializations_est": naive_dependency_pair_materializations_est,
        "region_final_readset_pair_storage_est": region_dependency_pair_storage_est,
        "dependency_pair_storage_reduction_est": 1.0 - region_dependency_pair_storage_est / naive_dependency_pair_materializations_est,
        "region_handle_propagations": region_handle_propagations,
        "commit_conflicts_expected": conflicts_expected,
        "commit_conflicts_detected": conflicts_detected,
        "commit_conflicts_escaped": conflicts_escaped,
        "same_value_race_conflicts": same_value_races,
        "retries": retries,
        "factors_created": factors_created,
        "post_commit_stale_serves_rejected": stale_serves_rejected,
        "post_commit_stale_serves_escaped": stale_serves_escaped,
        "invalidation_factor_nodes_touched": invalidation_touched,
        "median_commit_validation_ns_python": statistics.median(commit_ns),
        "median_seal_factor_creation_ns_python": statistics.median(seal_ns),
        "median_factor_admission_check_ns_python": statistics.median(admission_ns),
        "mechanism": (
            "All mutable-derived tensors inside one CKVM/J execution region carry only one RegionID. The transaction object owns the growing exact "
            "generation read-set once. Neural operations propagate RegionID without copying/unioning generation sets. At the Neural Commit Barrier "
            "the region read-set is validated exactly; only retained outputs are converted to a sealed CLFD FactorID. This moves temporal lineage "
            "bookkeeping to neural region boundaries rather than every tensor operation."
        ),
        "architectural_hypothesis": (
            "The earlier high lifecycle overhead is not inherent to exact freshness. CKCA can erase most per-operation lineage cost by treating J "
            "execution as an epistemic borrow region: dynamic reads accumulate centrally, ordinary internal tensor algebra carries a compact region "
            "handle, and exact generation sets are materialized only at commit/seal boundaries."
        ),
        "claim_boundary": (
            "Region/arena lifetimes, transactions, borrow regions, taint contexts and capability handles are established systems techniques. R336 "
            "tests their use to compress generation lineage through neural execution; production <5% overhead still requires integration with a "
            "real model runtime/kernel and must not be inferred solely from this Python metadata benchmark."
        ),
        "dod_status": "NOT_DOD; lifecycle-overhead architecture gate",
    }
    report["contract_pass"] = (
        value_mismatches == 0
        and final_readset_mismatches == 0
        and conflicts_detected == conflicts_expected
        and conflicts_escaped == 0
        and stale_serves_escaped == 0
        and stale_serves_rejected > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Epistemic Lifetime Region gate failed")


if __name__ == "__main__":
    main()
