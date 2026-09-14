from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

STAGE = "R336B-PACKED-EPISTEMIC-REGIONS"
PODS = int(os.environ.get("SO_R336B_PODS", "50000"))
PROGRAMS = int(os.environ.get("SO_R336B_PROGRAMS", "12000"))
MAX_READS = int(os.environ.get("SO_R336B_MAX_READS", "8"))
DEPTHS = tuple(int(x) for x in os.environ.get("SO_R336B_DEPTHS", "8,32,128,512").split(","))
REPORT_PATH = Path(os.environ.get("SO_R336B_REPORT", "r336b_report.json"))
SEED = 33620914

MASK32 = 0xFFFFFFFF


def op(v: int, y: int, code: int) -> int:
    if code == 0:
        return (v + y) & MASK32
    if code == 1:
        return v ^ y
    if code == 2:
        return (v * 1664525 + y * 1013904223) & MASK32
    return v if v >= y else y


def pure_compute(values: list[int], codes: list[int]) -> int:
    x = values[0]
    n = len(values)
    for i, code in enumerate(codes):
        x = op(x, values[(i + 1) % n], code)
    return x


def packed_region_compute(values: list[int], pids: list[int], generations: list[int], codes: list[int]):
    # Region metadata exists once at the execution boundary. The neural hot loop is
    # exactly ordinary arithmetic; it does not allocate Borrowed wrappers or union sets.
    readset = {pid: generations[pid] for pid in pids}
    x = pure_compute(values, codes)
    return x, readset


def naive_lineage_compute(values: list[int], pids: list[int], generations: list[int], codes: list[int]):
    vals = [
        (v, frozenset([(pid, generations[pid])]))
        for v, pid in zip(values, pids)
    ]
    x, deps = vals[0]
    n = len(vals)
    for i, code in enumerate(codes):
        y, ydeps = vals[(i + 1) % n]
        x = op(x, y, code)
        deps = deps | ydeps
    return x, deps


def percentile(values: list[float], p: float):
    xs = sorted(values)
    return xs[int(p * (len(xs) - 1))]


def benchmark_depth(rng: random.Random, generations: list[int], pod_values: list[int], depth: int):
    programs = []
    for _ in range(PROGRAMS):
        nreads = rng.randint(2, MAX_READS)
        pids = rng.sample(range(PODS), nreads)
        values = [pod_values[p] for p in pids]
        codes = [rng.randrange(4) for _ in range(depth)]
        programs.append((pids, values, codes))

    baseline_times = []
    region_times = []
    naive_times = []
    mismatches = 0
    readset_mismatches = 0

    # Interleaved order avoids giving one representation all of the warm-cache benefit.
    for pids, values, codes in programs:
        t0 = time.perf_counter_ns()
        b = pure_compute(values, codes)
        baseline_times.append(time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        r, rdeps = packed_region_compute(values, pids, generations, codes)
        region_times.append(time.perf_counter_ns() - t0)

        t0 = time.perf_counter_ns()
        n, ndeps = naive_lineage_compute(values, pids, generations, codes)
        naive_times.append(time.perf_counter_ns() - t0)

        mismatches += int(not (b == r == n))
        readset_mismatches += int(frozenset(rdeps.items()) != ndeps)

    med_b = statistics.median(baseline_times)
    med_r = statistics.median(region_times)
    med_n = statistics.median(naive_times)
    return {
        "depth": depth,
        "programs": PROGRAMS,
        "value_mismatches": mismatches,
        "readset_mismatches": readset_mismatches,
        "median_baseline_ns": med_b,
        "median_packed_region_ns": med_r,
        "median_naive_lineage_ns": med_n,
        "packed_region_overhead_fraction_vs_no_lineage": (med_r - med_b) / med_b,
        "naive_lineage_overhead_fraction_vs_no_lineage": (med_n - med_b) / med_b,
        "packed_region_speedup_vs_naive_lineage": med_n / med_r,
        "p99_baseline_ns": percentile(baseline_times, 0.99),
        "p99_packed_region_ns": percentile(region_times, 0.99),
        "p99_naive_lineage_ns": percentile(naive_times, 0.99),
    }


def lifecycle_stress(rng: random.Random, generations: list[int], pod_values: list[int]):
    expected = 0
    detected = 0
    escaped = 0
    same_value = 0
    commit_times = []
    admission_times = []
    sealed = []
    factor_live = []
    reverse: dict[tuple[int, int], list[int]] = {}

    def seal(readset: dict[int, int], value: int):
        fid = len(factor_live)
        factor_live.append(True)
        sealed.append((value, fid, tuple(sorted(readset.items()))))
        for dep in readset.items():
            reverse.setdefault(dep, []).append(fid)
        return fid

    def invalidate(pid: int, old_g: int):
        for fid in reverse.get((pid, old_g), ()):
            factor_live[fid] = False

    trials = 50000
    stale_rejected = 0
    stale_escaped = 0
    for _ in range(trials):
        nreads = rng.randint(2, MAX_READS)
        pids = rng.sample(range(PODS), nreads)
        values = [pod_values[p] for p in pids]
        codes = [rng.randrange(4) for _ in range(64)]
        value, readset = packed_region_compute(values, pids, generations, codes)

        race = rng.random() < 0.12
        if race:
            pid = rng.choice(pids)
            old_g = generations[pid]
            old_v = pod_values[pid]
            generations[pid] += 1
            if rng.random() < 0.45:
                pod_values[pid] = old_v
                same_value += 1
            else:
                pod_values[pid] = rng.randrange(1 << 24)
            invalidate(pid, old_g)
            expected += 1

        t0 = time.perf_counter_ns()
        ok = all(generations[pid] == g for pid, g in readset.items())
        commit_times.append(time.perf_counter_ns() - t0)
        if not ok:
            detected += 1
            # Local retry constructs a new region/readset against current state.
            values = [pod_values[p] for p in pids]
            value, readset = packed_region_compute(values, pids, generations, codes)
            ok = all(generations[pid] == g for pid, g in readset.items())
        elif race:
            escaped += 1
        if not ok:
            escaped += 1

        fid = seal(readset, value)
        if rng.random() < 0.10:
            pid = rng.choice(pids)
            old_g = generations[pid]
            old_v = pod_values[pid]
            generations[pid] += 1
            pod_values[pid] = old_v if rng.random() < 0.45 else rng.randrange(1 << 24)
            invalidate(pid, old_g)
            t0 = time.perf_counter_ns()
            live = factor_live[fid]
            admission_times.append(time.perf_counter_ns() - t0)
            if live:
                stale_escaped += 1
            else:
                stale_rejected += 1
        else:
            t0 = time.perf_counter_ns()
            _ = factor_live[fid]
            admission_times.append(time.perf_counter_ns() - t0)

    return {
        "trials": trials,
        "commit_conflicts_expected": expected,
        "commit_conflicts_detected": detected,
        "commit_conflicts_escaped": escaped,
        "same_value_race_conflicts": same_value,
        "post_commit_stale_serves_rejected": stale_rejected,
        "post_commit_stale_serves_escaped": stale_escaped,
        "median_commit_validation_ns": statistics.median(commit_times),
        "median_factor_admission_ns": statistics.median(admission_times),
    }


def main() -> None:
    rng = random.Random(SEED)
    generations = [1] * PODS
    pod_values = [rng.randrange(1 << 24) for _ in range(PODS)]

    depth_rows = [benchmark_depth(rng, generations, pod_values, d) for d in DEPTHS]
    life = lifecycle_stress(rng, generations, pod_values)

    all_correct = all(
        r["value_mismatches"] == 0 and r["readset_mismatches"] == 0
        for r in depth_rows
    )
    deepest = depth_rows[-1]
    mean_reads = (2 + MAX_READS) / 2
    naive_pairs = PROGRAMS * sum(DEPTHS) * mean_reads
    region_pairs = PROGRAMS * len(DEPTHS) * mean_reads

    report = {
        "stage": STAGE,
        "architecture_candidate": "Packed Epistemic Lifetime Regions (PELR)",
        "pods": PODS,
        "programs_per_depth": PROGRAMS,
        "depths": list(DEPTHS),
        "max_reads": MAX_READS,
        "depth_results": depth_rows,
        "deepest_depth": deepest["depth"],
        "deepest_packed_region_overhead_fraction_vs_no_lineage": deepest["packed_region_overhead_fraction_vs_no_lineage"],
        "deepest_naive_lineage_overhead_fraction_vs_no_lineage": deepest["naive_lineage_overhead_fraction_vs_no_lineage"],
        "deepest_packed_region_speedup_vs_naive_lineage": deepest["packed_region_speedup_vs_naive_lineage"],
        "estimated_naive_dependency_pair_materializations": naive_pairs,
        "estimated_region_readset_pair_storage": region_pairs,
        "estimated_dependency_pair_reduction": 1.0 - region_pairs / naive_pairs,
        "lifecycle_stress": life,
        "mechanism": (
            "R336's correctness result is retained but its Python Borrowed-object representation is removed. PELR treats RegionID as execution-frame "
            "metadata: the exact dynamic read-set is constructed once at the World Port boundary, while the interior J hot loop performs ordinary "
            "tensor/scalar operations with no dependency-set allocation or union. Commit validates the one region read-set and retained state receives "
            "one FactorID. Thus temporal lineage cost is boundary-amortized as neural compute depth grows."
        ),
        "architectural_hypothesis": (
            "Exact lifecycle control need not impose per-layer lineage work. A neural runtime can use region-scoped temporal typing so ordinary layer "
            "execution is metadata-free except for an implicit execution-context handle, with generation work paid only at read/commit/seal/admission "
            "boundaries. This is the concrete route for attacking the historical ~73% lifecycle-overhead failure without weakening correctness."
        ),
        "claim_boundary": (
            "This Python benchmark demonstrates representation scaling, not a production LLM serving overhead result. The <5% DoD remains open until "
            "the region handle is integrated into a real model runtime/kernel with realistic batch/sequence/concurrency conditions."
        ),
        "dod_status": "NOT_DOD; packed region representation and overhead-scaling gate",
    }
    report["contract_pass"] = (
        all_correct
        and life["commit_conflicts_detected"] == life["commit_conflicts_expected"]
        and life["commit_conflicts_escaped"] == 0
        and life["post_commit_stale_serves_escaped"] == 0
        and life["post_commit_stale_serves_rejected"] > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Packed Epistemic Lifetime Region gate failed")


if __name__ == "__main__":
    main()
