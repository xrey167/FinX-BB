from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

STAGE = "R328-VERSIONED-SYMLINK-CAPABILITIES"
ALIASES = int(os.environ.get("SO_R328_ALIASES", "200000"))
PODS = int(os.environ.get("SO_R328_PODS", "50000"))
PLANS = int(os.environ.get("SO_R328_PLANS", "250000"))
ALIASES_PER_PLAN = int(os.environ.get("SO_R328_ALIASES_PER_PLAN", "3"))
REBINDS = int(os.environ.get("SO_R328_REBINDS", "20000"))
PROBES = int(os.environ.get("SO_R328_PROBES", "1000000"))
REPORT_PATH = Path(os.environ.get("SO_R328_REPORT", "r328_report.json"))
SEED = 3280914


@dataclass
class Symlink:
    pod_id: int
    generation: int = 1


@dataclass(frozen=True)
class SymlinkCap:
    alias_id: int
    generation: int
    pod_id: int


@dataclass
class Plan:
    caps: tuple[SymlinkCap, ...]
    compiled_epoch: int


class Resolver:
    def __init__(self, rng: random.Random):
        self.links = [Symlink(rng.randrange(PODS)) for _ in range(ALIASES)]
        self.global_epoch = 1

    def cap(self, alias_id: int) -> SymlinkCap:
        s = self.links[alias_id]
        return SymlinkCap(alias_id, s.generation, s.pod_id)

    def verify(self, cap: SymlinkCap) -> bool:
        s = self.links[cap.alias_id]
        return s.generation == cap.generation and s.pod_id == cap.pod_id

    def rebind(self, alias_id: int, pod_id: int) -> tuple[int, int, int]:
        s = self.links[alias_id]
        old = (alias_id, s.generation, s.pod_id)
        s.generation += 1
        s.pod_id = pod_id
        self.global_epoch += 1
        return old


def main() -> None:
    rng = random.Random(SEED)
    resolver = Resolver(rng)
    reverse: list[list[int]] = [[] for _ in range(ALIASES)]
    plans: list[Plan] = []

    t0 = time.perf_counter()
    for pid in range(PLANS):
        aids = rng.sample(range(ALIASES), ALIASES_PER_PLAN)
        caps = tuple(resolver.cap(a) for a in aids)
        plans.append(Plan(caps, resolver.global_epoch))
        for a in aids:
            reverse[a].append(pid)
    build_seconds = time.perf_counter() - t0

    invalid = bytearray(PLANS)
    per_symlink_invalidations = 0
    global_epoch_invalidations_theoretical = 0
    same_target_rebinds = 0
    rebind_ns = []
    touched_per_rebind = []

    for _ in range(REBINDS):
        aid = rng.randrange(ALIASES)
        link = resolver.links[aid]
        same = rng.random() < 0.35
        new_pod = link.pod_id if same else rng.randrange(PODS)
        if same:
            same_target_rebinds += 1
        ts = time.perf_counter_ns()
        resolver.rebind(aid, new_pod)
        touched = 0
        for plan_id in reverse[aid]:
            if not invalid[plan_id]:
                invalid[plan_id] = 1
                per_symlink_invalidations += 1
                touched += 1
        rebind_ns.append(time.perf_counter_ns() - ts)
        touched_per_rebind.append(touched)
        global_epoch_invalidations_theoretical += PLANS

        # Lazily refresh a few touched plans to emulate ongoing use; their new caps
        # are registered in the same reverse lists because alias IDs did not change.
        if reverse[aid]:
            for plan_id in rng.sample(reverse[aid], min(3, len(reverse[aid]))):
                p = plans[plan_id]
                plans[plan_id] = Plan(tuple(resolver.cap(c.alias_id) for c in p.caps), resolver.global_epoch)
                invalid[plan_id] = 0

    # Probe plan admission. A generation-capability plan is valid iff every captured
    # Symlink generation still resolves to the same Pod. Global epoch policy instead
    # rejects every old plan after any unrelated alias change.
    stale_accepted = 0
    valid_rejected = 0
    full_oracle_mismatches = 0
    global_false_invalidations = 0
    probe_ns = []
    for _ in range(PROBES):
        i = rng.randrange(PLANS)
        p = plans[i]
        ts = time.perf_counter_ns()
        cap_valid = all(resolver.verify(c) for c in p.caps)
        probe_ns.append(time.perf_counter_ns() - ts)
        marked_invalid = bool(invalid[i])
        if cap_valid and marked_invalid:
            # Conservative reverse-index mark can be stale after plan refresh only if
            # bookkeeping is wrong; this should stay zero.
            valid_rejected += 1
        if not cap_valid and not marked_invalid:
            stale_accepted += 1
        if marked_invalid != (not cap_valid):
            full_oracle_mismatches += 1
        if p.compiled_epoch != resolver.global_epoch and cap_valid:
            global_false_invalidations += 1

    reduction = 1.0 - per_symlink_invalidations / max(1, global_epoch_invalidations_theoretical)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Versioned Symlink Capabilities (VSC) / per-binding plan lifetimes",
        "aliases": ALIASES,
        "pods": PODS,
        "plans": PLANS,
        "aliases_per_plan": ALIASES_PER_PLAN,
        "rebinds": REBINDS,
        "same_target_generation_rebinds": same_target_rebinds,
        "plan_admission_probes": PROBES,
        "build_seconds": build_seconds,
        "per_symlink_plan_invalidations": per_symlink_invalidations,
        "global_epoch_plan_invalidations_theoretical": global_epoch_invalidations_theoretical,
        "invalidation_reduction_vs_global_epoch": reduction,
        "mean_newly_invalidated_plans_per_rebind": statistics.mean(touched_per_rebind),
        "p99_newly_invalidated_plans_per_rebind": sorted(touched_per_rebind)[int(0.99 * (len(touched_per_rebind) - 1))],
        "median_rebind_plus_reverse_invalidation_ns_python": statistics.median(rebind_ns),
        "median_plan_capability_verify_ns_python": statistics.median(probe_ns),
        "stale_plans_accepted": stale_accepted,
        "valid_plans_rejected_by_reverse_index": valid_rejected,
        "reverse_index_vs_full_capability_oracle_mismatches": full_oracle_mismatches,
        "global_epoch_false_invalidations_seen_in_probes": global_false_invalidations,
        "contract_pass": stale_accepted == 0 and valid_rejected == 0 and full_oracle_mismatches == 0,
        "mechanism": (
            "Every Symlink/alias binding owns an independent monotonic generation. A compiled B plan captures capabilities only for the exact "
            "aliases it resolved. Rebinding one alias invalidates only plans that captured that Symlink generation; unrelated plans survive. "
            "Same-target rebinds still create a new temporal identity and therefore invalidate old capabilities."
        ),
        "architectural_hypothesis": (
            "Canonical value generations are insufficient by themselves: the linguistic-to-canonical binding is also mutable world state. "
            "Treating Symlink bindings as generation-scoped capabilities avoids global namespace epochs while preserving exact identity lifecycle."
        ),
        "dod_status": "NOT_DOD; Symlink plan-lifetime scalability gate",
        "claim_boundary": (
            "Versioned symbol tables, inodes/symlinks, dependency indexes and capability generations are established systems ideas. R328 "
            "tests their CKCA-specific role as exact B-plan identity lifetimes; standalone novelty is not claimed."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Versioned Symlink capability gate failed")


if __name__ == "__main__":
    main()
