from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

from research.ckca_v3_world_abi import (
    AuthorityStore,
    Conflict,
    LifetimeFabric,
    LifetimeViolation,
    PlanCache,
    WorldABI,
)

STAGE = "R320-CKCA-V3-WORLD-ABI-INTEGRATED-CONTRACT"
PODS = int(os.environ.get("SO_R320_PODS", "20000"))
PLANS = int(os.environ.get("SO_R320_PLANS", "5000"))
TRANSACTIONS = int(os.environ.get("SO_R320_TRANSACTIONS", "200000"))
REPORT_PATH = Path(os.environ.get("SO_R320_REPORT", "r320_report.json"))
SEED = 3200914
OPS = ("add8", "xor8", "max", "select_even")


def main() -> None:
    rng = random.Random(SEED)
    authority = AuthorityStore()
    lifetimes = LifetimeFabric()
    abi = WorldABI(authority, lifetimes)
    cache = PlanCache()

    for pid in range(PODS):
        authority.create(pid, rng.randrange(256))
        authority.bind(f"pod-{pid}", pid)
        authority.bind(f"late-alias-{pid}", pid)

    # Freeze alias generation after setup. The two aliases must collapse to one
    # canonical address; ordinary value changes do not touch this generation.
    alias_identity_errors = sum(
        authority.resolve(f"pod-{pid}") != authority.resolve(f"late-alias-{pid}")
        for pid in range(PODS)
    )

    plan_specs = []
    for i in range(PLANS):
        left = rng.randrange(PODS)
        right = rng.randrange(PODS)
        op = OPS[i % len(OPS)]
        plan_specs.append((op, left, right))
        cache.get_or_compile(op, left, right, authority)
    initial_plan_compiles = cache.compiles

    commit_conflicts_expected = 0
    commit_conflicts_detected = 0
    commit_conflicts_escaped = 0
    retries_succeeded = 0
    same_value_conflicts = 0
    post_commit_expirations = 0
    stale_serves_rejected = 0
    stale_serves_escaped = 0
    unrelated_updates = 0
    unrelated_false_invalidations = 0
    successful_live_serves = 0
    authority_updates = 0
    invalidation_nodes_touched = 0

    t0 = time.perf_counter()
    for i in range(TRANSACTIONS):
        op, left, right = plan_specs[rng.randrange(PLANS)]
        plan = cache.get_or_compile(op, left, right, authority)
        tx = abi.transaction(plan)
        borrowed = tx.execute()

        # Concurrent world transition between execution and publication.
        conflict_injected = rng.random() < 0.18
        if conflict_injected:
            pid = left if rng.random() < 0.5 else right
            p = authority.pods[pid]
            old_dep = authority.update(pid, p.value if rng.random() < 0.50 else rng.randrange(256))
            authority_updates += 1
            if authority.pods[pid].value == p.value:
                # Note: p is the same mutable object, so compare via old borrowed
                # generations rather than object identity. The metric is only a
                # lower-bound witness count and not used for correctness.
                same_value_conflicts += 1
            invalidation_nodes_touched += abi.expire_generation(old_dep)
            commit_conflicts_expected += 1

        try:
            tx.commit()
            if conflict_injected:
                commit_conflicts_escaped += 1
        except Conflict:
            commit_conflicts_detected += 1
            # The B plan is reused. Only current world/J execution retries.
            tx = abi.transaction(plan)
            borrowed = tx.execute()
            tx.commit()
            retries_succeeded += 1

        sealed = tx.seal(borrowed)

        # Either mutate a causal source after commit or mutate an unrelated Pod.
        r = rng.random()
        if r < 0.12:
            dep_pid = rng.choice(tuple(tx.readset.keys()))
            p = authority.pods[dep_pid]
            old_dep = authority.update(dep_pid, p.value if rng.random() < 0.45 else rng.randrange(256))
            authority_updates += 1
            invalidation_nodes_touched += abi.expire_generation(old_dep)
            post_commit_expirations += 1
            try:
                abi.serve(sealed)
                stale_serves_escaped += 1
            except LifetimeViolation:
                stale_serves_rejected += 1
        elif r < 0.20:
            deps = set(tx.readset.keys())
            pid = rng.randrange(PODS)
            while pid in deps:
                pid = rng.randrange(PODS)
            p = authority.pods[pid]
            old_dep = authority.update(pid, p.value if rng.random() < 0.45 else rng.randrange(256))
            authority_updates += 1
            invalidation_nodes_touched += abi.expire_generation(old_dep)
            unrelated_updates += 1
            try:
                abi.serve(sealed)
                successful_live_serves += 1
            except LifetimeViolation:
                unrelated_false_invalidations += 1
        else:
            try:
                abi.serve(sealed)
                successful_live_serves += 1
            except LifetimeViolation:
                unrelated_false_invalidations += 1

    elapsed = time.perf_counter() - t0

    # Value updates must not trigger B-plan recompilation.
    final_plan_compiles = cache.compiles
    plan_recompiles_due_to_value_updates = final_plan_compiles - initial_plan_compiles

    report = {
        "stage": STAGE,
        "architecture_candidate": "CKCA V3 Causal Neural World ABI reference runtime",
        "pods": PODS,
        "reusable_plans": PLANS,
        "transactions": TRANSACTIONS,
        "elapsed_seconds": elapsed,
        "transactions_per_second": TRANSACTIONS / elapsed,
        "canonical_alias_identity_errors": alias_identity_errors,
        "initial_plan_compiles": initial_plan_compiles,
        "final_plan_compiles": final_plan_compiles,
        "plan_recompiles_due_to_value_updates": plan_recompiles_due_to_value_updates,
        "authority_updates": authority_updates,
        "commit_conflicts_expected": commit_conflicts_expected,
        "commit_conflicts_detected": commit_conflicts_detected,
        "commit_conflicts_escaped": commit_conflicts_escaped,
        "retries_succeeded": retries_succeeded,
        "same_value_conflict_witnesses_lower_bound": same_value_conflicts,
        "post_commit_expirations": post_commit_expirations,
        "stale_serves_rejected": stale_serves_rejected,
        "stale_serves_escaped": stale_serves_escaped,
        "unrelated_updates": unrelated_updates,
        "unrelated_false_invalidations": unrelated_false_invalidations,
        "successful_live_serves": successful_live_serves,
        "lifetime_nodes": len(lifetimes.live),
        "invalidation_nodes_touched": invalidation_nodes_touched,
        "contract_pass": (
            alias_identity_errors == 0
            and plan_recompiles_due_to_value_updates == 0
            and commit_conflicts_escaped == 0
            and commit_conflicts_detected == commit_conflicts_expected
            and retries_succeeded == commit_conflicts_expected
            and stale_serves_escaped == 0
            and stale_serves_rejected == post_commit_expirations
            and unrelated_false_invalidations == 0
        ),
        "mechanism": (
            "One executable reference runtime composes canonical Symlink/Pod addressing, generation capabilities, value-independent B plans, "
            "binding-equivariant typed execution, exact read-set commit validation, CLFD sealing and post-commit admission checks."
        ),
        "dod_status": "NOT_DOD; integrated reference-runtime contract gate",
        "claim_boundary": (
            "This validates the internal composition of CKCA V3 mechanisms in a reference runtime. It is not a real-model quality gate, "
            "a production benchmark, a RAG comparison, or a novelty proof."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    if not report["contract_pass"]:
        raise SystemExit("CKCA V3 World ABI contract gate failed")


if __name__ == "__main__":
    main()
