from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

STAGE = "R338-PROOF-CARRYING-REQUALIFICATION"
INPUTS = int(os.environ.get("SO_R338_INPUTS", "2000"))
DERIVED = int(os.environ.get("SO_R338_DERIVED", "18000"))
UPDATES = int(os.environ.get("SO_R338_UPDATES", "12000"))
ORACLE_EVERY = int(os.environ.get("SO_R338_ORACLE_EVERY", "60"))
REPORT_PATH = Path(os.environ.get("SO_R338_REPORT", "r338_report.json"))
SEED = 3380914
MOD = 256

OPS = ("ADD", "XOR", "MAX", "MIN", "MUL", "SELECT_EVEN")


@dataclass
class Node:
    op: str
    parents: tuple[int, ...]
    value: int
    factor_id: int


class Factors:
    def __init__(self) -> None:
        self.next_id = 1
        self.live: dict[int, bool] = {}
        self.parents: dict[int, tuple[int, ...]] = {}
        self.ever_dead: set[int] = set()

    def mint(self, parents: tuple[int, ...] = ()) -> int:
        fid = self.next_id
        self.next_id += 1
        self.live[fid] = True
        self.parents[fid] = parents
        return fid

    def kill(self, fid: int) -> None:
        if self.live.get(fid, False):
            self.live[fid] = False
            self.ever_dead.add(fid)

    def assert_no_resurrection(self) -> None:
        for fid in self.ever_dead:
            if self.live.get(fid, False):
                raise AssertionError(f"dead factor resurrected: {fid}")


def eval_op(op: str, vals: tuple[int, ...]) -> int:
    a = vals[0]
    b = vals[1]
    if op == "ADD":
        return (a + b) % MOD
    if op == "XOR":
        return a ^ b
    if op == "MAX":
        return max(a, b)
    if op == "MIN":
        return min(a, b)
    if op == "MUL":
        return (a * b) % MOD
    if op == "SELECT_EVEN":
        return a if (b & 1) == 0 else b
    raise ValueError(op)


def certificate_unchanged(
    op: str,
    old_parent_vals: tuple[int, int],
    new_parent_vals: tuple[int, int],
    old_output: int,
) -> tuple[bool, str]:
    oa, ob = old_parent_vals
    na, nb = new_parent_vals

    # Factor-only rebasing: immediate semantic inputs did not change although their
    # temporal factors did. No value operation or neural rematerialization is needed.
    if oa == na and ob == nb:
        return True, "PARENTS_SEMANTICALLY_IDENTICAL"

    # Operation-specific certificates can establish output invariance from compact
    # typed summaries before touching any expensive downstream neural materializer.
    if op == "ADD":
        return (((na - oa) + (nb - ob)) % MOD == 0), "ADD_DELTA_ZERO"
    if op == "XOR":
        return ((oa ^ na) == (ob ^ nb)), "XOR_DELTA_CANCELS"
    if op == "MAX":
        return (max(na, nb) == old_output), "MAX_WINNER_BOUND"
    if op == "MIN":
        return (min(na, nb) == old_output), "MIN_WINNER_BOUND"
    if op == "MUL":
        return ((na * nb) % MOD == old_output), "MUL_MOD_INVARIANT"
    if op == "SELECT_EVEN":
        # Branch identity and selected typed payload are enough to certify reuse.
        selected = na if (nb & 1) == 0 else nb
        return (selected == old_output), "SELECT_BRANCH_INVARIANT"
    raise ValueError(op)


def descendants(start: int, reverse: list[list[int]]) -> list[int]:
    seen = {start}
    q = deque([start])
    out = []
    while q:
        x = q.popleft()
        for child in reverse[x]:
            if child not in seen:
                seen.add(child)
                q.append(child)
                out.append(child)
    out.sort()
    return out


def main() -> None:
    rng = random.Random(SEED)
    factors = Factors()
    total = INPUTS + DERIVED
    nodes: list[Node] = []
    reverse: list[list[int]] = [[] for _ in range(total)]

    # Canonical input generations. Input value bytes may repeat across generations.
    generations = [1] * INPUTS
    for _ in range(INPUTS):
        nodes.append(Node("INPUT", (), rng.randrange(MOD), factors.mint()))

    # Random typed semantic DAG. Parent IDs are always lower than child IDs, giving
    # an implicit topological order for exact change propagation.
    for nid in range(INPUTS, total):
        p1 = rng.randrange(nid)
        p2 = rng.randrange(nid)
        while p2 == p1:
            p2 = rng.randrange(nid)
        op = OPS[rng.randrange(len(OPS))]
        value = eval_op(op, (nodes[p1].value, nodes[p2].value))
        fid = factors.mint((nodes[p1].factor_id, nodes[p2].factor_id))
        nodes.append(Node(op, (p1, p2), value, fid))
        reverse[p1].append(nid)
        reverse[p2].append(nid)

    baseline_descendant_recomputes = 0
    actual_full_recomputes = 0
    certificate_requalifications = 0
    factor_only_rebases = 0
    op_certificate_rebases = 0
    conservative_certificate_misses = 0
    same_value_input_updates = 0
    same_value_updates_zero_full_recompute = 0
    semantic_changed_nodes = 0
    factors_rebased = 0
    oracle_mismatches = 0
    stale_current_factor_errors = 0
    dead_factor_resurrections = 0
    affected_sizes = []
    repair_ns = []
    cert_kinds: dict[str, int] = {}

    for u in range(UPDATES):
        input_id = rng.randrange(INPUTS)
        affected = descendants(input_id, reverse)
        affected_sizes.append(len(affected))
        baseline_descendant_recomputes += len(affected)

        # Snapshot old semantic values only for the affected subgraph. This is the
        # reference frame used by local invariance certificates.
        old_values = {input_id: nodes[input_id].value}
        for nid in affected:
            old_values[nid] = nodes[nid].value

        # Strict temporal invalidation happens first. No old factor is ever revived.
        old_input_factor = nodes[input_id].factor_id
        factors.kill(old_input_factor)
        for nid in affected:
            factors.kill(nodes[nid].factor_id)

        old_input_value = nodes[input_id].value
        same = rng.random() < 0.40
        new_input_value = old_input_value if same else rng.randrange(MOD)
        same_value_input_updates += int(same)
        generations[input_id] += 1
        nodes[input_id].value = new_input_value
        nodes[input_id].factor_id = factors.mint()

        update_full_recomputes = 0
        ts = time.perf_counter_ns()
        for nid in affected:
            node = nodes[nid]
            p1, p2 = node.parents
            old_parent_vals = (
                old_values.get(p1, nodes[p1].value),
                old_values.get(p2, nodes[p2].value),
            )
            new_parent_vals = (nodes[p1].value, nodes[p2].value)
            old_output = old_values[nid]

            unchanged, kind = certificate_unchanged(
                node.op, old_parent_vals, new_parent_vals, old_output
            )
            cert_kinds[kind] = cert_kinds.get(kind, 0) + int(unchanged)
            if unchanged:
                # Old lifetime stays dead. Reuse the exact physical semantic bytes,
                # but mint a fresh factor over the current parent factors.
                node.value = old_output
                node.factor_id = factors.mint(
                    (nodes[p1].factor_id, nodes[p2].factor_id)
                )
                certificate_requalifications += 1
                factors_rebased += 1
                if kind == "PARENTS_SEMANTICALLY_IDENTICAL":
                    factor_only_rebases += 1
                else:
                    op_certificate_rebases += 1
                continue

            # Certificate cannot prove invariance: perform a full typed recompute
            # (standing in for the expensive downstream J/neural materialization).
            new_output = eval_op(node.op, new_parent_vals)
            update_full_recomputes += 1
            actual_full_recomputes += 1
            semantic_changed_nodes += int(new_output != old_output)
            conservative_certificate_misses += int(new_output == old_output)
            node.value = new_output
            node.factor_id = factors.mint(
                (nodes[p1].factor_id, nodes[p2].factor_id)
            )
            factors_rebased += 1

        repair_ns.append(time.perf_counter_ns() - ts)
        if same and update_full_recomputes == 0:
            same_value_updates_zero_full_recompute += 1

        # The defining no-resurrection invariant: every killed factor remains dead,
        # while current nodes own newly minted live factors.
        try:
            factors.assert_no_resurrection()
        except AssertionError:
            dead_factor_resurrections += 1
        for nid in [input_id] + affected[:64]:
            stale_current_factor_errors += int(
                not factors.live.get(nodes[nid].factor_id, False)
            )

        # Periodic full-current recomputation oracle over the complete affected DAG.
        if u % ORACLE_EVERY == 0:
            oracle: dict[int, int] = {}
            for nid in affected:
                p1, p2 = nodes[nid].parents
                a = oracle.get(p1, nodes[p1].value)
                b = oracle.get(p2, nodes[p2].value)
                val = eval_op(nodes[nid].op, (a, b))
                oracle[nid] = val
                oracle_mismatches += int(val != nodes[nid].value)

    saved = baseline_descendant_recomputes - actual_full_recomputes
    report = {
        "stage": STAGE,
        "architecture_candidate": "Proof-Carrying Requalification (PCR) / Causal Factor Rebasing",
        "inputs": INPUTS,
        "derived_nodes": DERIVED,
        "updates": UPDATES,
        "same_value_input_generation_updates": same_value_input_updates,
        "same_value_updates_with_zero_full_recomputes": same_value_updates_zero_full_recompute,
        "baseline_descendant_recomputes": baseline_descendant_recomputes,
        "actual_full_recomputes": actual_full_recomputes,
        "full_recomputes_saved": saved,
        "full_recompute_reduction_fraction": saved / max(1, baseline_descendant_recomputes),
        "certificate_requalifications": certificate_requalifications,
        "factor_only_rebases": factor_only_rebases,
        "operation_certificate_rebases": op_certificate_rebases,
        "conservative_certificate_misses": conservative_certificate_misses,
        "semantic_changed_nodes": semantic_changed_nodes,
        "fresh_factors_minted_for_affected_nodes": factors_rebased,
        "certificate_kind_counts": cert_kinds,
        "oracle_mismatches": oracle_mismatches,
        "stale_current_factor_errors": stale_current_factor_errors,
        "dead_factor_resurrections": dead_factor_resurrections,
        "mean_affected_descendants_per_update": statistics.mean(affected_sizes),
        "p99_affected_descendants_per_update": sorted(affected_sizes)[int(0.99 * (len(affected_sizes) - 1))],
        "median_repair_ns_python": statistics.median(repair_ns),
        "mechanism": (
            "A generation transition kills the old CLFD factor unconditionally. CKCA then tries to prove that a derived semantic/neural payload is "
            "invariant under the transition using a typed operation certificate. If proven, the old physical bytes are reused but receive a newly "
            "minted factor over the current parent factors; the dead old factor is never revived. This rebasing can cascade: if a parent is "
            "requalified to identical semantic bytes, children can often be requalified by factor substitution alone without recomputing values."
        ),
        "novel_architecture_hypothesis": (
            "Temporal invalidation and physical recomputation need not be the same event. Exact lifecycle semantics can always invalidate the old "
            "generation while proof-carrying requalification preserves semantically invariant derived bytes under a fresh lifetime. This can make "
            "same-value ABA writes and operation-invariant updates cheap without weakening the rule that no old generation ever resurrects."
        ),
        "claim_boundary": (
            "Incremental computation, change propagation, memoization, algebraic invariants and proof/certificate systems are established. R338 "
            "tests the CKCA-specific distinction between permanently dead temporal factors and reusable physical neural/semantic bytes under a fresh "
            "proof-backed factor. General neural-state certificates remain an open problem; this gate uses typed deterministic operators."
        ),
        "dod_status": "NOT_DOD; proof-backed lifetime rebasing mechanism gate",
    }
    report["contract_pass"] = (
        oracle_mismatches == 0
        and stale_current_factor_errors == 0
        and dead_factor_resurrections == 0
        and same_value_updates_zero_full_recompute == same_value_input_updates
        and certificate_requalifications > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Proof-Carrying Requalification contract failed")


if __name__ == "__main__":
    main()
