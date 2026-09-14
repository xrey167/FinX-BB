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

STAGE = "R338B-PROOF-CARRYING-REQUALIFICATION"
INPUTS = int(os.environ.get("SO_R338B_INPUTS", "2000"))
DERIVED = int(os.environ.get("SO_R338B_DERIVED", "18000"))
UPDATES = int(os.environ.get("SO_R338B_UPDATES", "12000"))
ORACLE_EVERY = int(os.environ.get("SO_R338B_ORACLE_EVERY", "60"))
REPORT_PATH = Path(os.environ.get("SO_R338B_REPORT", "r338b_report.json"))
SEED = 33820914
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

    def mint(self, parents: tuple[int, ...] = ()) -> int:
        fid = self.next_id
        self.next_id += 1
        self.live[fid] = True
        self.parents[fid] = parents
        return fid

    def kill(self, fid: int) -> None:
        if not self.live.get(fid, False):
            raise AssertionError(f"attempted to kill non-live factor {fid}")
        self.live[fid] = False


def eval_op(op: str, a: int, b: int) -> int:
    if op == "ADD": return (a + b) % MOD
    if op == "XOR": return a ^ b
    if op == "MAX": return max(a, b)
    if op == "MIN": return min(a, b)
    if op == "MUL": return (a * b) % MOD
    if op == "SELECT_EVEN": return a if (b & 1) == 0 else b
    raise ValueError(op)


def certificate(op: str, oa: int, ob: int, na: int, nb: int, old_out: int):
    if oa == na and ob == nb:
        return True, "PARENT_VALUE_EQUIVALENCE"
    if op == "ADD":
        return (((na - oa) + (nb - ob)) % MOD == 0), "ADD_DELTA_ZERO"
    if op == "XOR":
        return ((oa ^ na) == (ob ^ nb)), "XOR_DELTA_CANCELS"
    if op == "MAX":
        return (max(na, nb) == old_out), "MAX_WINNER_CERT"
    if op == "MIN":
        return (min(na, nb) == old_out), "MIN_WINNER_CERT"
    if op == "MUL":
        return ((na * nb) % MOD == old_out), "MUL_MOD_CERT"
    if op == "SELECT_EVEN":
        return ((na if (nb & 1) == 0 else nb) == old_out), "SELECT_CERT"
    raise ValueError(op)


def find_descendants(start: int, reverse: list[list[int]]) -> list[int]:
    seen = {start}
    q = deque([start])
    out = []
    while q:
        n = q.popleft()
        for c in reverse[n]:
            if c not in seen:
                seen.add(c)
                q.append(c)
                out.append(c)
    out.sort()
    return out


def main() -> None:
    rng = random.Random(SEED)
    total = INPUTS + DERIVED
    factors = Factors()
    nodes: list[Node] = []
    reverse: list[list[int]] = [[] for _ in range(total)]
    generations = [1] * INPUTS

    for _ in range(INPUTS):
        nodes.append(Node("INPUT", (), rng.randrange(MOD), factors.mint()))

    for nid in range(INPUTS, total):
        p1 = rng.randrange(nid)
        p2 = rng.randrange(nid)
        while p2 == p1:
            p2 = rng.randrange(nid)
        op = OPS[rng.randrange(len(OPS))]
        value = eval_op(op, nodes[p1].value, nodes[p2].value)
        fid = factors.mint((nodes[p1].factor_id, nodes[p2].factor_id))
        nodes.append(Node(op, (p1, p2), value, fid))
        reverse[p1].append(nid)
        reverse[p2].append(nid)

    descendant_cache: dict[int, list[int]] = {}
    baseline_recomputes = 0
    actual_recomputes = 0
    requalifications = 0
    pure_factor_rebases = 0
    op_cert_rebases = 0
    cert_misses = 0
    semantic_changes = 0
    same_value_updates = 0
    same_value_zero_recompute = 0
    oracle_mismatches = 0
    dead_factor_resurrections = 0
    current_factor_errors = 0
    parent_factor_errors = 0
    affected_sizes = []
    repair_ns = []
    cert_counts: dict[str, int] = {}

    for u in range(UPDATES):
        src = rng.randrange(INPUTS)
        affected = descendant_cache.get(src)
        if affected is None:
            affected = find_descendants(src, reverse)
            descendant_cache[src] = affected
        affected_sizes.append(len(affected))
        baseline_recomputes += len(affected)

        old_values = {src: nodes[src].value}
        for nid in affected:
            old_values[nid] = nodes[nid].value

        killed_factors = [nodes[src].factor_id] + [nodes[n].factor_id for n in affected]
        for fid in killed_factors:
            factors.kill(fid)

        old_src = nodes[src].value
        same = rng.random() < 0.40
        new_src = old_src if same else rng.randrange(MOD)
        same_value_updates += int(same)
        generations[src] += 1
        nodes[src].value = new_src
        nodes[src].factor_id = factors.mint()

        this_recomputes = 0
        ts = time.perf_counter_ns()
        for nid in affected:
            node = nodes[nid]
            p1, p2 = node.parents
            oa = old_values.get(p1, nodes[p1].value)
            ob = old_values.get(p2, nodes[p2].value)
            na, nb = nodes[p1].value, nodes[p2].value
            old_out = old_values[nid]

            unchanged, kind = certificate(node.op, oa, ob, na, nb, old_out)
            if unchanged:
                node.value = old_out
                requalifications += 1
                cert_counts[kind] = cert_counts.get(kind, 0) + 1
                if kind == "PARENT_VALUE_EQUIVALENCE":
                    pure_factor_rebases += 1
                else:
                    op_cert_rebases += 1
            else:
                new_out = eval_op(node.op, na, nb)
                this_recomputes += 1
                actual_recomputes += 1
                semantic_changes += int(new_out != old_out)
                cert_misses += int(new_out == old_out)
                node.value = new_out
            node.factor_id = factors.mint((nodes[p1].factor_id, nodes[p2].factor_id))
        repair_ns.append(time.perf_counter_ns() - ts)
        if same and this_recomputes == 0:
            same_value_zero_recompute += 1

        # Local no-resurrection check is sufficient because factor IDs are monotonic:
        # killed IDs are never reissued, and all current nodes must own fresh live IDs.
        for fid in killed_factors:
            dead_factor_resurrections += int(factors.live.get(fid, False))
        for nid in [src] + affected:
            fid = nodes[nid].factor_id
            current_factor_errors += int(not factors.live.get(fid, False))
            if nid >= INPUTS:
                p1, p2 = nodes[nid].parents
                parent_factor_errors += int(
                    factors.parents[fid] != (nodes[p1].factor_id, nodes[p2].factor_id)
                )

        if u % ORACLE_EVERY == 0:
            oracle: dict[int, int] = {}
            for nid in affected:
                p1, p2 = nodes[nid].parents
                a = oracle.get(p1, nodes[p1].value)
                b = oracle.get(p2, nodes[p2].value)
                v = eval_op(nodes[nid].op, a, b)
                oracle[nid] = v
                oracle_mismatches += int(v != nodes[nid].value)

    saved = baseline_recomputes - actual_recomputes
    report = {
        "stage": STAGE,
        "architecture_candidate": "Proof-Carrying Requalification (PCR) / Causal Factor Rebasing",
        "inputs": INPUTS,
        "derived_nodes": DERIVED,
        "updates": UPDATES,
        "same_value_input_generation_updates": same_value_updates,
        "same_value_updates_with_zero_full_recompute": same_value_zero_recompute,
        "baseline_descendant_recomputes": baseline_recomputes,
        "actual_full_recomputes": actual_recomputes,
        "full_recomputes_saved": saved,
        "full_recompute_reduction_fraction": saved / max(1, baseline_recomputes),
        "certificate_requalifications": requalifications,
        "pure_factor_rebases": pure_factor_rebases,
        "operation_certificate_rebases": op_cert_rebases,
        "conservative_certificate_misses": cert_misses,
        "semantic_changed_nodes": semantic_changes,
        "certificate_kind_counts": cert_counts,
        "oracle_mismatches": oracle_mismatches,
        "dead_factor_resurrections": dead_factor_resurrections,
        "current_factor_errors": current_factor_errors,
        "parent_factor_rebase_errors": parent_factor_errors,
        "mean_affected_descendants_per_update": statistics.mean(affected_sizes),
        "p99_affected_descendants_per_update": sorted(affected_sizes)[int(0.99 * (len(affected_sizes)-1))],
        "median_repair_ns_python": statistics.median(repair_ns),
        "mechanism": (
            "Every generation transition kills the old factor first. A typed certificate may then prove that the derived semantic/neural payload is "
            "unchanged. The physical bytes are reused only under a newly minted factor whose immediate parents are the current fresh factors. If "
            "parent semantic bytes stayed identical, this rebasing cascades through the DAG without value recomputation. Thus temporal invalidation "
            "remains strict while physical recomputation is conditional on a proof of semantic change."
        ),
        "claim_boundary": (
            "This is typed deterministic incremental computation, not a certificate for arbitrary hidden neural states. Self-adjusting computation, "
            "memoization and change propagation have substantial prior art. The CKCA-specific research question is whether fresh temporal factors can "
            "be rebased over reused neural/semantic bytes without ever reviving the dead generation."
        ),
        "dod_status": "NOT_DOD; optimized proof-backed requalification gate",
    }
    report["contract_pass"] = (
        oracle_mismatches == 0
        and dead_factor_resurrections == 0
        and current_factor_errors == 0
        and parent_factor_errors == 0
        and same_value_zero_recompute == same_value_updates
        and requalifications > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("R338b Proof-Carrying Requalification contract failed")


if __name__ == "__main__":
    main()
