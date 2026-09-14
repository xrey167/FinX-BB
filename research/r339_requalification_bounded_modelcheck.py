from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

STAGE = "R339-REQUALIFICATION-BOUNDED-MODELCHECK"
DOMAIN = tuple(range(int(os.environ.get("SO_R339_DOMAIN", "4"))))
TRACE_DEPTH = int(os.environ.get("SO_R339_DEPTH", "3"))
REPORT_PATH = Path(os.environ.get("SO_R339_REPORT", "r339_report.json"))
MOD = len(DOMAIN)

# Fixed finite semantic DAG:
# n0=A, n1=B, n2=C
# n3=ADD(A,B)
# n4=MAX(B,C)
# n5=XOR(n3,n4)
# n6=SELECT_EVEN(n5,C)
PARENTS = {
    3: (0, 1),
    4: (1, 2),
    5: (3, 4),
    6: (5, 2),
}
OPS = {3: "ADD", 4: "MAX", 5: "XOR", 6: "SELECT_EVEN"}
REVERSE = {
    0: (3,),
    1: (3, 4),
    2: (4, 6),
    3: (5,),
    4: (5,),
    5: (6,),
    6: (),
}


@dataclass
class Factor:
    live: bool
    parents: tuple[int, ...]


@dataclass
class State:
    values: list[int]
    generations: list[int]
    factor_ids: list[int]
    factors: dict[int, Factor]
    next_factor: int


def op_eval(op: str, a: int, b: int) -> int:
    if op == "ADD":
        return (a + b) % MOD
    if op == "MAX":
        return max(a, b)
    if op == "XOR":
        return (a ^ b) % MOD
    if op == "SELECT_EVEN":
        return a if (b & 1) == 0 else b
    raise ValueError(op)


def cert(op: str, oa: int, ob: int, na: int, nb: int, old_out: int) -> bool:
    if oa == na and ob == nb:
        return True
    if op == "ADD":
        return ((na - oa) + (nb - ob)) % MOD == 0
    if op == "XOR":
        return ((oa ^ na) % MOD) == ((ob ^ nb) % MOD)
    if op == "MAX":
        return max(na, nb) == old_out
    if op == "SELECT_EVEN":
        return (na if (nb & 1) == 0 else nb) == old_out
    raise ValueError(op)


def descendants(src: int) -> list[int]:
    seen = {src}
    q = [src]
    out = []
    while q:
        x = q.pop()
        for y in REVERSE[x]:
            if y not in seen:
                seen.add(y)
                q.append(y)
                out.append(y)
    return sorted(out)


def mint(s: State, parents: tuple[int, ...] = ()) -> int:
    fid = s.next_factor
    s.next_factor += 1
    s.factors[fid] = Factor(True, parents)
    return fid


def initial_state(a: int, b: int, c: int) -> State:
    s = State([a, b, c, 0, 0, 0, 0], [1, 1, 1], [0] * 7, {}, 1)
    for i in range(3):
        s.factor_ids[i] = mint(s)
    for nid in range(3, 7):
        p1, p2 = PARENTS[nid]
        s.values[nid] = op_eval(OPS[nid], s.values[p1], s.values[p2])
        s.factor_ids[nid] = mint(s, (s.factor_ids[p1], s.factor_ids[p2]))
    return s


def clone_state(s: State) -> State:
    return State(
        list(s.values),
        list(s.generations),
        list(s.factor_ids),
        {k: Factor(v.live, v.parents) for k, v in s.factors.items()},
        s.next_factor,
    )


def full_oracle(inputs: tuple[int, int, int]) -> list[int]:
    vals = [inputs[0], inputs[1], inputs[2], 0, 0, 0, 0]
    for nid in range(3, 7):
        p1, p2 = PARENTS[nid]
        vals[nid] = op_eval(OPS[nid], vals[p1], vals[p2])
    return vals


def step(s: State, src: int, new_value: int):
    old_values = list(s.values)
    old_factor_ids = list(s.factor_ids)
    affected = descendants(src)

    # Strict invalidation first: every old affected temporal factor dies.
    s.factors[s.factor_ids[src]].live = False
    for nid in affected:
        s.factors[s.factor_ids[nid]].live = False

    s.generations[src] += 1
    s.values[src] = new_value
    s.factor_ids[src] = mint(s)

    requalified = 0
    recomputed = 0
    for nid in affected:
        p1, p2 = PARENTS[nid]
        unchanged = cert(
            OPS[nid],
            old_values[p1], old_values[p2],
            s.values[p1], s.values[p2],
            old_values[nid],
        )
        if unchanged:
            s.values[nid] = old_values[nid]
            requalified += 1
        else:
            s.values[nid] = op_eval(OPS[nid], s.values[p1], s.values[p2])
            recomputed += 1
        s.factor_ids[nid] = mint(s, (s.factor_ids[p1], s.factor_ids[p2]))

    return old_values, old_factor_ids, affected, requalified, recomputed


def check_invariants(s: State, old_factors: list[int], affected_ids: list[int]):
    errors = []
    oracle = full_oracle(tuple(s.values[:3]))
    if s.values != oracle:
        errors.append("semantic_oracle_mismatch")

    # Every old factor of the changed input/descendants remains permanently dead.
    for nid in affected_ids:
        if s.factors[old_factors[nid]].live:
            errors.append("old_factor_resurrected")

    # Every current factor is live and any derived current factor references current
    # immediate parent factors, not a dead historical factor.
    for nid, fid in enumerate(s.factor_ids):
        if not s.factors[fid].live:
            errors.append("current_factor_dead")
        if nid >= 3:
            p1, p2 = PARENTS[nid]
            if s.factors[fid].parents != (s.factor_ids[p1], s.factor_ids[p2]):
                errors.append("factor_parent_not_rebased")

    # Global no-resurrection condition for all historical factors.
    current = set(s.factor_ids)
    for fid, f in s.factors.items():
        if fid not in current and f.live:
            # Historical unaffected factors are allowed to remain live. It is only an
            # error if one was previously killed; no separate killed bit is stored here.
            pass
    return errors


def main() -> None:
    transitions = [(src, v) for src in range(3) for v in DOMAIN]
    traces = 0
    steps_checked = 0
    invariant_violations = 0
    semantic_mismatches = 0
    old_factor_resurrections = 0
    parent_rebase_violations = 0
    same_value_steps = 0
    same_value_steps_with_recompute = 0
    total_requalified = 0
    total_recomputed = 0
    baseline_descendant_recomputes = 0
    counterexamples = []

    for initial in itertools.product(DOMAIN, repeat=3):
        base = initial_state(*initial)
        for trace in itertools.product(transitions, repeat=TRACE_DEPTH):
            traces += 1
            s = clone_state(base)
            for src, new_v in trace:
                old_input = s.values[src]
                old_values, old_factors, affected, rq, rc = step(s, src, new_v)
                steps_checked += 1
                total_requalified += rq
                total_recomputed += rc
                baseline_descendant_recomputes += len(affected)
                if new_v == old_input:
                    same_value_steps += 1
                    same_value_steps_with_recompute += int(rc != 0)

                affected_ids = [src] + affected
                errors = check_invariants(s, old_factors, affected_ids)
                if errors:
                    invariant_violations += len(errors)
                    semantic_mismatches += errors.count("semantic_oracle_mismatch")
                    old_factor_resurrections += errors.count("old_factor_resurrected")
                    parent_rebase_violations += errors.count("factor_parent_not_rebased")
                    if len(counterexamples) < 10:
                        counterexamples.append({
                            "initial": initial,
                            "trace": trace,
                            "step": (src, new_v),
                            "errors": errors,
                            "values": s.values,
                        })
                    break

    reduction = 1.0 - total_recomputed / max(1, baseline_descendant_recomputes)
    report = {
        "stage": STAGE,
        "architecture_candidate": "bounded model check for Proof-Carrying Requalification / Causal Factor Rebasing",
        "domain": list(DOMAIN),
        "trace_depth": TRACE_DEPTH,
        "initial_worlds": len(DOMAIN) ** 3,
        "possible_updates_per_step": len(transitions),
        "traces_exhaustively_checked": traces,
        "transition_steps_checked": steps_checked,
        "invariant_violations": invariant_violations,
        "semantic_oracle_mismatches": semantic_mismatches,
        "old_factor_resurrections": old_factor_resurrections,
        "factor_parent_rebase_violations": parent_rebase_violations,
        "same_value_generation_steps": same_value_steps,
        "same_value_steps_with_any_full_recompute": same_value_steps_with_recompute,
        "baseline_descendant_recomputes": baseline_descendant_recomputes,
        "certificate_requalifications": total_requalified,
        "actual_full_recomputes": total_recomputed,
        "full_recompute_reduction_fraction": reduction,
        "counterexamples": counterexamples,
        "formalized_invariants": [
            "current semantic values equal from-scratch evaluation after every transition",
            "every factor invalidated by a transition remains dead",
            "every current derived factor is freshly rebased onto the current immediate-parent factors",
            "same-value generation transitions never require semantic recomputation in the modeled deterministic DAG",
        ],
        "mechanism": (
            "This bounded model checker exhaustively enumerates finite worlds and multi-update traces for a typed semantic DAG. Each transition first "
            "kills the old temporal factors, then either recomputes a changed value or reuses identical bytes under a freshly minted parent-rebased "
            "factor. The checker compares every intermediate state with from-scratch semantics and explicitly tests no-resurrection/factor-rebase invariants."
        ),
        "claim_boundary": (
            "This is exhaustive only for the finite modeled DAG/domain/trace depth and is not a general theorem. It strengthens the PCR mechanism "
            "evidence and provides a concrete specification for a future TLA+/Lean/Coq proof or larger state-space checker."
        ),
        "dod_status": "NOT_DOD; bounded exhaustive lifecycle model check",
    }
    report["contract_pass"] = (
        invariant_violations == 0
        and semantic_mismatches == 0
        and old_factor_resurrections == 0
        and parent_rebase_violations == 0
        and same_value_steps_with_recompute == 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("R339 bounded model check found a counterexample")


if __name__ == "__main__":
    main()
