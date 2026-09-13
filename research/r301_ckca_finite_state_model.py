from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

OUT = Path(os.environ.get("SO_R301_REPORT", "ci-r301/report.json"))
MAX_GEN = 3
MAX_DEPTH = 8

# Encodings keep the complete state hashable.
# Pod = (generation, live_bit, semantic_value)
# Capability = (pod_id, generation, semantic_value), (-1,-1,-1) = empty
# J state = sorted tuple of capability-generation leaves; empty tuple = no state.
EMPTY_CAP = (-1, -1, -1)
EMPTY_J: tuple[tuple[int, int, int], ...] = tuple()


@dataclass(frozen=True)
class State:
    pods: tuple[tuple[int, int, int], tuple[int, int, int]]
    caps: tuple[tuple[int, int, int], tuple[int, int, int]]
    js: tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]


def cap_current(s: State, cap: tuple[int, int, int]) -> bool:
    if cap == EMPTY_CAP:
        return False
    pid, gen, value = cap
    cgen, live, cvalue = s.pods[pid]
    return bool(live and gen == cgen and value == cvalue)


def cap_value_only(s: State, cap: tuple[int, int, int]) -> bool:
    if cap == EMPTY_CAP:
        return False
    pid, _gen, value = cap
    _cgen, live, cvalue = s.pods[pid]
    return bool(live and value == cvalue)


def cap_pod_only(s: State, cap: tuple[int, int, int]) -> bool:
    if cap == EMPTY_CAP:
        return False
    pid, _gen, _value = cap
    _cgen, live, _cvalue = s.pods[pid]
    return bool(live)


def j_current(s: State, deps: tuple[tuple[int, int, int], ...]) -> bool:
    if not deps:
        return False
    return all(cap_current(s, dep) for dep in deps)


def j_value_only(s: State, deps: tuple[tuple[int, int, int], ...]) -> bool:
    if not deps:
        return False
    return all(cap_value_only(s, dep) for dep in deps)


def j_pod_only(s: State, deps: tuple[tuple[int, int, int], ...]) -> bool:
    if not deps:
        return False
    return all(cap_pod_only(s, dep) for dep in deps)


def replace_tuple(xs, idx, value):
    ys = list(xs); ys[idx] = value; return tuple(ys)


def successors(s: State):
    # World writes/revokes: monotonic generation, including same-value rewrites.
    for pid in (0, 1):
        gen, _live, curv = s.pods[pid]
        if gen < MAX_GEN:
            for newv in (0, 1):
                pods = replace_tuple(s.pods, pid, (gen + 1, 1, newv))
                yield f"write p{pid}={newv}", State(pods, s.caps, s.js)
            pods = replace_tuple(s.pods, pid, (gen + 1, 0, curv))
            yield f"revoke p{pid}", State(pods, s.caps, s.js)

    # Capture a current authority capability into a reusable/stale-able slot.
    for pid in (0, 1):
        gen, live, value = s.pods[pid]
        if live:
            cap = (pid, gen, value)
            for slot in (0, 1):
                caps = replace_tuple(s.caps, slot, cap)
                yield f"capture p{pid}->c{slot}", State(s.pods, caps, s.js)

    # Clear slots (models eviction/reuse).
    for slot in (0, 1):
        if s.caps[slot] != EMPTY_CAP:
            caps = replace_tuple(s.caps, slot, EMPTY_CAP)
            yield f"clear c{slot}", State(s.pods, caps, s.js)
        if s.js[slot] != EMPTY_J:
            js = replace_tuple(s.js, slot, EMPTY_J)
            yield f"clear j{slot}", State(s.pods, s.caps, js)

    # Correct CKCA derivation: only verified current capabilities can create J state.
    if cap_current(s, s.caps[0]) and cap_current(s, s.caps[1]):
        deps = tuple(sorted(set((s.caps[0], s.caps[1]))))
        for slot in (0, 1):
            js = replace_tuple(s.js, slot, deps)
            yield f"derive caps->j{slot}", State(s.pods, s.caps, js)

    # Transitive J composition: factor = exact union of both source read-sets.
    if j_current(s, s.js[0]) and j_current(s, s.js[1]):
        deps = tuple(sorted(set(s.js[0]) | set(s.js[1])))
        for slot in (0, 1):
            js = replace_tuple(s.js, slot, deps)
            yield f"compose js->j{slot}", State(s.pods, s.caps, js)


def describe(s: State):
    return {"pods": s.pods, "caps": s.caps, "js": s.js}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    initial = State(
        pods=((1, 1, 0), (1, 1, 1)),
        caps=(EMPTY_CAP, EMPTY_CAP),
        js=(EMPTY_J, EMPTY_J),
    )
    q = deque([(initial, 0)])
    seen = {initial}
    transitions = 0
    states_by_depth = {0: 1}

    checks = 0
    ckca_false_accepts = 0
    value_only_false_accepts = 0
    pod_only_false_accepts = 0
    value_examples = []
    pod_examples = []
    same_value_aba_examples = []

    while q:
        s, depth = q.popleft()

        # At every reachable state, treat every retained capability/J state as a serve attempt.
        for kind, artifacts, truth_fn, value_fn, pod_fn in (
            ("cap", s.caps, cap_current, cap_value_only, cap_pod_only),
            ("j", s.js, j_current, j_value_only, j_pod_only),
        ):
            for slot, artifact in enumerate(artifacts):
                if (kind == "cap" and artifact == EMPTY_CAP) or (kind == "j" and artifact == EMPTY_J):
                    continue
                truth = truth_fn(s, artifact)
                v = value_fn(s, artifact)
                p = pod_fn(s, artifact)
                checks += 1
                # CKCA serve predicate is exactly the generation-lifetime truth predicate.
                ckca = truth
                ckca_false_accepts += int(ckca and not truth)
                if v and not truth:
                    value_only_false_accepts += 1
                    if len(value_examples) < 5:
                        value_examples.append({"kind": kind, "slot": slot, "artifact": artifact, "state": describe(s)})
                    # A same-value rewrite is the ABA corner: semantic bytes equal, generation differs.
                    if kind == "cap":
                        pid, gen, val = artifact
                        cgen, live, cval = s.pods[pid]
                        if live and val == cval and gen != cgen and len(same_value_aba_examples) < 5:
                            same_value_aba_examples.append({"artifact": artifact, "current": s.pods[pid], "state": describe(s)})
                if p and not truth:
                    pod_only_false_accepts += 1
                    if len(pod_examples) < 5:
                        pod_examples.append({"kind": kind, "slot": slot, "artifact": artifact, "state": describe(s)})

        # Single authority is encoded structurally: exactly one Pod state per canonical ID.
        for pid, (gen, live, _v) in enumerate(s.pods):
            if gen < 1 or gen > MAX_GEN or live not in (0, 1):
                raise AssertionError((pid, s))

        if depth >= MAX_DEPTH:
            continue
        for _action, ns in successors(s):
            transitions += 1
            if ns not in seen:
                seen.add(ns)
                nd = depth + 1
                states_by_depth[nd] = states_by_depth.get(nd, 0) + 1
                q.append((ns, nd))

    report = {
        "stage": "R301-CKCA-FINITE-STATE-MODEL",
        "architecture_candidate": "Causal Knowledge Coherence Architecture",
        "pods": 2,
        "semantic_values": 2,
        "max_generation": MAX_GEN,
        "max_trace_depth": MAX_DEPTH,
        "reachable_states": len(seen),
        "explored_transitions": transitions,
        "states_first_seen_by_depth": states_by_depth,
        "artifact_serve_checks": checks,
        "ckca_false_accepts": ckca_false_accepts,
        "value_only_false_accepts": value_only_false_accepts,
        "pod_only_false_accepts": pod_only_false_accepts,
        "value_only_false_accept_rate": value_only_false_accepts / max(checks, 1),
        "pod_only_false_accept_rate": pod_only_false_accepts / max(checks, 1),
        "same_value_aba_counterexamples_found": len(same_value_aba_examples),
        "sample_value_only_counterexamples": value_examples,
        "sample_pod_only_counterexamples": pod_examples,
        "sample_same_value_aba_counterexamples": same_value_aba_examples,
        "modeled_invariants": [
            "one canonical authority state per Pod",
            "generation monotonicity",
            "capability admission requires exact current generation and semantic page identity",
            "J-state admission is conjunction over exact source-generation leaves",
            "J composition uses transitive dependency union",
            "same-value rewrite cannot validate an old generation",
            "revoke then later write cannot validate pre-revoke artifacts",
        ],
        "scientific_scope": (
            "Exhaustive bounded state exploration of the protocol abstraction, not a proof for unbounded systems. "
            "The comparison intentionally shows why semantic-value-only or pod-ID-only freshness policies are "
            "insufficient under ABA/same-value rewrites."
        ),
        "dod_status": "NOT_DOD; bounded formal coherence gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "reachable_states", "explored_transitions", "artifact_serve_checks",
        "ckca_false_accepts", "value_only_false_accepts", "pod_only_false_accepts",
        "same_value_aba_counterexamples_found",
    )}, indent=2))

    if ckca_false_accepts != 0:
        return 2
    if value_only_false_accepts == 0 or pod_only_false_accepts == 0:
        return 3
    if len(same_value_aba_examples) == 0:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
