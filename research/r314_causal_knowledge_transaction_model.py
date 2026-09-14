from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

OUT = Path(os.environ.get("SO_R314_REPORT", "ci-r314/report.json"))
MAX_GEN = 3
MAX_DEPTH = 10
EMPTY_READ = (-1, -1, -1)
EMPTY_ARTIFACT = tuple()


@dataclass(frozen=True)
class State:
    # two Pods: (generation, semantic_value)
    pods: tuple[tuple[int, int], tuple[int, int]]
    # in-flight transaction phase: 0 idle, 1 read-A done, 2 read-B done
    phase: int
    read_a: tuple[int, int, int]
    read_b: tuple[int, int, int]
    # committed artifact dependencies, or empty
    artifact: tuple[tuple[int, int, int], ...]


def replace(xs, idx, value):
    ys = list(xs); ys[idx] = value; return tuple(ys)


def current_read(s: State, r: tuple[int, int, int]) -> bool:
    if r == EMPTY_READ:
        return False
    pid, gen, value = r
    cgen, cval = s.pods[pid]
    return gen == cgen and value == cval


def semantic_read(s: State, r: tuple[int, int, int]) -> bool:
    if r == EMPTY_READ:
        return False
    pid, _gen, value = r
    _cgen, cval = s.pods[pid]
    return value == cval


def artifact_live(s: State) -> bool:
    return bool(s.artifact) and all(current_read(s, r) for r in s.artifact)


def artifact_semantic_live(s: State) -> bool:
    return bool(s.artifact) and all(semantic_read(s, r) for r in s.artifact)


def successors(s: State):
    # World changes are always possible and generations are monotonic.
    for pid in (0, 1):
        gen, cur = s.pods[pid]
        if gen < MAX_GEN:
            for value in (0, 1):
                yield f"write p{pid}={value}", State(
                    replace(s.pods, pid, (gen + 1, value)),
                    s.phase, s.read_a, s.read_b, s.artifact,
                )

    # Start transaction / read A.
    if s.phase == 0:
        gen, val = s.pods[0]
        yield "read A", State(s.pods, 1, (0, gen, val), EMPTY_READ, s.artifact)

    # Read B later; a world update may have occurred in between.
    if s.phase == 1:
        gen, val = s.pods[1]
        yield "read B", State(s.pods, 2, s.read_a, (1, gen, val), s.artifact)

    # Correct CKCA commit: validate exact generations immediately before publication.
    if s.phase == 2:
        reads = tuple(sorted((s.read_a, s.read_b)))
        if all(current_read(s, r) for r in reads):
            yield "commit validated", State(s.pods, 0, EMPTY_READ, EMPTY_READ, reads)
        else:
            # Abort/retry J transaction, preserving B plan.
            yield "abort stale readset", State(s.pods, 0, EMPTY_READ, EMPTY_READ, s.artifact)

    # Eviction/clear artifact.
    if s.artifact:
        yield "clear artifact", State(s.pods, s.phase, s.read_a, s.read_b, EMPTY_ARTIFACT)


def naive_commit_possible(s: State) -> bool:
    return s.phase == 2


def naive_committed_artifact(s: State):
    if not naive_commit_possible(s):
        return EMPTY_ARTIFACT
    return tuple(sorted((s.read_a, s.read_b)))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    initial = State(pods=((1, 0), (1, 1)), phase=0, read_a=EMPTY_READ, read_b=EMPTY_READ, artifact=EMPTY_ARTIFACT)
    q = deque([(initial, 0)])
    seen = {initial}
    transitions = 0

    correct_mixed_generation_commits = 0
    naive_mixed_generation_commits = 0
    post_commit_stale_serves_if_no_lifetime = 0
    full_protocol_stale_serves = 0
    semantic_only_false_serves = 0
    same_value_aba_witnesses = []
    naive_mixed_examples = []
    post_commit_examples = []
    artifact_checks = 0
    transaction_commit_checks = 0

    while q:
        s, depth = q.popleft()

        # What would happen if a naive implementation committed every phase-2 read set?
        if s.phase == 2:
            transaction_commit_checks += 1
            naive = naive_committed_artifact(s)
            naive_valid = all(current_read(s, r) for r in naive)
            if not naive_valid:
                naive_mixed_generation_commits += 1
                if len(naive_mixed_examples) < 5:
                    naive_mixed_examples.append({"state": s, "naive_artifact": naive})
            # CKCA only commits if validation passes, by construction.
            if naive_valid is False and all(current_read(s, r) for r in naive):
                correct_mixed_generation_commits += 1

        if s.artifact:
            artifact_checks += 1
            exact = artifact_live(s)
            semantic = artifact_semantic_live(s)

            # A design with a commit barrier but no post-commit lifetime would serve any retained artifact.
            if not exact:
                post_commit_stale_serves_if_no_lifetime += 1
                if len(post_commit_examples) < 5:
                    post_commit_examples.append({"state": s})

            # Full protocol's serving predicate is exact generation lifetime.
            serve_full = exact
            full_protocol_stale_serves += int(serve_full and not exact)

            # Semantic-value-only lifetime is unsafe under same-value rewrite/ABA.
            if semantic and not exact:
                semantic_only_false_serves += 1
                if len(same_value_aba_witnesses) < 5:
                    same_value_aba_witnesses.append({"state": s})

        if depth >= MAX_DEPTH:
            continue
        for _action, ns in successors(s):
            transitions += 1
            if ns not in seen:
                seen.add(ns)
                q.append((ns, depth + 1))

    report = {
        "stage": "R314-CAUSAL-KNOWLEDGE-TRANSACTION-MODEL",
        "architecture_candidate": "Causal Knowledge Transaction (CKT)",
        "max_generation": MAX_GEN,
        "max_trace_depth": MAX_DEPTH,
        "reachable_states": len(seen),
        "explored_transitions": transitions,
        "transaction_commit_checks": transaction_commit_checks,
        "artifact_serve_checks": artifact_checks,
        "naive_mixed_generation_commits": naive_mixed_generation_commits,
        "validated_protocol_mixed_generation_commits": correct_mixed_generation_commits,
        "stale_serves_with_commit_barrier_but_no_post_commit_lifetime": post_commit_stale_serves_if_no_lifetime,
        "full_ckt_stale_serves": full_protocol_stale_serves,
        "semantic_value_only_false_serves": semantic_only_false_serves,
        "same_value_aba_witness_count": len(same_value_aba_witnesses),
        "sample_naive_mixed_generation_counterexamples": [str(x) for x in naive_mixed_examples],
        "sample_post_commit_stale_counterexamples": [str(x) for x in post_commit_examples],
        "sample_same_value_aba_counterexamples": [str(x) for x in same_value_aba_witnesses],
        "protocol": [
            "compile/reuse immutable B-plan",
            "read verified current generation capabilities lazily",
            "record exact dynamic generation read-set",
            "validate read-set at neural/answer commit barrier",
            "abort and retry J transaction on conflict without recompiling B-plan",
            "on successful commit, attach transitive generation lifetime factor to derived artifact",
            "serve retained artifact only while lifetime factor remains valid",
        ],
        "key_result": (
            "Commit-time validation and post-commit lifetime are complementary. The commit barrier prevents mixed/stale "
            "generations from being published while the transaction is running; lifetime coherence prevents an artifact "
            "that was valid at commit time from being served after a later source update."
        ),
        "prior_art_boundary": (
            "OCC/read-set validation and versioned derived-state invalidation are established systems techniques. R314 "
            "formalizes their required composition for CKCA neural artifacts; it is not standalone novelty proof."
        ),
        "scientific_scope": "Exhaustive bounded state model, not an unbounded formal proof.",
        "dod_status": "NOT_DOD; bounded neural-transaction protocol gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "reachable_states", "explored_transitions", "transaction_commit_checks", "artifact_serve_checks",
        "naive_mixed_generation_commits", "validated_protocol_mixed_generation_commits",
        "stale_serves_with_commit_barrier_but_no_post_commit_lifetime", "full_ckt_stale_serves",
        "semantic_value_only_false_serves", "same_value_aba_witness_count",
    )}, indent=2))

    if naive_mixed_generation_commits == 0:
        return 2
    if correct_mixed_generation_commits != 0:
        return 3
    if post_commit_stale_serves_if_no_lifetime == 0:
        return 4
    if full_protocol_stale_serves != 0:
        return 5
    if semantic_only_false_serves == 0 or len(same_value_aba_witnesses) == 0:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
