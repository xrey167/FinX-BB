"""E-000094: exact noninterference implies late-bound mutable-state factorization.

This is an architecture-level reduction/regression witness, not a novelty claim and
not a performance lower bound.  It deliberately tests finite deterministic operator
families so exact decision/output equality can be exhaustively enumerated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


POD_STATES = (-2, -1, 0, 1, 2, 3)  # revoked/deleted-like sentinels + generations
TRACES: tuple[tuple[int, ...], ...] = (
    (0, 0),          # no mutation
    (0, 1),          # update
    (1, 2, 3),       # repeated updates
    (2, -1),         # revoke-like
    (2, -2),         # delete/shred-like
    (0, -1, 2),      # revoke -> restore new generation
    (0, 1, 0),       # ABA/authority return
    (2, 3, 2),       # rollback
    (0, -2, 3, -1),  # delete/restore/revoke sequence
)


def _u(seed: int, *items: int, mod: int = 2**31 - 1) -> int:
    payload = ":".join(str(int(v)) for v in (seed, *items)).encode()
    raw = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(raw, "little") % mod


def static_state(seed: int, x: int) -> tuple[int, int, int, int]:
    """Large reusable state.  By construction it has no Pod-state argument."""
    return tuple(_u(seed + 11 * j, x, mod=997) for j in range(4))


def contaminated_state(seed: int, x: int, pod: int) -> tuple[int, int, int, int]:
    """Negative control: a purported reusable state that actually reads Pod state."""
    base = static_state(seed, x)
    return tuple((v + _u(seed + 101 * j, pod, mod=997)) % 997 for j, v in enumerate(base))


def mutable_substate(seed: int, reusable: Sequence[int], pod: int) -> tuple[int, int, int]:
    """Example cached state that genuinely depends on the mutable Pod."""
    s = sum(int(v) for v in reusable)
    return tuple(_u(seed + 211 * j, s, pod, mod=4093) for j in range(3))


def nonlinear_read(seed: int, reusable: Sequence[int], pod: int, query: int) -> tuple[int, int]:
    """Arbitrary nonlinear late-bound read/output operator G(F(x), p, q)."""
    a, b, c, d = (int(v) for v in reusable)
    # Nonlinear integer arithmetic intentionally includes interactions between static
    # state, current mutable state and query.  Exact form is not important.
    y0 = (a * (pod + 7) + b * (query + 5) + c * c + (pod + 3) * (query + 11) * d) % 65521
    y1 = (_u(seed + 701, a, b, c, d, pod, query, mod=65521) + y0 * y0) % 65521
    return y0, y1


def firewall_candidate(seed: int, x: int, pod: int, query: int) -> tuple[int, int]:
    reusable = static_state(seed, x)
    return nonlinear_read(seed, reusable, pod, query)


def late_bound_sidecar(seed: int, x: int, current_pod: int, query: int) -> tuple[int, int]:
    reusable = static_state(seed, x)
    return nonlinear_read(seed, reusable, current_pod, query)


def run(seeds: Iterable[int] = range(32), n_x: int = 12, n_q: int = 10) -> dict:
    equality_cases = 0
    equality_mismatches = 0
    trace_cases = 0
    trace_mismatches = 0

    contamination_pairs = 0
    contamination_detected = 0

    mutable_transition_cases = 0
    mutable_changed = 0

    for seed in seeds:
        seed = int(seed)
        for x in range(n_x):
            # Negative control: exact noninterference must fail somewhere for the
            # contaminated construction.
            states = [contaminated_state(seed, x, p) for p in POD_STATES]
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    contamination_pairs += 1
                    if states[i] != states[j]:
                        contamination_detected += 1

            r = static_state(seed, x)
            for p0 in POD_STATES:
                for p1 in POD_STATES:
                    if p0 == p1:
                        continue
                    mutable_transition_cases += 1
                    if mutable_substate(seed, r, p0) != mutable_substate(seed, r, p1):
                        mutable_changed += 1

            for pod in POD_STATES:
                for q in range(n_q):
                    equality_cases += 1
                    if firewall_candidate(seed, x, pod, q) != late_bound_sidecar(seed, x, pod, q):
                        equality_mismatches += 1

            for trace in TRACES:
                for q in range(n_q):
                    for pod in trace:
                        trace_cases += 1
                        if firewall_candidate(seed, x, pod, q) != late_bound_sidecar(seed, x, pod, q):
                            trace_mismatches += 1

    pass_reduction = (
        equality_mismatches == 0
        and trace_mismatches == 0
        and contamination_detected == contamination_pairs
        and mutable_changed == mutable_transition_cases
    )

    return {
        "experiment": "E-000094",
        "scope": "exact noninterference / late-binding factorization",
        "equality_cases": equality_cases,
        "equality_mismatches": equality_mismatches,
        "lifecycle_trace_cases": trace_cases,
        "lifecycle_trace_mismatches": trace_mismatches,
        "negative_control_pairs": contamination_pairs,
        "negative_control_detected": contamination_detected,
        "mutable_transition_cases": mutable_transition_cases,
        "mutable_substate_changed": mutable_changed,
        "kill_screen_pass": pass_reduction,
        "decision": (
            "KILL_NONINTERFERENCE_ALONE_AS_NOVELTY_SEAM"
            if pass_reduction
            else "REDUCTION_ASSAY_FAILED"
        ),
        "not_claimed": (
            "No latency lower bound; no claim against active state transforms, exact affected-work "
            "compression, or causal-lineage discovery."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-count", type=int, default=32)
    ap.add_argument("--contexts", type=int, default=12)
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rec = run(range(a.seed_count), a.contexts, a.queries)
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "e000094_noninterference_late_binding_reduction.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
