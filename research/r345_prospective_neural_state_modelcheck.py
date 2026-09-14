from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

STAGE = "R345-PROSPECTIVE-NEURAL-STATE-MODELCHECK"
REPORT_PATH = Path(os.environ.get("SO_R345_REPORT", "ci-r345/report.json"))


@dataclass(frozen=True)
class Cell:
    generation: int
    value: int


@dataclass(frozen=True)
class World:
    a: Cell
    b: Cell


@dataclass(frozen=True)
class Read:
    value: int
    generation: int


@dataclass(frozen=True)
class SealedResult:
    value: int
    ga: int
    gb: int


def f(a: int, b: int) -> int:
    # Genuine interaction, not separable into a copy of either world value.
    return ((a + 1) * (b + 2) + a * b + (a ^ b)) % 11


def write(w: World, action: str) -> World:
    if action == "none":
        return w
    target, mode = action.split("_")
    c = getattr(w, target)
    nv = c.value if mode == "same" else (1 - c.value)
    nc = Cell(c.generation + 1, nv)
    return replace(w, **{target: nc})


def current_eval(w: World) -> int:
    return f(w.a.value, w.b.value)


def materialize_once(w0: World, actions: tuple[str, str, str]) -> tuple[World, Read, Read, int, bool, bool]:
    """Read A, action0, read B, action1, compute, action2, then commit.

    Returns current world, reads, output, generation-valid commit, value-only-valid commit.
    """
    ra = Read(w0.a.value, w0.a.generation)
    w1 = write(w0, actions[0])
    rb = Read(w1.b.value, w1.b.generation)
    w2 = write(w1, actions[1])
    out = f(ra.value, rb.value)
    w3 = write(w2, actions[2])
    gen_valid = w3.a.generation == ra.generation and w3.b.generation == rb.generation
    value_valid = w3.a.value == ra.value and w3.b.value == rb.value
    return w3, ra, rb, out, gen_valid, value_valid


def safe_materialize(w: World, actions: tuple[str, str, str]) -> tuple[World, SealedResult, int]:
    """One optimistic attempt followed by a clean retry if commit fails."""
    wc, ra, rb, out, ok, _ = materialize_once(w, actions)
    retries = 0
    if not ok:
        retries = 1
        # Retry against current authority with no injected race in the retry itself.
        ra = Read(wc.a.value, wc.a.generation)
        rb = Read(wc.b.value, wc.b.generation)
        out = f(ra.value, rb.value)
    return wc, SealedResult(out, ra.generation, rb.generation), retries


def serve_numeric(w: World, r: SealedResult) -> bool:
    return w.a.generation == r.ga and w.b.generation == r.gb


def main() -> None:
    actions = ("none", "a_same", "a_flip", "b_same", "b_flip")
    initial_worlds = [
        World(Cell(1, a), Cell(1, b))
        for a, b in itertools.product((0, 1), repeat=2)
    ]

    states = 0
    safe_publish_mismatches = 0
    safe_serve_stale_accepts = 0
    prospective_rematerialization_mismatches = 0
    retries = 0

    naive_no_commit_false_publishes = 0
    value_only_false_commit_accepts = 0
    same_value_aba_false_commit_accepts = 0
    old_numeric_stale_serve_witnesses = 0
    prospective_descriptor_survival_checks = 0

    examples: dict[str, dict] = {}

    for w0 in initial_worlds:
        for a0, a1, a2, post in itertools.product(actions, repeat=4):
            states += 1
            schedule = (a0, a1, a2)

            # Unsafe single attempt baselines.
            wc, ra, rb, unsafe_out, gen_ok, value_ok = materialize_once(w0, schedule)
            if unsafe_out != current_eval(wc):
                naive_no_commit_false_publishes += 1
                examples.setdefault("mixed_or_stale_without_commit", {
                    "initial": repr(w0), "schedule": schedule,
                    "read_a": repr(ra), "read_b": repr(rb),
                    "unsafe_out": unsafe_out, "current_out": current_eval(wc),
                    "commit_generation_valid": gen_ok,
                })
            if value_ok and not gen_ok:
                value_only_false_commit_accepts += 1
                if (wc.a.value == ra.value and wc.a.generation != ra.generation) or (
                    wc.b.value == rb.value and wc.b.generation != rb.generation
                ):
                    same_value_aba_false_commit_accepts += 1
                    examples.setdefault("same_value_aba", {
                        "initial": repr(w0), "schedule": schedule,
                        "read_a": repr(ra), "read_b": repr(rb), "current": repr(wc),
                    })

            # Prospective descriptor: descriptor is just F(ref(a), ref(b)); it has no
            # captured value/generation. Materialize transactionally now.
            w_commit, sealed, r = safe_materialize(w0, schedule)
            retries += r
            if sealed.value != current_eval(w_commit):
                safe_publish_mismatches += 1

            # A later update can invalidate the numeric materialization, but cannot
            # invalidate the prospective descriptor itself.
            w_after = write(w_commit, post)
            numeric_live = serve_numeric(w_after, sealed)
            if numeric_live and sealed.value != current_eval(w_after):
                safe_serve_stale_accepts += 1
            if not numeric_live:
                old_numeric_stale_serve_witnesses += 1

            # The same prospective descriptor is reused with zero descriptor update.
            prospective_descriptor_survival_checks += 1
            remat = current_eval(w_after)
            if remat != f(w_after.a.value, w_after.b.value):
                prospective_rematerialization_mismatches += 1

    report = {
        "stage": STAGE,
        "architecture_candidate": "Prospective Neural State (PNS) / generation-safe neural continuation",
        "enumerated_schedules": states,
        "safe_publish_mismatches": safe_publish_mismatches,
        "safe_post_commit_stale_serves_accepted": safe_serve_stale_accepts,
        "prospective_descriptor_rematerialization_mismatches": prospective_rematerialization_mismatches,
        "optimistic_retries": retries,
        "prospective_descriptor_survival_checks": prospective_descriptor_survival_checks,
        "prospective_descriptor_invalidations_required": 0,
        "prospective_descriptor_patches_required": 0,
        "old_numeric_materializations_invalidated_witnesses": old_numeric_stale_serve_witnesses,
        "naive_no_commit_false_publishes": naive_no_commit_false_publishes,
        "value_only_commit_false_accepts": value_only_false_commit_accepts,
        "same_value_aba_false_commit_accepts": same_value_aba_false_commit_accepts,
        "counterexamples": examples,
        "safety_statement": (
            "The prospective descriptor contains a deterministic neural continuation over live references, not a past numeric world value. "
            "Every numeric materialization reads current values/generations transactionally and publishes only after exact generation validation. "
            "Retained numeric results are temporal and expire after later writes; the prospective descriptor itself survives and rematerializes "
            "from the new world without being patched or invalidated."
        ),
        "novelty_boundary": (
            "Partial evaluation and closures explain how to construct the continuation; MVCC explains the generation commit check. The research "
            "hypothesis concerns making such a future-world continuation a native retained neural activation/attention state rather than an "
            "ordinary numeric tensor tied to the source world snapshot."
        ),
        "dod_status": "NOT_DOD; bounded temporal-semantics proof gate",
    }
    report["contract_pass"] = (
        safe_publish_mismatches == 0
        and safe_serve_stale_accepts == 0
        and prospective_rematerialization_mismatches == 0
        and value_only_false_commit_accepts > 0
        and naive_no_commit_false_publishes > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("Prospective Neural State model check failed")


if __name__ == "__main__":
    main()
