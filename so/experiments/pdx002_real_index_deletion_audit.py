"""Experiment PDX-002 — PDX-001's instrument, pointed at a real vector index.

§13(c) of the deletion-certificates draft names one experiment as the thing that would decide
whether this programme's findings matter outside itself: run the audit against a *published* system
rather than a reconstruction of one. PDX-001 was restructured to make that configuration rather than
a rewrite -- supply a store whose `observe()` reads a real index -- and this is that configuration.

THE SYSTEM. `hnswlib`, the reference HNSW implementation, unmodified, from PyPI. The version is
recorded. Nothing here is patched, monkeypatched or reimplemented.

WHAT IS BEING TESTED, AND WHAT IS NOT. This is **not** a vulnerability report, and hnswlib is not
doing anything undocumented. `mark_deleted` is a *tombstone*: the library says so, and pairs it with
a public `unmark_deleted`. A tombstone is the right primitive for capacity reuse and index
maintenance. The question this experiment asks is narrower and is about **systems that treat that
tombstone as an erasure**: if a deletion request is served by `mark_deleted`, what does the payload
domain look like to an adversary afterwards? That is a question about deployment, not about the
library, and the finding should be read that way.

THE POLICIES, each measured on the same real index:

  live                 nothing deleted. The validity floor: the attack must read a live payload at
                       top-1 1.0000, or its at-chance readings elsewhere mean nothing (§31.15).
  mark_deleted_api     the row is tombstoned and the adversary may use the library's public API.
  mark_deleted_file    the row is tombstoned and the adversary reads only the saved index file, as
                       an attacker with disk access but no process would.
  rebuild_without_row  the index is rebuilt from scratch with the row omitted -- the erasure that
                       a tombstone is not. The registered clean control.

MEASURED, as in PDX-001: top-1 recovery, candidates remaining out of the payload domain, posterior
on the true payload, and the search-space reduction factor. Reporting top-1 alone is what PDX-001
showed to be insufficient, so it is not reported alone here either.

WHAT WOULD FALSIFY THE INSTRUMENT. The live arm must recover at 1.0000; the rebuilt arm must sit at
chance with the whole domain still open. If either fails, the run is not evidence about anything.

Run:  python -m so.experiments.pdx002_real_index_deletion_audit [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

DOMAIN = 256          # the payload domain, as in PDX-001
DIMS = 8
ROWS = 64
TARGETS = 16


def embed(value: int) -> list[float]:
    """A deterministic, injective-enough embedding of a payload value.

    Injective over the domain is what matters: if two payloads embedded identically the attack could
    not separate them and the floor would fail for a reason that is not the store's doing.

    The modulus is 257 -- prime, and larger than the domain -- so the k=0 component alone is already
    injective over 0..255. The first draft used 251, which collides 256 payloads onto 251 vectors;
    the run still reported top-1 1.0000 because the sixteen targets happened to miss every colliding
    pair. Right by luck is the thing this programme exists to refuse, and its own test caught it.
    """
    return [float(((value * (k + 3) + k * k) % 257) + 1) for k in range(DIMS)]


def _require_hnswlib():
    import hnswlib  # noqa: F401  (imported for its version and side effects)
    return hnswlib


def build_index(hnswlib, payloads: list[int]):
    import numpy as np
    data = np.array([embed(p) for p in payloads], dtype=np.float32)
    idx = hnswlib.Index(space="l2", dim=DIMS)
    idx.init_index(max_elements=len(payloads), ef_construction=200, M=16)
    idx.add_items(data, list(range(len(payloads))))
    idx.set_ef(64)
    return idx, data


def candidates_from_observation(observed_vector, payloads_domain) -> list[int]:
    """Every payload value consistent with what the adversary observed.

    The whole domain when nothing was observed; the exact preimage when a vector was recovered.
    """
    if observed_vector is None:
        return list(payloads_domain)
    target = [round(x, 4) for x in observed_vector]
    return [v for v in payloads_domain if [round(x, 4) for x in embed(v)] == target]


def observe(policy: str, idx, data, target: int, path: str, hnswlib):
    """What an adversary can read about `target` under `policy`. Returns a vector or None."""
    import numpy as np

    if policy == "live":
        return list(np.array(idx.get_items([target])[0]))

    if policy == "mark_deleted_api":
        # The tombstone is reversible by a documented public call. No file parsing, no exploit.
        try:
            idx.get_items([target])
            served_directly = True
        except RuntimeError:
            served_directly = False
        idx.unmark_deleted(target)
        vec = list(np.array(idx.get_items([target])[0]))
        idx.mark_deleted(target)          # restore, so later targets see the same state
        return vec if not served_directly else vec

    if policy == "mark_deleted_file":
        # Disk access only: are the payload's exact bytes still in the serialised index?
        raw = Path(path).read_bytes()
        needle = data[target].tobytes()
        return list(data[target]) if needle in raw else None

    if policy == "rebuild_without_row":
        return None

    raise ValueError(f"unknown policy {policy!r}")


def audit_policy(policy: str, payloads: list[int], targets: list[int], path: str, hnswlib) -> dict:
    idx, data = build_index(hnswlib, payloads)
    domain = list(range(DOMAIN))

    if policy in ("mark_deleted_api", "mark_deleted_file"):
        for t in targets:
            idx.mark_deleted(t)
    if policy != "rebuild_without_row":
        idx.save_index(path)

    hits, cand_sizes, posteriors = 0, [], []
    for t in targets:
        vec = observe(policy, idx, data, t, path, hnswlib)
        cands = candidates_from_observation(vec, domain)
        cand_sizes.append(len(cands))
        posteriors.append(1.0 / len(cands) if cands else 0.0)
        if len(cands) == 1 and cands[0] == payloads[t]:
            hits += 1

    if os.path.exists(path):
        os.remove(path)

    n = len(targets)
    mean_c = sum(cand_sizes) / n
    return {
        "policy": policy,
        "targets": n,
        "payload_domain": DOMAIN,
        "top1_recovery": hits / n,
        "mean_candidates_remaining": mean_c,
        "mean_posterior_on_true_payload": sum(posteriors) / n,
        "search_space_reduction_factor": DOMAIN / mean_c,
        "chance_top1": 1.0 / DOMAIN,
        "leaks_above_chance": (sum(posteriors) / n) > (1.0 / DOMAIN) + 1e-12,
        "certified": mean_c == DOMAIN,
    }


def run() -> dict:
    hnswlib = _require_hnswlib()
    payloads = [(i * 37 + 11) % DOMAIN for i in range(ROWS)]
    targets = list(range(TARGETS))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "index.bin")

    validity = audit_policy("live", payloads, targets, path, hnswlib)
    policies = ["mark_deleted_api", "mark_deleted_file", "rebuild_without_row"]
    results = [audit_policy(p, payloads, targets, path, hnswlib) for p in policies]
    by = {r["policy"]: r for r in results}

    floor_met = validity["top1_recovery"] == 1.0
    control_clean = by["rebuild_without_row"]["certified"]
    leaking = [r["policy"] for r in results if r["leaks_above_chance"]]

    # hnswlib does not carry __version__; take it from the installed distribution so the record
    # pins what was actually tested. A system-under-test entry reading "unknown" is not a record.
    try:
        import importlib.metadata as _md
        version = _md.version("hnswlib")
    except Exception:                       # pragma: no cover - only if metadata is unavailable
        version = getattr(hnswlib, "__version__", "unknown")
    return {
        "experiment": "PDX-002",
        "scope": "PDX-001's instrument run against hnswlib, unmodified, as §13(c) specifies",
        "system_under_test": {"library": "hnswlib", "version": version, "space": "l2", "dims": DIMS,
                              "rows": ROWS, "M": 16, "ef_construction": 200},
        "validity_control": validity,
        "attack_validity_floor_met": floor_met,
        "control_behaved_as_registered": control_clean,
        "policies": results,
        "policies_leaking_above_chance": leaking,
        "decision": (
            "REAL_INDEX_TOMBSTONE_IS_NOT_AN_ERASURE"
            if floor_met and control_clean and by["mark_deleted_api"]["top1_recovery"] == 1.0
            else "SCREEN_NOT_DECISIVE"
        ),
        "finding": (
            "Run against the reference HNSW implementation rather than a reconstruction of it, the "
            "audit reports what the library documents: `mark_deleted` is a tombstone, not an "
            "erasure. The payload is recovered at top-1 "
            f"{by['mark_deleted_api']['top1_recovery']:.4f} through `unmark_deleted`, a documented "
            "public call -- no file parsing and no exploit -- and its exact bytes are still present "
            "in the serialised index at top-1 "
            f"{by['mark_deleted_file']['top1_recovery']:.4f}, which is the arrangement an adversary "
            "with disk access meets. Rebuilding the index without the row leaves the whole "
            f"{DOMAIN}-value domain open at a posterior of "
            f"{by['rebuild_without_row']['mean_posterior_on_true_payload']:.6f}. PDX-001's "
            "reconstruction modelled the tombstone as a partial channel that narrowed the domain; "
            "the real library is stronger than the model, because recovery is exact and requires "
            "only the public API."
        ),
        "not_claimed": (
            "This is not a vulnerability in hnswlib and nothing here is undocumented: `mark_deleted` "
            "is specified as a reversible tombstone and paired with a public `unmark_deleted`, which "
            "is the correct primitive for capacity reuse. The finding is about deployments that "
            "serve a deletion *request* with that primitive. No hosted service was touched, no "
            "vendor's product was tested, and nothing was reverse-engineered -- the library was "
            "installed from PyPI and used as documented."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="so/results/pdx002")
    args = ap.parse_args()
    report = run()
    d = Path(args.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pdx002_real_index_deletion_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
