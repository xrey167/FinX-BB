"""E-000087 — marker validity contract and train/eval semantic alignment.

Decisive validity audit, NOT a novelty claim.  The historical generators call samples
"valid" without checking the already-defined mechanical validity radius.  Training uses
that label directly, while MVCC evaluation recomputes Euclidean validity.  This experiment:

1. measures the historical out-of-radius rate without changing anything;
2. installs a conditional/rejection sampler that keeps the SAME centre, Gaussian proposal,
   normalisation and valid_radius, but refuses to label an out-of-radius proposal valid;
3. checks WRITE/LINK/UPDATE/RELINK and training-bank marker labels against the mechanical rule;
4. reports proposal overhead so the correction cannot hide an impractical generator.

Old runs remain old evidence.  A corrected run must be retrained; this script never silently
reinterprets their metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import so.data as data
from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.mvcc import MVCCStore
from so.train import make_centre


def historical_valid_markers(rng: np.random.Generator, centre: np.ndarray, n: int,
                             scale: float = 0.05) -> np.ndarray:
    m = centre[None, :] + rng.normal(scale=scale, size=(n, centre.shape[0]))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def mechanical_valid(markers: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    if markers.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    return np.linalg.norm(markers.astype(float) - centre[None, :], axis=1) <= radius


def strict_valid_markers(rng: np.random.Generator, centre: np.ndarray, n: int,
                         scale: float = 0.05, valid_radius: float = 0.35,
                         max_rounds: int = 10000) -> np.ndarray:
    """Sample from the historical proposal conditioned on the existing validity predicate.

    This does not increase valid_radius or relabel rejected points.  It preserves the proposal
    distribution conditional on validity.  A pathological dimension/scale combination fails
    explicitly rather than silently producing invalid "valid" markers.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty((0, centre.shape[0]), dtype=float)
    out = np.empty((n, centre.shape[0]), dtype=float)
    remaining = np.arange(n)
    for _ in range(max_rounds):
        if remaining.size == 0:
            return out
        m = centre[None, :] + rng.normal(scale=scale, size=(remaining.size, centre.shape[0]))
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        ok = mechanical_valid(m, centre, valid_radius)
        if ok.any():
            out[remaining[ok]] = m[ok]
            remaining = remaining[~ok]
    raise RuntimeError(
        f"could not draw {remaining.size}/{n} valid markers after {max_rounds} rounds; "
        f"dim={centre.shape[0]} scale={scale} radius={valid_radius}"
    )


def strict_store_new_valid_marker(self: MVCCStore) -> np.ndarray:
    return strict_valid_markers(self.rng, self.marker_centre, 1,
                                scale=0.05, valid_radius=self.valid_radius)[0]


def install_strict_contract() -> None:
    """Patch only the run process; historical repository evidence stays immutable."""
    def training_sampler(rng, centre, n, scale=0.05):
        return strict_valid_markers(rng, centre, n, scale=scale, valid_radius=0.35)
    data.valid_markers = training_sampler
    MVCCStore.new_valid_marker = strict_store_new_valid_marker


def proposal_measure(seed: int, dim: int, draws: int, radius: float = 0.35) -> dict[str, Any]:
    centre = make_centre(seed, dim)
    rng = np.random.default_rng(870000 + seed * 1000 + dim)
    t0 = time.perf_counter_ns()
    old = historical_valid_markers(rng, centre, draws)
    historical_ns = time.perf_counter_ns() - t0
    old_valid = mechanical_valid(old, centre, radius)

    rng2 = np.random.default_rng(870000 + seed * 1000 + dim)
    t0 = time.perf_counter_ns()
    try:
        strict = strict_valid_markers(rng2, centre, draws, valid_radius=radius,
                                      max_rounds=2000)
        strict_ns = time.perf_counter_ns() - t0
        strict_valid = mechanical_valid(strict, centre, radius)
        strict_success = bool(strict_valid.all())
        strict_invalid_rate = float((~strict_valid).mean())
        overhead = float(strict_ns / max(historical_ns, 1))
    except RuntimeError:
        strict_ns = time.perf_counter_ns() - t0
        strict_success = False
        strict_invalid_rate = None
        overhead = None
    dist = np.linalg.norm(old - centre[None, :], axis=1)
    return {
        "seed": seed, "dim": dim, "draws": draws, "valid_radius": radius,
        "historical_invalid_rate": float((~old_valid).mean()),
        "historical_distance_quantiles": {
            "p50": float(np.quantile(dist, 0.50)), "p90": float(np.quantile(dist, 0.90)),
            "p95": float(np.quantile(dist, 0.95)), "p99": float(np.quantile(dist, 0.99)),
            "max": float(dist.max()),
        },
        "strict_success": strict_success,
        "strict_invalid_rate": strict_invalid_rate,
        "historical_ns": int(historical_ns), "strict_ns": int(strict_ns),
        "strict_over_historical_time": overhead,
    }


def semantic_cell(seed: int, dim: int, aliases: int = 8) -> dict[str, Any]:
    centre = make_centre(seed, dim)
    install_strict_contract()
    rng = np.random.default_rng(880000 + seed * 101 + dim)
    world, spec = E15.sample_alias_world(rng, n_base=256, n_groups=8,
                                         n_alias_per_group=aliases,
                                         n_entities=2048, n_relations=6, n_synonyms=2)
    store, kids = E15.load_arm(world, spec, centre, seed=890000 + seed, symlink=True)
    target, aa = spec.groups[0]

    def active_markers_valid() -> bool:
        for cell in store.cells.values():
            if not cell.versions or cell.active is None:
                continue
            if not store.marker_valid(cell.active.marker):
                return False
        return True

    initial_ok = active_markers_valid()
    old_obj = int(world.index[target])
    new_obj = int((old_obj + 17) % world.n_entities)
    store.update(kids[target], new_obj)
    update_ok = active_markers_valid()
    update_view = bank_from_store(store)
    update_propagates = (
        update_view.index_view.get(target) == new_obj
        and all(update_view.index_view.get(a) == new_obj for a in aa)
    )

    # A relink must also create a mechanically valid new LINK version.
    other_target, _ = spec.groups[1]
    store.relink(kids[aa[0]], kids[other_target])
    relink_ok = active_markers_valid()

    # Training generator: every row labelled marker_valid=True must satisfy the same radius.
    trng = np.random.default_rng(900000 + seed)
    train_bank = E15.bank_with_links(trng, world, spec, centre,
                                    p_revoked=0.20, p_shred=0.10, p_stale=0.05,
                                    p_dangling=0.05)
    physical = mechanical_valid(train_bank.marker, centre, 0.35)
    labelled = np.asarray(train_bank.marker_valid, dtype=bool)
    label_false_positive = int(np.logical_and(labelled, ~physical).sum())
    label_false_negative = int(np.logical_and(~labelled, physical).sum())

    # SHRED remains deliberately invalid and must not be "fixed" into validity.
    fresh_store, fresh_kids = E15.load_arm(world, spec, centre, seed=910000 + seed, symlink=True)
    fresh_store.shred(fresh_kids[target])
    shredded_marker_is_invalid = not fresh_store.marker_valid(
        fresh_store.cells[fresh_kids[target]].version_obj(
            fresh_store.cells[fresh_kids[target]].active_version).marker
    )

    checks = {
        "initial_active_versions_mechanically_valid": initial_ok,
        "update_version_mechanically_valid": update_ok,
        "single_target_update_propagates_all_aliases": update_propagates,
        "relink_version_mechanically_valid": relink_ok,
        "training_valid_labels_have_no_physical_false_positive": label_false_positive == 0,
        "training_invalid_labels_have_no_physical_false_negative": label_false_negative == 0,
        "shred_still_mechanically_invalid": shredded_marker_is_invalid,
    }
    return {
        "seed": seed, "dim": dim, "aliases": aliases,
        "label_false_positive": label_false_positive,
        "label_false_negative": label_false_negative,
        "checks": checks, "pass": all(checks.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--dims", type=int, nargs="*", default=[8, 16, 32, 64, 128])
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    proposal = [proposal_measure(s, d, a.draws) for d in a.dims for s in a.seeds]
    # Strict lifecycle contract is expected to be practical at the dimensions actually used in
    # current models/tests.  128D is retained as a historical failure-rate measurement only.
    semantics = [semantic_cell(s, d) for d in (16, 32, 64) for s in a.seeds]
    marker16 = [r["historical_invalid_rate"] for r in proposal if r["dim"] == 16]
    marker32 = [r["historical_invalid_rate"] for r in proposal if r["dim"] == 32]
    marker64 = [r["historical_invalid_rate"] for r in proposal if r["dim"] == 64]
    rec = {
        "experiment": "E-000087",
        "title": "marker validity train/eval contract",
        "result": "contract_fixed" if all(r["pass"] for r in semantics) else "correction_failed",
        "proposal_measurements": proposal,
        "semantic_cells": semantics,
        "summary": {
            "historical_invalid_rate_mean_dim16": float(np.mean(marker16)),
            "historical_invalid_rate_mean_dim32": float(np.mean(marker32)),
            "historical_invalid_rate_mean_dim64": float(np.mean(marker64)),
            "strict_semantic_cells_all_pass": all(r["pass"] for r in semantics),
        },
        "breakthrough": False,
        "classification": "validity/evaluation correction; no novelty credit",
        "important_boundary": (
            "AdapterConfig marker_dim is 16 in E-000077, so this result does NOT explain the 0.94 seed. "
            "The old E-000077 all-seed gate remains failed until independently rerun; no historical metric is upgraded."
        ),
        "unchanged_semantics": "valid_radius remains 0.35; invalid markers remain invalid; only out-of-radius proposals stop being labelled valid",
    }
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "e000087_marker_validity_contract.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["summary"]["strict_semantic_cells_all_pass"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
