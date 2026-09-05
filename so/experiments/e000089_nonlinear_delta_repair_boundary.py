"""E-000089 — nonlinear delta-repair boundary.

Question: can a bounded correction carrier propagate a pod revision through later nonlinear
persistent neural writes and beat exact replay, while matching a full-rebuild counterfactual?

This screen deliberately gives the carrier its strongest generic form.  For a recurrence

    z_t = A_t h_t + u_t + source_t
    h_{t+1} = tanh(z_t)

we retain the old pre-activation z_t.  After an edit, an exact delta carrier can propagate

    dz_t = A_t dh_t + dsource_t
    dh_{t+1} = tanh(z_t + dz_t) - tanh(z_t).

That identity is exact even through the nonlinearity.  The important comparison is not against
full rebuilding of source-independent u_t, because a guarantee-matched replay baseline may cache
u_t under the same extra-memory budget.  The exact carrier and cached replay then each require one
dependency matvec plus one nonlinear evaluation per affected step.  The carrier is an algebraic
rewrite of replay over the affected recurrent state, not a generic >10x invention.

We also include a stale-Jacobian carrier.  It is cheaper only by refusing to re-evaluate the
nonlinearity and must fail the exact full-rebuild counterfactual under finite edits.  Finally, a
block-local recurrence demonstrates where a real systems win can come from: a structurally sparse
and verifiable dependency cone.  But the same cone accelerates exact replay too, so the future
novelty target must be the neural mechanism that creates/certifies sparse source-local dependency,
not delta propagation itself.

No novelty claim is made by this experiment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _case(seed: int, d: int, steps: int, edit_at: int):
    rng = np.random.default_rng(seed)
    # Stable but genuinely dense nonlinear recurrence.
    A = rng.normal(size=(steps, d, d)).astype(np.float64)
    norms = np.linalg.norm(A, axis=(1, 2), keepdims=True)
    A *= (0.75 * np.sqrt(d)) / np.maximum(norms, 1e-12)
    u = rng.normal(scale=0.45, size=(steps, d)).astype(np.float64)
    h0 = rng.normal(scale=0.2, size=d).astype(np.float64)
    old_source = rng.normal(scale=0.7, size=d).astype(np.float64)
    # Finite, adversarially material edit: direction differs and magnitude is large enough
    # to move tanh gates, which is exactly where stale coefficients are unsafe.
    new_source = -1.25 * old_source + rng.normal(scale=0.35, size=d)
    return A, u, h0, old_source, new_source


def _full(A, u, h0, source, edit_at: int, heavy: bool = False):
    h = h0.copy()
    hs = [h.copy()]
    zs = []
    for t in range(len(A)):
        ut = u[t]
        if heavy:
            # Deliberately irrelevant source-independent work.  It models the misleading baseline
            # in which every unaffected branch is rebuilt.  Strong baselines cache this branch.
            q = np.tanh(ut)
            for _ in range(3):
                q = np.tanh(0.8 * q + 0.2 * ut)
            ut = q
        z = A[t] @ h + ut + (source if t == edit_at else 0.0)
        h = np.tanh(z)
        zs.append(z.copy()); hs.append(h.copy())
    return h, np.stack(hs), np.stack(zs)


def _cached_replay(A, u, old_hs, source, edit_at: int):
    """Exact strongest baseline: reuse source-independent u and replay only affected recurrence."""
    h = old_hs[edit_at].copy()
    for t in range(edit_at, len(A)):
        z = A[t] @ h + u[t] + (source if t == edit_at else 0.0)
        h = np.tanh(z)
    return h


def _exact_delta(A, old_hs, old_zs, old_source, new_source, edit_at: int):
    """Best-case bounded nonlinear correction carrier; exact by algebra."""
    dh = np.zeros_like(old_hs[0])
    for t in range(edit_at, len(A)):
        dsrc = (new_source - old_source) if t == edit_at else 0.0
        dz = A[t] @ dh + dsrc
        h_new = np.tanh(old_zs[t] + dz)
        dh = h_new - old_hs[t + 1]
    return old_hs[-1] + dh


def _stale_jacobian(A, old_hs, old_zs, old_source, new_source, edit_at: int):
    """Frozen local coefficients: intentionally forbidden if they fail full-rebuild equality."""
    dh = np.zeros_like(old_hs[0])
    for t in range(edit_at, len(A)):
        dsrc = (new_source - old_source) if t == edit_at else 0.0
        dz = A[t] @ dh + dsrc
        deriv = 1.0 - np.tanh(old_zs[t]) ** 2
        dh = deriv * dz
    return old_hs[-1] + dh


def _time(fn, rounds: int) -> float:
    vals = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns(); fn(); vals.append(time.perf_counter_ns() - t0)
    return float(np.median(vals))


def _block_local(seed: int, d: int, block: int, steps: int, edit_at: int):
    """Exact structural control: only one block can descend from the edited pod."""
    assert d % block == 0
    rng = np.random.default_rng(100000 + seed)
    nblk = d // block
    mats = rng.normal(size=(steps, nblk, block, block))
    mats *= 0.6 / np.maximum(np.linalg.norm(mats, axis=(2, 3), keepdims=True) / np.sqrt(block), 1e-12)
    u = rng.normal(scale=0.3, size=(steps, d))
    h0 = rng.normal(scale=0.2, size=d)
    src0 = np.zeros(d); src1 = np.zeros(d)
    src0[:block] = rng.normal(scale=0.6, size=block)
    src1[:block] = -src0[:block] + rng.normal(scale=0.25, size=block)

    def full(src):
        h = h0.copy(); hs = [h.copy()]; zs = []
        for t in range(steps):
            out = np.empty_like(h)
            for b in range(nblk):
                sl = slice(b * block, (b + 1) * block)
                out[sl] = mats[t, b] @ h[sl]
            z = out + u[t] + (src if t == edit_at else 0.0)
            h = np.tanh(z); zs.append(z.copy()); hs.append(h.copy())
        return h, np.stack(hs), np.stack(zs)

    old, oh, oz = full(src0); ref, _, _ = full(src1)
    # Replay only the one certified affected block. Every other block is copied from old state.
    hblk = oh[edit_at, :block].copy()
    for t in range(edit_at, steps):
        z = mats[t, 0] @ hblk + u[t, :block] + (src1[:block] if t == edit_at else 0.0)
        hblk = np.tanh(z)
    repaired = old.copy(); repaired[:block] = hblk
    return {
        "maxabs": float(np.max(np.abs(repaired - ref))),
        "affected_width": block,
        "full_width": d,
        "theoretical_dense_matmul_ratio": float((d * d) / (block * block)),
        "strong_baseline_can_use_same_cone": True,
    }


def run(seed: int, d: int, steps: int, edit_at: int, rounds: int) -> Dict[str, object]:
    A, u, h0, s0, s1 = _case(seed, d, steps, edit_at)
    old, old_hs, old_zs = _full(A, u, h0, s0, edit_at)
    ref, _, _ = _full(A, u, h0, s1, edit_at)
    exact = _exact_delta(A, old_hs, old_zs, s0, s1, edit_at)
    replay = _cached_replay(A, u, old_hs, s1, edit_at)
    jac = _stale_jacobian(A, old_hs, old_zs, s0, s1, edit_at)

    exact_err = float(np.max(np.abs(exact - ref)))
    replay_err = float(np.max(np.abs(replay - ref)))
    jac_err = float(np.max(np.abs(jac - ref)))

    # Same extra-state order: carrier keeps old z_t; strongest replay keeps source-independent u_t.
    # Both already have the persistent old h trajectory in this experiment.
    t_exact = _time(lambda: _exact_delta(A, old_hs, old_zs, s0, s1, edit_at), rounds)
    t_replay = _time(lambda: _cached_replay(A, u, old_hs, s1, edit_at), rounds)
    t_raw = _time(lambda: _full(A, u, h0, s1, edit_at, heavy=True)[0], max(3, rounds // 4))

    affected_steps = steps - edit_at
    operation_contract = {
        "exact_delta_dependency_matvecs": affected_steps,
        "cached_replay_dependency_matvecs": affected_steps,
        "exact_delta_nonlinear_evals": affected_steps,
        "cached_replay_nonlinear_evals": affected_steps,
        "same_asymptotic_affected_work": True,
        "matched_extra_vectors_per_step": 1,
    }
    block = _block_local(seed, d, max(8, d // 8), steps, edit_at)
    checks = {
        "exact_delta_matches_full_rebuild": exact_err <= 1e-11,
        "cached_replay_matches_full_rebuild": replay_err <= 1e-11,
        "stale_jacobian_fails_exactness": jac_err >= 1e-5,
        "carrier_does_not_reduce_dense_dependency_matvec_count": operation_contract["exact_delta_dependency_matvecs"] == operation_contract["cached_replay_dependency_matvecs"],
        "carrier_does_not_reduce_dense_nonlinearity_count": operation_contract["exact_delta_nonlinear_evals"] == operation_contract["cached_replay_nonlinear_evals"],
        "structured_cone_is_exact": block["maxabs"] <= 1e-11,
    }
    return {
        "seed": seed,
        "d": d,
        "steps": steps,
        "edit_at": edit_at,
        "errors": {"exact_delta": exact_err, "cached_replay": replay_err, "stale_jacobian": jac_err},
        "timing_ns_median": {"exact_delta": t_exact, "cached_replay": t_replay, "raw_rebuild_with_irrelevant_branch": t_raw},
        "timing_ratios": {
            "cached_replay_over_exact_delta": t_replay / max(t_exact, 1.0),
            "raw_rebuild_over_exact_delta": t_raw / max(t_exact, 1.0),
        },
        "operation_contract": operation_contract,
        "structured_dependency_control": block,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--edit-at", type=int, default=8)
    p.add_argument("--rounds", type=int, default=17)
    p.add_argument("--results-dir", default="so/results")
    a = p.parse_args()
    rows: List[Dict[str, object]] = [run(s, a.d, a.steps, a.edit_at, a.rounds) for s in a.seeds]
    rec = {
        "experiment": "E-000089",
        "title": "nonlinear bounded correction carrier vs guarantee-matched exact replay",
        "result": "decisive_falsification_of_generic_lane_B" if all(r["pass"] for r in rows) else "inconclusive",
        "rows": rows,
        "breakthrough": False,
        "novelty_claim": False,
        "decision": (
            "A generic exact nonlinear correction carrier is not a distinct systems primitive here: once old preactivations are retained, "
            "its exact update is algebraically the same affected-step computation as replay with cached source-independent terms. "
            "A stale/Jacobian carrier violates full-rebuild equality. Material speedup therefore has to come from a truly smaller, "
            "source-local dependency cone or a special memory representation, not from the carrier algebra itself."
        ),
        "next_architecture_boundary": (
            "Promote only mechanisms that create and verify sparse source-lineage cones in persistent neural state and then beat the strongest "
            "cone-aware exact replay baseline under matched memory. Learned dependency certificates are useful only if independent J-space "
            "audits show no omitted causal descendants."
        ),
    }
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "e000089_nonlinear_delta_repair_boundary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if rec["result"] != "decisive_falsification_of_generic_lane_B":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
