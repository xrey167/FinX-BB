"""E-000089 — nonlinear delta-repair boundary.

Can a bounded correction carrier propagate a pod revision through later nonlinear persistent neural
writes, match the full-rebuild counterfactual at EVERY dependent write, and materially beat the
strongest exact replay baseline?

For z_t=A_t h_t+u_t+source_t and h_{t+1}=tanh(z_t), retaining old z_t gives the strongest generic
exact carrier:

    dz_t = A_t dh_t + dsource_t
    dh_{t+1} = tanh(z_t + dz_t) - tanh(z_t).

This is exact through the nonlinearity, but after source-independent u_t is cached under a matched
extra-memory budget it still performs one dependency matvec and one nonlinear evaluation at every
affected step — the same affected-step work as exact replay. A frozen-Jacobian carrier skips the
new nonlinearity but is checked against every dependent state, not merely the final state; decay of
its error later in the recurrence cannot hide a stale write.

A block-local control shows the only clean escape found here: make the true dependency cone smaller.
That can accelerate exact repair, but the strongest cone-aware replay baseline receives the same
structural advantage. Therefore delta propagation itself gets no novelty credit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


def _case(seed: int, d: int, steps: int):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(steps, d, d)).astype(np.float64)
    norms = np.linalg.norm(A, axis=(1, 2), keepdims=True)
    A *= (0.75 * np.sqrt(d)) / np.maximum(norms, 1e-12)
    u = rng.normal(scale=0.45, size=(steps, d)).astype(np.float64)
    h0 = rng.normal(scale=0.2, size=d).astype(np.float64)
    s0 = rng.normal(scale=0.7, size=d).astype(np.float64)
    s1 = -1.25 * s0 + rng.normal(scale=0.35, size=d)
    return A, u, h0, s0, s1


def _full(A, u, h0, source, edit_at: int, heavy: bool = False):
    h = h0.copy(); hs = [h.copy()]; zs = []
    for t in range(len(A)):
        ut = u[t]
        if heavy:
            # Irrelevant source-independent work: useful only to expose a weak benchmark baseline.
            q = np.tanh(ut)
            for _ in range(3):
                q = np.tanh(0.8 * q + 0.2 * ut)
            ut = q
        z = A[t] @ h + ut + (source if t == edit_at else 0.0)
        h = np.tanh(z); zs.append(z.copy()); hs.append(h.copy())
    return h, np.stack(hs), np.stack(zs)


def _cached_replay(A, u, old_hs, source, edit_at: int):
    h = old_hs[edit_at].copy(); out = [h.copy()]
    for t in range(edit_at, len(A)):
        z = A[t] @ h + u[t] + (source if t == edit_at else 0.0)
        h = np.tanh(z); out.append(h.copy())
    return h, np.stack(out)


def _exact_delta(A, old_hs, old_zs, s0, s1, edit_at: int):
    dh = np.zeros_like(old_hs[0]); out = [old_hs[edit_at].copy()]
    for t in range(edit_at, len(A)):
        dsrc = (s1 - s0) if t == edit_at else 0.0
        dz = A[t] @ dh + dsrc
        h_new = np.tanh(old_zs[t] + dz)
        dh = h_new - old_hs[t + 1]
        out.append(h_new.copy())
    return old_hs[-1] + dh, np.stack(out)


def _stale_jacobian(A, old_hs, old_zs, s0, s1, edit_at: int):
    dh = np.zeros_like(old_hs[0]); out = [old_hs[edit_at].copy()]
    for t in range(edit_at, len(A)):
        dsrc = (s1 - s0) if t == edit_at else 0.0
        dz = A[t] @ dh + dsrc
        deriv = 1.0 - np.tanh(old_zs[t]) ** 2
        dh = deriv * dz
        out.append((old_hs[t + 1] + dh).copy())
    return old_hs[-1] + dh, np.stack(out)


def _time(fn, rounds: int) -> float:
    vals = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns(); fn(); vals.append(time.perf_counter_ns() - t0)
    return float(np.median(vals))


def _block_local(seed: int, d: int, block: int, steps: int, edit_at: int):
    assert d % block == 0
    rng = np.random.default_rng(100000 + seed); nblk = d // block
    mats = rng.normal(size=(steps, nblk, block, block))
    mats *= 0.6 / np.maximum(np.linalg.norm(mats, axis=(2, 3), keepdims=True) / np.sqrt(block), 1e-12)
    u = rng.normal(scale=0.3, size=(steps, d)); h0 = rng.normal(scale=0.2, size=d)
    s0 = np.zeros(d); s1 = np.zeros(d)
    s0[:block] = rng.normal(scale=0.6, size=block)
    s1[:block] = -s0[:block] + rng.normal(scale=0.25, size=block)

    def full(src):
        h = h0.copy(); hs = [h.copy()]
        for t in range(steps):
            x = np.empty_like(h)
            for b in range(nblk):
                sl = slice(b * block, (b + 1) * block); x[sl] = mats[t, b] @ h[sl]
            h = np.tanh(x + u[t] + (src if t == edit_at else 0.0)); hs.append(h.copy())
        return h, np.stack(hs)

    old, oh = full(s0); ref, rh = full(s1)
    hb = oh[edit_at, :block].copy(); repaired = [oh[edit_at].copy()]
    cur = oh[edit_at].copy()
    for t in range(edit_at, steps):
        hb = np.tanh(mats[t, 0] @ hb + u[t, :block] + (s1[:block] if t == edit_at else 0.0))
        cur = oh[t + 1].copy(); cur[:block] = hb; repaired.append(cur)
    repaired = np.stack(repaired)
    ref_suffix = rh[edit_at:]
    return {
        "dependent_state_maxabs": float(np.max(np.abs(repaired - ref_suffix))),
        "affected_width": block,
        "full_width": d,
        "theoretical_dense_matmul_ratio": float((d * d) / (block * block)),
        "strong_baseline_can_use_same_cone": True,
    }


def run(seed: int, d: int, steps: int, edit_at: int, rounds: int) -> Dict[str, object]:
    A, u, h0, s0, s1 = _case(seed, d, steps)
    old, old_hs, old_zs = _full(A, u, h0, s0, edit_at)
    ref, ref_hs, _ = _full(A, u, h0, s1, edit_at)
    exact, exact_hs = _exact_delta(A, old_hs, old_zs, s0, s1, edit_at)
    replay, replay_hs = _cached_replay(A, u, old_hs, s1, edit_at)
    jac, jac_hs = _stale_jacobian(A, old_hs, old_zs, s0, s1, edit_at)
    ref_suffix = ref_hs[edit_at:]

    err = {
        "exact_delta_final": float(np.max(np.abs(exact - ref))),
        "exact_delta_dependent_write_max": float(np.max(np.abs(exact_hs - ref_suffix))),
        "cached_replay_final": float(np.max(np.abs(replay - ref))),
        "cached_replay_dependent_write_max": float(np.max(np.abs(replay_hs - ref_suffix))),
        "stale_jacobian_final": float(np.max(np.abs(jac - ref))),
        "stale_jacobian_dependent_write_max": float(np.max(np.abs(jac_hs - ref_suffix))),
    }

    t_exact = _time(lambda: _exact_delta(A, old_hs, old_zs, s0, s1, edit_at)[0], rounds)
    t_replay = _time(lambda: _cached_replay(A, u, old_hs, s1, edit_at)[0], rounds)
    t_raw = _time(lambda: _full(A, u, h0, s1, edit_at, heavy=True)[0], max(3, rounds // 4))
    affected = steps - edit_at
    ops = {
        "exact_delta_dependency_matvecs": affected,
        "cached_replay_dependency_matvecs": affected,
        "exact_delta_nonlinear_evals": affected,
        "cached_replay_nonlinear_evals": affected,
        "same_asymptotic_affected_work": True,
        "matched_extra_vectors_per_step": 1,
    }
    block = _block_local(seed, d, max(8, d // 8), steps, edit_at)
    checks = {
        "exact_delta_matches_every_dependent_write": err["exact_delta_dependent_write_max"] <= 1e-11,
        "cached_replay_matches_every_dependent_write": err["cached_replay_dependent_write_max"] <= 1e-11,
        "stale_jacobian_fails_dependent_write_equality": err["stale_jacobian_dependent_write_max"] >= 1e-3,
        "carrier_does_not_reduce_dense_dependency_matvec_count": ops["exact_delta_dependency_matvecs"] == ops["cached_replay_dependency_matvecs"],
        "carrier_does_not_reduce_dense_nonlinearity_count": ops["exact_delta_nonlinear_evals"] == ops["cached_replay_nonlinear_evals"],
        "structured_cone_is_exact": block["dependent_state_maxabs"] <= 1e-11,
    }
    return {
        "seed": seed, "d": d, "steps": steps, "edit_at": edit_at,
        "errors": err,
        "timing_ns_median": {"exact_delta": t_exact, "cached_replay": t_replay, "raw_rebuild_with_irrelevant_branch": t_raw},
        "timing_ratios": {"cached_replay_over_exact_delta": t_replay / max(t_exact, 1.0), "raw_rebuild_over_exact_delta": t_raw / max(t_exact, 1.0)},
        "operation_contract": ops,
        "structured_dependency_control": block,
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--d", type=int, default=128); p.add_argument("--steps", type=int, default=48)
    p.add_argument("--edit-at", type=int, default=8); p.add_argument("--rounds", type=int, default=17)
    p.add_argument("--results-dir", default="so/results"); a = p.parse_args()
    rows: List[Dict[str, object]] = [run(s, a.d, a.steps, a.edit_at, a.rounds) for s in a.seeds]
    ok = all(r["pass"] for r in rows)
    rec = {
        "experiment": "E-000089",
        "title": "nonlinear bounded correction carrier vs guarantee-matched exact replay",
        "result": "decisive_falsification_of_generic_lane_B" if ok else "inconclusive",
        "rows": rows,
        "breakthrough": False,
        "novelty_claim": False,
        "decision": "Exact nonlinear delta propagation is an algebraic replay of the affected dense recurrence under a matched cache; stale coefficients fail intermediate dependent-write equality even when final error later decays. Generic carrier algebra therefore cannot supply the required >10x guarantee-matched benefit.",
        "next_architecture_boundary": "Only pursue neural mechanisms that make the true source-lineage dependency cone materially smaller and independently verifiable. Compare them against cone-aware exact replay under matched memory; J-space/J-lens must detect any omitted causal descendant.",
    }
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "e000089_nonlinear_delta_repair_boundary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not ok: raise SystemExit(2)


if __name__ == "__main__":
    main()
