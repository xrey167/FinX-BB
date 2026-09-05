"""E-000056 — can a LEARNED acceptance function host the store's lifecycle predicate?

This discharges two questions the programme registered against itself and never ran:

  * §31.12 left one open: the marker gate's operational radius is 0.90 against a declared
    0.35, and "making it implement the predicate is a change to the data, not to the
    architecture: draw unsigned markers from the whole region beyond the declared radius,
    the annulus included.  That needs a training run and is not yet evaluated."
  * §31.47 registered this experiment and did not run it: "whether the learned
    acceptance function can host the freshness predicate at all ... a rotation of the marker
    centre large enough for the frozen gate to reject the previous epoch is, on the face of
    the geometry, also large enough for it to reject the current one."

THE QUESTION.  A pod's lifecycle is declared by the store as a predicate V over exported
material (here `MVCCStore.marker_valid`: ``||m - kappa|| <= valid_radius``, 0.35).  A design
that attests "at the point of neural consumption" asks a *learned* component -- the marker
gate, a 16 -> 64 -> 1 MLP frozen with the rest of the model -- to be the custodian of V.
Can it be?

ARMS.  The gate is trained standalone, which is exactly what it is: `so.model` reads it as a
function of the marker alone, and E-000021 scored 11 recorded checkpoints that way.  The
standalone reproduction is anchored against the recorded full-model numbers (E-000029:
operational radius 0.90, annulus accept 1.0000; E-000021: false accept 8.49e-04) before any
new arm is read -- if the anchor misses, nothing below is evidence about this architecture.

  A. MARGIN     the programme's own training distribution: signed at ~0.194, unsigned
                rejection-sampled beyond 0.7.  Reproduces E-000029.
  B. BOUNDARY   §31.12's proposed fix, implemented honestly: the negative class constructed
                ON THE SHELLS uniformly over [0.35, sqrt2], annulus included.  (A rejection
                sampler cannot do this: in 16 dimensions a uniform unit vector sits at
                ||m-kappa|| ~ sqrt2, so the annulus is empty in practice -- which is the
                original defect, not a repair.)
  C. EPOCH      the frozen arm-A gate, asked to host a rotating credential: for each centre
                displacement d, is the new epoch usable (accept >= 0.95) and is an epoch at
                that displacement revocable (accept <= 0.05)?  A displacement that is both
                is a hostable revocable epoch.  Every epoch must be usable while current, so
                a usable-and-revocable displacement is what a rotation scheme needs.
  D. LIVE       the steelman: give the gate liveness -- the epoch centre as a second input --
                and train it to accept a marker iff it was minted under THAT centre.
  E. ORACLE     hand the gate the store's own computation (the distance to the live centre)
                as an input feature.  The upper bound on what delegation can buy.
  F. CAPACITY   arm B at 4x and 16x the width and at 5x the budget, against the objection that
                the boundary arm's error rate is undertraining rather than a property of
                learning a hard predicate from samples.

PRE-REGISTERED BARS.  The claim under test is a LIMITATION, so each bar is stated in the
direction that would have to hold for delegation to be sound; the limitation is supported
only if the marked bars fail and the structural ones hold.  Evaluated on the WORST seed.

  anchor_margin_accept_at_0p70       >= 0.99   (arm A reproduces E-000029 row 0.70 = 0.9999)
  anchor_margin_accept_at_0p90       <= 0.00   (arm A reproduces E-000029 row 0.90 = 0.0000)
  anchor_margin_annulus_accept       >= 0.99   (arm A reproduces E-000029 annulus = 1.0000)
  anchor_margin_false_accept         <= 2e-3   (arm A reproduces E-000021 = 8.49e-04)

  boundary_annulus_accept            <= 0.05   delegation is faithful to V           [expect FAIL]
  boundary_false_accept              <= 1e-3   ... and as reliable as the quoted rate [expect FAIL]
  fidelity_reliability_ratio         >= 10     the two cannot be had together        [structural]
  frozen_hostable_revocable_epochs   >= 1      a frozen gate can host a rotation      [expect FAIL]
  usable_revocable_gap               >= 0.05   ... and misses by a margin, not a hair [structural]
  live_current_accept                >= 0.95   liveness restores capability           [expect PASS]
  live_stale_accept                  <= 1e-4   ... and revocation is sound            [expect FAIL]
  oracle_false_accept                <= 1e-3   exactness needs the store's own computation
  capacity_best_false_accept         <= 1e-3   a bigger/longer-trained gate closes the gap  [expect FAIL]

PRE-REGISTRATION CORRECTION (recorded, not tuned away).  The anchor was first written as
``operational_radius == 0.90``.  That is E-000029's number on E-000029's grid, which steps in
0.1; this experiment sweeps in 0.05, where the first all-reject shell legitimately lands at
0.85 on some seeds -- E-000029 measured 0.80 -> 0.2191 and 0.90 -> 0.0000 and never tested
0.85.  The first run failed the anchor at 0.85 for exactly that reason, on an artefact of the
grid rather than on the gate.  The anchor is now stated on the two shells E-000029 actually
recorded, plus its annulus row and E-000021's rate, which is grid-independent.  No other bar
was touched and no arm changed; the first run's numbers for every non-anchor bar are the ones
reported below.

NOT CLAIMED.  Nothing here claims novelty for: learned gating, verification losses, marker
or capability schemes, MVCC, versioned pointers, cache invalidation, leases, epochs, nonces,
freshness-requires-liveness (Needham-Schroeder and the whole authentication literature),
approximation error of learned classifiers, or margin-based generalisation.  The measured
object is narrower: WHERE the lifecycle-authority boundary can lie in an architecture whose
admission decision is a learned function of exported material.

    python -m so.experiments.e000056_learned_custodian --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data import valid_markers, invalid_markers

D = 16
DECLARED_RADIUS = 0.35
MAX_CHORD = float(np.sqrt(2.0))
BANDS = [round(0.05 * i, 2) for i in range(1, 29)]

BARS = {
    "anchor_margin_accept_at_0p70": (">=", 0.99),
    "anchor_margin_accept_at_0p90": ("<=", 0.0),
    "anchor_margin_annulus_accept": (">=", 0.99),
    "anchor_margin_false_accept": ("<=", 2e-3),
    "boundary_annulus_accept": ("<=", 0.05),
    "boundary_false_accept": ("<=", 1e-3),
    "fidelity_reliability_ratio": (">=", 10.0),
    "frozen_hostable_revocable_epochs": (">=", 1),
    "usable_revocable_gap": (">=", 0.05),
    "live_current_accept": (">=", 0.95),
    "live_stale_accept": ("<=", 1e-4),
    "oracle_false_accept": ("<=", 1e-3),
    "capacity_best_false_accept": ("<=", 1e-3),
}

ANCHOR_BARS = (
    "anchor_margin_accept_at_0p70",
    "anchor_margin_accept_at_0p90",
    "anchor_margin_annulus_accept",
    "anchor_margin_false_accept",
)

# Which bars must FAIL for the limitation to be supported, and which must HOLD.
LIMITATION_REQUIRES_FAIL = (
    "boundary_false_accept",
    "frozen_hostable_revocable_epochs",
    "live_stale_accept",
    "capacity_best_false_accept",
)
LIMITATION_REQUIRES_PASS = (
    "fidelity_reliability_ratio",
    "usable_revocable_gap",
    "live_current_accept",
)


# --------------------------------------------------------------------------- geometry


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_centre(seed: int) -> np.ndarray:
    """The store's marker centre, derived exactly as `MVCCStore` derives it (10_000 + seed)."""
    return _unit(np.random.default_rng(10_000 + seed).normal(size=D))


def shell(rng: np.random.Generator, centre: np.ndarray, dist, n: int) -> np.ndarray:
    """n unit vectors at exact chord distance `dist` from `centre` (scalar or length-n array)."""
    v = rng.normal(size=(n, D))
    v -= (v @ centre)[:, None] * centre[None, :]
    v = _unit(v)
    cos = 1.0 - np.asarray(dist, dtype=float).reshape(-1, 1) ** 2 / 2.0
    return cos * centre[None, :] + np.sqrt(np.clip(1.0 - cos ** 2, 0.0, None)) * v


def deleted_region(rng: np.random.Generator, centre: np.ndarray, n: int) -> np.ndarray:
    """The whole region the store calls deleted: shells uniform over [declared, sqrt2]."""
    return shell(rng, centre, rng.uniform(DECLARED_RADIUS, MAX_CHORD, size=n), n)


# --------------------------------------------------------------------------- the gate


def gate_mlp(din: int = D, width: int = 64, depth: int = 1) -> nn.Module:
    """The marker gate of `so.model.KnowledgeModel` (16 -> 64 -> 1), optionally deeper/wider."""
    layers: List[nn.Module] = [nn.Linear(din, width), nn.GELU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.GELU()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


def _fit(gate: nn.Module, sampler, seed: int, steps: int, lr: float = 1e-3) -> nn.Module:
    """Class-balanced verification loss, the E-000010 recipe (unsigned class weighted 5x)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for _ in range(steps):
        x, y, w = sampler(rng)
        loss = nn.functional.binary_cross_entropy_with_logits(gate(x), y, weight=w)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return gate


def _accept(gate: nn.Module, m: np.ndarray, extra: Optional[np.ndarray] = None) -> float:
    with torch.no_grad():
        x = torch.tensor(m, dtype=torch.float32)
        if extra is not None:
            x = torch.cat([x, torch.tensor(extra, dtype=torch.float32)], dim=1)
        return float((torch.sigmoid(gate(x)).numpy().ravel() > 0.5).mean())


def _labelled(pos: np.ndarray, neg: np.ndarray, neg_weight: float = 5.0):
    x = torch.tensor(np.concatenate([pos, neg]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]), dtype=torch.float32)[:, None]
    w = torch.tensor(np.concatenate([np.ones(len(pos)), neg_weight * np.ones(len(neg))]), dtype=torch.float32)[:, None]
    return x, y, w


def _band_curve(gate: nn.Module, rng: np.random.Generator, centre: np.ndarray, n: int) -> List[float]:
    return [_accept(gate, shell(rng, centre, np.full(n, d), n)) for d in BANDS]


def _operational_radius(curve: List[float]) -> Optional[float]:
    """The first shell at which the gate accepts nothing -- E-000029's definition."""
    return next((BANDS[i] for i, a in enumerate(curve) if a == 0.0), None)


# --------------------------------------------------------------------------- the arms


def run(seed: int, steps: int = 4000, batch: int = 512, n_eval: int = 200_000) -> Dict:
    centre = make_centre(seed)
    out: Dict[str, object] = {"seed": seed, "declared_radius": DECLARED_RADIUS}

    # --- A. MARGIN: the programme's own distribution ------------------------
    def margin_sampler(rng):
        npos = int(batch * 0.95)
        return _labelled(valid_markers(rng, centre, npos), invalid_markers(rng, centre, batch - npos))

    gate_a = _fit(gate_mlp(), margin_sampler, seed, steps)
    rng = np.random.default_rng(700 + seed)
    curve_a = _band_curve(gate_a, rng, centre, 4000)
    annulus = shell(rng, centre, rng.uniform(0.36, 0.69, size=40_000), 40_000)
    out["margin"] = {
        "band_accept": curve_a,
        "operational_radius": _operational_radius(curve_a),
        "annulus_accept": _accept(gate_a, annulus),
        "signed_accept": _accept(gate_a, valid_markers(rng, centre, 50_000)),
        "false_accept_rejection_sampled": _accept(gate_a, invalid_markers(rng, centre, n_eval)),
        "false_accept_declared_region": _accept(gate_a, deleted_region(rng, centre, n_eval)),
    }

    # --- B. BOUNDARY: §31.12's fix, constructed on the shells ---------------
    def boundary_sampler(rng):
        npos = int(batch * 0.95)
        return _labelled(valid_markers(rng, centre, npos), deleted_region(rng, centre, batch - npos))

    gate_b = _fit(gate_mlp(), boundary_sampler, seed, steps)
    curve_b = _band_curve(gate_b, rng, centre, 4000)
    out["boundary"] = {
        "band_accept": curve_b,
        "operational_radius": _operational_radius(curve_b),
        "annulus_accept": _accept(gate_b, annulus),
        "signed_accept": _accept(gate_b, valid_markers(rng, centre, 50_000)),
        "false_accept_declared_region": _accept(gate_b, deleted_region(rng, centre, n_eval)),
    }

    # --- C. EPOCH: can the FROZEN arm-A gate host a rotating credential? ----
    grid = []
    for d in BANDS:
        ct = shell(rng, centre, np.full(1, d), 1)[0]
        grid.append({"centre_distance": d, "epoch_accept": _accept(gate_a, valid_markers(rng, ct, 4000))})
    usable = [g["centre_distance"] for g in grid if g["epoch_accept"] >= 0.95]
    revocable = [g["centre_distance"] for g in grid if g["epoch_accept"] <= 0.05]
    feasible = sorted(set(usable) & set(revocable))
    out["epoch"] = {
        "grid": grid,
        "max_usable_distance": max(usable) if usable else None,
        "min_revocable_distance": min(revocable) if revocable else None,
        "feasible_displacements": feasible,
        "hostable_revocable_epochs": len(feasible),
        "usable_revocable_gap": (min(revocable) - max(usable)) if (usable and revocable) else None,
    }

    # --- D. LIVE: the steelman -- give the gate liveness --------------------
    def live_sampler(rng):
        half = batch // 2
        cur = _unit(rng.normal(size=(half, D)))
        fresh = _unit(cur + rng.normal(scale=0.05, size=(half, D)))
        other_live = _unit(rng.normal(size=(batch - half, D)))
        other_mint = _unit(rng.normal(size=(batch - half, D)))
        stale = _unit(other_mint + rng.normal(scale=0.05, size=(batch - half, D)))
        return _labelled(np.concatenate([fresh, cur], 1), np.concatenate([stale, other_live], 1), neg_weight=1.0)

    gate_d = _fit(gate_mlp(din=2 * D), live_sampler, seed, steps + 2000)
    live = _unit(rng.normal(size=(n_eval, D)))
    fresh = _unit(live + rng.normal(scale=0.05, size=(n_eval, D)))
    stale_centre = _unit(rng.normal(size=(n_eval, D)))
    stale = _unit(stale_centre + rng.normal(scale=0.05, size=(n_eval, D)))
    out["live"] = {
        "current_accept": _accept(gate_d, fresh, extra=live),
        "stale_accept": _accept(gate_d, stale, extra=live),
        "n": n_eval,
    }

    # --- E. ORACLE: hand it the store's own computation ---------------------
    def oracle_feat(m, c):
        return np.concatenate([m, np.linalg.norm(m - c[None, :], axis=1, keepdims=True)], axis=1)

    def oracle_sampler(rng):
        npos = int(batch * 0.95)
        pos = oracle_feat(valid_markers(rng, centre, npos), centre)
        neg = oracle_feat(deleted_region(rng, centre, batch - npos), centre)
        return _labelled(pos, neg)

    gate_e = _fit(gate_mlp(din=D + 1), oracle_sampler, seed, steps)
    out["oracle"] = {
        "signed_accept": _accept(gate_e, oracle_feat(valid_markers(rng, centre, 30_000), centre)),
        "false_accept_declared_region": _accept(gate_e, oracle_feat(deleted_region(rng, centre, n_eval), centre)),
    }

    # --- F. CAPACITY: is arm B's error rate just undertraining? -------------
    capacity = []
    for width, depth, mult in ((256, 2, 1), (256, 2, 5), (1024, 2, 1)):
        g = _fit(gate_mlp(width=width, depth=depth), boundary_sampler, seed, steps * mult)
        capacity.append({
            "width": width, "depth": depth, "steps": steps * mult,
            "signed_accept": _accept(g, valid_markers(rng, centre, 30_000)),
            "false_accept_declared_region": _accept(g, deleted_region(rng, centre, n_eval)),
        })
    out["capacity"] = capacity
    best = min(c["false_accept_declared_region"] for c in capacity
               + [{"false_accept_declared_region": out["boundary"]["false_accept_declared_region"]}])

    out["metrics"] = {
        "anchor_margin_accept_at_0p70": curve_a[BANDS.index(0.70)],
        "anchor_margin_accept_at_0p90": curve_a[BANDS.index(0.90)],
        "anchor_margin_annulus_accept": out["margin"]["annulus_accept"],
        "anchor_margin_false_accept": out["margin"]["false_accept_rejection_sampled"],
        "boundary_annulus_accept": out["boundary"]["annulus_accept"],
        "boundary_false_accept": out["boundary"]["false_accept_declared_region"],
        "fidelity_reliability_ratio": (
            out["boundary"]["false_accept_declared_region"] / out["margin"]["false_accept_rejection_sampled"]
            if out["margin"]["false_accept_rejection_sampled"] > 0 else float("inf")
        ),
        "frozen_hostable_revocable_epochs": out["epoch"]["hostable_revocable_epochs"],
        "usable_revocable_gap": out["epoch"]["usable_revocable_gap"],
        "live_current_accept": out["live"]["current_accept"],
        "live_stale_accept": out["live"]["stale_accept"],
        "oracle_false_accept": out["oracle"]["false_accept_declared_region"],
        "capacity_best_false_accept": best,
    }
    return out


# --------------------------------------------------------------------------- scoring


def _worst(values: List[float], op: str):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return min(vals) if op in (">=", "==") else max(vals)


def score(per_seed: List[Dict]) -> Dict:
    checks, observed = {}, {}
    for name, (op, threshold) in BARS.items():
        vals = [s["metrics"][name] for s in per_seed]
        w = _worst(vals, op)
        observed[name] = w
        if w is None:
            checks[name] = False
        elif op == ">=":
            checks[name] = w >= threshold
        elif op == "<=":
            checks[name] = w <= threshold
        else:
            checks[name] = w == threshold
    anchor_ok = all(checks[k] for k in ANCHOR_BARS)
    limitation = (
        anchor_ok
        and all(not checks[k] for k in LIMITATION_REQUIRES_FAIL)
        and all(checks[k] for k in LIMITATION_REQUIRES_PASS)
    )
    return {
        "checks": checks,
        "observed_worst_seed": observed,
        "bars": {k: {"op": v[0], "threshold": v[1]} for k, v in BARS.items()},
        "anchor_reproduces_recorded_gate": anchor_ok,
        "limitation_supported": limitation,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--results-dir", default="so/results")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run(s, steps=args.steps, n_eval=args.n_eval) for s in args.seeds]
    record = {
        "experiment": "E-000056",
        "question": "Can a learned acceptance function host the store's lifecycle predicate?",
        "discharges": ["ledger §31.12 open question", "ledger §31.47 registration"],
        "per_seed": per_seed,
        **score(per_seed),
        "interpretation_limit": (
            "The gate is trained standalone, which is what so.model reads it as (a function of the "
            "marker alone) and how E-000021 scored 11 recorded checkpoints. The anchor bars check "
            "that this reproduces the recorded full-model gate before any other arm is read. Nothing "
            "here measures the trained adapter's reading behaviour, and nothing here is a security "
            "claim: an adversary who can choose the marker is not modelled."
        ),
        "not_claimed": [
            "learned gating", "verification losses", "marker/capability schemes", "MVCC",
            "versioned pointers", "cache invalidation", "leases", "epochs", "nonces",
            "freshness-requires-liveness", "approximation error of learned classifiers",
        ],
    }
    out = Path(args.results_dir) / "e000056_learned_custodian.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"observed_worst_seed": record["observed_worst_seed"],
                      "checks": record["checks"],
                      "anchor_reproduces_recorded_gate": record["anchor_reproduces_recorded_gate"],
                      "limitation_supported": record["limitation_supported"]}, indent=1, default=float))
    if not record["anchor_reproduces_recorded_gate"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
