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

THE SETTING.  A pod's lifecycle is declared by the store as an exact predicate V over exported
material (`MVCCStore.marker_valid`: ``||m - kappa|| <= valid_radius``, 0.35).  A design that
attests "at the point of neural consumption" asks a *learned* component -- the marker gate, a
16 -> 64 -> 1 MLP frozen with the rest of the model, whose output multiplies the injected
payload (`so/model.py`: ``v_f = v_f * g``) -- to be the custodian of V.

A FRAMING THIS EXPERIMENT REFUTED, RECORDED BEFORE ITS RESULTS
--------------------------------------------------------------
The first version of this file was built to support the sentence *"revocation cannot be
delegated to a frozen LEARNED acceptance function"*, with the EPOCH arm as its evidence.  That
sentence is withdrawn, and the arm is kept only because its controls are what refuted it:

  1. The arm's metric was **arithmetically unsatisfiable**.  It selected `usable` as the grid
     points with ``epoch_accept >= 0.95`` and `revocable` as those with ``epoch_accept <= 0.05``,
     then intersected them -- one scalar cannot satisfy both, so `hostable_revocable_epochs` was
     0 for every function that has ever existed, learned or not.  A bar that could not have come
     out the other way is not a pre-registration.
  2. The headline numbers are reproduced **to the digit by the store's own hand-written exact
     predicate**, which is neither learned nor neural (see the CONTROL checkers below): the same
     0.20 gap, the same 0 hostable epochs, the same accept(previous epoch) = 1.0.  So the word
     carrying the conclusion was "frozen", never "learned", and the 0.20 gap is the grid-quantised
     width of the transition band of any radial predicate under the 0.05 mint noise of
     `so.data.valid_markers` -- a statistic of the sampler, not a property of the gate.
  3. The variable that actually moves is **liveness of the reference**, which §31.47 had already
     concluded by an economic argument citing Redell 1974, Gray & Cheriton 1989, Kung & Robinson
     1981 and Sandberg 1985, and which the general result (a verifier reading only the presented
     artefact cannot implement a time-varying predicate) has owned since Needham-Schroeder 1978.

The arm is therefore rewritten to measure the variable that moves, with the controls that make
it interpretable.  What this experiment still establishes on its own evidence is §31.12's answer
(arms A, B, F) and the liveness measurements (arms D, E) -- neither of which is a novelty claim.

ARMS.  The gate is trained standalone, which is what `so.model` reads it as (a function of the
marker alone) and how E-000021 scored eleven recorded checkpoints.  The anchor bars check that
this reproduces the recorded full-model gate before any other arm is read.

  A. MARGIN     the programme's own training distribution: signed at ~0.194, unsigned
                rejection-sampled beyond 0.7.  Reproduces E-000029.
  B. BOUNDARY   §31.12's proposed fix, implemented honestly: the negative class constructed
                ON THE SHELLS uniformly over [0.35, sqrt2], annulus included.  (A rejection
                sampler cannot do this: in 16 dimensions a uniform unit vector sits at
                ||m-kappa|| ~ sqrt2, so the annulus is empty in practice -- which is the
                original defect, not a repair.)
  C. EPOCH      four checkers over the same centre-rotation grid, so the comparison is the
                measurement: (i) the frozen learned gate; (ii) the store's exact predicate at
                the FROZEN centre -- a non-learned, non-neural control; (iii) a hard sphere at
                the gate's own operational radius -- a second non-learned control; (iv) the
                store's exact predicate reading the LIVE centre.  Reported per checker:
                how many grid rotations admit a revocable epoch.
  D. LIVE       the steelman: give the learned gate liveness -- the epoch centre as a second
                input -- and train it to accept a marker iff it was minted under THAT centre.
                Swept over capacity, because arm F shows capacity matters for arm B.
  E. ORACLE     hand the gate the store's own computation (the distance to the live centre)
                as an input feature.  The upper bound on what delegation can buy.
  F. CAPACITY   arm B at 4x and 16x the width and at 5x the budget, against the objection that
                the boundary arm's error rate is undertraining rather than a property of
                learning a hard predicate from samples.

PRE-REGISTERED BARS.  Evaluated on the WORST seed.

  anchor_margin_accept_at_0p70       >= 0.99   (arm A reproduces E-000029 row 0.70 = 0.9999)
  anchor_margin_accept_at_0p90       <= 0.00   (arm A reproduces E-000029 row 0.90 = 0.0000)
  anchor_margin_annulus_accept       >= 0.99   (arm A reproduces E-000029 annulus = 1.0000)
  anchor_margin_false_accept         <= 2e-3   (arm A reproduces E-000021 = 8.49e-04)

  boundary_annulus_accept            <= 0.05   §31.12's fix makes the gate faithful to V
  boundary_false_accept              <= 1e-3   ... at the gate's own capacity
  capacity_best_false_accept         <= 1e-3   ... or at any capacity swept here
  live_current_accept                >= 0.95   a gate given liveness keeps capability
  live_stale_accept_best             <= 1e-5   ... and its revocation residual vanishes
  oracle_false_accept                <= 1e-3   exactness when handed the store's computation

  CONTROLS, reported and NOT scored as evidence for anything about learning, because the
  non-learned checkers are expected to match the gate exactly:
  epoch_hostable_frozen_gate, epoch_hostable_store_frozen, epoch_hostable_sphere_frozen,
  epoch_hostable_store_live, epoch_gap_frozen_gate, epoch_gap_store_frozen.

PRE-REGISTRATION CORRECTIONS (recorded, not tuned away).
  1. The anchor was first written as ``operational_radius == 0.90`` -- E-000029's number on
     E-000029's 0.1 grid.  This experiment sweeps in 0.05, where the first all-reject shell
     legitimately lands at 0.85 on some seeds; E-000029 measured 0.80 -> 0.2191 and
     0.90 -> 0.0000 and never tested 0.85.  The first run failed the anchor on that grid
     artefact rather than on the gate.  The anchor is now stated on the two shells E-000029
     actually recorded, plus its annulus row and E-000021's rate.  No arm changed.
  2. `frozen_hostable_revocable_epochs >= 1` and `usable_revocable_gap >= 0.05` were scored as
     evidence in the first run.  Both are forced (see the refutation above); they are demoted to
     controls, and the arm gained the three checkers that show what they measure.

NOT CLAIMED.  Nothing here claims novelty for: learned gating, verification losses, marker or
capability schemes, MVCC, versioned pointers, cache invalidation, leases, epochs, nonces,
freshness-requires-liveness (Needham-Schroeder and the whole authentication literature),
revocable proof systems (Christ & Bonneau, FC 2023 / IACR ePrint 2022/1478), the approximation
error of learned classifiers, or margin-based generalisation.

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

from ..data import invalid_markers, valid_markers

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
    "capacity_best_false_accept": ("<=", 1e-3),
    "live_current_accept": (">=", 0.95),
    "live_stale_accept_best": ("<=", 1e-5),
    "oracle_false_accept": ("<=", 1e-3),
}

ANCHOR_BARS = (
    "anchor_margin_accept_at_0p70",
    "anchor_margin_accept_at_0p90",
    "anchor_margin_annulus_accept",
    "anchor_margin_false_accept",
)

# Reported, never scored: the non-learned checkers are expected to match the gate, which is the
# whole point of having them.
CONTROL_METRICS = (
    "epoch_hostable_frozen_gate",
    "epoch_hostable_store_frozen",
    "epoch_hostable_sphere_frozen",
    "epoch_hostable_store_live",
    "epoch_gap_frozen_gate",
    "epoch_gap_store_frozen",
)


# --------------------------------------------------------------------------- geometry


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_centre(seed: int) -> np.ndarray:
    """The store's marker centre, derived as `MVCCStore` derives it (10_000 + seed)."""
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


# --------------------------------------------------------------------------- arm C


def _epoch_scan(per_d) -> Dict:
    """One checker over the rotation grid.

    `per_d(d)` returns `(new, prev)` for ONE rotation of the marker centre by chord distance d:
      `new`  -- rate at which markers minted at the rotated centre are accepted NOW (the new
                epoch's capability);
      `prev` -- rate at which the PREVIOUS epoch's markers are still accepted after that rotation
                (what revocation has to drive down).

    Both must come from the SAME rotated centre.  Drawing a fresh rotation for each collapses
    `new`, because two centres each at distance d from the origin can be up to 2d from each other.

    A rotation hosts a revocable epoch when the new epoch works and the old one is refused.  For
    any checker that reads only the presented marker, `prev` does not depend on d at all, which is
    exactly why it cannot revoke -- and why this must be compared against a checker that reads the
    live centre before anything is concluded.
    """
    rows = []
    for d in BANDS:
        new, prev = per_d(d)
        rows.append({"centre_distance": d, "new": new, "prev": prev})
    usable = [r["centre_distance"] for r in rows if r["new"] >= 0.95]
    revoked = [r["centre_distance"] for r in rows if r["prev"] <= 0.05]
    hostable = sorted(set(usable) & set(revoked))
    return {
        "grid": rows,
        "max_usable_distance": max(usable) if usable else None,
        "min_revoking_distance": min(revoked) if revoked else None,
        "hostable_revocable_epochs": len(hostable),
        "gap": (min(revoked) - max(usable)) if (usable and revoked) else None,
    }


def _epoch_arm(gate: nn.Module, centre: np.ndarray, rng: np.random.Generator, n: int = 4000) -> Dict:
    """Four checkers, same grid.  The comparison is the measurement."""
    op_radius = 0.90  # the gate's recorded operational radius (E-000029)

    def minted(c, k=n):
        return valid_markers(rng, c, k)

    def rotated(d):
        return shell(rng, centre, np.full(1, d), 1)[0]

    prev_markers = minted(centre)
    # One rotation per grid point, shared by every checker, so the four scans are comparable.
    rotations = {d: rotated(d) for d in BANDS}
    new_markers = {d: minted(c_t) for d, c_t in rotations.items()}

    def _within(m, c, r):
        return float((np.linalg.norm(m - c[None, :], axis=1) <= r).mean())

    # (i) the frozen learned gate
    gate_scan = _epoch_scan(lambda d: (_accept(gate, new_markers[d]), _accept(gate, prev_markers)))

    # (ii) + (iii) non-learned, non-neural controls at the frozen centre
    store_scan = _epoch_scan(lambda d: (_within(new_markers[d], centre, DECLARED_RADIUS),
                                        _within(prev_markers, centre, DECLARED_RADIUS)))
    sphere_scan = _epoch_scan(lambda d: (_within(new_markers[d], centre, op_radius),
                                         _within(prev_markers, centre, op_radius)))

    # (iv) the same exact predicate, reading the LIVE centre
    live_scan = _epoch_scan(lambda d: (_within(new_markers[d], rotations[d], DECLARED_RADIUS),
                                       _within(prev_markers, rotations[d], DECLARED_RADIUS)))

    return {
        "frozen_gate": gate_scan,
        "store_frozen": store_scan,
        "sphere_frozen": sphere_scan,
        "store_live": live_scan,
        "note": (
            "The three frozen checkers are expected to agree; that agreement is the result. Only the "
            "live-reference checker hosts a revocable epoch, so the variable is liveness, not learning."
        ),
    }


# --------------------------------------------------------------------------- the run


def run(seed: int, steps: int = 4000, batch: int = 512, n_eval: int = 200_000) -> Dict:
    centre = make_centre(seed)
    out: Dict[str, object] = {"seed": seed, "declared_radius": DECLARED_RADIUS}

    # --- A. MARGIN ----------------------------------------------------------
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

    # --- B. BOUNDARY --------------------------------------------------------
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

    # --- C. EPOCH, with controls -------------------------------------------
    out["epoch"] = _epoch_arm(gate_a, centre, rng)

    # --- D. LIVE, swept over capacity --------------------------------------
    def live_sampler(rng):
        half = batch // 2
        cur = _unit(rng.normal(size=(half, D)))
        fresh = _unit(cur + rng.normal(scale=0.05, size=(half, D)))
        other_live = _unit(rng.normal(size=(batch - half, D)))
        other_mint = _unit(rng.normal(size=(batch - half, D)))
        stale = _unit(other_mint + rng.normal(scale=0.05, size=(batch - half, D)))
        return _labelled(np.concatenate([fresh, cur], 1), np.concatenate([stale, other_live], 1), neg_weight=1.0)

    live = _unit(rng.normal(size=(n_eval, D)))
    fresh = _unit(live + rng.normal(scale=0.05, size=(n_eval, D)))
    stale = _unit(_unit(rng.normal(size=(n_eval, D))) + rng.normal(scale=0.05, size=(n_eval, D)))
    live_rows = []
    for width, depth, mult in ((64, 1, 1), (256, 2, 1), (256, 2, 5)):
        g = _fit(gate_mlp(din=2 * D, width=width, depth=depth), live_sampler, seed, steps * mult)
        live_rows.append({
            "width": width, "depth": depth, "steps": steps * mult,
            "current_accept": _accept(g, fresh, extra=live),
            "stale_accept": _accept(g, stale, extra=live),
        })
    out["live"] = {"sweep": live_rows, "n": n_eval}

    # --- E. ORACLE ----------------------------------------------------------
    def oracle_feat(m, c):
        return np.concatenate([m, np.linalg.norm(m - c[None, :], axis=1, keepdims=True)], axis=1)

    def oracle_sampler(rng):
        npos = int(batch * 0.95)
        return _labelled(oracle_feat(valid_markers(rng, centre, npos), centre),
                         oracle_feat(deleted_region(rng, centre, batch - npos), centre))

    gate_e = _fit(gate_mlp(din=D + 1), oracle_sampler, seed, steps)
    out["oracle"] = {
        "signed_accept": _accept(gate_e, oracle_feat(valid_markers(rng, centre, 30_000), centre)),
        "false_accept_declared_region": _accept(gate_e, oracle_feat(deleted_region(rng, centre, n_eval), centre)),
    }

    # --- F. CAPACITY --------------------------------------------------------
    capacity = []
    for width, depth, mult in ((256, 2, 1), (256, 2, 5), (1024, 2, 1)):
        g = _fit(gate_mlp(width=width, depth=depth), boundary_sampler, seed, steps * mult)
        capacity.append({
            "width": width, "depth": depth, "steps": steps * mult,
            "signed_accept": _accept(g, valid_markers(rng, centre, 30_000)),
            "false_accept_declared_region": _accept(g, deleted_region(rng, centre, n_eval)),
        })
    out["capacity"] = capacity

    ep = out["epoch"]
    out["metrics"] = {
        "anchor_margin_accept_at_0p70": curve_a[BANDS.index(0.70)],
        "anchor_margin_accept_at_0p90": curve_a[BANDS.index(0.90)],
        "anchor_margin_annulus_accept": out["margin"]["annulus_accept"],
        "anchor_margin_false_accept": out["margin"]["false_accept_rejection_sampled"],
        "boundary_annulus_accept": out["boundary"]["annulus_accept"],
        "boundary_false_accept": out["boundary"]["false_accept_declared_region"],
        "capacity_best_false_accept": min(
            [c["false_accept_declared_region"] for c in capacity]
            + [out["boundary"]["false_accept_declared_region"]]
        ),
        "live_current_accept": min(r["current_accept"] for r in live_rows),
        "live_stale_accept_best": min(r["stale_accept"] for r in live_rows),
        "oracle_false_accept": out["oracle"]["false_accept_declared_region"],
        # controls, reported not scored
        "epoch_hostable_frozen_gate": ep["frozen_gate"]["hostable_revocable_epochs"],
        "epoch_hostable_store_frozen": ep["store_frozen"]["hostable_revocable_epochs"],
        "epoch_hostable_sphere_frozen": ep["sphere_frozen"]["hostable_revocable_epochs"],
        "epoch_hostable_store_live": ep["store_live"]["hostable_revocable_epochs"],
        "epoch_gap_frozen_gate": ep["frozen_gate"]["gap"],
        "epoch_gap_store_frozen": ep["store_frozen"]["gap"],
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
        checks[name] = False if w is None else (w >= threshold if op == ">=" else
                                                w <= threshold if op == "<=" else w == threshold)
    for name in CONTROL_METRICS:
        vals = [s["metrics"][name] for s in per_seed if s["metrics"][name] is not None]
        observed[name] = max(vals) if vals else None

    anchor_ok = all(checks[k] for k in ANCHOR_BARS)
    # §31.12's fix works if the boundary moves onto the declared predicate at SOME swept capacity.
    s31_12_fix_works = anchor_ok and checks["capacity_best_false_accept"]
    # The refuted framing: does "learned" do any work in the epoch arm?  It does only if the
    # learned gate differs from the non-learned frozen controls.
    learnedness_is_the_variable = any(
        s["metrics"]["epoch_hostable_frozen_gate"] != s["metrics"]["epoch_hostable_store_frozen"]
        for s in per_seed
    )
    liveness_is_the_variable = all(
        (s["metrics"]["epoch_hostable_store_live"] or 0) > (s["metrics"]["epoch_hostable_store_frozen"] or 0)
        for s in per_seed
    )
    return {
        "checks": checks,
        "observed_worst_seed": observed,
        "bars": {k: {"op": v[0], "threshold": v[1]} for k, v in BARS.items()},
        "anchor_reproduces_recorded_gate": anchor_ok,
        "s31_12_fix_works": s31_12_fix_works,
        "learnedness_is_the_variable": learnedness_is_the_variable,
        "liveness_is_the_variable": liveness_is_the_variable,
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
        "withdrawn_framing": (
            "'Revocation cannot be delegated to a frozen LEARNED acceptance function' is withdrawn. "
            "Its metric was arithmetically unsatisfiable and its numbers are reproduced by the store's "
            "own hand-written predicate; the variable is liveness of the reference, not learning, which "
            "§31.47 already concluded and Needham-Schroeder 1978 and Christ & Bonneau (FC 2023) own."
        ),
        "per_seed": per_seed,
        **score(per_seed),
        "interpretation_limit": (
            "The gate is trained standalone, which is what so.model reads it as (a function of the "
            "marker alone) and how E-000021 scored 11 recorded checkpoints. The anchor bars check that "
            "this reproduces the recorded full-model gate before any other arm is read. Nothing here "
            "measures the trained adapter's reading behaviour, and nothing here is a security claim: an "
            "adversary who can choose the marker is not modelled."
        ),
        "not_claimed": [
            "learned gating", "verification losses", "marker/capability schemes", "MVCC",
            "versioned pointers", "cache invalidation", "leases", "epochs", "nonces",
            "freshness-requires-liveness", "revocable proof systems (Christ & Bonneau, FC 2023)",
            "approximation error of learned classifiers",
        ],
    }
    out = Path(args.results_dir) / "e000056_learned_custodian.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({k: record[k] for k in ("observed_worst_seed", "checks",
                                             "anchor_reproduces_recorded_gate", "s31_12_fix_works",
                                             "learnedness_is_the_variable", "liveness_is_the_variable")},
                     indent=1, default=float))
    if not record["anchor_reproduces_recorded_gate"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
