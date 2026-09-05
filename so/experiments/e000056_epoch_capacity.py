"""Experiment E-000056 -- can a frozen learned acceptance gate host a freshness predicate?

THE QUESTION, AND WHERE IT COMES FROM. A parallel research branch audited this architecture and
narrowed its novelty candidate to one requirement: neural material derived from an authorized read
(a materialized Bank, a routing distribution, a cached activation) must not act as a bearer
capability -- when it is consumed later, its authority must be revalidated against the live referent,
and a stale generation must fail closed (ledger 31.47). Every mechanism in that sentence is owned:
NFS file handles carry a generation number and a stale one returns ESTALE (Sandberg et al., 1985),
leases bound the validity of a cached right (Gray and Cheriton, SOSP 1989), optimistic concurrency
control validates at commit (Kung and Robinson, 1981), and revocation by indirection through a
versioned pointer is Redell (1974). The IETF draft the other branch cites (PAMSPEC, July 2026)
already declares derived representations non-authoritative.

What is NOT owned is the question this substrate can answer, because it is a property of a LEARNED
component rather than of a database: **where can the freshness check live?** In this architecture the
reader's acceptance of a row is a trained network -- ``marker_gate``, a 16 -> 64 -> 1 MLP over the
row's marker -- whose operational radius was measured at 0.90 against a declared 0.35 (E-000029), and
whose acceptance survives a change of marker SCHEME without retraining: content-derived markers (an
HMAC of the exported content, drawn into the same normal(0.05) family around the same centre) are
accepted at 1.000 with KL <= 0.0001 (E-000053). So the gate is indifferent to which sample of the
family signs a row. The freshness question is whether it can be made to care about WHICH EPOCH signed
it, without being retrained.

THE PROPOSITION, WHICH IS PRIOR TO THE MEASUREMENT AND SETTLES IT. ``gate_logits`` is a pure function
of one argument: the row's own marker (``so/model.py:176``, ``so/llm_adapter.py:200``). A retained
bank carries its markers with it, unchanged. Therefore the gate's verdict on a retained row is
constant in time, and NO signing schedule -- epochs, rotating keys, shrinking leases, generation
numbers in the marker -- can make a row that the gate accepted when it was written stop being accepted
later, because nothing about the row or the gate changes. A freshness predicate needs an input that
changes between the write and the read; the gate has none. So the freshness check cannot live in this
learned reader, by construction of its interface, and any design that promises attestation "at the
point of neural consumption" must either give the reader live state (an epoch nonce mixed in at
materialisation, which is the store touching every row, which is materialising the bank) or place the
check in the store before materialisation. That is a one-line argument, not a result, and it is stated
here first so that nothing below is read as discovering it.

WHAT IS THEREFORE MEASURED, AND WHAT CAN STILL FAIL. What remains is a description of the instrument
that the argument's corollary rests on, and two rows that could come out otherwise. The corollary --
"a store cannot cycle epochs through disjoint accepting regions" -- is stronger than the argument
needs, and it is the one geometric escape somebody could propose: if the accepting set were
disconnected, a store could sign each epoch into a different component. It would still not work (a
stale marker sits in whichever component accepted it), and the geometry is reported to show what the
acceptance region actually is rather than to decide anything. The rows that CAN fail are arm K -- the
gate accepts two different HMAC keys of the same content alike, which is E-000053's finding from the
other side and which would re-open E-000053 if it failed -- and the band structure itself, which no
recorded run has ever looked at.

ARMS (all evaluation-only; nothing is trained, nothing is written to a store):
  R  RADIAL PROFILE. Accept rate over 200 shells of chord distance from the centre, sampled by
     construction (E-000029's ``shell``): the number of maximal accept bands along the radius, the
     operational radius, and whether the profile is monotone.
  T  TANGENTIAL PROFILE. The same count along great circles at a fixed radius inside the accepting
     cap, in 64 random directions: an accept region can be non-radial, and a store could cycle epochs
     by DIRECTION at constant radius. Reports the fraction of each circle that accepts and the number
     of alternations.
  E  THE EPOCH TEST ITSELF, the row that decides. For a grid of radii and for random direction pairs,
     sign a live bank at epoch e and a stale bank at epoch e-1 and measure the gate's acceptance of
     each. Two schedules: EQUIDISTANT (two directions at the same radius -- the scheme a key rotation
     gives) and INWARD (successive radii -- the scheme a shrinking lease gives). An epoch pair
     SUCCEEDS iff accept(live) >= 0.99 and accept(stale) <= 0.01. The capacity is the longest chain
     of successive epochs that all succeed.
  K  THE KEY-ROTATION CONTROL. Sign the same content under two different HMAC keys, both drawn into
     the family E-000053 uses. If both are accepted (expected: they are), then a scheme that keeps
     the live bank readable cannot make the stale bank unreadable by key alone, which is the same
     conclusion from the other side.
  C  THE COST ROW, read only if E finds a capacity above 1: reading accuracy and KL to the recorded
     reader when live rows are signed at the epoch radius rather than at the centre.

WHAT IS NOT CLAIMED. Nothing about mechanisms: epochs, leases, generation numbers, revocation by
indirection and fail-closed validation are all owned and cited above. Nor is the proposition claimed
as a finding -- it is a property of a function signature, and this file exists to state it where the
code is, to pin it with the criteria below, and to record what the acceptance region of a trained gate
looks like. The epoch-capacity rows are BY CONSTRUCTION at most one and are reported, never scored as
a claim.

Run:  python -m so.experiments.e000056_epoch_capacity [--families e000010 e000014] [--threads 1]
      python -m so.experiments.e000056_epoch_capacity --quick --results-dir /path   (a smoke run)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256, checkpoint_path
from so.experiments.e000021_gate_error_rates import FAMILIES, gate_scores
from so.experiments.e000029_marker_geometry import shell
from so.model import ModelConfig, MutableKnowledgeTransformer

N_SHELLS = 200
SHELL_MAX = 2.0
N_PER_SHELL = 4_000
N_CIRCLES = 64
N_PER_CIRCLE = 360
N_EPOCH_ROWS = 2_000
SIGMA = 0.05                      # MVCCStore.MARKER_SCALE: a signed marker is normalise(c + N(0, sigma^2))
ACCEPT = 0.5                      # the gate's own decision threshold (sigmoid > 0.5)
LIVE_BAR, STALE_BAR = 0.99, 0.01


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def epoch_centre(centre: np.ndarray, rng: np.random.Generator, dist: float) -> np.ndarray:
    """One epoch's signing centre: a unit vector at chord distance ``dist`` from the store's centre."""
    return shell(rng, centre, dist, 1)[0]


def sign(rng: np.random.Generator, epoch_c: np.ndarray, n: int, sigma: float = SIGMA) -> np.ndarray:
    """``MVCCStore.new_valid_marker`` with the epoch's centre in place of the store's."""
    return _unit(epoch_c[None, :] + rng.normal(scale=sigma, size=(n, epoch_c.shape[0])))


def hmac_markers(key: bytes, contents: List[str], centre: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """E-000053's content-derived scheme: HMAC(key, content) seeds the draw around the same centre."""
    out = []
    for c in contents:
        h = hashlib.blake2b(c.encode(), key=key, digest_size=32).digest()
        r = np.random.default_rng(int.from_bytes(h[:8], "big"))
        out.append(centre + r.normal(scale=sigma, size=centre.shape[0]))
    return _unit(np.asarray(out))


def bands(accept: np.ndarray, thresh: float = ACCEPT) -> List[Tuple[int, int]]:
    """Maximal runs of accepting positions in a profile -- the count is the epoch capacity along it."""
    hot = accept > thresh
    out, start = [], None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        elif not h and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(hot) - 1))
    return out


def radial_profile(model, centre: np.ndarray, rng: np.random.Generator, n_shells: int,
                   n_per: int) -> Dict[str, Any]:
    dists = np.linspace(SHELL_MAX / n_shells, SHELL_MAX, n_shells)
    acc = []
    for d in dists:
        m = shell(rng, centre, float(d), n_per)
        acc.append(float((gate_scores(model, m) > ACCEPT).mean()) if m.shape[0] else 0.0)
    acc = np.asarray(acc)
    bd = bands(acc)
    hot = acc > ACCEPT
    monotone = bool(np.all(np.diff(hot.astype(int)) <= 0))          # accept ... then reject, never back
    first_zero = float(dists[np.argmax(acc == 0.0)]) if (acc == 0.0).any() else float("nan")
    return {"distances": dists.tolist(), "accept": acc.tolist(), "n_bands": len(bd),
            "bands": [[float(dists[a]), float(dists[b])] for a, b in bd],
            "monotone_accept_then_reject": monotone, "operational_radius": first_zero}


def tangential_profile(model, centre: np.ndarray, rng: np.random.Generator, radius: float,
                       n_circles: int, n_per: int) -> Dict[str, Any]:
    """Accept along great circles at fixed chord distance: is the accepting set a cap, or patchy?"""
    d = centre.shape[0]
    n_bands_seen, frac = [], []
    cos_t = 1.0 - radius ** 2 / 2.0
    sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t ** 2)))
    for _ in range(n_circles):
        u = rng.normal(size=(2, d))
        u -= (u @ centre)[:, None] * centre[None, :]
        u = _unit(u)
        u[1] -= (u[1] @ u[0]) * u[0]
        u = _unit(u)
        phi = np.linspace(0.0, 2.0 * np.pi, n_per, endpoint=False)
        tang = np.cos(phi)[:, None] * u[0][None, :] + np.sin(phi)[:, None] * u[1][None, :]
        m = cos_t * centre[None, :] + sin_t * tang
        a = (gate_scores(model, _unit(m)) > ACCEPT).astype(float)
        n_bands_seen.append(len(bands(a)))
        frac.append(float(a.mean()))
    return {"radius": radius, "n_bands_mean": float(np.mean(n_bands_seen)),
            "n_bands_max": int(np.max(n_bands_seen)), "accept_fraction_mean": float(np.mean(frac)),
            "accept_fraction_min": float(np.min(frac)), "accept_fraction_max": float(np.max(frac))}


def epoch_test(model, centre: np.ndarray, rng: np.random.Generator, radii: List[float],
               n_rows: int) -> Dict[str, Any]:
    """E: can a live epoch be accepted while the epoch before it is rejected?"""
    out: Dict[str, Any] = {}
    # EQUIDISTANT: two directions at the same radius (a key rotation)
    eq = []
    for r in radii:
        a_live = float((gate_scores(model, sign(rng, epoch_centre(centre, rng, r), n_rows)) > ACCEPT).mean())
        a_stale = float((gate_scores(model, sign(rng, epoch_centre(centre, rng, r), n_rows)) > ACCEPT).mean())
        eq.append({"radius": r, "accept_live": a_live, "accept_stale": a_stale,
                   "succeeds": bool(a_live >= LIVE_BAR and a_stale <= STALE_BAR)})
    out["equidistant"] = eq
    out["equidistant_successes"] = int(sum(e["succeeds"] for e in eq))
    out["equidistant_max_gap"] = float(max(abs(e["accept_live"] - e["accept_stale"]) for e in eq))
    # INWARD: successive radii, stale one step further out (a shrinking lease)
    inward, chain, best = [], 0, 0
    for i in range(1, len(radii)):
        r_stale, r_live = radii[i - 1], radii[i]
        a_live = float((gate_scores(model, sign(rng, epoch_centre(centre, rng, r_live), n_rows)) > ACCEPT).mean())
        a_stale = float((gate_scores(model, sign(rng, epoch_centre(centre, rng, r_stale), n_rows)) > ACCEPT).mean())
        ok = bool(a_live >= LIVE_BAR and a_stale <= STALE_BAR)
        inward.append({"radius_live": r_live, "radius_stale": r_stale, "accept_live": a_live,
                       "accept_stale": a_stale, "succeeds": ok})
        chain = chain + 1 if ok else 0
        best = max(best, chain)
    out["inward"] = inward
    out["inward_longest_chain"] = best
    out["capacity"] = max(best, out["equidistant_successes"])
    return out


def key_rotation(model, centre: np.ndarray, n_rows: int) -> Dict[str, Any]:
    """K: two HMAC keys over the same content, both drawn into E-000053's family."""
    contents = [json.dumps(["FACT", i, 0, (i * 7) % 256]) for i in range(n_rows)]
    a = gate_scores(model, hmac_markers(b"epoch-0-key", contents, centre))
    b = gate_scores(model, hmac_markers(b"epoch-1-key", contents, centre))
    return {"accept_key0": float((a > ACCEPT).mean()), "accept_key1": float((b > ACCEPT).mean()),
            "score_mean_key0": float(a.mean()), "score_mean_key1": float(b.mean()),
            "max_abs_score_gap": float(np.abs(np.sort(a) - np.sort(b)).max())}


def measure(name: str, seed: int, quick: bool) -> Dict[str, Any]:
    path = checkpoint_path(name, seed)
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    ck = torch.load(path, weights_only=False)
    model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    centre = np.asarray(ck["centre"], dtype=float)
    rng = np.random.default_rng(56_000 + seed)
    n_shells = 20 if quick else N_SHELLS
    n_per = 400 if quick else N_PER_SHELL
    n_circ = 4 if quick else N_CIRCLES
    n_pc = 60 if quick else N_PER_CIRCLE
    n_rows = 200 if quick else N_EPOCH_ROWS
    m: Dict[str, Any] = {"family": name, "seed": seed, "checkpoint": f"{name}_seed{seed}",
                         "checkpoint_sha256": _sha256(path), "marker_dim": int(centre.shape[0])}
    rad = radial_profile(model, centre, rng, n_shells, n_per)
    m["radial"] = rad
    m["radial/n_bands"] = float(rad["n_bands"])
    m["radial/monotone"] = float(rad["monotone_accept_then_reject"])
    m["radial/operational_radius"] = float(rad["operational_radius"])
    inner = max(0.05, 0.5 * (rad["operational_radius"] if np.isfinite(rad["operational_radius"]) else 0.9))
    tan = tangential_profile(model, centre, rng, inner, n_circ, n_pc)
    m["tangential"] = tan
    m["tangential/n_bands_max"] = float(tan["n_bands_max"])
    m["tangential/accept_fraction_min"] = float(tan["accept_fraction_min"])
    radii = [round(x, 4) for x in np.linspace(0.05, 1.6, 8 if quick else 32)]
    ep = epoch_test(model, centre, rng, radii, n_rows)
    m["epoch"] = ep
    m["epoch/capacity"] = float(ep["capacity"])
    m["epoch/equidistant_max_gap"] = float(ep["equidistant_max_gap"])
    m["epoch/inward_longest_chain"] = float(ep["inward_longest_chain"])
    k = key_rotation(model, centre, n_rows)
    m["key"] = k
    m["key/accept_key0"] = k["accept_key0"]
    m["key/accept_key1"] = k["accept_key1"]
    m["key/accept_gap"] = abs(k["accept_key0"] - k["accept_key1"])
    return m


# Worst checkpoint. Fixed before the run.
CRITERIA: Dict[str, Tuple[str, float]] = {
    # V: the gate is the one the record measured -- a single accepting cap with a finite radius
    "radial/operational_radius": ("<=", 2.0),
    "radial/n_bands": ("<=", 1.0),
    "tangential/n_bands_max": ("<=", 1.0),
    # K: the acceptance is a property of the region, not of the signing key (E-000053 from the other side)
    "key/accept_key0": (">=", 0.99),
    "key/accept_key1": (">=", 0.99),
    "key/accept_gap": ("<=", 0.01),
    # E: recorded, not scored -- at most one epoch, by construction of the gate's signature
    "epoch/capacity": ("<=", 1.0),
}

DECISION_RULE = (
    "Worst checkpoint over every family and seed. The epoch rows cannot come out otherwise -- the "
    "gate is a pure function of the row's own marker, so a retained row's verdict is constant in time "
    "-- and are recorded, not scored: `epoch/capacity` at most 1 and `epoch/equidistant_max_gap` are "
    "reported for the record. What is read: VOID if the radial profile has no finite operational "
    "radius (the gate accepts everywhere, and E-000029's instrument reading is wrong). RE-OPENS "
    "E-000053 if arm K fails -- if the gate does not accept two HMAC keys of the same content alike "
    "(`key/accept_key*` below 0.99 or a gap over 0.01), then E-000053's 1.000 acceptance was a "
    "property of one key and the content-marker option is re-measured before anything else is read. "
    "DISCONNECTED if the radial or tangential band count exceeds 1: the accepting set of a trained "
    "gate is not a cap, which is a fact about the instrument worth recording (it still does not buy a "
    "freshness predicate, per the proposition). CAP otherwise: one accepting region, of the recorded "
    "radius, indifferent to the signing key -- the instrument behind E-000029, E-000053 and the "
    "proposition, in one table. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quick:
        os.environ["SO_RESULT_SUFFIX"] = "-smoke"
    per: List[Dict[str, Any]] = []
    for name in args.families:
        for seed in FAMILIES[name]:
            if not checkpoint_path(name, seed).exists():
                print(f"  (skipping missing {name}_seed{seed})", flush=True)
                continue
            m = measure(name, seed, args.quick)
            per.append(m)
            print(f"  {name} seed {seed}: radial bands {m['radial/n_bands']:.0f} r={m['radial/operational_radius']:.3f} "
                  f"| tangential bands {m['tangential/n_bands_max']:.0f} | epoch capacity {m['epoch/capacity']:.0f} "
                  f"(equidistant gap {m['epoch/equidistant_max_gap']:.3f}) | key accept "
                  f"{m['key/accept_key0']:.3f}/{m['key/accept_key1']:.3f}", flush=True)
    if not per:
        raise SystemExit("no checkpoints found")
    keys = sorted(k for k in per[0] if isinstance(per[0][k], (int, float)) and k not in ("seed",))
    agg = ledger.aggregate(per, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    rows = [[m["checkpoint"], f"{m['radial/n_bands']:.0f}", f"{m['radial/operational_radius']:.3f}",
             "yes" if m["radial/monotone"] else "no", f"{m['tangential/n_bands_max']:.0f}",
             f"{m['tangential/accept_fraction_min']:.3f}", f"{m['epoch/capacity']:.0f}",
             f"{m['epoch/equidistant_max_gap']:.3f}", f"{m['key/accept_key0']:.3f} / {m['key/accept_key1']:.3f}"]
            for m in per]
    tbl = ledger.table(["checkpoint", "radial accept bands", "operational radius", "monotone",
                        "tangential bands (max)", "min accept fraction on a circle", "epoch capacity",
                        "equidistant gap", "accept, key 0 / key 1"], rows)
    record = {"experiment": "E-000056", "title": "the epoch capacity of a frozen learned acceptance gate",
              "evidence_level": "E4", "families": args.families, "quick": args.quick,
              "trains_nothing": True, "decision_rule": DECISION_RULE, "per_checkpoint": per,
              "aggregate": agg, "criteria": check,
              "control": "E-000029's radial geometry and E-000021's gate scores, unchanged; E-000053's "
                         "content-marker family for the key arm"}
    md = [f"# E-000056 — {record['title']}", "",
          "Recorded checkpoints, nothing trained, nothing written to a store. The question is where a",
          "version check can live in a memory-augmented model of this shape: a frozen gate can carry as",
          "many epochs as its acceptance function has disjoint accepting regions. Worst checkpoint.", "",
          tbl, "", "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check, basis="worst checkpoint"), ""]
    text = "\n".join(md)
    path = None
    if not args.quick:
        path = ledger.save("e000056_epoch_capacity", record, text)
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        name = "e000056_epoch_capacity" + os.environ.get("SO_RESULT_SUFFIX", "")
        record.setdefault("environment", ledger.environment())
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(ledger._to_jsonable(record), f, indent=1, sort_keys=True)
        with open(os.path.join(args.results_dir, name + ".md"), "w") as f:
            f.write(text.rstrip("\n") + "\n")
        path = path or os.path.join(args.results_dir, name + ".md")
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
