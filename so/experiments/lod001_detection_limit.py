"""LOD-001 -- the detection limit of an accessibility audit, in units of retained memory.

WHY THIS EXISTS. WSC-001 (docs/novelty/audit-siting-claim.md) recorded that at E-000063's capture site
no readout of any kind separates a LIVE pod from one that was never written, while one block later
every readout separates them at 0.66 to 0.92. It separated "the audit is blind here" from "nothing is
here" with a MEDIATOR THAT IS A NORM -- how far the state moves -- and its own "how it could be wrong"
names that as the weakness: a norm is not a decodability measure.

The measurement a norm stands in for is this one. An audit that reports "no residual trace after
deletion" is asserting something about every state between a live memory and none: that had a residue
been there, it would have been seen. That assertion has a name in every measuring discipline outside
this one -- a DETECTION LIMIT -- and it has never been stated for an accessibility audit of a language
model, because a residue is not a thing a parametric memory can be given in a known amount. Here the
memory is external and its write is a tensor this harness holds, so it can be:

  the write-side ladder   alpha in [0, 1] scales the adapter's write (so/llm_adapter.py,
                          ``set_inject_scale``). Exact, and an idealisation.
  the store-side ladder   the marker of the pod's active version is rotated to chord distance c from
                          the store's centre, which moves the neural gate g in ``encode_bank`` and
                          attenuates the payload TOWARD ' unknown'. This is the residue an
                          INCOMPLETELY APPLIED SHRED actually leaves, and c = 0 is the live pod while
                          a full SHRED is an unsigned marker far outside the valid radius.

Both ladders are run because a reviewer is right to ask whether the first is an arbitrary intervention
rather than a deletion residue; the second is a lifecycle operation applied by halves, and if the two
ladders put the detection limit in the same place, the first is not an artefact of its own arbitrariness.

WHAT IS MEASURED. For every rung, at every candidate site, the SAME probes WSC-001 trains -- five
feature families, trained on the live pod's states only and applied unchanged -- score the rung
against the never-written control. The detection limit is the smallest rung at which a family's
transfer probe separates the rung from NEVER by the bar E-000063 already uses for its own validity row
(0.30). Beside it, in the same units, the BEHAVIOURAL limit: the smallest rung at which the model's
answer returns the pod's true object. Those two numbers in the same units are the field's own premise
-- output forgetting is insufficient, so audit the internals -- as a quantity rather than a slogan.

WHAT IS NOT CLAIMED HERE. Nothing about the lens, lens-based auditing, probing, projection, external
memory, pods, pointer aliases, dose-response calibration, or limits of detection as an idea: the
pre-registration at docs/novelty/lod001-preregister.md carries the boundary list. This file measures.

Run:  SO_BOS=1 python -m so.experiments.lod001_detection_limit --seeds 0 1 2 --threads 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000063_workspace_pod_certificate as E63
from so.experiments import wsc001_workspace_share as WSC
from so.jlens import jlens_vectors

# The write-side ladder. Dense below 0.25 because WSC-001's ratio row (0.032) says that is where a
# first-read-site residue would live if the certificate's site could see one at all.
ALPHA_LADDER: Tuple[float, ...] = (0.0, 0.03125, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0)

# The store-side ladder, in chord distance of the marker from the store's centre. Ledger 31.48 R3
# measured this gate's acceptance on three checkpoints: 1.0000 at every chord to 0.7, 0.0055/0.0935/
# 0.0430 at 0.8, and 0.0000 at 0.9 and beyond. The rungs bracket that transition.
CHORD_LADDER: Tuple[float, ...] = (0.0, 0.5, 0.7, 0.75, 0.8, 0.825, 0.85, 0.9, 1.0)

# The bar a rung must clear to count as DETECTED. Not a new number: it is the bar E-000063's own
# validity row uses to decide that its instrument can see a live pod (`jprobe active - never >= .30`).
# Stating the detection limit against a different bar would make it incomparable to the certificate it
# is about.
DELTA: float = 0.30

FAMILIES = ("jspace", "random", "pca", "unembed", "raw")
SITE_NAMES = tuple(f"site{l}" for l in WSC.SITES) + ("final",)


def rotate_marker(marker: np.ndarray, centre: np.ndarray, chord: float,
                  rng: np.random.Generator) -> np.ndarray:
    """A marker at chord distance ``chord`` from ``centre``, in the plane of a random orthogonal axis.

    The valid markers this store draws are unit vectors a short way from the centre (MARKER_SCALE
    0.05), so a rotation of the CENTRE by the requested chord, not of the marker, is what puts a row at
    a stated distance; the marker's own small offset is preserved by adding it back and renormalising.
    Chord distance rather than angle because that is the coordinate ledger 31.48 R3 measured the gate's
    acceptance in, so the rungs here and the acceptance curve there are the same axis.
    """
    c = np.asarray(centre, dtype=float)
    c = c / np.linalg.norm(c)
    m = np.asarray(marker, dtype=float)
    offset = m - (m @ c) * c                       # the marker's own component off the centre
    if chord <= 0.0:
        return m.copy()
    u = rng.normal(size=c.shape)
    u = u - (u @ c) * c
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        u = np.zeros_like(c); u[0] = 1.0; u = u - (u @ c) * c; nu = np.linalg.norm(u)
    u = u / nu
    theta = 2.0 * np.arcsin(min(max(chord / 2.0, 0.0), 1.0))
    c2 = np.cos(theta) * c + np.sin(theta) * u     # the rotated centre, chord away from c
    out = c2 + offset
    return out / np.linalg.norm(out)


def dose_probe(train: torch.Tensor, arms: Dict[str, torch.Tensor], y: torch.Tensor,
               groups: torch.Tensor, n_class: int, steps: int = 300) -> Dict[str, float]:
    """E-000063's transfer probe, generalised to many arms: train on ``train`` only, score every arm.

    The probe is fitted on the LIVE pod's states and its exact weights are applied unchanged to every
    rung and to the never-written control, which is what stops the alias text in the prompt from being
    read as recovered memory (E-000063's stated reason for the design). Folds are by template, as
    there.
    """
    out: Dict[str, List[float]] = {k: [] for k in arms}
    for g in groups.unique():
        tr, te = groups != g, groups == g
        mu = train[tr].mean(0); sd = train[tr].std(0).clamp(min=1e-5)
        xt = (train - mu) / sd
        w = torch.zeros(train.shape[1], n_class, requires_grad=True)
        b = torch.zeros(n_class, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=.08)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.cross_entropy(xt[tr] @ w + b, y[tr]) + 1e-3 * w.pow(2).mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            for name, x in arms.items():
                xa = (x - mu) / sd
                out[name].append(float(((xa[te] @ w + b).argmax(-1) == y[te]).float().mean()))
    return {k: float(np.mean(v)) for k, v in out.items()}


def limit_from_curve(ladder: Sequence[float], values: Sequence[float], delta: float,
                     increasing: bool = True) -> Dict[str, Any]:
    """The smallest rung clearing ``delta``, and the interpolated crossing between bracketing rungs.

    ``detected`` is False when NO rung clears the bar, which is the outcome that matters: an audit whose
    limit is above the top of the ladder could not have seen the memory even when all of it was there.
    The interpolated number is DESCRIPTIVE -- it assumes a straight line between two rungs and the
    curve is not required to be one -- and the discrete rung is what any sentence is read from.
    """
    pairs = sorted(zip(ladder, values), key=lambda t: t[0], reverse=not increasing)
    for i, (x, v) in enumerate(pairs):
        if v >= delta:
            if i == 0:
                return {"detected": True, "rung": float(x), "interpolated": float(x), "at_first_rung": True}
            x0, v0 = pairs[i - 1]
            t = 0.0 if v == v0 else (delta - v0) / (v - v0)
            return {"detected": True, "rung": float(x), "at_first_rung": False,
                    "interpolated": float(x0 + t * (x - x0))}
    return {"detected": False, "rung": None, "interpolated": None, "at_first_rung": False,
            "max_observed": float(max(values)) if len(values) else float("nan")}


def _arm_states(gk, cap, store, texts, alpha: Optional[float]):
    if alpha is not None:
        gk.model.set_inject_scale(alpha)
    try:
        return WSC.eval_states(gk, cap, store, texts)
    finally:
        if alpha is not None:
            gk.model.set_inject_scale(None)


def run_seed(gk: E8.GPT2Knowledge, centre: np.ndarray, seed: int, n_groups: int,
             modes: Sequence[str], verbose: bool = True) -> Dict[str, Any]:
    """One seed: both ladders, both address modes, every site, every family."""
    rng = np.random.default_rng(63000 + seed)
    world, spec = E15.sample_alias_world(rng, 420, max(n_groups * 2, 48), 2, gk.n_entities, 4,
                                         E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, 64000 + seed, symlink=True)
    selected, seen = [], set()
    for target, aliases in spec.groups:
        obj = int(world.index[target])
        if obj not in seen:
            selected.append((target, aliases, obj)); seen.add(obj)
        if len(selected) >= n_groups:
            break
    n = len(selected)
    obj_ids = [int(gk.model.entity_token_ids[o]) for _, _, o in selected]
    enc = gk.tok(WSC.LENS_CORPUS, return_tensors="pt", padding=True)
    w_out = gk.model.w_out
    jl: Dict[str, Any] = {f"site{l}": jlens_vectors(gk.model.lm, l + 1, obj_ids, enc["input_ids"],
                                                    enc["attention_mask"], w_out, batch=4)
                          for l in WSC.SITES}

    class _Identity:                      # at the final state J = I, so the lens IS the unembedding row
        vectors = F.normalize(w_out[torch.as_tensor(obj_ids)].float(), dim=-1)
    jl["final"] = _Identity()

    arm_names = (["never", "shred"]
                 + [f"a{a:g}" for a in ALPHA_LADDER]
                 + [f"c{c:g}" for c in CHORD_LADDER])
    cap = WSC.MultiCapture(gk)
    acc: Dict[Tuple[str, str, str], List[torch.Tensor]] = {}
    ans: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    med: Dict[Tuple[str, str, str], List[float]] = {}
    ys: Dict[str, List[int]] = {m: [] for m in modes}
    gs: Dict[str, List[int]] = {m: [] for m in modes}
    truths: Dict[str, List[int]] = {m: [] for m in modes}
    gates: Dict[str, List[float]] = {}
    achieved: Dict[str, List[float]] = {}

    for label, (target, aliases, obj) in enumerate(selected):
        never = copy.deepcopy(store)
        for ak in aliases:
            if kids[ak] in never.cells:
                never.delete(kids[ak])
        if kids[target] in never.cells:
            never.delete(kids[target])

        # The store-side rungs, built once per pod: a copy whose target cell's ACTIVE version carries a
        # marker at the requested chord from the centre. Nothing else about the store changes, and the
        # copy is discarded, so the recorded store is never mutated by this ladder.
        chord_stores: Dict[float, Any] = {}
        mrng = np.random.default_rng(91000 + 97 * seed + label)
        for c in CHORD_LADDER:
            st = copy.deepcopy(store)
            cell = st.cells[kids[target]]
            v = cell.version_obj(cell.active_version)
            v.marker = rotate_marker(v.marker, centre, float(c), mrng)
            chord_stores[c] = st

        for mode in modes:
            keys = aliases if mode == "alias" else [target]
            texts, tg = E63.texts_for(keys, gk.names, E63.TEMPLATES)

            sa, aa, inj_a = _arm_states(gk, cap, store, texts, None)          # live pod, trained write
            per_arm = {"a1": (sa, aa, inj_a)}
            sn, an, _ = _arm_states(gk, cap, never, texts, None)
            per_arm["never"] = (sn, an, None)

            store.shred(kids[target])
            ss, as_, inj_s = _arm_states(gk, cap, store, texts, None)
            store.resign(kids[target])
            per_arm["shred"] = (ss, as_, inj_s)

            for a in ALPHA_LADDER:
                if a == 1.0:
                    continue
                per_arm[f"a{a:g}"] = _arm_states(gk, cap, store, texts, float(a))
            for c in CHORD_LADDER:
                per_arm[f"c{c:g}"] = _arm_states(gk, cap, chord_stores[c], texts, None)

            for arm, (st_, an_, _) in per_arm.items():
                for site in st_:
                    acc.setdefault((mode, site, arm), []).append(st_[site])
                ans.setdefault((mode, arm), []).append(an_)
            for site in sa:
                med.setdefault((mode, site, "shred_moves"), []).append(float((sa[site] - ss[site]).abs().max()))
                med.setdefault((mode, site, "never_moves"), []).append(float((sa[site] - sn[site]).abs().max()))
            if inj_a is not None and inj_s is not None:
                for j, l in enumerate(gk.model.cfg.read_layers):
                    med.setdefault((mode, f"write{l}", "shred_moves"), []).append(
                        float((inj_a[:, j] - inj_s[:, j]).abs().max()))
                    med.setdefault((mode, f"write{l}", "norm_active"), []).append(
                        float(inj_a[:, j].norm(dim=-1).mean()))

            ys[mode].extend([label] * len(texts)); gs[mode].extend(tg)
            truths[mode].extend([obj] * len(texts))

        # The gate the store-side ladder actually produced, recorded so the chord rungs are read in the
        # quantity that reaches the payload rather than in the coordinate they were requested in.
        for c in CHORD_LADDER:
            b = E63.bank_from_store(chord_stores[c])
            row = int(np.where(np.asarray(b.kid) == kids[target])[0][0])
            with torch.no_grad():
                g = gk.model.encode_bank(b.tensors())["gate"]
            gates.setdefault(f"c{c:g}", []).append(float(g[row]))
            # The chord actually achieved. A valid marker sits a short way OFF the centre already
            # (MARKER_SCALE 0.05), so rotating the centre by c leaves the marker at c plus that
            # scatter; the requested rung is the knob and this is the coordinate the gate sees.
            mk = np.asarray(b.marker[row], dtype=float)
            ctr = np.asarray(centre, dtype=float); ctr = ctr / np.linalg.norm(ctr)
            achieved.setdefault(f"c{c:g}", []).append(float(np.linalg.norm(mk - ctr)))
        if verbose and (label + 1) % 4 == 0:
            print(f"  seed {seed}: {label + 1}/{n} pods", flush=True)
    cap.close()

    out: Dict[str, Any] = {"seed": seed, "n_pods": n, "chance": 1.0 / n, "arms": arm_names,
                           "alpha_ladder": list(ALPHA_LADDER), "chord_ladder": list(CHORD_LADDER),
                           "delta": DELTA, "gate_by_chord": {k: float(np.mean(v)) for k, v in gates.items()},
                           "achieved_chord": {k: float(np.mean(v)) for k, v in achieved.items()}}
    for (mode, site, what), v in med.items():
        out[f"{mode}/{site}/{what}"] = float(np.mean(v))

    for mode in modes:
        y = torch.tensor(ys[mode], dtype=torch.long)
        g = torch.tensor(gs[mode], dtype=torch.long)
        truth = torch.tensor(truths[mode], dtype=torch.long)
        beh = {arm: float((torch.cat(v) == truth).float().mean()) for (m_, arm), v in ans.items() if m_ == mode}
        for arm, v in beh.items():
            out[f"{mode}/answer/{arm}"] = v
        for site in SITE_NAMES:
            states = {arm: torch.cat(acc[(mode, site, arm)]) for arm in arm_names}
            fam_feats = _families(states, jl[site].vectors, w_out[torch.as_tensor(obj_ids)], n, seed)
            for fam, feats in fam_feats.items():
                p = dose_probe(feats["a1"], feats, y, g, n)
                for arm, v in p.items():
                    out[f"{mode}/{site}/{fam}/{arm}"] = v
                    out[f"{mode}/{site}/{fam}/{arm}_minus_never"] = v - p["never"]
                for axis, ladder in (("alpha", ALPHA_LADDER), ("chord", CHORD_LADDER)):
                    pre = "a" if axis == "alpha" else "c"
                    # A chord rung is a LARGER residue when the chord is SMALLER, so the chord ladder is
                    # searched from the top down; alpha from the bottom up. Both then read "the smallest
                    # residue this readout can see".
                    xs = [x for x in ladder]
                    vs = [p[f"{pre}{x:g}"] - p["never"] for x in xs]
                    lim = limit_from_curve(xs, vs, DELTA, increasing=(axis == "alpha"))
                    out[f"{mode}/{site}/{fam}/LOD_{axis}"] = lim
                    out[f"{mode}/{site}/{fam}/curve_{axis}"] = [float(v_) for v_ in vs]
            if verbose:
                j = out[f"{mode}/{site}/jspace/LOD_alpha"]
                print(f"  seed {seed} {mode:6s} {site:6s} jspace live={out[f'{mode}/{site}/jspace/a1_minus_never']:+.3f} "
                      f"LOD_alpha={'>1' if not j['detected'] else format(j['rung'], '.4g')}", flush=True)
        for axis, ladder in (("alpha", ALPHA_LADDER), ("chord", CHORD_LADDER)):
            pre = "a" if axis == "alpha" else "c"
            xs = [x for x in ladder]
            vs = [beh[f"{pre}{x:g}"] - beh["never"] for x in xs]
            out[f"{mode}/answer/LOD_{axis}"] = limit_from_curve(xs, vs, DELTA, increasing=(axis == "alpha"))
            out[f"{mode}/answer/curve_{axis}"] = [float(v_) for v_ in vs]
    return out


def _families(states: Dict[str, torch.Tensor], atoms: torch.Tensor, unembed: torch.Tensor,
              k: int, seed: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """WSC-001's five families over an arbitrary set of arms, with the bases fitted on the LIVE arm.

    ``pca`` is the top-k of the live pod's states and NOT refitted per rung: a basis refitted on each
    rung would be a different readout at every rung and the ladder would not be one curve.
    """
    def cos(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1) @ b.t()
    a = F.normalize(atoms.float(), dim=-1)
    u = F.normalize(unembed.float(), dim=-1)
    d = states["a1"].shape[1]
    r = WSC.random_basis(d, k, seed=seed).t()
    mu = states["a1"].float().mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(states["a1"].float() - mu, full_matrices=False)
    pca = v[:k]
    return {
        "jspace": {s: cos(x, a) for s, x in states.items()},
        "random": {s: cos(x, r) for s, x in states.items()},
        "pca": {s: cos(x, pca) for s, x in states.items()},
        "unembed": {s: cos(x, u) for s, x in states.items()},
        "raw": {s: x.float() for s, x in states.items()},
    }


def _lod_str(l: Any, top: float) -> str:
    """A limit as a reader should see it: the rung, or '> top' when no rung cleared the bar."""
    if not isinstance(l, dict):
        return "?"
    if not l.get("detected"):
        m = l.get("max_observed")
        return f"**> {top:g}**" + (f" (max {m:+.3f})" if isinstance(m, float) else "")
    return f"{l['rung']:g}"


def _worst_lod(per_seed: List[Dict[str, Any]], key: str, top: float, increasing: bool) -> Dict[str, Any]:
    """The worst seed's limit: for a detection limit, worst = LEAST SENSITIVE = the largest rung.

    An undetected seed is worst of all and dominates: one seed on which the audit could not see the
    memory at any dose is not averaged away by two on which it could.
    """
    vals = [s.get(key) for s in per_seed if isinstance(s.get(key), dict)]
    if not vals:
        return {"detected": False, "rung": None, "missing": True}
    if any(not v.get("detected") for v in vals):
        und = [v for v in vals if not v.get("detected")]
        return {"detected": False, "rung": None,
                "max_observed": max((v.get("max_observed", float("-inf")) for v in und)),
                "n_undetected": len(und), "n_seeds": len(vals)}
    rungs = [v["rung"] for v in vals]
    worst = max(rungs) if increasing else min(rungs)
    return {"detected": True, "rung": float(worst), "spread": [float(min(rungs)), float(max(rungs))],
            "n_seeds": len(vals)}


def summarise(rec: Dict[str, Any]) -> str:
    """The record as the tables a reader checks the pre-registered rows against."""
    per = rec["per_seed"]
    seeds = rec["seeds"]
    delta = rec["delta"]
    a_top, c_top = max(rec["alpha_ladder"]), min(rec["chord_ladder"])
    out: List[str] = []
    out.append(f"# LOD-001 — the detection limit of an accessibility audit\n")
    out.append(f"Seeds {seeds}, {per[0]['n_pods']} pods per seed, delta = {delta}, "
               f"modes {rec['modes']}. Worst seed reported in every summary row; a limit is 'worst' "
               f"when it is the LEAST sensitive, and an undetected seed dominates.\n")
    out.append(f"Pre-registration: `docs/novelty/lod001-preregister.md`. Every bar in it was written "
               f"before this file existed.\n")

    out.append("\n## Validity rows (worst seed)\n")
    out.append("| row | observed | bar | |")
    out.append("|---|---|---|---|")
    v1 = min(s.get("alias/answer/a1", float("nan")) for s in per)
    v2 = min(s.get("alias/final/raw/a1_minus_never", float("nan")) for s in per)
    v3a = max(s.get("alias/answer/a0", 0.0) - s.get("alias/answer/never", 0.0) for s in per)
    v3b = max(abs(s.get("alias/final/raw/a0_minus_never", 0.0)) for s in per)
    g0 = min(s["gate_by_chord"].get("c0", float("nan")) for s in per)
    g1 = max(s["gate_by_chord"].get("c1", float("nan")) for s in per)
    for name, obs, op, bar in (("V1 alias/answer/a1", v1, ">=", 0.80),
                               ("V2 alias/final/raw a1-never", v2, ">=", 0.30),
                               ("V3a alias/answer a0-never", v3a, "<=", 0.05),
                               ("V3b |alias/final/raw a0-never|", v3b, "<=", 0.10),
                               ("V4a gate at chord 0", g0, ">=", 0.90),
                               ("V4b gate at chord 1.0", g1, "<=", 0.10)):
        ok = (obs >= bar) if op == ">=" else (obs <= bar)
        out.append(f"| `{name}` | {obs:.4f} | {op} {bar} | {'PASS' if ok else '**FAIL**'} |")

    for mode in rec["modes"]:
        out.append(f"\n## Detection limits — {mode} mode, worst seed\n")
        out.append(f"A cell is the smallest rung whose separation from the never-written control "
                   f"reaches {delta}. `> {a_top:g}` means no rung cleared it: the audit could not have "
                   f"seen the memory with all of it present.\n")
        out.append("| site | " + " | ".join(FAMILIES) + " | answer |")
        out.append("|---|" + "---|" * (len(FAMILIES) + 1))
        ans = _worst_lod(per, f"{mode}/answer/LOD_alpha", a_top, True)
        for site in SITE_NAMES:
            cells = [_lod_str(_worst_lod(per, f"{mode}/{site}/{f}/LOD_alpha", a_top, True), a_top)
                     for f in FAMILIES]
            out.append(f"| `{site}` | " + " | ".join(cells) + f" | {_lod_str(ans, a_top)} |")
        out.append(f"\nStore-side ladder (marker chord; a SMALLER chord is a LARGER residue, so the "
                   f"limit is the LARGEST chord still detected and `< {c_top:g}` means none was):\n")
        out.append("| site | " + " | ".join(FAMILIES) + " | answer |")
        out.append("|---|" + "---|" * (len(FAMILIES) + 1))
        ansc = _worst_lod(per, f"{mode}/answer/LOD_chord", c_top, False)
        for site in SITE_NAMES:
            cells = [_lod_str(_worst_lod(per, f"{mode}/{site}/{f}/LOD_chord", c_top, False), c_top)
                     for f in FAMILIES]
            out.append(f"| `{site}` | " + " | ".join(cells) + f" | {_lod_str(ansc, c_top)} |")

        out.append(f"\n### The curves — {mode}, `jspace` and the answer, per seed\n")
        out.append("| seed | site | " + " | ".join(f"a={a:g}" for a in rec["alpha_ladder"]) + " |")
        out.append("|---|---|" + "---|" * len(rec["alpha_ladder"]))
        for s in per:
            for site in SITE_NAMES:
                c = s.get(f"{mode}/{site}/jspace/curve_alpha")
                if c:
                    out.append(f"| {s['seed']} | `{site}` | " + " | ".join(f"{v:+.3f}" for v in c) + " |")
            c = s.get(f"{mode}/answer/curve_alpha")
            if c:
                out.append(f"| {s['seed']} | answer | " + " | ".join(f"{v:+.3f}" for v in c) + " |")

    out.append("\n## Mediators (worst seed), the rows WSC-001 reports beside every probe number\n")
    out.append("| mode | site | shred_moves | never_moves |")
    out.append("|---|---|---|---|")
    for mode in rec["modes"]:
        for site in list(SITE_NAMES) + [f"write{l}" for l in (8, 10)]:
            sm = [s.get(f"{mode}/{site}/shred_moves") for s in per]
            nm = [s.get(f"{mode}/{site}/never_moves") for s in per]
            if any(x is not None for x in sm):
                f = lambda xs: f"{max(x for x in xs if x is not None):.3f}" if any(x is not None for x in xs) else "-"
                out.append(f"| {mode} | `{site}` | {f(sm)} | {f(nm)} |")

    out.append("\n## The store-side ladder, in the quantity the payload sees\n")
    out.append("| requested chord | achieved chord | gate |")
    out.append("|---|---|---|")
    for c in rec["chord_ladder"]:
        k = f"c{c:g}"
        ach = float(np.mean([s["achieved_chord"][k] for s in per if k in s.get("achieved_chord", {})]))
        gt = float(np.mean([s["gate_by_chord"][k] for s in per if k in s.get("gate_by_chord", {})]))
        out.append(f"| {c:g} | {ach:.3f} | {gt:.4f} |")
    out.append(f"\nRun in {rec['seconds']:.0f}s.\n")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-groups", type=int, default=16)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--modes", nargs="*", default=["alias", "direct"])
    ap.add_argument("--results-dir", default="so/results/lod001")
    ap.add_argument("--suffix", default="_bos")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()
    per_seed = []
    for s in a.seeds:
        gk, centre, ck = WSC.load_bos_adapter(s, a.suffix)
        print(f"=== LOD-001 seed {s} ({ck}) ===", flush=True)
        r = run_seed(gk, centre, s, a.n_groups, a.modes)
        r["checkpoint"] = ck
        per_seed.append(r)
    rec = {"experiment": "LOD-001", "seeds": list(a.seeds), "n_groups": a.n_groups,
           "modes": list(a.modes), "delta": DELTA, "bos": os.environ.get("SO_BOS", "") == "1",
           "alpha_ladder": list(ALPHA_LADDER), "chord_ladder": list(CHORD_LADDER),
           "per_seed": per_seed, "seconds": time.time() - t0}
    p = Path(a.results_dir); p.mkdir(parents=True, exist_ok=True)
    (p / "lod001_detection_limit.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    (p / "lod001_detection_limit.md").write_text(summarise(rec), encoding="utf-8")
    print(f"wrote {p / 'lod001_detection_limit.json'} in {rec['seconds']:.0f}s", flush=True)
    return rec


if __name__ == "__main__":
    main()
