"""Experiment E-000053 -- history-independent markers, measured at the reader.

E-000051 (ledger §31.41) found that ``MVCCStore`` draws every row's marker from a seeded generator
whose POSITION encodes how many writes preceded it, so a store that wrote a pod and then evicted every
row of it (CASCADE) differs from one that never wrote it (NEVER) in the markers of every row written
after the pod, and the frozen E-000015 reader separates the two on bystander queries at AUC 0.948 at
KL 0.000 -- E-000046's exported-level history independence of 1.0000 did not reach the reader. The fix
was named there and not run: derive markers from content, never from a position in a stream.

THE OPTION. ``MVCCStore(content_markers=True)``: a row's marker is an HMAC of its EXPORTED content
(kind, key, object or pointed-at key) under a per-store secret, mapped into the same distribution the
generator draws from (normal, scale 0.05 around the centre, normalised). Default off; the recorded runs
are untouched; ``bank()``, ``state_hash`` and the gate are unchanged. The mechanism is owned
(content-derived / deterministic signatures; content-addressable storage) and is not claimed.
``so/tests/test_content_markers.py`` pins the store half: ``check_history_independence`` reports
``markers_equal`` after CASCADE under the option and not without it.

WHAT IS BY CONSTRUCTION, SAID BEFORE THE RUN. With equal content (E-000046), equal markers (the option,
the unit test) and the same row order (both stores write facts then links in world order), the CASCADE(p)
and NEVER(p) banks are bit-identical and the reader never consumes ``kid``; the reader's outputs are
then identical and the CASCADE-vs-NEVER AUC is 0.500 by the tie rule. That closure is the sentence's
clause, and it is reported (M1) as a PIPELINE CHECK -- it can fail only through the implementation (a
link marker derived from a cell id, a blanked row keeping its old marker) -- and not as the finding.

WHAT IS MEASURED, AND CAN FAIL EITHER WAY.
  R1 the frozen reader accepts a re-signed bank as its own. The reader's gate is a LEARNED function of
     the marker and was trained on generator-drawn markers; the derived markers are a different sample
     of the same family. LIVE under the option against LIVE under the generator, same content, every
     marker different: gate acceptance, top-1 agreement and max KL on classes (i) and (ii), and the
     reading accuracy of the whole bank. A bad map into the valid region, or a gate that reads the
     sample, fails this; then the channel closes only by retraining.
  R2 the row-count floor does not move. ADD2 vs PERM (E-000051's 0.9646, worst seed 0.9602) re-run
     under the option: per seed within +-0.05 of the recorded value and still >= 0.90.
  M2 what remains once the marker channel is closed. BLANK/DANGLE vs MATCHED on (ii): within +-0.05 of
     E-000051's 0.8174 / 0.8687 and below the ADD2 floor -- the residue AUCs were never marker-borne
     (the matched bank shares every marker) and must not change.
  S1 the side effect, registered. Two rows with identical exported content carry identical markers: a
     second write of a pod's target fact (DUP) has marker distance 0.000 to the first under the option
     (> 0 without). At the reader, DUP against DUPX (the duplicate re-signed with an independent
     valid draw, per pod) asks whether the identical-marker signing of a duplicate is legible on
     bystander queries. E-000035's "duplicated" arm writes copies under their OWN keys, so its markers
     stay distinct under the option; the collision is at exact content only.

READER. The synthetic E-000015 reader (checkpoints e000015_deref1_seed{0,1,2}), three seeds, 100 pods
per seed, five-fold cross-validated Mann-Whitney AUCs as in E-000051 (``run_reader_seed`` reused,
``content_markers=True``). Trains nothing. The GPT-2 half is queued with E-000051's.

Run:  python -m so.experiments.e000053_hi_markers_reader [--seeds 0 1 2] [--n-pods 100] [--threads 1]
      python -m so.experiments.e000053_hi_markers_reader --quick --seeds 0 --n-pods 6 --threads 1   (smoke)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store, valid_markers
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000051_residue_reader as E51

RECORDED = os.path.join(os.path.dirname(__file__), "..", "results", "e000051_residue_reader-syn.json")
RECORDED_KEYS = ("add2/auc_ii", "add2/auc_iii", "blank_matched/auc_ii", "dangle_matched/auc_ii", "cascade_soft/auc_ii")
ACCURACY_KEYS = 400          # keys read for the whole-bank accuracy check (base and alias, fixed draw)


def recorded_per_seed() -> Dict[int, Dict[str, float]]:
    """E-000051's per-seed synthetic numbers, the reference every 'does not move' row is read against."""
    if not os.path.exists(RECORDED):
        return {}
    with open(RECORDED) as f:
        rec = json.load(f)
    return {int(m["seed"]): {k: float(m[k]) for k in RECORDED_KEYS if k in m}
            for m in rec.get("per_reader_seed", []) if m.get("reader") == "syn"}


@torch.no_grad()
def reader_controls(S_hi: E51.Setting, S_gen: E51.Setting) -> Dict[str, float]:
    """R1 and S1: the same world, the same pods, the same queries, two marker schemes."""
    out: Dict[str, float] = {}
    model, world = S_hi.model, S_hi.world
    assert np.array_equal(S_hi.live.subject, S_gen.live.subject) and np.array_equal(S_hi.live.obj, S_gen.live.obj)
    # --- the gate on the re-signed bank
    g_hi = model.gate(S_hi.live.tensors()["marker"]).numpy().reshape(-1)
    g_gen = model.gate(S_gen.live.tensors()["marker"]).numpy().reshape(-1)
    out["live/gate_accept"] = float((g_hi >= 0.5).mean())
    out["live/gate_absdiff_max"] = float(np.abs(g_hi - g_gen).max())
    out["live/marker_absdiff_max"] = float(np.abs(S_hi.live.marker - S_gen.live.marker).max())   # must be > 0: two schemes
    # --- answers on the pods' own keys (i) and on bystanders (ii), re-signed against generator
    for cls in ("i", "ii"):
        kls, agrees = [], []
        for p in S_hi.pods:
            qs, _ = S_hi.queries(p, cls)
            kl, agree, _ = E51.kl_top1(S_hi.read(S_hi.live, qs, cls), S_gen.read(S_gen.live, qs, cls))
            kls.append(kl); agrees.append(agree)
        out[f"live/kl_max_{cls}"] = float(max(kls)); out[f"live/top1_agree_{cls}"] = float(np.mean(agrees))
    # --- whole-bank reading accuracy under both schemes
    r = np.random.default_rng(17000 + S_hi.seed)
    keys = [f.key for f in world.facts]
    keys = [keys[int(i)] for i in r.choice(len(keys), min(ACCURACY_KEYS, len(keys)), replace=False)]
    truth = np.array([world.index[k] for k in keys])
    for name, S in (("hi", S_hi), ("gen", S_gen)):
        ans = np.asarray(E15.predict(model, S.live, world, [E15._q1(world, k) for k in keys]).answers)
        out[f"live/acc_{name}"] = float((ans == truth).mean())
    out["live/acc_absdelta"] = abs(out["live/acc_hi"] - out["live/acc_gen"])
    # --- S1: the duplicate row
    f_dup, f_dupx, kls, agrees, d_hi, d_gen = [], [], [], [], [], []
    for pi, p in enumerate(S_hi.pods):
        s = S_hi._clone(); k2 = s.write(p.target[0], p.target[1], p.obj, provenance="dup")
        k1 = S_hi.kids[p.target]
        d_hi.append(float(np.linalg.norm(s.cells[k1].versions[0].marker - s.cells[k2].versions[0].marker)))
        sg = S_gen._clone(); g2 = sg.write(p.target[0], p.target[1], p.obj, provenance="dup")
        d_gen.append(float(np.linalg.norm(sg.cells[S_gen.kids[p.target]].versions[0].marker
                                          - sg.cells[g2].versions[0].marker)))
        dup = bank_from_store(s)
        row = int(np.nonzero(dup.kid == k2)[0][0])
        m2 = dup.marker.copy()
        m2[row] = valid_markers(np.random.default_rng(19000 + S_hi.seed * 1000 + pi), S_hi.centre, 1)[0]
        dupx = dataclasses.replace(dup, marker=m2)
        qs, objs = S_hi.queries(p, "ii")
        lg_d, lg_x = S_hi.read(dup, qs, "ii"), S_hi.read(dupx, qs, "ii")
        f_dup.append(S_hi.feats(lg_d, objs, "ii")); f_dupx.append(S_hi.feats(lg_x, objs, "ii"))
        kl, agree, _ = E51.kl_top1(lg_d, lg_x); kls.append(kl); agrees.append(agree)
    out["dup/pair_marker_dist_max"] = float(max(d_hi)); out["dup/pair_marker_dist_gen_min"] = float(min(d_gen))
    with torch.enable_grad():          # cv_auc fits a probe; the reads above are under no_grad
        out["dup/auc_ii"] = E51.cv_auc(np.stack(f_dup), np.stack(f_dupx), S_hi.seed) if len(S_hi.pods) >= 5 else float("nan")
    out["dup/kl_max_ii"] = float(max(kls)); out["dup/top1_agree_ii"] = float(np.mean(agrees))
    return out


def run_seed(seed: int, n_pods: int, threads: int, n_hardgate: int, rec: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    t0 = time.time()
    m = E51.run_reader_seed("syn", seed, n_pods, threads, n_hardgate, content_markers=True)
    S_hi = E51.Setting("syn", seed, n_pods, threads, content_markers=True)
    S_gen = E51.Setting("syn", seed, n_pods, threads, content_markers=False)
    m.update(reader_controls(S_hi, S_gen))
    for k in RECORDED_KEYS:
        r = rec.get(seed, {}).get(k)
        m[f"{k}_recorded"] = float("nan") if r is None else r
        m[f"{k}_absdelta"] = float("nan") if r is None else abs(float(m[k]) - r)
    m["residue_below_add2"] = float(max(m["blank_matched/auc_ii"], m["dangle_matched/auc_ii"]) <= m["add2/auc_ii"])
    m["seconds_total"] = time.time() - t0
    print(f"  seed {seed}: cascade ii {m['cascade_soft/auc_ii']:.3f} (recorded {m['cascade_soft/auc_ii_recorded']:.3f}) | "
          f"add2 ii {m['add2/auc_ii']:.3f} (d {m['add2/auc_ii_absdelta']:.3f}) | live top1 ii {m['live/top1_agree_ii']:.3f} "
          f"KL {m['live/kl_max_ii']:.4f} gate {m['live/gate_accept']:.3f} | dup dist {m['dup/pair_marker_dist_max']:.3f} "
          f"auc {m['dup/auc_ii']:.3f} | markers_equal cascade {m['store/cascade/markers_equal']:.2f}  ({m['seconds_total']:.0f}s)",
          flush=True)
    return m


CRITERIA = {
    # validity, as in E-000051
    "present/auc_i": (">=", 0.95),                                          # V1 the adversary sees a live row
    "perm/auc_ii": ("<=", 0.60), "perm/auc_iii": ("<=", 0.60),              # V2 float floor
    "store/cascade/exported_hi": (">=", 0.999), "store/cascade/markers_equal": (">=", 0.999),   # V3 the store half
    "live/marker_absdiff_max": (">=", 1e-3),                                # V4 the two schemes differ (no silent no-op)
    # R1 the frozen reader reads the re-signed bank as its own -- the control that can fail
    "live/gate_accept": (">=", 0.99), "live/top1_agree_i": (">=", 0.98), "live/top1_agree_ii": (">=", 0.98),
    "live/kl_max_i": ("<=", 0.05), "live/kl_max_ii": ("<=", 0.05), "live/acc_absdelta": ("<=", 0.01),
    # R2 the row-count floor must not move
    "add2/auc_ii": (">=", 0.90), "add2/auc_ii_absdelta": ("<=", 0.05),
    # M1 the closure (a pipeline check: implied by V2 + V3, see the docstring)
    "cascade_soft/auc_i": ("<=", 0.60), "cascade_soft/auc_ii": ("<=", 0.60), "cascade_soft/auc_iii": ("<=", 0.60),
    "enc/cascade_never_maxabs": ("<=", 1e-6),
    # M2 what remains: the residue arms unchanged, below the floor
    "blank_matched/auc_ii_absdelta": ("<=", 0.05), "dangle_matched/auc_ii_absdelta": ("<=", 0.05),
    "residue_below_add2": (">=", 1.0),
    # S1 the side effect, at the store: identical content, identical marker
    "dup/pair_marker_dist_max": ("<=", 1e-9), "dup/pair_marker_dist_gen_min": (">=", 0.05),
}

DECISION_RULE = (
    "Synthetic reader, three seeds, worst seed. VOID if present/auc_i < 0.95, if perm AUC > 0.60, or if the "
    "store half fails (cascade markers_equal < 1 or exported HI < 1: the option is not history independent at "
    "the store and nothing downstream is read). NOT USABLE if R1 fails (gate acceptance < 0.99, top-1 agreement "
    "< 0.98 or KL > 0.05 nats between the re-signed and the generator-signed live bank, or accuracy moved by more "
    "than 0.01): the frozen reader does not accept content-derived markers as its own, and the marker channel "
    "closes only by retraining -- that is the negative sentence. NOT COMPARABLE if R2 fails (add2 AUC outside "
    "+-0.05 of E-000051's per-seed value or below 0.90): the option changed the reader's arithmetic and the "
    "E-000051 table cannot be re-read under it. With V and R passing: the POSITIVE sentence needs M1 (cascade vs "
    "never <= 0.60 on every class with interface residual 0.000 -- reported as the pipeline check it is) AND M2 "
    "(blank/dangle vs matched within +-0.05 of the record and at or below the add2 floor). If M1 fails with V3 "
    "passing, a channel other than content and marker reaches the reader (row order, a placeholder column) and is "
    "named. S1 is registered, not decided on: the store-level pair distance is 0.000 by the mechanism; dup/auc_ii "
    "<= 0.60 records the identical-marker side effect as invisible at the reader, >= 0.75 as legible, the grey "
    "zone as inconclusive at this n. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-pods", type=int, default=100)
    ap.add_argument("--n-hardgate", type=int, default=20)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--quick", action="store_true", help="reduced sizes: written with a -smoke suffix, not a record")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.quick:
        args.n_hardgate = min(args.n_hardgate, 3)
    rec = recorded_per_seed()
    per = [run_seed(seed, args.n_pods, args.threads, args.n_hardgate, rec) for seed in args.seeds]
    rows = [{k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and k != "seed"} for m in per]
    keys = sorted(set.intersection(*[set(r) for r in rows]))
    agg = ledger.aggregate(rows, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    def cell(k, f="mean"):
        return f"{agg[k][f]:.3f}" if k in agg else "-"

    md = ["# E-000053 — history-independent markers, measured at the reader", "",
          f"Synthetic E-000015 reader, seeds {args.seeds}, {args.n_pods} pods per seed, ``MVCCStore(content_markers=True)``, "
          "trains nothing" + ("; REDUCED SIZES (--quick): not a record" if args.quick else "") + ". AUCs are E-000051's "
          "five-fold cross-validated Mann-Whitney statistics; 'recorded' is E-000051's value on the same seed with the "
          "generator scheme.", "",
          ledger.table(["arm (positive vs reference)", "AUC (i)", "AUC (ii) bystanders", "AUC (iii) generic",
                        "recorded (ii)", "max KL (ii)", "top-1 agree (ii)"],
                       [[f"{arm}: {pos} vs {ref}", cell(f"{arm}/auc_i"), cell(f"{arm}/auc_ii"), cell(f"{arm}/auc_iii"),
                         cell(f"{arm}/auc_ii_recorded"), cell(f"{arm}/kl_max_ii", "max"), cell(f"{arm}/top1_agree_ii", "min")]
                        for arm, (pos, ref, _) in E51.ARMS.items()]), "",
          ledger.table(["store-level (mean over pods)", "exported HI", "residue rows", "markers equal"],
                       [[n, cell(f"store/{n}/exported_hi"), cell(f"store/{n}/residue_rows"), cell(f"store/{n}/markers_equal")]
                        for n in ("cascade", "blank", "dangle")]), "",
          ledger.table(["R1: the frozen reader on the re-signed live bank", "value (worst seed)"],
                       [["gate acceptance", cell("live/gate_accept", "min")], ["max |gate delta|", cell("live/gate_absdiff_max", "max")],
                        ["top-1 agreement (i) / (ii)", f"{cell('live/top1_agree_i', 'min')} / {cell('live/top1_agree_ii', 'min')}"],
                        ["max KL (i) / (ii)", f"{cell('live/kl_max_i', 'max')} / {cell('live/kl_max_ii', 'max')}"],
                        ["reading accuracy, derived / generator", f"{cell('live/acc_hi', 'min')} / {cell('live/acc_gen', 'min')}"]]), "",
          ledger.table(["S1: the duplicate row (identical content)", "value"],
                       [["pair marker distance, option / generator", f"{cell('dup/pair_marker_dist_max', 'max')} / {cell('dup/pair_marker_dist_gen_min', 'min')}"],
                        ["DUP vs DUPX AUC (ii)", cell("dup/auc_ii", "max")], ["max KL (ii) / top-1", f"{cell('dup/kl_max_ii', 'max')} / {cell('dup/top1_agree_ii', 'min')}"]]),
          "", f"Interface residual cascade vs never {cell('enc/cascade_never_maxabs', 'max')} (E-000051: 0.0116); hard-gate "
          f"check encoding {cell('hardgate/enc_maxabs', 'max')}, logits {cell('hardgate/logit_maxabs', 'max')}.", "",
          "## The rule, fixed before the run", "", DECISION_RULE, "", "## Pre-registered criteria", "",
          ledger.criteria_table(check), ""]
    record = {"experiment": "E-000053", "title": "history-independent markers, measured at the reader",
              "evidence_level": "E5", "trains_nothing": True, "seeds": args.seeds, "n_pods": args.n_pods,
              "quick": args.quick, "decision_rule": DECISION_RULE, "per_seed": per, "aggregate": agg, "criteria": check}
    name = "e000053_hi_markers_reader" + ("-smoke" if args.quick else "")
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(record, f, indent=1, default=float)
        path = os.path.join(args.results_dir, name + ".md")
        with open(path, "w") as f:
            f.write("\n".join(md))
    else:
        path = ledger.save(name, record, "\n".join(md))
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
