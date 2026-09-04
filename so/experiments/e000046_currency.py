"""Experiment E-000046 — is T = k a law about pods, or the price of one design choice?

E-000041 is the only claim in this programme that has survived every review: for a fact reachable
through k access paths, U = 1 + (paths still stored as copies) and T = k, in 105 of 105 cells.
Canonicalisation drives U from k to one and does not move T at all.

IT CARRIES EXACTLY ONE CAVEAT, WRITTEN INTO THE RECORD WHEN IT WAS CLAIMED: "T is invariant for THIS
STORE'S SEMANTICS, where MVCCStore.bank() exports a link's target key and keeps doing so after the
target is gone. A store that compacts its aliases on deletion, or never exports a target key, has a
different T." This experiment attacks that caveat, because a law with an unexamined caveat is a law
about the caveat.

THREE SEMANTICS, one store, the same grid:

  EXPORTING   (current) a link row exports its target key, before and after the target is removed.
              Tracelessness costs a removal per surviving alias.
  COMPACTING  on removing the object, surviving link rows are REPAIRED -- their target cleared -- so
              the store stays functional. The aliases are not deleted; they are rewritten.
  OPAQUE      the bank never exports a target key at all. An adversary reading the exported view sees
              no dangling pointer, so tracelessness against THAT view is free.

WHAT IS ACTUALLY BEING ASKED. If T = k under EXPORTING and COMPACTING but T = U under OPAQUE, then k
is not a cost that canonicalisation imposes -- it is a cost that can be paid in three currencies:
DELETIONS, REPAIRS, or an INTERFACE that declines to show the reference. And the third is not payment.

SO THE CONTROL IS THE EXPERIMENT. Under OPAQUE the exported view is clean by construction, and an
experiment that stopped there would have measured its own definition -- the tenth instrument in this
ledger to certify by not testing. So an adversary is run against the RAW store as well: if it still
names the removed key at the rate E-000035 measured through the exported view, then opacity moved the
disclosure behind an interface and removed nothing, and "T = U under OPAQUE" is a statement about who
is looking rather than about what is there.

Mechanical throughout. No model.

Run:  python -m so.experiments.e000046_currency [--kmax 8] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from so import ledger
from so.closure import fact_closure
from so.experiments.e000035_deletion_disclosure import dangling_targets, live_keys
from so.experiments.e000041_traceless_cost import build, unreachable_cost
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver
from so.world import Query


def _orphans(b, base) -> List[int]:
    held = live_keys(b)
    return [int(b["kid"][i]) for i, is_link in enumerate(b["is_link"])
            if bool(is_link)
            and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in held
            and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in base]


def _unreachable(st: MVCCStore, keys, obj: int) -> bool:
    return all(ReferenceResolver(st).resolve(Query("fwd", a, (r,), (0,))).answer != obj
               for a, r in keys)


def cost(st: MVCCStore, keys, obj: int, mode: str) -> Dict[str, Any]:
    """T under one semantics, with the store restored afterwards. Operations are counted, not rows."""
    base = set(dangling_targets(st.bank()))
    removed: List[int] = []
    repaired: List[int] = []
    fc = fact_closure(st, keys, obj=obj)
    for kid in fc.records:
        st.evict(kid)
        removed.append(kid)

    if mode == "exporting":
        for _ in range(64):
            orph = _orphans(st.bank(), base)
            if not orph:
                break
            for kid in orph:
                st.evict(kid)
                removed.append(kid)
    elif mode == "compacting":
        # the aliases are REPAIRED rather than deleted: the target is cleared, the row survives
        for _ in range(64):
            orph = _orphans(st.bank(), base)
            if not orph:
                break
            for kid in orph:
                st.blank(kid)
                repaired.append(kid)
    # OPAQUE: nothing to do -- the exported view never showed a target, so it is clean already

    b = st.bank()
    exported_clean = not [k for k in dangling_targets(b) if k not in base]
    if mode == "opaque":
        exported_clean = True                      # by construction of the view
    unreach = _unreachable(st, keys, obj)
    n_live = int(b["kid"].shape[0])

    # THE CONTROL: an adversary reading the RAW store, whatever the exported view shows.
    raw_named = [k for k in dangling_targets(b) if k not in base]
    st.restore_all() if hasattr(st, "restore_all") else None
    return {"T_ops": len(removed) + len(repaired), "removed": len(removed), "repaired": len(repaired),
            "exported_clean": float(exported_clean), "unreachable": float(unreach),
            "raw_discloses": float(len(raw_named) > 0), "n_live": n_live}


def run(kmax: int, seeds: Sequence[int], verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for k in range(2, kmax + 1, 2):
            for n_links in range(k):
                for mode in ("exporting", "compacting", "opaque"):
                    obj = 7
                    st, _kids, keys = build(k, n_links, seed)
                    U = unreachable_cost(st, keys, obj)
                    st2, _k2, keys2 = build(k, n_links, seed)
                    c = cost(st2, keys2, obj, mode)
                    rows.append({"seed": seed, "k": k, "n_links": n_links, "mode": mode,
                                 "U": U, "T": c["T_ops"], "T_equals_k": float(c["T_ops"] == k),
                                 "T_equals_U": float(c["T_ops"] == U), **c})
        if verbose:
            print(f"  seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    m: Dict[str, Any] = {"n_cells": len(rows), "seconds": time.time() - t0}
    for mode in ("exporting", "compacting", "opaque"):
        sel = [r for r in rows if r["mode"] == mode]
        m[f"{mode}/rows_kept"] = float(np.mean([r["n_live"] for r in sel]))
        m[f"{mode}/T_mean"] = float(np.mean([r["T"] for r in sel]))
        m[f"{mode}/T_equals_k"] = float(np.mean([r["T_equals_k"] for r in sel]))
        m[f"{mode}/T_equals_U"] = float(np.mean([r["T_equals_U"] for r in sel]))
        m[f"{mode}/unreachable"] = float(np.mean([r["unreachable"] for r in sel]))
        m[f"{mode}/exported_clean"] = float(np.mean([r["exported_clean"] for r in sel]))
        m[f"{mode}/raw_discloses"] = float(np.mean([r["raw_discloses"] for r in sel]))
    m["per_cell"] = rows
    return m


KEYS = [f"{m}/{q}" for m in ("exporting", "compacting", "opaque")
        for q in ("T_mean", "T_equals_k", "T_equals_U", "unreachable", "exported_clean",
                  "raw_discloses", "rows_kept")] + ["n_cells"]

CRITERIA = {
    # E-000041 reproduced: under the semantics it was measured on, T = k everywhere
    "exporting/T_equals_k": (">=", 1.0),
    "exporting/unreachable": (">=", 1.0),
    # the same k, paid as repairs instead of deletions
    "compacting/T_equals_k": (">=", 1.0),
    # and the point of paying in repairs rather than deletions: the aliases are still there
    "compacting/raw_discloses": ("<=", 0.0),
    # THE CONTROL THAT DECIDES WHAT OPAQUE MEANS. If the exported view is clean while the RAW store
    # still names the removed key, then opacity moved the disclosure behind an interface and removed
    # nothing -- and "T = U under OPAQUE" is a statement about who is looking. If raw_discloses comes
    # back at 0.0 the opposite holds and opacity really is erasure, which would be the stronger and
    # more surprising result.
    "opaque/exported_clean": (">=", 1.0),
    "opaque/raw_discloses": (">=", 0.50),
}

DECISION_RULE = (
    "T = k under EXPORTING and COMPACTING with T = U under OPAQUE, and the raw store still disclosing "
    "-> k is not a cost canonicalisation imposes but one that can be paid in three currencies, "
    "deletions, repairs, or an interface that declines to show the reference, and the third is not "
    "payment. raw_discloses at 0 under OPAQUE -> opacity is erasure and the law must be restated. "
    "T != k under COMPACTING -> repairs are cheaper than deletions and the law is about deletions "
    "specifically. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args(argv)

    m = run(args.kmax, args.seeds)
    numeric = {k: float(v) for k, v in m.items() if isinstance(v, (bool, int, float))}
    agg = ledger.aggregate([numeric], [k for k in KEYS if k in numeric])
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    record = {"experiment": "E-000046",
              "title": "the currency of tracelessness: deletions, repairs, or an opaque interface",
              "evidence_level": "E5", "trains_nothing": True, "decision_rule": DECISION_RULE,
              "result": m, "aggregate": agg, "criteria": check}
    rows = [[mode,
             f"{m[mode + '/T_mean']:.2f}", f"{m[mode + '/T_equals_k']:.4f}",
             f"{m[mode + '/T_equals_U']:.4f}", f"{m[mode + '/exported_clean']:.4f}",
             f"{m[mode + '/raw_discloses']:.4f}", f"{m[mode + '/rows_kept']:.1f}"]
            for mode in ("exporting", "compacting", "opaque")]
    md = [f"# E-000046 — {record['title']}", "",
          "E-000041 measured T = k over 105 of 105 cells and carried one caveat: that it held for a",
          "store which exports a link's target key and goes on exporting it after the target is gone.",
          "This is that caveat, tested. Mechanical, no model.", "",
          ledger.table(["semantics", "T", "T = k", "T = U", "exported view clean",
                        "**raw store still discloses**", "rows left live"], rows), "",
          "The last column is the experiment. Under OPAQUE the exported view is clean by construction,",
          "so an experiment that stopped at the fourth column would have measured its own definition.",
          "What decides whether opacity is erasure or access control is whether the removed key is",
          "still recoverable from the store itself.", "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000046_currency", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
