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
from so.audit import certify_traceless
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

    # THE CONTROL: an adversary reading the RAW STORE, whatever the exported view shows. The first
    # version of this read dangling_targets(bank()) and called it raw -- but bank() IS the exported
    # view, so it was the same check twice under two names, and it made EVICT look traceless when it
    # is not. certify_traceless walks store.cells and asks whether the removed key is still held by a
    # surviving version, which is the question. EVICT retains the row's data on purpose, so an evicted
    # alias goes on holding the removed key: clean in the view, not in the store.
    cert = certify_traceless(st, keys, obj, baseline=tuple(base),
                             ops=len(removed) + len(repaired), n_live_before=0)
    raw_named = [] if cert.raw_clean else list(cert.dangling)
    # THE CHECK THE FIRST TWO VERSIONS DID NOT HAVE (ledger §31.35). "Raw clean" is referential
    # cleanliness: no surviving version holds the removed key. It is not history independence, the
    # property §31.31 adopted as the meaning of "traceless": a store that blanked its aliases still
    # holds rows that exist only because the fact once did, and is therefore distinguishable from a
    # store that never wrote it. check_history_independence builds that never-wrote store and
    # compares, at the exported level (bank()) and at the raw level (cells, log, ids).
    hi = cert.history
    st.restore_all() if hasattr(st, "restore_all") else None
    return {"T_ops": len(removed) + len(repaired), "removed": len(removed), "repaired": len(repaired),
            "exported_clean": float(exported_clean), "unreachable": float(unreach),
            "raw_discloses": float(len(raw_named) > 0), "n_live": n_live,
            "exported_hi": float(hi.exported_hi), "markers_equal": float(hi.markers_equal),
            "raw_hi": float(hi.raw_hi), "residue_rows": float(hi.residue_rows)}


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
        m[f"{mode}/exported_hi"] = float(np.mean([r["exported_hi"] for r in sel]))
        m[f"{mode}/markers_equal"] = float(np.mean([r["markers_equal"] for r in sel]))
        m[f"{mode}/raw_hi"] = float(np.mean([r["raw_hi"] for r in sel]))
        m[f"{mode}/residue_rows"] = float(np.mean([r["residue_rows"] for r in sel]))
        # POST-HOC, LABELLED AS SUCH: the pre-registered exported_hi rows for COMPACTING and OPAQUE
        # came back at 0.2000 rather than 0.0, and the 0.2 is exactly the cells with n_links = 0 --
        # a pod of copies has no alias to blank or to leave dangling, so evicting its closure leaves
        # nothing behind under every semantics. The registered criterion did not condition on
        # n_links >= 1 and FAILS as registered; this is the same quantity over the cells the
        # prediction was about, reported beside it and not in its place.
        linked = [r for r in sel if r["n_links"] >= 1]
        m[f"{mode}/exported_hi_linked"] = float(np.mean([r["exported_hi"] for r in linked]))
        m[f"{mode}/residue_rows_linked"] = float(np.mean([r["residue_rows"] for r in linked]))
        m[f"{mode}/n_linked_cells"] = float(len(linked))
    m["per_cell"] = rows
    return m


KEYS = [f"{m}/{q}" for m in ("exporting", "compacting", "opaque")
        for q in ("T_mean", "T_equals_k", "T_equals_U", "unreachable", "exported_clean",
                  "raw_discloses", "rows_kept", "exported_hi", "markers_equal", "raw_hi",
                  "residue_rows", "exported_hi_linked", "residue_rows_linked", "n_linked_cells")] + ["n_cells"]

CRITERIA = {
    # E-000041 reproduced: under the semantics it was measured on, T = k everywhere
    "exporting/T_equals_k": (">=", 1.0),
    "exporting/unreachable": (">=", 1.0),
    # the same k, paid as repairs instead of deletions
    "compacting/T_equals_k": (">=", 1.0),
    # and the point of paying in repairs rather than deletions: the aliases are still there
    "compacting/raw_discloses": ("<=", 0.0),
    # and the correction the certificate forced: EVICT is clean in the VIEW and not in the STORE
    "exporting/raw_discloses": (">=", 0.50),
    # THE CONTROL THAT DECIDES WHAT OPAQUE MEANS. If the exported view is clean while the RAW store
    # still names the removed key, then opacity moved the disclosure behind an interface and removed
    # nothing -- and "T = U under OPAQUE" is a statement about who is looking. If raw_discloses comes
    # back at 0.0 the opposite holds and opacity really is erasure, which would be the stronger and
    # more surprising result.
    "opaque/exported_clean": (">=", 1.0),
    "opaque/raw_discloses": (">=", 0.50),
    # HISTORY INDEPENDENCE, added for the third run (ledger §31.35) and fixed before it. The
    # prediction that inverts §31.30: the rows blanking KEEPS are the residue, so compacting is not
    # history independent even at the exported level, while exporting -- evicting every row of the
    # pod -- leaves bank() identical to a store that never held the fact. And no semantics reaches
    # raw history independence, because an MVCC store keeps its log and its evicted cells on purpose.
    "compacting/exported_hi": ("<=", 0.0),
    "opaque/exported_hi": ("<=", 0.0),
    "exporting/exported_hi": (">=", 1.0),
    "exporting/raw_hi": ("<=", 0.0),
    "compacting/raw_hi": ("<=", 0.0),
}

DECISION_RULE = (
    "T = k under EXPORTING and COMPACTING with T = U under OPAQUE, and the raw store still disclosing "
    "-> k is not a cost canonicalisation imposes but one that can be paid in three currencies, "
    "deletions, repairs, or an interface that declines to show the reference, and the third is not "
    "payment. raw_discloses at 0 under OPAQUE -> opacity is erasure and the law must be restated. "
    "T != k under COMPACTING -> repairs are cheaper than deletions and the law is about deletions "
    "specifically. Fixed before the run. THIRD RUN: exported_hi at 0 under COMPACTING and at 1 under "
    "EXPORTING -> 'referentially clean' and 'history independent' are different properties, repair "
    "buys the first and deletion the second, and §31.30's 'strictly stronger AND less destructive' "
    "is withdrawn. exported_hi at 1 under COMPACTING -> blanked rows are not a residue and §31.30 "
    "stands. raw_hi at 1 anywhere -> the fresh-store comparison is broken, since the log alone "
    "distinguishes the two stores. Fixed before the third run.")


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
             f"{m[mode + '/raw_discloses']:.4f}", f"{m[mode + '/rows_kept']:.1f}",
             f"{m[mode + '/exported_hi']:.4f}", f"{m[mode + '/residue_rows']:.1f}",
             f"{m[mode + '/raw_hi']:.4f}"]
            for mode in ("exporting", "compacting", "opaque")]
    md = [f"# E-000046 — {record['title']}", "",
          "E-000041 measured T = k over 105 of 105 cells and carried one caveat: that it held for a",
          "store which exports a link's target key and goes on exporting it after the target is gone.",
          "This is that caveat, tested. Mechanical, no model.", "",
          ledger.table(["semantics", "T", "T = k", "T = U", "exported view clean",
                        "raw store still discloses", "rows left live",
                        "**history independent (exported)**", "residue rows", "history independent (raw)"],
                       rows), "",
          "`raw store still discloses` is referential: does a surviving version still hold the removed",
          "key. Under OPAQUE the exported view is clean by construction, so an experiment that stopped",
          "at the fourth column would have measured its own definition. `history independent (exported)`",
          "is the property §31.31 adopted as the meaning of traceless (Naor and Teague 2001, Def. 2.1):",
          "is `bank()` identical to that of a store that never held the fact. `residue rows` counts the",
          "exported rows that exist only because it did. `history independent (raw)` compares",
          "`store.cells`, the operation log and the next id as well, and an MVCC store fails it by",
          "design. The first two versions of this report had only the referential column and read it",
          "as the history-independence one (ledger §31.35).", "",
          "## Post hoc, labelled as such: the same column over cells that have an alias", "",
          "The registered `exported_hi` rows for COMPACTING and OPAQUE came back at 0.2000, not 0.0, and",
          "FAIL as registered. The 0.2 is exactly the cells with `n_links = 0` -- a pod made of copies",
          "has no alias to blank or to leave dangling, so evicting its closure leaves nothing behind",
          "under every semantics. The criterion should have conditioned on `n_links >= 1`; it did not,",
          "and it is not rewritten. The same quantity over the cells the prediction was about:", "",
          ledger.table(["semantics", "cells with an alias", "history independent (exported)", "residue rows"],
                       [[mode, f"{m[mode + '/n_linked_cells']:.0f}",
                         f"{m[mode + '/exported_hi_linked']:.4f}",
                         f"{m[mode + '/residue_rows_linked']:.1f}"]
                        for mode in ("exporting", "compacting", "opaque")]), "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000046_currency", record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
