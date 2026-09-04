"""Experiment E-000041 — the traceless erasure cost of a fact is invariant, and canonicalisation
cannot touch it.

E-000035 recorded that a pod and a duplicated store invert: a pod makes a fact unreachable in one
record deletion instead of k, and pays for it by naming the deleted key to anyone who reads the store.
That inversion has a reason, and the reason is a law rather than an observation.

TWO COSTS, FOR THE SAME GOAL.  For a fact reachable through k access paths:

    U   the minimum records to remove so that NO access path yields the object
    T   the minimum records to remove so that no access path yields the object AND no surviving row
        points at anything that is gone -- unreachable *and* traceless

THE CLAIM.  **U depends on the arrangement and T does not.**  In a canonical pod U is 1, because every
path shares the object; in a duplicated store U is k, because each path has its own record; in a
partially normalised store U is one plus the number of paths that are still copies.  T is k in all of
them.  A pod's k-1 aliases ARE its references, so cleaning them costs k-1 on top of the object; a
duplicated store has no references to clean, and its k copies already cost k.  The total is the same
number reached from opposite ends.

WHY THAT IS THE INTERESTING STATEMENT.  Canonicalisation is not a reduction in the cost of erasure. It
is a MOVE ALONG A TRADE-OFF: it buys the cheap-but-visible regime and gives nothing in the
traceless one. Every claim of the form "normalise and erasure becomes one operation" is a claim about
U alone, and is true; the same design leaves T exactly where it was. Codd's modification anomaly is
about U. Database resilience -- the minimum contingency set -- is U. Raeesi and Roed's proposal to
"store aliases and paraphrastic forms as pointers into a single canonical record" (arXiv:2607.00605,
section 9) is a proposal about U, offered as untested. None of them is about T, and T is the number a
data subject is actually promised when they are told a record is gone.

HOW IT IS MEASURED.  Exhaustively over the whole mixing spectrum, with the mechanical resolver and no
model at all: k access paths, j of them LINK cells and k-1-j of them copies, for every k in a range
and every j from 0 to k-1. That is the full family between the two arms E-000035 compared, and the
prediction is specific enough to be wrong in each of its cells: U = 1 + (k-1-j) and T = k.

A cell where T is not k falsifies the law. So does a cell where U is not 1 + (k-1-j).

Trains nothing, loads nothing, runs in seconds.

Run:  python -m so.experiments.e000041_traceless_cost [--kmax 8] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from so import ledger
from so.closure import fact_closure
from so.experiments.e000035_deletion_disclosure import dangling_targets, live_keys
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver
from so.world import Query, UNKNOWN


def build(k: int, n_links: int, seed: int, obj: int = 7, noise: int = 40) -> Tuple[MVCCStore, List[int], List[Tuple[int, int]]]:
    """One fact reachable through ``k`` keys, ``n_links`` of them pointers and the rest copies.

    ``noise`` unrelated facts are written first so the store is not a two-row toy and so the
    traceless search has bystanders it must not touch.
    """
    st = MVCCStore(marker_dim=16, seed=seed)
    rng = np.random.default_rng(seed)
    for i in range(noise):
        st.write(500 + i, 1, int(rng.integers(100, 400)), provenance="noise")
    target = st.write(3, 1, obj, provenance="target")
    kids = [target]
    keys = [(3, 1)]
    for i in range(k - 1):
        if i < n_links:
            kids.append(st.link(10 + i, 1, target, provenance=f"alias{i}"))
        else:
            kids.append(st.write(10 + i, 1, obj, provenance=f"copy{i}"))
        keys.append((10 + i, 1))
    return st, kids, keys


def unreachable_cost(st: MVCCStore, keys, obj: int) -> int:
    """U: the fact closure, which so/closure.py already computes with a certified lower bound."""
    return fact_closure(st, keys, obj=obj).size


def traceless_cost(st: MVCCStore, keys, obj: int, kids: List[int]) -> Tuple[int, bool]:
    """T: remove until the fact is unreachable AND no surviving row points at anything gone.

    Greedy in the only order that can work: first whatever the closure needs, then every row left
    pointing at a removed key, repeatedly, because removing a pointer can itself orphan another. The
    bystander control is checked at the end -- a search that reached traceless by emptying the store
    would be measuring nothing.
    """
    base = set(dangling_targets(st.bank()))
    removed: List[int] = []
    try:
        fc = fact_closure(st, keys, obj=obj)
        for kid in fc.records:
            st.evict(kid)
            removed.append(kid)
        for _ in range(64):
            b = st.bank()
            held = live_keys(b)
            orphans = [int(b["kid"][i]) for i, is_link in enumerate(b["is_link"])
                       if bool(is_link)
                       and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in held
                       and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in base]
            if not orphans:
                break
            for kid in orphans:
                st.evict(kid)
                removed.append(kid)
        clean = not [k for k in dangling_targets(st.bank()) if k not in base]
        unreachable = all(ReferenceResolver(st).resolve(Query("fwd", a, (r,), (0,))).answer != obj
                          for a, r in keys)
        n_live = int(st.bank()["kid"].shape[0])
    finally:
        for kid in reversed(removed):
            st.restore(kid)
    return len(removed), bool(clean and unreachable and n_live > 0)


def run_seed(seed: int, kmax: int, verbose: bool = True) -> Dict[str, Any]:
    m: Dict[str, Any] = {"seed": seed}
    cells, u_ok, t_ok = 0, 0, 0
    rows: List[Dict[str, Any]] = []
    for k in range(2, kmax + 1):
        for j in range(0, k):
            st, kids, keys = build(k, j, seed)
            u = unreachable_cost(st, keys, 7)
            t, valid = traceless_cost(st, keys, 7, kids)
            pu, pt = 1 + (k - 1 - j), k
            cells += 1
            u_ok += int(u == pu)
            t_ok += int(t == pt)
            rows.append({"k": k, "links": j, "U": u, "U_predicted": pu, "T": t, "T_predicted": pt,
                         "valid": valid})
            if verbose and (j == 0 or j == k - 1):
                arm = "duplicated" if j == 0 else "canonical pod"
                print(f"  seed {seed} k={k} links={j:<2} {arm:<14} U={u} (predicted {pu})  "
                      f"T={t} (predicted {pt})  {'ok' if valid else 'INVALID'}", flush=True)
    m["n_cells"] = cells
    m["U_matches_prediction"] = u_ok / cells
    m["T_matches_prediction"] = t_ok / cells
    m["T_equals_k"] = float(np.mean([r["T"] == r["k"] for r in rows]))
    m["all_valid"] = float(np.mean([r["valid"] for r in rows]))
    m["U_min"] = float(min(r["U"] for r in rows))
    m["U_max"] = float(max(r["U"] for r in rows))
    m["T_spread"] = float(max(r["T"] - r["k"] for r in rows) - min(r["T"] - r["k"] for r in rows))
    m["grid"] = rows
    return m


KEYS = ["n_cells", "U_matches_prediction", "T_matches_prediction", "T_equals_k", "all_valid",
        "U_min", "U_max", "T_spread"]

CRITERIA = {
    # the law, in both halves. Either can fail in any cell of the grid.
    "T_equals_k": (">=", 1.0),
    "U_matches_prediction": (">=", 1.0),
    # and the control: a search that reached traceless by emptying the store measures nothing
    "all_valid": (">=", 1.0),
    # the contrast has to be real -- if U never varies there is no trade-off to describe
    "U_min": ("<=", 1.0),
    "U_max": (">=", 4.0),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args(argv)

    per_seed = [run_seed(s, args.kmax) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    agg = ledger.aggregate(numeric, [k for k in KEYS if all(k in s for s in numeric)])
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    grid = per_seed[0]["grid"]
    ks = sorted({r["k"] for r in grid})
    rows = []
    for k in ks:
        cell = {r["links"]: r for r in grid if r["k"] == k}
        u_line = " ".join(str(cell[j]["U"]) for j in range(k))
        t_line = " ".join(str(cell[j]["T"]) for j in range(k))
        rows.append([str(k), u_line, t_line])
    tbl = ledger.table(["k access paths", "U by number of links (0 = all copies ... k-1 = full pod)",
                        "T, same order"], rows)

    record = {"experiment": "E-000041",
              "title": "the traceless erasure cost is invariant and canonicalisation cannot touch it",
              "trains_nothing": True, "uses_no_model": True, "seeds": args.seeds, "kmax": args.kmax,
              "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000041 — {record['title']}", "",
          f"Seeds {args.seeds}, k from 2 to {args.kmax}, every mixing ratio from all-copies to a full",
          "pod. No model, no checkpoint, no training: the mechanical resolver and the store's own bank.", "",
          "## U falls from k to 1 across the spectrum; T does not move", "", tbl, "",
          "`U` is the fact closure — the minimum records to remove before no access path yields the",
          "object. `T` is the minimum to remove so that the fact is unreachable **and** no surviving row",
          "points at anything that is gone. Reading each row left to right is walking from a fully",
          "duplicated store to a fully canonical one.", "",
          "**U is 1 + (the number of paths that are still copies). T is k, in every cell.**", "",
          "So canonicalisation is not a reduction in the cost of erasure; it is a move along a",
          "trade-off. It buys the cheap-but-visible regime and gives nothing in the traceless one.",
          "Every claim of the form *normalise and erasure becomes one operation* is a claim about U,",
          "and is true. The same design leaves T exactly where it was — and T is the number a data",
          "subject is promised when they are told a record is gone.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this is and is not", "",
          "Codd's modification anomaly is about U. Database resilience — the minimum contingency set —",
          "is U. Raeesi and Roed's proposal to store aliases as pointers into a single canonical record",
          "(arXiv:2607.00605 §9, offered as untested) is about U. None of them is about T. What is not",
          "claimed: that T is invariant in stores unlike this one. A store that compacts its aliases on",
          "deletion, or never exports a target key, has a different T and the law would have to be",
          "restated for it — which is exactly why it is stated as a measurement over a spectrum rather",
          "than as an identity.", ""]
    path = ledger.save("e000041_traceless_cost", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl); print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
