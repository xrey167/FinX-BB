"""E-000064 -- temporal semantics of symlinks to an MVCC canonical pod.

Question: does a pointer alias behave like an identity reference to ONE canonical mutable knowledge
object across UPDATE/ROLLBACK/REVOKE/SHRED/EVICT/RESTORE/DELETE, or can a stale payload copy remain
reachable? This is a prerequisite for the stronger Workspace-Native Versioned Indirection thesis.

This experiment is not a novelty claim: MVCC and pointers are old. It falsifies the substrate if any
alias diverges from the canonical pod's currently committed/readable state.

Registered contract:
  UPDATE: every alias follows the new active version with zero relinks.
  ROLLBACK: every alias follows the selected historical version with zero relinks.
  REVOKE: root and every alias resolve UNKNOWN.
  RESTORE after revoke: root and aliases resolve the active version again.
  SHRED: root and every alias resolve UNKNOWN when marker validity is respected.
  RESIGN: reachability returns without relinking.
  EVICT: root and aliases resolve UNKNOWN while versions remain retained.
  RESTORE after evict: reachability returns.
  DELETE: root is gone and every alias remains dangling permanently; restore is forbidden.

Run: python -m so.experiments.e000064_versioned_symlink_semantics --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from so.mvcc import MVCCStore

UNKNOWN = None


def values(store: MVCCStore, keys: list[tuple[int,int]]) -> list[int|None]:
    view = store.index_view(respect_markers=True)
    return [view.get(k) for k in keys]


def all_value(store: MVCCStore, keys: list[tuple[int,int]], value: int|None) -> bool:
    return all(v == value for v in values(store, keys))


def run(seed: int, aliases: int) -> Dict[str, object]:
    s = MVCCStore(seed=seed)
    root_key = (100, 0)
    root = s.write(*root_key, 10, provenance="pod")
    keys = [root_key]
    links = []
    for i in range(aliases):
        k = (200 + i, 0)
        links.append(s.link(*k, root, provenance="alias"))
        keys.append(k)

    checks: Dict[str,bool] = {}
    checks["initial"] = all_value(s, keys, 10)

    v2 = s.update(root, 20)
    checks["update_follows"] = v2 == 2 and all_value(s, keys, 20)
    checks["update_zero_relinks"] = sum(op == "relink" for op, _ in s.log) == 0

    v3 = s.update(root, 30)
    checks["second_update_follows"] = v3 == 3 and all_value(s, keys, 30)
    s.rollback(root, 1)
    checks["rollback_follows"] = all_value(s, keys, 10)
    s.rollback(root, 3)
    checks["rollback_forward_follows"] = all_value(s, keys, 30)

    s.revoke(root)
    checks["revoke_closes_all"] = all_value(s, keys, UNKNOWN)
    s.restore(root)
    checks["restore_reopens_all"] = all_value(s, keys, 30)

    s.shred(root)
    checks["shred_closes_all"] = all_value(s, keys, UNKNOWN)
    s.resign(root)
    checks["resign_reopens_all"] = all_value(s, keys, 30)

    n_versions_before = len(s.cells[root].versions)
    s.evict(root)
    checks["evict_closes_all"] = all_value(s, keys, UNKNOWN)
    checks["evict_retains_history"] = len(s.cells[root].versions) == n_versions_before
    s.restore(root)
    checks["restore_after_evict"] = all_value(s, keys, 30)

    # Delete is the irreversible generation boundary in the current store: kid is never reused.
    s.delete(root)
    checks["delete_dangles_all"] = all_value(s, keys, UNKNOWN)
    checks["delete_retains_alias_rows"] = all(k in s.cells for k in links)
    restore_forbidden = False
    try:
        s.restore(root)
    except KeyError:
        restore_forbidden = True
    checks["delete_restore_forbidden"] = restore_forbidden

    # The link versions must never contain a payload object; LINK_OBJ is not the target object.
    checks["aliases_pointer_only"] = all(
        all(v.obj == -1 and v.target is not None for v in s.cells[k].versions) for k in links
    )
    passed = all(checks.values())
    return {"seed": seed, "aliases": aliases, "pass": passed, "checks": checks,
            "root_versions_before_delete": n_versions_before,
            "relinks": sum(op == "relink" for op, _ in s.log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0,1,2,3,4])
    ap.add_argument("--aliases", type=int, nargs="*", default=[1,4,16,64])
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows: List[Dict[str,object]] = [run(seed, n) for seed in a.seeds for n in a.aliases]
    rec = {"experiment":"E-000064", "claim":"substrate temporal contract, not novelty",
           "all_pass": all(bool(x["pass"]) for x in rows), "rows": rows}
    p=Path(a.results_dir); p.mkdir(parents=True, exist_ok=True)
    (p/"e000064_versioned_symlink_semantics.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({"all_pass":rec["all_pass"], "runs":len(rows)}, indent=2))
    if not rec["all_pass"]:
        for r in rows:
            if not r["pass"]:
                print("FAIL", r)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
