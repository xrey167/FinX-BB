"""E-000061 -- canonical pod vs duplicated aliases: closure and mutation scaling.

This is deliberately NOT a novelty claim. Canonicalisation/pointer sharing is prior art and is
already acknowledged in so.closure. E-000061 establishes the substrate quantitatively so later
J-space/versioned-indirection experiments can state exactly what the pod contributes and what it
does not.

For k aliases we compare:
  DUP: k+1 independent FACT rows all carrying the same object.
  POD: one FACT row plus k LINK rows pointing at that canonical cell.

Pre-registered structural predictions:
  * fact deletion closure is k+1 for DUP and 1 for POD;
  * changing the fact everywhere costs k+1 payload mutations for DUP and 1 for POD;
  * evicting the canonical payload makes every POD alias unresolved in one operation;
  * evicting one duplicate leaves k other routes live.

Run: python -m so.experiments.e000061_pod_closure_scaling --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from so.closure import fact_closure, pod_keys
from so.mvcc import MVCCStore


def make_duplicate(seed: int, aliases: int, obj: int = 777) -> tuple[MVCCStore, list[tuple[int,int]], list[int]]:
    s = MVCCStore(seed=seed)
    keys, kids = [], []
    for i in range(aliases + 1):
        key = (1000 + i, 0)
        kids.append(s.write(key[0], key[1], obj, provenance="duplicate"))
        keys.append(key)
    return s, keys, kids


def make_pod(seed: int, aliases: int, obj: int = 777) -> tuple[MVCCStore, list[tuple[int,int]], int, list[int]]:
    s = MVCCStore(seed=seed)
    root_key = (1000, 0)
    root = s.write(root_key[0], root_key[1], obj, provenance="canonical-pod")
    keys = [root_key]
    links = []
    for i in range(1, aliases + 1):
        key = (1000 + i, 0)
        links.append(s.link(key[0], key[1], root, provenance="symlink-alias"))
        keys.append(key)
    return s, keys, root, links


def one(seed: int, aliases: int) -> Dict[str, float]:
    dup, dkeys, dkids = make_duplicate(seed, aliases)
    pod, pkeys, root, links = make_pod(seed + 10000, aliases)

    dclose = fact_closure(dup, dkeys, obj=777, max_records=aliases + 4)
    pclose = fact_closure(pod, pkeys, obj=777, max_records=aliases + 4)

    # Mutation anomaly: one logical fact update.
    new_obj = 778
    for kid in dkids:
        dup.update(kid, new_obj)
    pod.update(root, new_obj)
    dup_update_ops = aliases + 1
    pod_update_ops = 1
    dup_consistent = all(dup.index_view().get(k) == new_obj for k in dkeys)
    pod_consistent = all(pod.index_view().get(k) == new_obj for k in pkeys)

    # One-operation erasure reachability.
    dup.evict(dkids[0])
    pod.evict(root)
    dup_live_after_one = sum(dup.index_view().get(k) == new_obj for k in dkeys)
    pod_live_after_one = sum(pod.index_view().get(k) == new_obj for k in pkeys)

    # Identity closure helper must cover every direct alias plus the root.
    pod_key_count = len(pod_keys(make_pod(seed + 20000, aliases)[0], 1))

    expected_dup = aliases + 1
    ok = (
        dclose.size == expected_dup and dclose.optimal and
        pclose.size == 1 and pclose.optimal and
        dup_update_ops == expected_dup and pod_update_ops == 1 and
        dup_consistent and pod_consistent and
        dup_live_after_one == aliases and pod_live_after_one == 0 and
        pod_key_count == aliases + 1
    )
    return {
        "seed": seed, "aliases": aliases,
        "dup_closure": dclose.size, "pod_closure": pclose.size,
        "dup_lower_bound": dclose.lower_bound, "pod_lower_bound": pclose.lower_bound,
        "dup_optimal": float(dclose.optimal), "pod_optimal": float(pclose.optimal),
        "dup_update_ops": dup_update_ops, "pod_update_ops": pod_update_ops,
        "dup_live_after_one": dup_live_after_one, "pod_live_after_one": pod_live_after_one,
        "pod_key_count": pod_key_count, "pass": float(ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0,1,2,3,4])
    ap.add_argument("--aliases", nargs="*", type=int, default=[1,2,4,8,16,32,64,128])
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows: List[Dict[str,float]] = [one(s, k) for s in a.seeds for k in a.aliases]
    all_pass = all(bool(r["pass"]) for r in rows)
    rec = {
        "experiment": "E-000061",
        "claim": "substrate control only; canonicalisation/pointer sharing is prior art",
        "all_pass": all_pass,
        "rows": rows,
    }
    p = Path(a.results_dir); p.mkdir(parents=True, exist_ok=True)
    (p / "e000061_pod_closure_scaling.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("aliases | dup closure | pod closure | dup live after one | pod live after one")
    for k in a.aliases:
        r = next(x for x in rows if x["seed"] == a.seeds[0] and x["aliases"] == k)
        print(f"{k:7d} | {int(r['dup_closure']):11d} | {int(r['pod_closure']):11d} | {int(r['dup_live_after_one']):18d} | {int(r['pod_live_after_one']):18d}")
    print(f"ALL_PASS={all_pass}")
    if not all_pass:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
