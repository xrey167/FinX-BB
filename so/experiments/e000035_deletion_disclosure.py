"""Experiment E-000035 — a pod's aliases are signposts, and a deletion leaves them pointing at it.

E-000032 measures what canonicalisation buys: the fact closure falls from k to one, so a record-level
certificate composes into a fact-level one. E-000025 prices what it costs the reader: 0.0954 for
sharing and 0.0688 for training the dereference. Neither asks what it costs the SUBJECT of the deleted
fact, and there is a cost, because the two arrangements do not leave the same trace behind.

THE ASYMMETRY. Delete one of k duplicated copies and the store is a store with k-1 copies: nothing in
it says a deletion happened, and nothing says where. Delete a pod's object and every one of its k-1
aliases is still there, still a LINK row, still carrying the removed cell's key in
`bank["link_subject"]` and `bank["link_relation"]` -- because `MVCCStore.bank()` keeps it deliberately,
so the model has to discover the miss rather than being handed it by the control plane (E-000015's
recorded design). Each surviving alias is therefore a signpost reading *a record stood at (s, r) and
is gone*.

That is a deletion-disclosure channel, it exists only in the canonical arrangement, and no recorded
experiment has measured it. Under some threat models it is harmless -- the alias's own key was public
anyway. Under the one this programme cares about it is not: "was there a record about this person, and
was it deleted" is exactly the question an erasure guarantee is supposed to make unanswerable, and the
answer here is legible to anyone who can read the bank, without touching the model.

WHAT IS MEASURED, mechanically, with no model anywhere:

  disclosed        after the removal, can an adversary reading only the bank name the deleted key?
                   A key is disclosed when some surviving row points at a key that no live row holds.
  candidates       how many keys the bank leaves consistent with "this is where the deletion was".
                   One is a full disclosure; the store's whole key space is none.
  false_positives  keys the adversary would name that were never deleted -- a dangling pointer that
                   was always dangling, which E-000015 puts in the training distribution on purpose.

AND THE CLOSURE INVERTS. E-000032 measures the closure for one guarantee -- how many records must go
before no query yields the object -- and finds one for a pod and k for k duplicates. Ask for the OTHER
guarantee, that the bank shows no evidence a deletion happened at that key, and the same two stores
swap places: a pod's aliases must go too, so it costs k, while a duplicated store costs the one record
you were removing anyway. Both numbers are measured here, side by side, because a claim that quotes
only the first is quoting the half that flatters the design:

  | guarantee                     | canonical pod | duplicated |
  |-------------------------------|---------------|------------|
  | unreachable to the reader     | 1 record      | k records  |
  | no trace left in the bank     | k records     | 1 record   |

THE MITIGATION, AND ITS PRICE. Blanking a dangling pointer's key closes the channel. It also removes
the thing E-000015's alias criteria are about: with the key blanked, an alias to a removed target is
indistinguishable from an alias to key (0, 0), so `delete_target/alias_unknown` stops being a
discovery and becomes a tautology. Both arms are measured here so the trade is a number rather than an
argument.

Trains nothing, needs no checkpoint, runs in seconds.

Run:  python -m so.experiments.e000035_deletion_disclosure [--seeds 0 1 2] [--n-groups 100]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from so import ledger
from so.closure import fact_closure, pod_keys
from so.experiments.e000015_symlink_cells import EVAL, load_arm, sample_alias_world
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver
from so.train import make_centre

ARMS = ("canonical", "duplicated")


def live_keys(bank: Dict[str, np.ndarray]) -> set:
    """The keys the bank shows as held by a row, which is all an adversary reading it can see."""
    return {(int(s), int(r)) for s, r in zip(bank["subject"], bank["relation"])}


def dangling_targets(bank: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    """Every key a LINK row points at that no row in the bank holds.

    This is the adversary's whole method, and it needs nothing but the bank: a pointer with no
    referent is a pointer at something that was removed.
    """
    held = live_keys(bank)
    out: List[Tuple[int, int]] = []
    for i, is_link in enumerate(bank["is_link"]):
        if not bool(is_link):
            continue
        key = (int(bank["link_subject"][i]), int(bank["link_relation"][i]))
        if key not in held and key != (int(bank["subject"][i]), int(bank["relation"][i])):
            out.append(key)
    return out


def run_seed(seed: int, n_groups: int, verbose: bool = True) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, 16)
    world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
    stores = {"canonical": load_arm(world, spec, centre, seed, symlink=True),
              "duplicated": load_arm(world, spec, centre, seed, symlink=False)}
    chosen = [spec.groups[int(i)][0] for i in rng.permutation(len(spec.groups))[:n_groups]]

    m: Dict[str, Any] = {"seed": seed, "n_groups": len(chosen),
                         "key_space": world.n_entities * world.n_relations}
    t0 = time.time()

    # the baseline an adversary faces BEFORE any deletion: dangling pointers that were always dangling
    for arm, (store, _) in stores.items():
        base = dangling_targets(store.bank())
        m[f"{arm}/baseline_dangling"] = float(len(base))

    for arm in ARMS:
        store, kids = stores[arm]
        base = set(dangling_targets(store.bank()))
        disclosed, candidates, false_pos = [], [], []
        for t_key in chosen:
            store.evict(kids[t_key])
            # DISTINCT keys: k aliases of one pod all point at the same removed key, and counting the
            # rows instead of the keys would report k candidates where the adversary has one
            found = sorted({k for k in dangling_targets(store.bank()) if k not in base})
            disclosed.append(float(t_key in found))
            # what the bank narrows the deletion down to: the new dangling keys, or the whole key
            # space when there are none, because then nothing points at the removal at all
            candidates.append(float(len(found)) if found else float(m["key_space"]))
            false_pos.append(float(len([k for k in found if k != t_key])))
            store.restore(kids[t_key])
        m[f"{arm}/deleted_key_disclosed"] = float(np.mean(disclosed))
        m[f"{arm}/candidate_keys_mean"] = float(np.mean(candidates))
        m[f"{arm}/uniquely_identified"] = float(np.mean([d == 1.0 and c == 1.0
                                                         for d, c in zip(disclosed, candidates)]))
        m[f"{arm}/false_positive_keys"] = float(np.mean(false_pos))
        if verbose:
            print(f"  seed {seed} {arm:<11} deleted key disclosed "
                  f"{m[f'{arm}/deleted_key_disclosed']:.4f}  uniquely identified "
                  f"{m[f'{arm}/uniquely_identified']:.4f}  candidates "
                  f"{m[f'{arm}/candidate_keys_mean']:.1f} of {m['key_space']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # the OTHER closure: how many records must go before the bank shows no evidence of the deletion.
    # Greedy and exact here, because the signposts are exactly the rows that point at what was removed.
    for arm in ARMS:
        store, kids = stores[arm]
        base = set(dangling_targets(store.bank()))
        sizes = []
        for t_key in chosen:
            removed = [kids[t_key]]
            store.evict(kids[t_key])
            for _ in range(16):
                new_dangling = [k for k in dangling_targets(store.bank()) if k not in base]
                if not new_dangling:
                    break
                # every row pointing at a key that is gone is itself a signpost, so it goes next
                b = store.bank()
                held = live_keys(b)
                signposts = [int(b["kid"][i]) for i, l in enumerate(b["is_link"])
                             if bool(l)
                             and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in held
                             and (int(b["link_subject"][i]), int(b["link_relation"][i]))
                             not in base]
                if not signposts:
                    break
                for kid in signposts:
                    store.evict(kid)
                    removed.append(kid)
            sizes.append(float(len(removed)))
            for kid in reversed(removed):
                store.restore(kid)
        m[f"{arm}/trace_closure_mean"] = float(np.mean(sizes))
        m[f"{arm}/trace_closure_max"] = float(np.max(sizes))
        if verbose:
            print(f"  seed {seed} {arm:<11} records to leave NO TRACE: {np.mean(sizes):.2f} "
                  f"(max {np.max(sizes):.0f})", flush=True)

    # the mitigation, and what it costs: with the pointer blanked, a removed target and a pointer to
    # nothing are the same row, so the channel closes and so does the alias's own miss-discovery
    store, kids = stores["canonical"]
    closed, indistinguishable = [], []
    for t_key in chosen[: min(25, len(chosen))]:
        store.evict(kids[t_key])
        b = store.bank()
        blanked = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
        held = live_keys(b)
        for i, is_link in enumerate(b["is_link"]):
            if bool(is_link) and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in held:
                blanked["link_subject"][i] = 0
                blanked["link_relation"][i] = 0
        closed.append(float(t_key not in dangling_targets(blanked)))
        # every blanked alias now carries the same key, so nothing distinguishes which target went
        rows = [i for i, l in enumerate(blanked["is_link"])
                if bool(l) and (int(blanked["link_subject"][i]), int(blanked["link_relation"][i])) == (0, 0)]
        indistinguishable.append(float(len(rows) > 0 and len({(int(blanked["link_subject"][i]),
                                                               int(blanked["link_relation"][i]))
                                                              for i in rows}) == 1))
        store.restore(kids[t_key])
    m["blanked/channel_closed"] = float(np.mean(closed))
    m["blanked/pointers_indistinguishable"] = float(np.mean(indistinguishable))
    m["seconds"] = time.time() - t0
    return m


KEYS = (["key_space", "n_groups", "blanked/channel_closed", "blanked/pointers_indistinguishable"] +
        [f"{a}/{x}" for a in ARMS
         for x in ("deleted_key_disclosed", "candidate_keys_mean", "uniquely_identified",
                   "false_positive_keys", "baseline_dangling", "trace_closure_mean",
                   "trace_closure_max")])

CRITERIA = {
    # the channel, and the control that says it is a property of the arrangement and not of the store
    "canonical/deleted_key_disclosed": (">=", 0.95),
    "canonical/uniquely_identified": (">=", 0.90),
    "duplicated/deleted_key_disclosed": ("<=", 0.05),
    # a duplicated store leaves the adversary the whole key space, which is what "no signpost" means
    "duplicated/candidate_keys_mean": (">=", 1000.0),
    # the inversion: the pod pays for leaving no trace exactly what the duplicated store pays for
    # being unreachable, and vice versa
    "canonical/trace_closure_mean": (">=", 3.0),
    "duplicated/trace_closure_mean": ("<=", 1.0),
    # and the mitigation has to actually close it, or it is not a mitigation
    "blanked/channel_closed": (">=", 1.0),
    "blanked/pointers_indistinguishable": (">=", 1.0),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-groups", type=int, default=100)
    args = ap.parse_args(argv)

    per_seed = [run_seed(s, args.n_groups) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = [[arm,
             f"{agg[f'{arm}/deleted_key_disclosed']['mean']:.4f}",
             f"{agg[f'{arm}/uniquely_identified']['mean']:.4f}",
             f"{agg[f'{arm}/candidate_keys_mean']['mean']:.1f}",
             f"{agg[f'{arm}/false_positive_keys']['mean']:.2f}",
             f"{agg[f'{arm}/baseline_dangling']['mean']:.1f}"]
            for arm in ARMS]
    tbl = ledger.table(["store", "deleted key disclosed", "uniquely identified",
                        "candidate keys left", "false positives", "dangling before any deletion"], rows)
    inv = ledger.table(["guarantee", "canonical pod", "duplicated"],
                       [["unreachable to the reader (E-000032)", "1.00", "3.00"],
                        ["no trace left in the bank (here)",
                         f"{agg['canonical/trace_closure_mean']['mean']:.2f}",
                         f"{agg['duplicated/trace_closure_mean']['mean']:.2f}"]])

    record = {"experiment": "E-000035",
              "title": "a pod's aliases are signposts, and a deletion leaves them pointing at it",
              "trains_nothing": True, "uses_no_model": True, "seeds": args.seeds,
              "n_groups": args.n_groups, "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000035 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_groups} pods per seed. No model, no checkpoint, no training:",
          "the adversary reads `MVCCStore.bank()` and nothing else, and names every key a LINK row points",
          "at that no row holds.", "",
          "## What a deletion leaves behind", "", tbl, "",
          "`dangling before any deletion` is the control: E-000015 puts pointers to nothing in the",
          "training distribution on purpose, so some dangle without any deletion having happened, and",
          "only the NEW ones are counted as disclosure.", "",
          "The asymmetry is the finding. Deleting one of k duplicated copies leaves a store with k-1",
          "copies and no trace of the operation — the adversary is left the whole key space. Deleting a",
          "pod's object leaves every alias still pointing at it, and `MVCCStore.bank()` keeps that key",
          "deliberately so the model has to discover the miss rather than be handed it. Each surviving",
          "alias is therefore a signpost reading *a record stood here and is gone*.", "",
          "## The closure inverts with the guarantee", "", inv, "",
          "E-000032 measures the first row: how many records must go before no query yields the object.",
          "This experiment measures the second: how many before the bank shows no evidence a deletion",
          "happened there. The same two stores swap places. A pod's aliases are the signposts, so they",
          "must go too; a duplicated store costs the one record you were removing anyway. Quoting only",
          "the first row would be quoting the half that flatters the design.", "",
          "## The mitigation, and what it costs", "",
          f"Blanking a dangling pointer's key closes the channel "
          f"({agg['blanked/channel_closed']['min']:.4f}) and makes every such pointer identical "
          f"({agg['blanked/pointers_indistinguishable']['min']:.4f}). It also removes what E-000015's",
          "alias criteria are about: with the key blanked, an alias to a removed target is",
          "indistinguishable from an alias to key (0, 0), so discovering the miss stops being a",
          "discovery. The trade is recorded as a number rather than argued.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this does not show", "",
          "It is a property of this store's bank, not of canonicalisation in general: a store that",
          "compacts its aliases on deletion, or that never exports the target key, has no such channel.",
          "It says nothing about whether the disclosure matters, which is a threat-model question. And",
          "it measures the bank an adversary can read, not what the model exposes to one who cannot.", ""]
    path = ledger.save("e000035_deletion_disclosure", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
