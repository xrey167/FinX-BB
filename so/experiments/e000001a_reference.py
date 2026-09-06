"""Experiment E-000001-A — mechanical reference implementation.

Purpose (architecture document section 19): establish the intended knowledge
semantics — addressing, composition, provenance, updates, rollback, locality,
alternative paths and replay determinism — in a controlled reference system
*before* any neural model is trained.  Scale: 5 seeds × 1,000 cells.

Run:  python -m so.experiments.e000001a_reference
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.world import UNKNOWN, World, fill_random, inject_alternative_paths

CONFIG: Dict[str, Any] = dict(
    n_entities=256, n_relations=4, n_synonyms=2, n_cells=1000, n_alt_structures=25,
    n_2hop=500, n_3hop=500, n_broken=100, n_lifecycle=100, n_locality_updates=100,
    n_locality_revokes=50, n_locality_multihop=300, n_alt_pairs=100, seeds=[0, 1, 2, 3, 4],
)


def build(seed: int, cfg: Dict[str, Any]):
    rng = np.random.default_rng(seed)
    empty = World(cfg["n_entities"], cfg["n_relations"], cfg["n_synonyms"], [])
    structured = inject_alternative_paths(rng, empty, cfg["n_alt_structures"])
    world = fill_random(rng, structured, cfg["n_cells"])
    assert len(world.facts) == cfg["n_cells"], len(world.facts)
    store = MVCCStore(seed=seed)
    kids = load_world(store, world)
    return rng, world, store, kids


def run_seed(seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    rng, world, store, kids = build(seed, cfg)
    res = ReferenceResolver(store)
    m: Dict[str, Any] = {"seed": seed, "n_cells": len(world.facts)}

    # ---- direct addressing: every cell answers with itself as provenance
    ok = prov_ok = 0
    for f in world.facts:
        q = world.make_query(rng, "fwd", f.subject, [f.relation])
        r = res.resolve(q)
        ok += int(r.answer == f.obj)
        prov_ok += int(r.trace == (kids[f.key],))
    m["direct"] = ok / len(world.facts)
    prov_total, prov_hits = len(world.facts), prov_ok

    # ---- multi-hop composition (answerable) and broken paths (must be UNKNOWN)
    for hops, key in ((2, "hop2"), (3, "hop3")):
        qs = world.sample_queries(rng, cfg[f"n_{hops}hop"], hops, "fwd", require_answer=True)
        hits = 0
        for q in qs:
            gt = world.answer(q)
            r = res.resolve(q)
            hits += int(r.answer == gt.answer)
            prov_hits += int(r.trace == tuple(kids[e] for e in gt.edges))
            prov_total += 1
        m[key] = hits / len(qs)
        broken = world.sample_queries(rng, cfg["n_broken"], hops, "fwd", require_answer=False)
        m[f"{key}_broken_unknown"] = sum(res.resolve(q).answer == UNKNOWN for q in broken) / len(broken)
    m["provenance"] = prov_hits / prov_total

    # ---- lifecycle: update -> derived answer follows -> rollback -> revoke -> restore
    cells = [world.facts[int(i)] for i in rng.choice(len(world.facts), size=cfg["n_lifecycle"], replace=False)]
    steps = 0
    passed = 0
    for f in cells:
        kid = kids[f.key]
        new_obj = int((f.obj + 1 + rng.integers(0, world.n_entities - 1)) % world.n_entities)
        q = world.make_query(rng, "fwd", f.subject, [f.relation])
        # a 2-hop query that passes through this cell as its *first* edge (if any)
        nxt = [r2 for r2 in range(world.n_relations) if (f.obj, r2) in world.index]
        store.update(kid, new_obj)
        steps += 1; passed += int(res.resolve(q).answer == new_obj)
        if nxt:
            r2 = int(nxt[0])
            q2 = world.make_query(rng, "fwd", f.subject, [f.relation, r2])
            expected_index = dict(world.index)
            expected_index[f.key] = new_obj          # the updated cell may itself lie on the second hop
            expect = world.follow(f.subject, [f.relation, r2], expected_index).answer
            steps += 1; passed += int(res.resolve(q2).answer == expect)
        store.rollback(kid, 1)
        steps += 1; passed += int(res.resolve(q).answer == f.obj)
        store.revoke(kid)
        steps += 1; passed += int(res.resolve(q).answer == UNKNOWN)
        store.restore(kid)
        steps += 1; passed += int(res.resolve(q).answer == f.obj)
        store.shred(kid)
        steps += 1; passed += int(res.resolve(q).answer == UNKNOWN)
        store.resign(kid)
        steps += 1; passed += int(res.resolve(q).answer == f.obj)
    m["update_rollback"] = passed / steps
    m["lifecycle_steps"] = steps

    # ---- locality: mutate a target set, everything outside it must be unchanged, then undo
    all_direct = [world.make_query(rng, "fwd", f.subject, [f.relation]) for f in world.facts]
    snapshot = [res.resolve(q).answer for q in all_direct]
    target_idx = rng.choice(len(world.facts), size=cfg["n_locality_updates"] + cfg["n_locality_revokes"], replace=False)
    target_keys = {world.facts[int(i)].key for i in target_idx}
    multihop = [q for q in world.sample_queries(rng, 4 * cfg["n_locality_multihop"], 2, "fwd")
                if not (set(world.answer(q).edges) & target_keys)][: cfg["n_locality_multihop"]]
    mh_before = [res.resolve(q).answer for q in multihop]
    for j, i in enumerate(target_idx):
        f = world.facts[int(i)]
        if j < cfg["n_locality_updates"]:
            store.update(kids[f.key], int((f.obj + 1) % world.n_entities))
        else:
            store.revoke(kids[f.key])
    after = [res.resolve(q).answer for q in all_direct]
    unchanged = sum(a == b for q, a, b in zip(all_direct, snapshot, after) if (q.start, q.path[0]) not in target_keys)
    n_outside = len(world.facts) - len(target_keys)
    changed_targets = sum(a != b for q, a, b in zip(all_direct, snapshot, after) if (q.start, q.path[0]) in target_keys)
    mh_after = [res.resolve(q).answer for q in multihop]
    m["locality"] = (unchanged + sum(a == b for a, b in zip(mh_before, mh_after))) / (n_outside + len(multihop))
    m["locality_targets_changed"] = changed_targets / len(target_keys)
    for j, i in enumerate(target_idx):
        f = world.facts[int(i)]
        if j < cfg["n_locality_updates"]:
            store.rollback(kids[f.key], 1)
        else:
            store.restore(kids[f.key])
    m["locality_undo_exact"] = float([res.resolve(q).answer for q in all_direct] == snapshot)

    # ---- alternative paths: revoking one edge breaks only the path that uses it
    pairs = world.alternative_path_pairs(rng, cfg["n_alt_pairs"])
    alt_ok = 0
    for q1, q2, edge in pairs:
        target = world.answer(q1).answer
        kid = kids[edge]
        good = res.resolve(q1).answer == target and res.resolve(q2).answer == target
        store.revoke(kid)
        good = good and res.resolve(q1).answer == UNKNOWN and res.resolve(q2).answer == target
        store.restore(kid)
        good = good and res.resolve(q1).answer == target and res.resolve(q2).answer == target
        alt_ok += int(good)
    m["alternative_path"] = alt_ok / len(pairs)
    m["alternative_pairs"] = len(pairs)

    # ---- replay determinism: rebuild from the operation log
    clone = store.clone_by_replay()
    res_clone = ReferenceResolver(clone)
    deviations = sum(res.resolve(q).answer != res_clone.resolve(q).answer for q in all_direct)
    deviations += int(clone.state_hash() != store.state_hash())
    m["replay_deviation"] = deviations
    m["operations"] = len(store.log)
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=CONFIG["seeds"])
    args = ap.parse_args(argv)
    cfg = dict(CONFIG, seeds=list(args.seeds))
    per_seed = [run_seed(s, cfg) for s in cfg["seeds"]]
    keys = ["direct", "hop2", "hop3", "hop2_broken_unknown", "hop3_broken_unknown", "provenance",
            "update_rollback", "locality", "locality_targets_changed", "locality_undo_exact",
            "alternative_path", "replay_deviation"]
    agg = ledger.aggregate(per_seed, keys)
    all_pass = all(agg[k]["min"] == 1.0 for k in keys if k != "replay_deviation") and agg["replay_deviation"]["max"] == 0
    record = {
        "experiment": "E-000001-A", "title": "Mechanical reference implementation",
        "evidence_level": "E3", "deletion_level": "F1",
        "claim": "The intended knowledge semantics (addressing, composition, provenance, update, rollback, "
                 "revoke/restore, marker shredding, locality, alternative paths, replay determinism) are "
                 "internally coherent in a mechanical reference over the mutable knowledge layer.",
        "not_claimed": "Nothing about neural networks; this experiment contains no learned component.",
        "config": cfg, "per_seed": per_seed, "aggregate": agg, "all_pass": all_pass,
    }
    rows = [(k, ledger.pct(agg[k]["mean"]) if k != "replay_deviation" else f"{agg[k]['max']:.0f}",
             ledger.pct(agg[k]["min"]) if k != "replay_deviation" else "-") for k in keys]
    md = "\n".join([
        "# E-000001-A — Mechanical reference implementation", "",
        f"Evidence level: **E3** ({ledger.EVIDENCE_LEVELS['E3']}). Deletion level exercised: **F1** (routing removal) "
        "plus marker shredding at the mechanical level.", "",
        f"Seeds: {cfg['seeds']} · cells per seed: {cfg['n_cells']} · all tests passed: **{all_pass}**", "",
        ledger.table(["Measure", "Mean over seeds", "Worst seed"], rows), "",
        "Per seed:", "",
        ledger.table(["seed"] + keys, [[s["seed"]] + [s[k] for k in keys] for s in per_seed]), "",
        "Interpretation: establishes that the desired semantics are coherent in the controlled reference system. "
        "It does not show that a trained neural network reproduces them (that is E-000001-B).",
    ])
    path = ledger.save("e000001a_reference", record, md)
    print(md)
    print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
