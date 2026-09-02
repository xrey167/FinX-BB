"""The test families for the neural model (experiment E-000001-B and successors).

Every family compares the *model's* behaviour with the *mechanical reference*
over the very same ``MVCCStore``: the reference defines what the answer and
the provenance trace must be after each lifecycle operation, the model has to
reproduce it.  Families: direct, 2-hop, 3-hop, broken paths, provenance,
lifecycle (update / rollback / revoke / restore / shred / resign), locality,
alternative paths, replay determinism, noise robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import Bank, bank_from_store, encode_queries
from .mvcc import MVCCStore
from .reference import ReferenceResolver, load_world
from .world import Query, UNKNOWN, World, fill_random, inject_alternative_paths


@dataclass
class Predictions:
    answers: np.ndarray            # (N,) entity id or UNKNOWN
    traces: List[Tuple[int, ...]]  # kids read per hop (only confident reads)
    routing: np.ndarray            # (N, H, C+1) routing distributions
    logits: np.ndarray             # (N, n_entities + 1)


def build_eval_world(seed: int, n_entities: int, n_relations: int, n_synonyms: int, n_cells: int,
                     n_alt_structures: int, marker_centre: np.ndarray):
    rng = np.random.default_rng(seed)
    empty = World(n_entities, n_relations, n_synonyms, [])
    structured = inject_alternative_paths(rng, empty, n_alt_structures)
    world = fill_random(rng, structured, n_cells)
    store = MVCCStore(marker_dim=marker_centre.shape[0], seed=seed, marker_centre=marker_centre)
    kids = load_world(store, world)
    return rng, world, store, kids


@torch.no_grad()
def predict(model, store: MVCCStore, world: World, queries: Sequence[Query], noise: float = 0.0,
            cell_mask: Optional[np.ndarray] = None, batch_size: int = 256, seed: int = 0,
            bank: Optional[Bank] = None, confident: float = 0.5) -> Predictions:
    bank = bank_from_store(store) if bank is None else bank
    tensors = bank.tensors()
    gen = torch.Generator().manual_seed(seed)
    mask_t = None if cell_mask is None else torch.as_tensor(cell_mask, dtype=torch.bool)
    H = model.cfg.max_hops
    answers, traces, routings, logits_all = [], [], [], []
    for i in range(0, len(queries), batch_size):
        chunk = list(queries[i: i + batch_size])
        batch = encode_queries(chunk, bank, world, H)
        logits, routing, _ = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid,
                                   noise=noise, generator=gen, cell_mask=mask_t)
        pred = logits.argmax(-1).numpy()
        pred = np.where(pred == world.n_entities, UNKNOWN, pred)
        r = routing.numpy()
        C = bank.size
        for j, q in enumerate(chunk):
            tr: List[int] = []
            for t in range(q.hops):
                p = r[j, t]
                k = int(p.argmax())
                if k >= C or p[k] < confident:
                    break
                tr.append(int(bank.kid[k]))
            traces.append(tuple(tr))
        answers.append(pred)
        routings.append(r)
        logits_all.append(logits.numpy())
    return Predictions(np.concatenate(answers), traces, np.concatenate(routings), np.concatenate(logits_all))


def agree(pred: Predictions, ref: ReferenceResolver, queries: Sequence[Query]) -> Tuple[float, float]:
    """(answer agreement, exact trace agreement) between model and reference."""
    ans = prov = 0
    for a, tr, q in zip(pred.answers, pred.traces, queries):
        r = ref.resolve(q)
        ans += int(a == r.answer)
        prov += int(tr == r.trace)
    return ans / len(queries), prov / len(queries)


def run_suite(model, seed: int, cfg: Dict[str, Any], marker_centre: np.ndarray,
              noise_levels: Sequence[float] = (0.0, 0.05, 0.1, 0.16, 0.2, 0.24, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5)) -> Dict[str, Any]:
    rng, world, store, kids = build_eval_world(seed, cfg["n_entities"], cfg["n_relations"], cfg["n_synonyms"],
                                               cfg["n_cells"], cfg["n_alt_structures"], marker_centre)
    ref = ReferenceResolver(store)
    m: Dict[str, Any] = {"seed": seed, "n_cells": len(world.facts)}

    def q1(f) -> Query:
        return world.make_query(rng, "fwd", f.subject, [f.relation])

    # ---- direct + provenance
    direct_qs = [q1(f) for f in world.facts]
    p = predict(model, store, world, direct_qs)
    m["direct"], prov_direct = agree(p, ref, direct_qs)
    m["direct_unknown_rate"] = float((p.answers == UNKNOWN).mean())
    prov_hits, prov_total = prov_direct * len(direct_qs), len(direct_qs)

    # ---- multi-hop
    for hops, key in ((2, "hop2"), (3, "hop3")):
        qs = world.sample_queries(rng, cfg[f"n_{hops}hop"], hops, "fwd", require_answer=True)
        pp = predict(model, store, world, qs)
        m[key], pr = agree(pp, ref, qs)
        prov_hits += pr * len(qs); prov_total += len(qs)
        broken = world.sample_queries(rng, cfg["n_broken"], hops, "fwd", require_answer=False)
        pb = predict(model, store, world, broken)
        m[f"{key}_broken_unknown"] = float((pb.answers == UNKNOWN).mean())
    m["provenance"] = prov_hits / prov_total

    # ---- reverse queries (shared knowledge object accessed from the other side)
    rev = world.sample_queries(rng, cfg.get("n_rev", 300), 1, "rev")
    m["reverse"], _ = agree(predict(model, store, world, rev), ref, rev)

    # ---- lifecycle: every operation must be reproduced exactly (compared with the reference)
    cells = [world.facts[int(i)] for i in rng.choice(len(world.facts), size=cfg["n_lifecycle"], replace=False)]
    counters = {k: [0, 0] for k in ("update", "update_derived", "rollback", "revoke", "restore", "shred", "resign")}

    def check(name: str, q: Query) -> None:
        a = int(predict(model, store, world, [q]).answers[0])
        counters[name][0] += int(a == ref.resolve(q).answer)
        counters[name][1] += 1

    for f in cells:
        kid = kids[f.key]
        q = q1(f)
        new_obj = int((f.obj + 1 + rng.integers(0, world.n_entities - 1)) % world.n_entities)
        nxt = [r2 for r2 in range(world.n_relations) if (f.obj, r2) in world.index]
        store.update(kid, new_obj); check("update", q)
        if nxt:
            check("update_derived", world.make_query(rng, "fwd", f.subject, [f.relation, int(nxt[0])]))
        store.rollback(kid, 1); check("rollback", q)
        store.revoke(kid); check("revoke", q)
        store.restore(kid); check("restore", q)
        store.shred(kid); check("shred", q)
        store.resign(kid); check("resign", q)
    for k, (hit, tot) in counters.items():
        m[k] = hit / tot if tot else float("nan")
    m["update_rollback"] = sum(v[0] for v in counters.values()) / sum(v[1] for v in counters.values())

    # ---- locality
    snapshot = predict(model, store, world, direct_qs).answers
    n_t = cfg["n_locality_updates"] + cfg["n_locality_revokes"]
    target_idx = rng.choice(len(world.facts), size=n_t, replace=False)
    target_keys = {world.facts[int(i)].key for i in target_idx}
    multihop = [q for q in world.sample_queries(rng, 4 * cfg["n_locality_multihop"], 2, "fwd")
                if not (set(world.answer(q).edges) & target_keys)][: cfg["n_locality_multihop"]]
    mh_before = predict(model, store, world, multihop).answers
    for j, i in enumerate(target_idx):
        f = world.facts[int(i)]
        if j < cfg["n_locality_updates"]:
            store.update(kids[f.key], int((f.obj + 1) % world.n_entities))
        else:
            store.revoke(kids[f.key])
    after = predict(model, store, world, direct_qs).answers
    outside = np.array([(q.start, q.path[0]) not in target_keys for q in direct_qs])
    mh_after = predict(model, store, world, multihop).answers
    m["locality"] = (float((snapshot[outside] == after[outside]).sum()) + float((mh_before == mh_after).sum())) / (outside.sum() + len(multihop))
    ref_after = np.array([ref.resolve(q).answer for q in direct_qs])
    m["locality_targets_correct"] = float((after[~outside] == ref_after[~outside]).mean())
    for j, i in enumerate(target_idx):
        f = world.facts[int(i)]
        if j < cfg["n_locality_updates"]:
            store.rollback(kids[f.key], 1)
        else:
            store.restore(kids[f.key])
    m["locality_undo_exact"] = float(np.array_equal(predict(model, store, world, direct_qs).answers, snapshot))

    # ---- alternative paths
    pairs = world.alternative_path_pairs(rng, cfg["n_alt_pairs"])
    alt_ok = 0
    for qa, qb, edge in pairs:
        target = world.answer(qa).answer
        kid = kids[edge]
        pa = predict(model, store, world, [qa, qb]).answers
        good = pa[0] == target and pa[1] == target
        store.revoke(kid)
        pa = predict(model, store, world, [qa, qb]).answers
        good = good and pa[0] == UNKNOWN and pa[1] == target
        store.restore(kid)
        pa = predict(model, store, world, [qa, qb]).answers
        good = good and pa[0] == target and pa[1] == target
        alt_ok += int(good)
    m["alternative_path"] = alt_ok / len(pairs)

    # ---- replay determinism: rebuild the layer from the log, predictions must be identical
    clone = store.clone_by_replay()
    a1 = predict(model, store, world, direct_qs).answers
    a2 = predict(model, clone, world, direct_qs).answers
    a3 = predict(model, store, world, direct_qs).answers
    m["replay_deviation"] = int((a1 != a2).sum() + (a1 != a3).sum())

    # ---- noise robustness on direct queries (bank-level Gaussian perturbation)
    ref_direct = np.array([ref.resolve(q).answer for q in direct_qs])
    m["noise"] = {}
    for lvl in noise_levels:
        pn = predict(model, store, world, direct_qs, noise=float(lvl), seed=seed)
        m["noise"][f"{lvl:.2f}"] = float((pn.answers == ref_direct).mean())
    return m


SUITE_KEYS = ["direct", "hop2", "hop3", "hop2_broken_unknown", "hop3_broken_unknown", "provenance", "reverse",
              "update", "update_derived", "rollback", "revoke", "restore", "shred", "resign", "update_rollback",
              "locality", "locality_targets_correct", "locality_undo_exact", "alternative_path", "replay_deviation"]
