"""Experiment E-000028 — the channel SHRED does not close.

SHRED is the programme's strongest deletion operation and the basis of its F4 claim: the marker is
destroyed, the gate closes, the payload becomes unreadable, and every recorded attack on the value
channel comes back at chance (E-000010, E-000019).

But ``shred()`` writes only ``Version.marker`` (so/mvcc.py) and leaves the row ACTIVE, and in
``encode_bank`` (so/model.py) the routing keys are computed from the payload *before* the gate is
applied and are never gated:

    k_f = k_fwd(LN(s + r))          # from subject and relation
    k_r = k_rev(LN(o + r))          # FROM THE OBJECT
    v_f = v_f * g                   # only the values are gated

So a shredded cell's reverse key is still a deterministic function of the object that was shredded.
Two consequences follow, one already visible in the record and one never tested:

  1. The hop-0 routing distribution is bit-identical before and after SHRED. It is: E-000019 records
     ``active/routing_mass_on_target`` and ``shred/routing_mass_on_target`` as the same float in
     every seed. The ledger noted that routing alone does not separate the two; it did not follow the
     observation to its consequence.
  2. An attacker who can see routing can sweep candidate objects through a REVERSE query and ask
     which candidate steers the read onto the target's column. Nothing in the gate opposes this,
     because the gate never touches keys.

The threat model is the battery's own: the attacker knows the cell exists and knows its subject and
relation, and wants the object. The target's column is not assumed known — it is located from public
behaviour, by reading which column the ordinary forward question routes to (that mass is 0.997 after
SHRED, which is the point).

Controls that decide what the number means:
  active     the same sweep before SHRED. If it fails here the attack is simply weak and the
             shredded result says nothing.
  revoked    a revoked cell is not routable at all, so no candidate can put mass on it.
  evicted    E-000030's prescription: the row leaves the addressable bank while the store keeps the
             payload. It should behave like a deleted cell against this attack and unlike a shredded
             one, AND still be restorable -- which is the thing DELETE cannot offer.
  deleted    a deleted cell has no column.

Trains nothing: it scores the recorded E-000010 checkpoints, the F4 result itself.

Run:  python -m so.experiments.e000028_key_channel [--seeds 0 1 2 3 4] [--n-targets 100]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store, encode_queries
from so.experiments.common import fresh_world, position_of_kid
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, checkpoint_path, train_or_load
from so.model import ModelConfig
from so.train import TrainConfig
from so.world import Query

N_TARGETS = 100


@torch.no_grad()
def routing_of(model, store, world, queries: Sequence[Query], batch_size: int = 256) -> np.ndarray:
    """(N, H, C+1) routing for ``queries`` against the store as it stands."""
    bank = bank_from_store(store)
    tensors = bank.tensors()
    out = []
    for i in range(0, len(queries), batch_size):
        chunk = list(queries[i: i + batch_size])
        b = encode_queries(chunk, bank, world, model.cfg.max_hops)
        _, routing, _ = model(tensors, b.mode, b.start, b.rels, b.hop_valid)
        out.append(routing.numpy())
    return np.concatenate(out)


def locate_columns(model, store, world, targets) -> np.ndarray:
    """Where each target sits, as the attacker would find it: the column its own question routes to."""
    q = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]
    r = routing_of(model, store, world, q)
    # hop 0 is the read of a 1-hop query in this model; the later hops are padding and route nowhere
    return r[:, 0, :].argmax(-1)


def sweep(model, store, world, targets, columns: np.ndarray, n_entities: int) -> Dict[str, np.ndarray]:
    """For each target, the mass every candidate object puts on the target's column, via a reverse query.

    Returns the predicted object per target and the rank of the true one.
    """
    preds, ranks, margins = [], [], []
    for i, f in enumerate(targets):
        cand = [Query("rev", o, (f.relation,), (world.surface_of(f.relation, 0),)) for o in range(n_entities)]
        r = routing_of(model, store, world, cand)
        mass = r[:, 0, int(columns[i])]                       # hop 0 of a 1-hop reverse read
        order = np.argsort(-mass)
        preds.append(int(order[0]))
        ranks.append(int(np.where(order == f.obj)[0][0]))
        top = np.sort(mass)[::-1]
        margins.append(float(top[0] - top[1]))
    return {"pred": np.array(preds), "rank": np.array(ranks), "margin": np.array(margins)}


def score(res: Dict[str, np.ndarray], truth: np.ndarray, n_entities: int, tag: str) -> Dict[str, float]:
    return {f"{tag}/object_top1": float((res["pred"] == truth).mean()),
            f"{tag}/object_top5": float((res["rank"] < 5).mean()),
            f"{tag}/object_mean_rank": float(res["rank"].mean()),
            f"{tag}/margin_mean": float(res["margin"].mean())}


def run_seed(seed: int, n_targets: int, steps: int, verbose: bool = True) -> Dict[str, Any]:
    path = checkpoint_path("e000010", seed)
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}; run the E-000010 arm of e000009_verification_gate first")
    out = train_or_load("e000010", seed, ModelConfig(),
                        TrainConfig(seed=seed, n_steps=steps, gate_weight=5.0, gate_balanced=True))
    model, centre = out["model"], out["centre"]
    model.cfg.hard_gate = False
    rng, world, store, kids, _ = fresh_world(900 + seed, centre)
    facts = list(world.facts)
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:n_targets]]
    truth = np.array([f.obj for f in targets])
    n_ent = world.n_entities

    m: Dict[str, Any] = {"seed": seed, "checkpoint_sha256": _sha256(path), "n_targets": len(targets),
                         "n_entities": n_ent, "chance_top1": 1.0 / n_ent,
                         "chance_mean_rank": (n_ent - 1) / 2.0}
    t0 = time.time()
    cols = locate_columns(model, store, world, targets)
    m["column_located"] = float(np.mean([int(c == position_of_kid(store, kids[f.key]))
                                         for c, f in zip(cols, targets)]))
    m.update(score(sweep(model, store, world, targets, cols, n_ent), truth, n_ent, "active"))
    m["active/column_mass"] = float(np.mean([routing_of(model, store, world,
        [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),))])[0, 0, int(c)]
        for f, c in zip(targets[:20], cols[:20])]))
    if verbose:
        print(f"  seed {seed} active   top1 {m['active/object_top1']:.4f}  rank {m['active/object_mean_rank']:.1f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    for f in targets:
        store.shred(kids[f.key])
    cols_s = locate_columns(model, store, world, targets)
    m["shred/column_located"] = float(np.mean([int(a == b) for a, b in zip(cols_s, cols)]))
    m.update(score(sweep(model, store, world, targets, cols_s, n_ent), truth, n_ent, "shred"))
    if verbose:
        print(f"  seed {seed} shred    top1 {m['shred/object_top1']:.4f}  rank {m['shred/object_mean_rank']:.1f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    for f in targets:
        store.revoke(kids[f.key])
    # a revoked row is not routable, so the forward question no longer names it: the attacker follows
    # wherever the routing now goes, which is the best they can do
    m.update(score(sweep(model, store, world, targets, locate_columns(model, store, world, targets), n_ent),
                   truth, n_ent, "revoke"))
    for f in targets:
        store.restore(kids[f.key])

    # EVICT: out of the addressable bank, payload retained in the store. The certificate says this
    # should behave like DELETE against the attack and unlike SHRED, while keeping the data.
    for f in targets:
        store.evict(kids[f.key])
    m.update(score(sweep(model, store, world, targets, locate_columns(model, store, world, targets), n_ent),
                   truth, n_ent, "evict"))
    m["evict/payload_retained"] = float(np.mean([1.0 if store.cells[kids[f.key]].versions else 0.0
                                                 for f in targets]))
    for f in targets:
        store.restore(kids[f.key])
    m["evict/restore_reads_again"] = float(np.mean(
        [1.0 if store.resolve_key(f.key)[0] == f.obj else 0.0 for f in targets]))
    for f in targets:
        store.revoke(kids[f.key])
    for f in targets:
        store.restore(kids[f.key])
        store.delete(kids[f.key])
    # after DELETE the rows are gone and the remaining columns have shifted, so the attacker re-locates
    # from scratch exactly as in every other condition
    m.update(score(sweep(model, store, world, targets, locate_columns(model, store, world, targets), n_ent),
                   truth, n_ent, "delete"))
    m["seconds"] = time.time() - t0
    if verbose:
        print(f"  seed {seed} revoke   top1 {m['revoke/object_top1']:.4f} | delete top1 "
              f"{m['delete/object_top1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return m


KEYS = ["column_located", "shred/column_located", "active/column_mass",
        "evict/payload_retained", "evict/restore_reads_again"] + [f"{c}/{k}" for c in ("active", "shred", "revoke", "evict", "delete")
                             for k in ("object_top1", "object_top5", "object_mean_rank", "margin_mean")]

# Pre-registered. The claim under test is that SHRED closes the value channel and NOT the key channel,
# so the interesting outcome is the one where "active" and "shred" agree and both beat chance.
CRITERIA = {
    "column_located": (">=", 0.90),          # the attacker can find the column from public behaviour
    "active/object_top1": (">=", 0.20),      # validity: the sweep must work on a live cell
}
LEAK = {"shred/object_top1": ("<=", 1.0 / 256),   # what F4 would require of the key channel
        "evict/object_top1": ("<=", 1.0 / 256),  # and what EVICT should deliver instead
        "evict/payload_retained": (">=", 1.0),   # while keeping the data, which DELETE does not
        "evict/restore_reads_again": (">=", 1.0)}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_targets, args.steps) for s in args.seeds]
    keys = [k for k in KEYS if all(k in s for s in per_seed)]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    leak = ledger.check_criteria(agg, {k: v for k, v in LEAK.items() if k in agg})
    n_ent = per_seed[0]["n_entities"]
    sizes = {k: args.n_targets for k in keys if k.endswith(("object_top1", "object_top5", "column_located"))}
    rows = ledger.ci_rows(per_seed, keys, sizes,
                          lower_is_better=[k for k in keys if k.startswith(("shred/", "revoke/", "delete/"))
                                           and k.endswith(("object_top1", "object_top5"))])

    head = ["condition", "object recovered top-1", "top-5", "mean rank", "chance top-1", "chance rank"]
    body = [[c,
             f"{ledger.worst(agg[f'{c}/object_top1'], c != 'active'):.4f}",
             f"{ledger.worst(agg[f'{c}/object_top5'], c != 'active'):.4f}",
             f"{ledger.worst(agg[f'{c}/object_mean_rank'], False):.1f}",
             f"{1.0 / n_ent:.4f}", f"{(n_ent - 1) / 2.0:.1f}"]
            for c in ("active", "shred", "revoke", "evict", "delete") if f"{c}/object_top1" in agg]
    tbl = ledger.table(head, body)

    record = {"experiment": "E-000028", "title": "the channel SHRED does not close",
              "trains_nothing": True, "seeds": args.seeds, "n_targets": args.n_targets,
              "per_seed": per_seed, "aggregate": agg, "criteria": check, "key_channel_leak": leak}
    md = [f"# E-000028 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.n_targets} targets, the recorded E-000010 checkpoints, no training.",
          "The attacker knows a cell's subject and relation, finds its column from the routing of the",
          "ordinary forward question, then sweeps every candidate object through a REVERSE query and",
          "takes the candidate that steers the read onto that column.", "",
          "## Object recovery through the key channel (worst seed)", "", tbl, "",
          "`active` is the validity control: if the sweep cannot recover a live object, the shredded",
          "number means nothing. `revoke` and `delete` remove the row from routing altogether.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What F4 would require of this channel", "", ledger.criteria_table(leak), "",
          "## All measures", "", ledger.table(ledger.CI_HEADERS, rows), ""]
    path = ledger.save("e000028_key_channel", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    print(ledger.criteria_table(leak))
    return record


if __name__ == "__main__":
    main()
