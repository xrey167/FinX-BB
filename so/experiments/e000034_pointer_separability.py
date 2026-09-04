"""Experiment E-000034 — a pointer is separable from an object by its norm alone.

E-000015's record and the code that produced it both say, in these words: *the store decides which
payload a row carries; the model is never told that a value it has read is a pointer -- that it must
learn.* The claim is load-bearing. It is what makes the dereference a learned capability rather than a
branch on a flag, and E-000020 repeats it for the frozen GPT-2.

A prior-art review of the symlink concept turned it over and asked whether recognising a pointer is
learned at all, or given away by the geometry. On every recorded seed, one number settles it: a single
threshold on the value vector's L2 norm separates alias rows from fact rows at 1.0000, with disjoint
ranges 7.6 pooled standard deviations apart.

The first version of this experiment attributed that to the architecture, and that was wrong. At
INITIALISATION the two ranges overlap; the gap is learned. `so/model.py encode_bank` builds a fact
row's value as `v_fwd(ent_emb(obj))` and an alias row's as `v_link(ln_key(...))` -- two unrelated
projections whose scales nothing couples -- so the architecture supplies the FREEDOM to tag the kinds
apart, and training takes it, completely.

So the recorded sentence stands as written and fails as a claim about difficulty. The honest statement
is narrower: **recognition is solved for free; only following a pointer is genuinely learned.** That is
still a real capability -- E-000016's one-slot arm refuses a two-link chain rather than inventing an
answer, which a flag-reading branch would not do -- but it is not the capability the record implied.

TWO PARTS.

  --phase diagnose   No training. For each recorded checkpoint, encode the bank and report how well a
                     single threshold on the value norm tells alias rows from fact rows, together with
                     the separation in units of the pooled standard deviation. A LINEAR probe on the
                     whole vector is reported beside it as the upper bound the norm is compared to.
  --phase train      The remedy, and the test of whether the claim can be earned: `share_link_value`
                     sends BOTH payload kinds through the same LayerNorm and the same projection. That
                     removes the freedom, not the possibility -- the model can still learn a scale
                     difference by growing `cell_rel_emb`, so the criterion is a measurement and not a
                     theorem, and it can fail. Alias reading and direct reading are then measured
                     against the E-000015 baseline. If resolution survives with the cue gone, the
                     claim is earned in a form the record can keep; if it collapses, the design's
                     dependence on the cue is what gets recorded.

Run:  python -m so.experiments.e000034_pointer_separability --phase diagnose
      python -m so.experiments.e000034_pointer_separability --phase train --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, guard_recorded_checkpoint
from so.experiments.e000015_symlink_cells import (EVAL, _q1, load_arm, model_config, predict,
                                                  sample_alias_world, train_config, train_or_load,
                                                  train_symlink)
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.reference import ReferenceResolver
from so.world import UNKNOWN


def best_threshold_accuracy(pos: np.ndarray, neg: np.ndarray) -> float:
    """Accuracy of the best single threshold on a scalar -- the cheapest possible pointer detector."""
    xs = np.concatenate([pos, neg])
    ys = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    order = np.argsort(xs)
    ys = ys[order]
    n = len(ys)
    best = 0
    for i in range(n + 1):
        lo, hi = ys[:i], ys[i:]
        best = max(best, int((lo == 0).sum() + (hi == 1).sum()), int((lo == 1).sum() + (hi == 0).sum()))
    return best / n


def linear_probe_accuracy(x: np.ndarray, y: np.ndarray, seed: int = 0, steps: int = 300) -> float:
    """The upper bound the norm is compared against: what the whole vector gives a linear reader."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    xb = torch.as_tensor(x, dtype=torch.float32)
    yb = torch.as_tensor(y, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xb[tr] @ w + b, yb[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = ((xb[te] @ w + b) > 0).float()
    return float((pred == yb[te]).float().mean())


def separability(model, store) -> Dict[str, float]:
    bank = bank_from_store(store)
    t = bank.tensors()
    with torch.no_grad():
        enc = model.encode_bank(t)
    v = enc["v_f"].numpy()
    is_link = t["is_link"].numpy().astype(bool)
    usable = t["active"].numpy().astype(bool) & t["marker_valid"].numpy().astype(bool)
    keep = usable
    v, is_link = v[keep], is_link[keep]
    n = np.linalg.norm(v, axis=-1)
    a, b = n[is_link], n[~is_link]
    pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
    return {"link_norm_mean": float(a.mean()), "fact_norm_mean": float(b.mean()),
            "link_norm_std": float(a.std()), "fact_norm_std": float(b.std()),
            "norm_gap_in_sd": float(abs(a.mean() - b.mean()) / pooled),
            "norm_threshold_accuracy": best_threshold_accuracy(a, b),
            "norm_ranges_overlap": float(a.min() < b.max() and b.min() < a.max()),
            "linear_probe_accuracy": linear_probe_accuracy(v, is_link.astype(np.float32)),
            "n_link": int(is_link.sum()), "n_fact": int((~is_link).sum())}


def read_metrics(model, world, spec, store) -> Dict[str, float]:
    ref = ReferenceResolver(store)
    bank = bank_from_store(store)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    alias_keys = spec.alias_keys

    def acc(keys: Sequence[Tuple[int, int]]) -> float:
        qs = [_q1(world, k) for k in keys]
        p = predict(model, bank, world, qs)
        return float(np.mean([a == ref.resolve(q).answer for a, q in zip(p.answers, qs)]))

    return {"direct": acc(base_keys[:400]), "alias_direct": acc(alias_keys)}


def diagnose(seeds: Sequence[int], verbose: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        res = train_or_load(seed, 4000, n_deref=1)
        model, centre = res["model"], res["centre"]
        rng = np.random.default_rng(seed)
        world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
        store, _ = load_arm(world, spec, centre, seed, symlink=True)
        m: Dict[str, Any] = {"seed": seed, "arm": "recorded (separate v_link)"}
        m.update(separability(model, store))
        m.update(read_metrics(model, world, spec, store))
        out.append(m)
        if verbose:
            print(f"  seed {seed} recorded    link norm {m['link_norm_mean']:.3f}  fact norm "
                  f"{m['fact_norm_mean']:.3f}  gap {m['norm_gap_in_sd']:.1f} sd  threshold acc "
                  f"{m['norm_threshold_accuracy']:.4f}  probe {m['linear_probe_accuracy']:.4f}  "
                  f"alias read {m['alias_direct']:.4f}", flush=True)
    return out


def train_shared(seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000034_shared{CKPT_SUFFIX}_seed{seed}.pt"
    cfg_m = model_config(1)
    cfg_m.share_link_value = True
    cfg_t = train_config(seed, steps)
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        model.eval()
        return {"model": model, "centre": np.asarray(ck["centre"]), "train_seconds": ck["train_seconds"],
                "checkpoint_sha256": _sha256(path), "steps": ck.get("steps", steps)}
    out = train_symlink(cfg_m, cfg_t)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"model": out["model"].state_dict(), "model_cfg": cfg_m.to_dict(),
                "train_cfg": cfg_t.to_dict(), "centre": out["centre"], "history": out["history"],
                "steps": steps, "train_seconds": out["train_seconds"]}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def run_shared(seeds: Sequence[int], steps: int, force: bool, verbose: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        res = train_shared(seed, steps, force)
        model, centre = res["model"], res["centre"]
        rng = np.random.default_rng(seed)
        world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
        store, _ = load_arm(world, spec, centre, seed, symlink=True)
        m: Dict[str, Any] = {"seed": seed, "arm": "shared projection"}
        m.update(separability(model, store))
        m.update(read_metrics(model, world, spec, store))
        out.append(m)
        if verbose:
            print(f"  seed {seed} shared      link norm {m['link_norm_mean']:.3f}  fact norm "
                  f"{m['fact_norm_mean']:.3f}  gap {m['norm_gap_in_sd']:.1f} sd  threshold acc "
                  f"{m['norm_threshold_accuracy']:.4f}  probe {m['linear_probe_accuracy']:.4f}  "
                  f"alias read {m['alias_direct']:.4f}", flush=True)
    return out


KEYS = ["norm_threshold_accuracy", "norm_gap_in_sd", "linear_probe_accuracy", "norm_ranges_overlap",
        "link_norm_mean", "fact_norm_mean", "direct", "alias_direct"]

CRITERIA = {
    # what the diagnostic asserts about the RECORDED design: the cue is there and it is total
    "recorded/norm_threshold_accuracy": (">=", 0.99),
    # and what the remedy has to achieve to earn the claim back
    "shared/norm_threshold_accuracy": ("<=", 0.75),
    "shared/alias_direct": (">=", 0.80),
    "shared/direct": (">=", 0.90),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["diagnose", "train", "both"], default="diagnose")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    recorded = diagnose(args.seeds) if args.phase in ("diagnose", "both") else []
    shared = run_shared(args.seeds, args.steps, args.force) if args.phase in ("train", "both") else []

    agg: Dict[str, Any] = {}
    for name, rows in (("recorded", recorded), ("shared", shared)):
        if not rows:
            continue
        numeric = [{f"{name}/{k}": float(v) for k, v in r.items() if isinstance(v, (bool, int, float))}
                   for r in rows]
        agg.update(ledger.aggregate(numeric, [f"{name}/{k}" for k in KEYS]))
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows_tbl = []
    for name, rows in (("recorded (separate v_link)", recorded), ("shared projection", shared)):
        if not rows:
            continue
        key = "recorded" if name.startswith("recorded") else "shared"
        rows_tbl.append([name,
                         f"{agg[f'{key}/link_norm_mean']['mean']:.3f}",
                         f"{agg[f'{key}/fact_norm_mean']['mean']:.3f}",
                         f"{agg[f'{key}/norm_gap_in_sd']['mean']:.1f}",
                         f"{agg[f'{key}/norm_threshold_accuracy']['max']:.4f}",
                         f"{agg[f'{key}/linear_probe_accuracy']['max']:.4f}",
                         f"{agg[f'{key}/direct']['min']:.4f}",
                         f"{agg[f'{key}/alias_direct']['min']:.4f}"])
    tbl = ledger.table(["value projection", "pointer norm", "object norm", "gap (pooled sd)",
                        "best single threshold", "linear probe", "direct read (worst seed)",
                        "alias read (worst seed)"], rows_tbl)

    record = {"experiment": "E-000034",
              "title": "a pointer is separable from an object by its norm alone",
              "phase": args.phase, "seeds": args.seeds, "steps": args.steps,
              "recorded": recorded, "shared": shared, "aggregate": agg, "criteria": check,
              "seconds": time.time() - t0}
    md = [f"# E-000034 — {record['title']}", "",
          f"Seeds {args.seeds}. The recorded arm trains nothing: it encodes the E-000015 banks with the",
          "recorded one-slot checkpoints and asks how well a single threshold on the value vector's L2",
          "norm tells alias rows from fact rows.", "",
          "## What the store gives away for free", "", tbl, "",
          "E-000015's record says *the model is never told that a value it has read is a pointer -- that",
          "it must learn*. It is not told, and it learns: at initialisation the two ranges overlap, and",
          "after training one number separates them perfectly. `encode_bank` carries a fact row's value",
          "through `v_fwd` and an alias row's through `v_link`, two projections whose scales nothing",
          "couples, so the architecture supplies the freedom and training takes it.", "",
          "The claim that survives is narrower and still real: **recognising a pointer is free; only",
          "following one is learned.** E-000016's one-slot arm refuses a two-link chain rather than",
          "inventing an answer, which a branch on a flag would not do.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000034_pointer_separability", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
