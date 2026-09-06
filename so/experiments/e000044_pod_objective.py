"""Experiment E-000044 — train the pod objective, and price it.

E-000043 diagnosed. This constructs.

WHAT E-000043 ESTABLISHED. On a frozen model the deletion subspaces of seventeen facts overlap 0.4118
more than a matched null while 92% of the dimension budget sits unused, so the clean-deletion capacity
bound ``n <= d/s`` is nowhere near binding and the failure is not capacity. It is ALLOCATION. And the
sharing is in the ADDRESSING rather than the content -- a fact's own content direction sits +0.1594
above the null, its phrasing machinery +0.4024 -- which is the symlink stated in activation space:
what a store keeps in separate records, the object and the keys that reach it, a representation keeps
in one subspace, so a deletion aimed at the content pays its collateral to the addressing.

A DIAGNOSIS IS NOT A METHOD. If allocation is really the problem then an objective should fix it, and
if it cannot then "allocation, not capacity" was the wrong reading. Two arms, same world, same seeds,
one difference (``so/pod.py``):

    POD      pull every access path of one fact onto one carrier -- many keys, one object -- so the
             deletion closure is 1 rather than the number of ways the fact can be asked
    PRIVATE  push the carriers of different facts apart, hinged at the larger of the Welch bound and
             the centring floor, so removing one leaves the others standing

THE PRICE IS PRE-REGISTERED, because it is the obvious way to cheat. An objective that orthogonalises
the carriers by destroying the task has proved nothing, so ``accuracy_ratio >= 0.95`` is a criterion
that can fail and is reported whether or not it does. A second obvious cheat is silencing everything:
a "deletion" that works because the model no longer answers anything is caught by requiring the
baseline arm to answer in the first place and by reporting collateral, which is accuracy on facts
nobody asked to delete.

WHAT WOULD FALSIFY THE READING. If arm B's excess overlap does not fall, allocation is not trainable
by this objective and the E-000043 verdict loses its constructive half. If it falls but accuracy falls
with it, the dimensions were not free after all and the capacity reading was closer to right than the
allocation one. Both outcomes are recorded rather than retried.

Run:  python -m so.experiments.e000044_pod_objective [--steps 1200] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.capacity import allocation, subspace_overlap
from so.data import bank_from_world, encode_queries
from so.model import ModelConfig
from so.pod import pod_queries
from so.train import TrainConfig, train
from so.workspace import project_out
from so.world import World

N_ENT, N_REL, N_SYN, N_FACTS = 128, 4, 4, 320


def fact_basis(res_self: torch.Tensor, res_others: torch.Tensor) -> torch.Tensor:
    """E-000043's basis: the shared content direction first, then the phrasing spread's PCs."""
    spec = res_self - res_others
    centred = spec - spec.mean(0, keepdim=True)
    u, _, _ = torch.linalg.svd(centred, full_matrices=False)
    rows = [spec.mean(0)] + [centred.t() @ u[:, i] for i in range(centred.shape[0] - 1)]
    return torch.stack([r / r.norm().clamp(min=1e-8) for r in rows])


@torch.no_grad()
def measure(model, world: World, bank, tensors, facts: Sequence[Tuple[int, int]],
            rng: np.random.Generator, max_hops: int, n_ent: int) -> Dict[str, Any]:
    """The E-000043 measurement, plus the deletion it was a diagnosis of."""
    qs, fact_ids = pod_queries(world, facts, rng)
    batch = encode_queries(qs, bank, world, max_hops)
    logits, _, extras = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid)
    h = extras["hidden"]
    acc = float((logits.argmax(-1) == batch.target).float().mean())
    d = h.shape[1]
    n_f = int(fact_ids.max()) + 1
    per = {f: h[fact_ids == f] for f in range(n_f)}

    V = {}
    for f in range(n_f):
        others = torch.stack([per[g].mean(0) for g in range(n_f) if g != f]).mean(0)
        V[f] = fact_basis(per[f], others.expand_as(per[f]))

    g = torch.Generator().manual_seed(0)
    rnd = {f: torch.randn(per[f].shape[0], d, generator=g) for f in range(n_f)}
    Vn = {}
    for f in range(n_f):
        others = torch.stack([rnd[g_] .mean(0) for g_ in range(n_f) if g_ != f]).mean(0)
        Vn[f] = fact_basis(rnd[f], others.expand_as(rnd[f]))

    def mo(m, keys):
        pr = [subspace_overlap(m[a], m[b]) for i, a in enumerate(keys) for b in keys[i + 1:]]
        return float(np.mean(pr)) if pr else float("nan")

    keys = list(range(n_f))
    ov_full, nu_full = mo(V, keys), mo(Vn, keys)
    ov_c = mo({f: V[f][:1] for f in keys}, keys); nu_c = mo({f: Vn[f][:1] for f in keys}, keys)
    ov_a = mo({f: V[f][1:] for f in keys}, keys); nu_a = mo({f: Vn[f][1:] for f in keys}, keys)

    # the deletion the geometry is a diagnosis of, via the readout-side hook
    A, dims, colls, sil = {}, [], [], []
    for f in keys:
        rows = (fact_ids == f)
        tgt = batch.target[rows]
        chosen: List[int] = []
        for i in range(V[f].shape[0]):
            chosen.append(i)
            dirs = V[f][torch.as_tensor(chosen)]
            lg, _, _ = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid,
                             hidden_edit=lambda x, dd=dirs: project_out(x, dd))
            if float((lg.argmax(-1)[rows] == tgt).float().mean()) <= 0.25:
                break
        dirs = V[f][torch.as_tensor(chosen)]
        lg, _, _ = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid,
                         hidden_edit=lambda x, dd=dirs: project_out(x, dd))
        after = float((lg.argmax(-1)[rows] == tgt).float().mean())
        other = ~rows
        colls.append(float((lg.argmax(-1)[other] == batch.target[other]).float().mean()))
        dims.append(len(chosen))
        sil.append(float(after <= 0.25))
        if after <= 0.25:
            A[f] = dirs

    names = list(A)
    An = {f: Vn[f][:A[f].shape[0]] for f in names}
    alloc = allocation([A[f] for f in names], d, null_overlap=mo(An, names))
    return {"accuracy": acc, "n_facts": n_f, "d": d,
            "closure": float(np.mean(dims)), "silenced_rate": float(np.mean(sil)),
            "collateral": float(np.mean(colls)),
            "overlap_full": ov_full, "null_full": nu_full, "excess_full": ov_full - nu_full,
            "excess_content": ov_c - nu_c, "excess_address": ov_a - nu_a,
            "address_over_content": (ov_a - nu_a) - (ov_c - nu_c),
            "excess_deletion": alloc.excess if alloc.excess is not None else float("nan"),
            "pressure": alloc.pressure}


def run_arm(seed: int, steps: int, pod_w: float, priv_w: float, threads: int,
            verbose: bool) -> Dict[str, Any]:
    if threads:
        torch.set_num_threads(threads)
    rng = np.random.default_rng(9000 + seed)
    world = World.sample(rng, N_ENT, N_REL, N_FACTS, N_SYN)
    mc = ModelConfig(n_entities=N_ENT, n_relations=N_REL, n_surface=N_REL * N_SYN)
    tc = TrainConfig(seed=seed, n_steps=steps, fixed_world=True, n_entities=N_ENT,
                     n_relations=N_REL, n_synonyms=N_SYN, p_revoked=0.0, p_shred=0.0, p_stale=0.0,
                     mix={"fwd1": 1.0}, log_every=max(steps // 3, 1),
                     pod_weight=pod_w, private_weight=priv_w, pod_facts=24)
    out = train(mc, tc, world_override=world, verbose=verbose)
    model, centre = out["model"], out["centre"]
    bank = bank_from_world(np.random.default_rng(seed), world, centre, 0.0, 0.0, 0.0)
    facts = [(int(f.subject), int(f.relation)) for f in world.facts[:16]]
    m = measure(model, world, bank, bank.tensors(), facts, rng, mc.max_hops, N_ENT)
    m["train_seconds"] = out["train_seconds"]
    return m


KEYS = ["accuracy", "closure", "silenced_rate", "collateral", "overlap_full", "null_full",
        "excess_full", "excess_content", "excess_address", "address_over_content",
        "excess_deletion", "pressure"]

CRITERIA = {
    # there must be a model to speak of, or every geometry number below is about noise
    "A/accuracy": (">=", 0.60),
    # THE PRICE, which is the obvious way to cheat and is therefore pre-registered
    "accuracy_ratio": (">=", 0.95),
    # THE CLAIM: the objective moves the allocation the diagnosis said was the problem
    "excess_full_drop": (">=", 0.10),
    # and it should show up where the deletion is actually paid
    "collateral_gain": (">=", 0.05),
}

DECISION_RULE = (
    "excess_full_drop >= 0.10 with accuracy_ratio >= 0.95 -> allocation is trainable and E-000043's "
    "verdict has its constructive half. A drop that only arrives with accuracy_ratio below 0.95 -> the "
    "dimensions were not free and the capacity reading was closer to right. No drop -> allocation is "
    "not trainable by this objective, and the diagnosis stands without a remedy. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--pod", type=float, default=1.0)
    ap.add_argument("--private", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args(argv)

    t0 = time.time()
    per_seed = []
    for seed in args.seeds:
        a = run_arm(seed, args.steps, 0.0, 0.0, args.threads, verbose=True)
        b = run_arm(seed, args.steps, args.pod, args.private, args.threads, verbose=True)
        row = {"seed": seed}
        row.update({f"A/{k}": a[k] for k in KEYS})
        row.update({f"B/{k}": b[k] for k in KEYS})
        row["accuracy_ratio"] = b["accuracy"] / max(a["accuracy"], 1e-9)
        row["excess_full_drop"] = a["excess_full"] - b["excess_full"]
        row["excess_address_drop"] = a["excess_address"] - b["excess_address"]
        row["collateral_gain"] = b["collateral"] - a["collateral"]
        row["closure_drop"] = a["closure"] - b["closure"]
        per_seed.append(row)
        print(f"  seed {seed}: acc {a['accuracy']:.3f} -> {b['accuracy']:.3f} | excess "
              f"{a['excess_full']:+.4f} -> {b['excess_full']:+.4f} | collateral "
              f"{a['collateral']:.3f} -> {b['collateral']:.3f} | closure {a['closure']:.2f} -> "
              f"{b['closure']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    keys = [k for k in per_seed[0] if k != "seed"]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    record = {"experiment": "E-000044",
              "title": "training the pod objective: is the allocation failure fixable, and at what price",
              "evidence_level": "E4", "seeds": args.seeds, "steps": args.steps,
              "pod_weight": args.pod, "private_weight": args.private,
              "decision_rule": DECISION_RULE, "per_seed": per_seed, "aggregate": agg,
              "criteria": check, "seconds": time.time() - t0}
    rows = [[k.replace("A/", ""), f"{agg['A/' + k]['mean']:.4f}", f"{agg['B/' + k]['mean']:.4f}"]
            for k in ["accuracy", "closure", "silenced_rate", "collateral", "excess_full",
                      "excess_content", "excess_address", "pressure"] if "A/" + k in agg]
    md = [f"# E-000044 — {record['title']}", "",
          f"Two arms on the same worlds and seeds, {args.steps} steps, {len(args.seeds)} seed(s). Arm B",
          "adds the pod objective (`so/pod.py`): every access path of one fact pulled onto one carrier,",
          "carriers of different facts pushed apart, hinged at the larger of the Welch bound and the",
          "centring floor. Everything else is identical.", "",
          "## Arm A (baseline) against arm B (pod objective)", "",
          ledger.table(["measure", "A", "B"], rows), "",
          "## The differences the criteria are written on", "",
          ledger.table(["measure", "mean over seeds", "worst seed"],
                       [["accuracy ratio B/A (the price)", f"{agg['accuracy_ratio']['mean']:.4f}",
                         f"{agg['accuracy_ratio']['min']:.4f}"],
                        ["drop in excess overlap", f"{agg['excess_full_drop']['mean']:+.4f}",
                         f"{agg['excess_full_drop']['min']:+.4f}"],
                        ["drop in excess ADDRESSING overlap",
                         f"{agg['excess_address_drop']['mean']:+.4f}",
                         f"{agg['excess_address_drop']['min']:+.4f}"],
                        ["gain in bystander accuracy under deletion",
                         f"{agg['collateral_gain']['mean']:+.4f}",
                         f"{agg['collateral_gain']['min']:+.4f}"],
                        ["drop in closure size", f"{agg['closure_drop']['mean']:+.4f}",
                         f"{agg['closure_drop']['min']:+.4f}"]]), "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    name = "e000044_pod_objective" + (f"-{args.tag}" if args.tag else "")
    path = ledger.save(name, record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
