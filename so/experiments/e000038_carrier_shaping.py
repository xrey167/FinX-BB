"""Experiment E-000038 — the carriers are not fact-specific; shape them until they are.

E-000037's instrument measured the workspace closure of a fact in a frozen GPT-2 and returned a number
that is useless in exactly the way its collateral was built to expose: for France -> Paris, eight of
eight candidate directions had to go, and by then bystander accuracy had fallen from 1.0000 to 0.0000.
The directions that carry one fact are not private to it. Facts are superposed and the model rebuilds
what is projected away.

So there is nothing in a residual stream whose removal is the fact and only the fact. A store has such
a thing -- a record -- and a pod makes it canonical: one object, many links, delete the object and
every access path dies. **A representation has no inode.** This experiment asks whether one can be
trained in, and it is the symlink pod moved one level down.

TWO PROPERTIES, FOUR ARMS, AND NEITHER PROPERTY IS SUFFICIENT ALONE.

  PRIVATE   ``so.carrier.privacy_loss``: the per-object read directions ``v_fwd(ent_emb(o))`` pushed
            towards mutual near-orthogonality, hinged at the WELCH BOUND so the loss is never asked to
            beat a theorem. For 256 objects in 128 dimensions that floor is 0.0626, and a carrier set
            at coherence c leaves a fraction c^2 of any other carrier removed with it -- about 0.4
            percent -- so the collateral bound is a consequence of the coherence rather than a hope.
  TIED      ``so.carrier.ablation_loss``: with the carrier projected out of the readout state, the
            model must answer UNKNOWN. This is the certificate as a training signal -- if the answer
            survives the removal of the carrier, the carrier was not where the fact was.

                    | not tied                   | tied
      not private   | closure k, collateral high  | closure 1, collateral high
      private       | closure k, collateral low   | closure 1, collateral low   <- the prediction

The obvious objection to the tying loss is that it teaches the model to detect the ablation and play
dead. That is why it is never reported without the collateral: the same projection is applied while
the rest of the batch keeps its ordinary answer loss, so a model that plays dead pays for it in the
same step, and the record shows both numbers side by side.

WHERE THE ABLATION ACTS, STATED PLAINLY BECAUSE IT BOUNDS THE CLAIM. Both the loss and the measurement
project the READOUT state out of the carrier. That is one intervention point, not the whole network,
so what is established is that the readout cannot recover the object -- not that no layer holds it.
A mid-network ablation is the stronger claim and is not made here.

Run:  python -m so.experiments.e000038_carrier_shaping [--seeds 0 1 2] [--steps 4000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.carrier import ablate_carrier, carriers, welch_bound
from so.data import bank_from_world
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, guard_recorded_checkpoint
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.train import TrainConfig, make_centre, train
from so.workspace import workspace_closure
from so.world import Query, World

ARMS = {"neither": (0.0, 0.0), "private": (1.0, 0.0), "tied": (0.0, 0.5), "both": (1.0, 0.5)}
N_ENTITIES, N_RELATIONS, N_SYNONYMS, D_MODEL = 256, 6, 2, 128
N_TARGETS, N_BYSTANDERS, N_CANDIDATES = 40, 40, 8


def model_config() -> ModelConfig:
    return ModelConfig(n_entities=N_ENTITIES, n_relations=N_RELATIONS, n_surface=N_RELATIONS * N_SYNONYMS,
                       d_model=D_MODEL, max_hops=3)


def train_config(seed: int, steps: int, privacy: float, ablation: float) -> TrainConfig:
    return TrainConfig(seed=seed, n_steps=steps, n_entities=N_ENTITIES, n_relations=N_RELATIONS,
                       n_synonyms=N_SYNONYMS, n_cells_min=600, n_cells_max=850, route_weight=0.5,
                       gate_weight=5.0, gate_balanced=True, log_every=500,
                       carrier_privacy=privacy, carrier_ablation=ablation)


def train_or_load(arm: str, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000038_{arm}{CKPT_SUFFIX}_seed{seed}.pt"
    cfg_m, cfg_t = model_config(), train_config(seed, steps, *ARMS[arm])
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        if ck.get("steps", steps) >= steps:
            model = MutableKnowledgeTransformer(ModelConfig(**ck["model_cfg"]))
            model.load_state_dict(ck["model"])
            model.eval()
            return {"model": model, "centre": np.asarray(ck["centre"]), "history": ck["history"],
                    "train_seconds": ck["train_seconds"], "checkpoint_sha256": _sha256(path)}
    out = train(cfg_m, cfg_t, verbose=False)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"model": out["model"].state_dict(), "model_cfg": cfg_m.to_dict(),
                "train_cfg": cfg_t.to_dict(), "centre": out["centre"], "history": out["history"],
                "steps": steps, "train_seconds": out["train_seconds"]}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


@torch.no_grad()
def hidden_and_answer(model, tensors, batch) -> Tuple[torch.Tensor, np.ndarray]:
    logits, _routing, extras = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid)
    return extras["hidden"], logits.argmax(-1).numpy()


def measure(model, centre, seed: int) -> Dict[str, Any]:
    """Closure and collateral at the readout, on a world the model never trained on."""
    from so.data import encode_queries
    rng = np.random.default_rng(9000 + seed)
    world = World.sample(rng, N_ENTITIES, N_RELATIONS, 800, N_SYNONYMS)
    bank = bank_from_world(rng, world, centre, 0.0, 0.0, 0.0)
    tensors = bank.tensors()
    facts = list(world.facts)
    pick = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in pick[:N_TARGETS]]
    bystanders = [facts[int(i)] for i in pick[N_TARGETS:N_TARGETS + N_BYSTANDERS]]

    def qs_of(fs, surface):
        return [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, surface),)) for f in fs]

    # one forward per surface form; the ablation is at the readout, so nothing needs re-running
    hs, truths = [], []
    for k in range(N_SYNONYMS):
        b = encode_queries(qs_of(targets, k), bank, world, model.cfg.max_hops)
        h, _ = hidden_and_answer(model, tensors, b)
        hs.append(h)
        truths.append(np.array([f.obj for f in targets]))
    b_by = encode_queries(qs_of(bystanders, 0), bank, world, model.cfg.max_hops)
    h_by, _ = hidden_and_answer(model, tensors, b_by)
    truth_by = np.array([f.obj for f in bystanders])

    with torch.no_grad():
        V = carriers(model)
        Vn = F.normalize(V, dim=-1)
        gram = (Vn @ Vn.t())
        off = gram - torch.diag(torch.diagonal(gram))
        base_read = float(np.mean([(model.readout(h).argmax(-1).numpy() == t).mean()
                                   for h, t in zip(hs, truths)]))
        base_by = float((model.readout(h_by).argmax(-1).numpy() == truth_by).mean())

        def answers_for(i, dirs):
            """The target's answer under this ablation, one entry per surface form."""
            out = []
            for h in hs:
                x = h[i: i + 1]
                for d in dirs:
                    x = ablate_carrier(x, V[d][None])
                out.append(int(model.readout(x).argmax(-1)))
            return out

        def collateral(dirs):
            x = h_by
            for d in dirs:
                x = ablate_carrier(x, V[d][None])
            return float((model.readout(x).argmax(-1).numpy() == truth_by).mean())

        sizes, colls, exhausted = [], [], []
        for i, f in enumerate(targets):
            own = int(f.obj)
            near = torch.argsort(off[own].abs(), descending=True)[:N_CANDIDATES - 1].tolist()
            cand = [own] + [int(x) for x in near]
            wc = workspace_closure(lambda d, i=i: answers_for(i, d), cand, own, N_SYNONYMS,
                                   max_dirs=len(cand), workload=f"{N_SYNONYMS} surface forms",
                                   lens="the object's own read direction v_fwd(ent_emb(o))",
                                   collateral_with=collateral, bound=False)
            sizes.append(float(wc.size))
            colls.append(float(wc.collateral) if wc.collateral is not None else float("nan"))
            exhausted.append(float(wc.exhausted))

    return {"read_before": base_read, "bystander_before": base_by,
            "closure_mean": float(np.mean(sizes)), "closure_max": float(np.max(sizes)),
            "exhausted_rate": float(np.mean(exhausted)),
            "collateral_after": float(np.nanmean(colls)),
            "collateral_cost": base_by - float(np.nanmean(colls)),
            "coherence_max": float(off.abs().max()), "coherence_rms": float((off ** 2).mean().sqrt()),
            "welch_bound": welch_bound(N_ENTITIES, D_MODEL)}


def run_seed(seed: int, steps: int, force: bool, verbose: bool = True) -> Dict[str, Any]:
    m: Dict[str, Any] = {"seed": seed}
    t0 = time.time()
    for arm in ARMS:
        out = train_or_load(arm, seed, steps, force)
        r = measure(out["model"], out["centre"], seed)
        for k, v in r.items():
            m[f"{arm}/{k}"] = v
        if verbose:
            print(f"  seed {seed} {arm:<8} read {r['read_before']:.4f}  closure {r['closure_mean']:.2f} "
                  f"(max {r['closure_max']:.0f})  collateral {r['bystander_before']:.4f} -> "
                  f"{r['collateral_after']:.4f}  coherence {r['coherence_max']:.4f} "
                  f"(Welch {r['welch_bound']:.4f})  ({time.time() - t0:.0f}s)", flush=True)
    m["seconds"] = time.time() - t0
    return m


KEYS = [f"{a}/{k}" for a in ARMS for k in
        ("read_before", "closure_mean", "closure_max", "collateral_after", "collateral_cost",
         "coherence_max", "coherence_rms", "exhausted_rate")]

CRITERIA = {
    # attack validity: every arm must read the fact before anything is ablated, or its closure is noise
    "neither/read_before": (">=", 0.85),
    "both/read_before": (">=", 0.85),
    # the contrast that can fail: if the untouched model ALREADY has private carriers, there is
    # nothing to shape and the whole experiment is answered in the negative
    "neither/collateral_cost": (">=", 0.15),
    # what shaping is supposed to buy
    "both/closure_mean": ("<=", 1.20),
    "both/collateral_cost": ("<=", 0.05),
    "both/coherence_max": ("<=", 0.20),
    # and it must not be bought with capability
    "both/read_before": (">=", 0.85),
}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.steps, args.force) for s in args.seeds]
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (bool, int, float))} for s in per_seed]
    keys = [k for k in KEYS if all(k in s for s in numeric)]
    agg = ledger.aggregate(numeric, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = [[arm,
             f"{agg[f'{arm}/read_before']['min']:.4f}",
             f"{agg[f'{arm}/closure_mean']['mean']:.2f}",
             f"{agg[f'{arm}/collateral_cost']['max']:.4f}",
             f"{agg[f'{arm}/coherence_max']['max']:.4f}"] for arm in ARMS]
    tbl = ledger.table(["arm", "reads the fact (worst seed)", "closure", "collateral cost (worst seed)",
                        "carrier coherence (worst seed)"], rows)

    record = {"experiment": "E-000038",
              "title": "the carriers are not fact-specific; shape them until they are",
              "seeds": args.seeds, "steps": args.steps, "arms": list(ARMS),
              "welch_bound": welch_bound(N_ENTITIES, D_MODEL),
              "per_seed": per_seed, "aggregate": agg, "criteria": check}
    md = [f"# E-000038 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.steps} steps, four arms trained identically apart from one loss",
          f"term each. {N_ENTITIES} objects in {D_MODEL} dimensions, so exact mutual orthogonality is",
          f"impossible and the Welch bound {welch_bound(N_ENTITIES, D_MODEL):.4f} is the floor the",
          "privacy loss is hinged at. Closure and collateral are measured on a world the model never",
          "trained on, by projecting the READOUT state out of carrier directions.", "",
          "## Shaping the carrier", "", tbl, "",
          "`closure` is how many carrier directions must go before neither surface form yields the",
          "object. `collateral cost` is what the same ablation costs on facts nobody asked to delete --",
          "the number that made E-000037's frozen-GPT-2 measurement useless, at 1.0000 to 0.0000. The",
          "pair is the result; either alone can be made to look good.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## What this does not establish", "",
          "The ablation acts at the READOUT, which is where the training loss acts. What is shown is",
          "that the readout cannot recover the object, not that no layer holds it; a mid-network",
          "ablation is the stronger claim and is not made. Training a model so that interventions are",
          "clean is not new -- codebook features and Backpack LMs do it for interpretability -- and the",
          "minimum direction count against LINEAR predictors is settled by LEACE. What is measured here",
          "is deletability as the objective and (closure, collateral) as the pair that reports it.", ""]
    path = ledger.save("e000038_carrier_shaping", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(tbl)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
