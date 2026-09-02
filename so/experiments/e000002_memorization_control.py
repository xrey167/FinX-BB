"""Experiment E-000002 — weight-memorisation control (the "copy" problem).

Ledger sections 9, 16 and 28: key destruction or routing removal proves
nothing if the network has *copied* the information into its weights.  This
experiment measures exactly that.  Three training regimes of the same
architecture are compared:

    resampled        world re-sampled every step (E-000001-B models)
    fixed_routing    ONE fixed world for all steps, knowledge layer available
    fixed_no_routing ONE fixed world, knowledge layer removed (must memorise)

Measured on the training world (fixed regimes) or a fresh world (resampled):
direct accuracy with the layer intact, accuracy with the ENTIRE layer masked
(what the weights know on their own), and the leak after revoking 100 target
cells (fraction still answered correctly).

Run:  python -m so.experiments.e000002_memorization_control
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.experiments.common import accuracy, answers, load_base_model, unknown_rate
from so.experiments.e000001b_mini_transformer import CHECKPOINTS
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.mvcc import MVCCStore
from so.reference import load_world
from so.train import TrainConfig, make_centre, train
from so.world import World

N_TARGETS = 100


def train_fixed(seed: int, steps: int, use_routing: bool, world: World, force: bool = False):
    name = f"e000002_fixed_{'routing' if use_routing else 'noroute'}_seed{seed}.pt"
    path = CHECKPOINTS / name
    mc = ModelConfig(use_routing=use_routing)
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
        model.load_state_dict(ck["state_dict"]); model.eval()
        return model, ck["centre"], ck["train_seconds"]
    out = train(mc, TrainConfig(seed=seed, n_steps=steps, fixed_world=True), world_override=world)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["model"].state_dict(), "centre": out["centre"], "model_config": mc.to_dict(),
                "train_seconds": out["train_seconds"]}, path)
    return out["model"], out["centre"], out["train_seconds"]


def measure(model, world: World, centre: np.ndarray, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    store = MVCCStore(seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = world.facts
    qs = [world.make_query(rng, "fwd", f.subject, [f.relation]) for f in facts]
    truth = [f.obj for f in facts]
    m: Dict[str, float] = {}
    a = answers(model, store, world, qs)
    m["direct"] = accuracy(a, truth)
    no_bank = np.zeros(len(facts), dtype=bool)
    a_nb = answers(model, store, world, qs, cell_mask=no_bank)
    m["bank_removed_acc"] = accuracy(a_nb, truth)          # what the weights know without the layer
    m["bank_removed_unknown"] = unknown_rate(a_nb)
    idx = rng.choice(len(facts), size=N_TARGETS, replace=False)
    t_qs, t_truth = [qs[i] for i in idx], [truth[i] for i in idx]
    c_idx = [i for i in range(len(facts)) if i not in set(idx.tolist())]
    c_qs, c_truth = [qs[i] for i in c_idx], [truth[i] for i in c_idx]
    m["target_before"] = accuracy(answers(model, store, world, t_qs), t_truth)
    for i in idx:
        store.revoke(kids[facts[int(i)].key])
    a_t = answers(model, store, world, t_qs)
    m["target_after_revoke_leak"] = accuracy(a_t, t_truth)  # still correct = leaked from weights
    m["target_after_revoke_unknown"] = unknown_rate(a_t)
    m["control_after_revoke"] = accuracy(answers(model, store, world, c_qs), c_truth)
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    conditions = ["resampled", "fixed_routing", "fixed_no_routing"]
    results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in conditions}
    for seed in args.seeds:
        rng = np.random.default_rng(500 + seed)
        fixed_world = World.sample(rng, 256, 4, 1000, 2)
        # resampled: E-000001-B model, measured on the fixed world (which it has never seen)
        base = load_base_model(seed)
        m = measure(base["model"], fixed_world, base["centre"], 900 + seed); m["seed"] = seed
        results["resampled"].append(m); print("resampled", seed, m, flush=True)
        for cond, use_routing in (("fixed_routing", True), ("fixed_no_routing", False)):
            model, centre, secs = train_fixed(seed, args.steps, use_routing, fixed_world, args.force)
            m = measure(model, fixed_world, centre, 900 + seed); m["seed"] = seed; m["train_seconds"] = secs
            # general capability on a world the fixed models never saw
            other = World.sample(np.random.default_rng(600 + seed), 256, 4, 1000, 2)
            m["fresh_world_direct"] = measure(model, other, centre, 950 + seed)["direct"]
            results[cond].append(m); print(cond, seed, m, flush=True)
    keys = ["direct", "bank_removed_acc", "target_before", "target_after_revoke_leak", "target_after_revoke_unknown",
            "control_after_revoke"]
    agg = {c: ledger.aggregate(results[c], keys) for c in conditions}
    check = ledger.check_criteria(
        {f"{c}/{k}": v for c in conditions for k, v in agg[c].items()},
        {"resampled/bank_removed_acc": ("<=", 0.02), "resampled/target_after_revoke_leak": ("<=", 0.02),
         "resampled/control_after_revoke": (">=", 0.99), "fixed_no_routing/target_after_revoke_leak": (">=", 0.5),
         "fixed_no_routing/direct": (">=", 0.5)})
    record = {
        "experiment": "E-000002", "title": "Weight-memorisation control (copy problem)",
        "evidence_level": "E4", "deletion_level": None,
        "claim": "Revocation in the knowledge layer only deletes what the weights have not copied. With re-sampled "
                 "worlds the weights hold no facts (layer masked -> nothing answered) and revocation leaves no "
                 "leak; with a fixed training world the core copies facts into its weights and revocation leaks; "
                 "without a knowledge layer everything is weight-encoded and revocation is impossible.",
        "not_claimed": "No statement about unlearning facts already encoded in weights (that is exactly the "
                       "regime this mechanism avoids by construction).",
        "config": {"seeds": args.seeds, "fixed_steps": args.steps, "n_targets": N_TARGETS},
        "caveats": "Only 'fixed_routing' is an empirical control: 'resampled' cannot memorise by construction and "
                   "'fixed_no_routing' cannot read the layer by construction. The fixed-world regimes see the same "
                   "random lifecycle states per step as the re-sampled regime (only the world is held fixed), so the "
                   "no-routing model receives inconsistent labels for revoked/shredded cells and settles on the "
                   "majority label. Fixed regimes are trained for fewer steps than the re-sampled E-000001-B models.",
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_condition": results, "aggregate": agg,
    }
    rows = [(c, ledger.pct(agg[c]["direct"]["mean"]), ledger.pct(agg[c]["bank_removed_acc"]["mean"]),
             ledger.pct(agg[c]["target_after_revoke_leak"]["mean"]), ledger.pct(agg[c]["target_after_revoke_unknown"]["mean"]),
             ledger.pct(agg[c]["control_after_revoke"]["mean"])) for c in conditions]
    md = "\n".join([
        "# E-000002 — Weight-memorisation control (copy problem)", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}). Seeds: {args.seeds}. "
        f"Fixed-world regimes trained for {args.steps} steps; resampled regime = E-000001-B models.", "",
        ledger.table(["training regime", "direct (layer intact)", "layer fully masked", "target leak after REVOKE",
                      "target UNKNOWN after REVOKE", "control after REVOKE"], rows), "",
        "Reading: 'layer fully masked' is what the weights answer on their own. A leak after REVOKE is knowledge "
        "that survived in the weights — the copy problem the ledger warns about (sections 9, 28). The mechanism's "
        "deletion guarantee therefore depends on the training regime keeping facts out of the weights. "
        f"n = {N_TARGETS} targets per seed (leak of 0 in 300 pooled trials -> failure rate below 1.3% at 95%).", "",
        "Pre-registered criteria (worst seed; leak-type metrics use the max):", "", ledger.criteria_table(check), "",
        f"Caveats: {record['caveats']}", "",
        "Per seed:", "",
        "\n\n".join(f"**{c}**\n\n" + ledger.table(["seed"] + keys, [[s["seed"]] + [s[k] for k in keys] for s in results[c]])
                    for c in conditions),
    ])
    path = ledger.save("e000002_memorization_control", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
