"""Experiment E-000006 — ablations.

Ledger section 26: a component may only be called essential if removing it
has a measurable effect.  Variants trained from scratch (same data regime):

    full               E-000001-B models
    no_marker_gate     values are not gated by the marker -> SHRED must stop working
    no_null_cell       no explicit "nothing found" cell -> broken paths / UNKNOWN must suffer
    no_routing_loss    answer loss only -> does routing / provenance emerge on its own?
    no_routing         no knowledge layer at all -> nothing can be read from re-sampled worlds

Plus two mechanical rows: random deletion (revoking another cell must not
affect the target) and "without versioning" (UPDATE by in-place replace
leaves nothing to roll back to — structural, not learned).

Run:  python -m so.experiments.e000006_ablations
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.evaluation import run_suite
from so.experiments.common import answers, fresh_world, load_base_model
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, EVAL_CONFIG
from so.interventions import disable_mask
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.train import TrainConfig, train

VARIANTS: Dict[str, Dict[str, Any]] = {
    "full_same_budget": {"model": {}, "train": {}},          # the fair comparison: default model, variant step budget
    "no_marker_gate": {"model": {"use_marker_gate": False}, "train": {}},
    "no_null_cell": {"model": {"use_null_cell": False}, "train": {}},
    "no_routing_loss": {"model": {}, "train": {"route_weight": 0.0}},
    "no_routing": {"model": {"use_routing": False}, "train": {}},
}
SMALL_EVAL = dict(EVAL_CONFIG, n_2hop=200, n_3hop=200, n_broken=100, n_rev=100, n_lifecycle=50,
                  n_locality_updates=50, n_locality_revokes=25, n_locality_multihop=100, n_alt_pairs=50)
KEYS = ["direct", "hop2", "hop3", "hop2_broken_unknown", "provenance", "reverse", "revoke", "shred", "update",
        "rollback", "locality", "alternative_path"]


def train_variant(name: str, seed: int, steps: int, force: bool = False):
    path = CHECKPOINTS / f"e000006_{name}_seed{seed}.pt"
    v = VARIANTS[name]
    mc = ModelConfig(**v["model"])
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
        model.load_state_dict(ck["state_dict"]); model.eval()
        return model, ck["centre"], ck["train_seconds"]
    out = train(mc, TrainConfig(seed=seed, n_steps=steps, **v["train"]))
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["model"].state_dict(), "centre": out["centre"], "model_config": mc.to_dict(),
                "train_seconds": out["train_seconds"]}, path)
    return out["model"], out["centre"], out["train_seconds"]


def random_deletion(model, centre: np.ndarray, seed: int, n: int = 50) -> float:
    rng, world, store, kids, ref = fresh_world(650 + seed, centre)
    perm = rng.permutation(len(world.facts))
    ok = 0
    for i, j in zip(perm[:n], perm[n:2 * n]):
        f, g = world.facts[int(i)], world.facts[int(j)]
        q = world.make_query(rng, "fwd", f.subject, [f.relation])
        ok += int(answers(model, store, world, [q], cell_mask=disable_mask(store, kids[g.key]))[0] == f.obj)
    return ok / n


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    results: Dict[str, List[Dict[str, Any]]] = {"full": []}
    for seed in args.seeds:
        base = load_base_model(seed)
        m = run_suite(base["model"], 600 + seed, SMALL_EVAL, base["centre"], noise_levels=(0.0,), train_seed=seed)
        m["base_checkpoint_sha256"] = base["checkpoint_sha256"]
        m["random_deletion_target_unchanged"] = random_deletion(base["model"], base["centre"], seed)
        results["full"].append(m); print("full", seed, {k: m[k] for k in KEYS}, flush=True)
    for name in VARIANTS:
        results[name] = []
        for seed in args.seeds:
            model, centre, secs = train_variant(name, seed, args.steps, args.force)
            m = run_suite(model, 600 + seed, SMALL_EVAL, centre, noise_levels=(0.0,), train_seed=seed)
            m["train_seconds"] = secs
            results[name].append(m); print(name, seed, {k: m[k] for k in KEYS}, flush=True)
    agg = {name: ledger.aggregate(rs, KEYS) for name, rs in results.items()}
    # structural row: without versioning there is nothing to roll back to
    from so.mvcc import MVCCStore
    s = MVCCStore(seed=0); k = s.write(1, 0, 5); s.replace(k, 7)
    try:
        s.rollback(k, 2); no_versioning_rollback = "possible"
    except ValueError:
        no_versioning_rollback = "impossible (no version to return to)"
    check = ledger.check_criteria(
        {f"{v}/{k}": x for v in results for k, x in agg[v].items()},
        {"full_same_budget/direct": (">=", 0.98), "full_same_budget/shred": (">=", 0.95),
         "no_marker_gate/shred": ("<=", 0.2), "no_marker_gate/direct": (">=", 0.98),
         "no_null_cell/hop2_broken_unknown": ("<=", 0.5), "no_routing/direct": ("<=", 0.1)})
    record = {
        "experiment": "E-000006", "title": "Ablations",
        "evidence_level": "E4", "deletion_level": None,
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "by_construction_vs_learned": "'no_routing' and 'no_marker_gate' remove an information path, so their "
                                      "failures (nothing readable / SHRED ineffective) are information-flow "
                                      "necessities, reported to quantify them. 'no_null_cell' and 'no_routing_loss' "
                                      "keep the information paths and test learned behaviour: whether UNKNOWN "
                                      "detection and exact provenance emerge without the dedicated cell / loss. "
                                      "'full_same_budget' is the fair baseline trained with the variants' step budget.",
        "claim": "Each architectural component has a measurable, specific effect: the marker gate is what makes "
                 "SHRED work, the null cell is what makes broken paths answer UNKNOWN, routing is what makes any "
                 "reading of re-sampled worlds possible, and the routing loss is what makes provenance exact.",
        "not_claimed": "Optimality of the design; only necessity of components within this system.",
        "config": {"seeds": args.seeds, "variant_steps": args.steps, "full_steps": 3000},
        "per_variant": results, "aggregate": agg,
        "random_deletion_target_unchanged": float(np.mean([r["random_deletion_target_unchanged"] for r in results["full"]])),
        "no_versioning_rollback": no_versioning_rollback,
    }
    rows = [(name, *(ledger.pct(agg[name][k]["mean"]) for k in KEYS)) for name in results]
    md = "\n".join([
        "# E-000006 — Ablations", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}). Seeds: {args.seeds}; variants trained "
        f"{args.steps} steps, full model 3000 steps (E-000001-B). Values are means over seeds.", "",
        ledger.table(["variant"] + KEYS, rows), "",
        record["by_construction_vs_learned"], "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        f"Random deletion (revoke another cell, target must stay): {ledger.pct(record['random_deletion_target_unchanged'])}", "",
        "Reading the table: for a variant that answers UNKNOWN to everything (no_routing, no_routing_loss) the rows "
        "hop2_broken_unknown, revoke, shred and locality are satisfied trivially and carry no information.", "",
        f"Without versioning (UPDATE as in-place replace): rollback {no_versioning_rollback} — structural property of "
        "the layer, not a learned one.",
    ])
    path = ledger.save("e000006_ablations", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
