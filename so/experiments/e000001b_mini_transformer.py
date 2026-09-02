"""Experiment E-000001-B — trained Mini-Transformer over the mutable knowledge layer.

Objective (architecture document section 20): determine whether learned neural
computation can interact with the controlled knowledge mechanism while
preserving the required semantics.  5 seeds; each seed trains its own model on
re-sampled worlds and is then evaluated on a fresh world with the full suite
plus a noise sweep.

Run:  python -m so.experiments.e000001b_mini_transformer [--steps N] [--seeds ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.evaluation import SUITE_KEYS, run_suite
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.train import TrainConfig, train

EVAL_CONFIG: Dict[str, Any] = dict(
    n_entities=256, n_relations=4, n_synonyms=2, n_cells=1000, n_alt_structures=25,
    n_2hop=500, n_3hop=500, n_broken=100, n_rev=300, n_lifecycle=100, n_locality_updates=100,
    n_locality_revokes=50, n_locality_multihop=300, n_alt_pairs=100,
)
CHECKPOINTS = ledger.RESULTS_DIR / "checkpoints"


def checkpoint_path(name: str, seed: int) -> Path:
    return CHECKPOINTS / f"{name}_seed{seed}.pt"


def train_or_load(name: str, seed: int, model_cfg: ModelConfig, train_cfg: TrainConfig, force: bool = False):
    path = checkpoint_path(name, seed)
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return {"model": model, "centre": ck["centre"], "history": ck["history"], "train_config": ck["train_config"],
                "model_config": ck["model_config"], "train_seconds": ck["train_seconds"], "loaded": True}
    out = train(model_cfg, train_cfg)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": out["model"].state_dict(), "centre": out["centre"], "history": out["history"],
                "train_config": out["train_config"], "model_config": out["model_config"],
                "train_seconds": out["train_seconds"]}, path)
    out["loaded"] = False
    return out


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    model_cfg = ModelConfig()
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"=== seed {seed}: training ===", flush=True)
        out = train_or_load("e000001b", seed, model_cfg, TrainConfig(seed=seed, n_steps=args.steps), force=args.force)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = run_suite(out["model"], 100 + seed, EVAL_CONFIG, out["centre"])
        m["train_seconds"] = out["train_seconds"]
        m["final_train_loss"] = out["history"][-1]["loss"] if out["history"] else None
        per_seed.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "noise"}, flush=True)
        print("noise:", m["noise"], flush=True)
    agg = ledger.aggregate(per_seed, SUITE_KEYS)
    noise_levels = list(per_seed[0]["noise"].keys())
    noise_agg = {lvl: float(np.mean([s["noise"][lvl] for s in per_seed])) for lvl in noise_levels}
    core = ["direct", "hop2", "hop3", "provenance", "revoke", "update", "rollback", "shred", "locality",
            "alternative_path"]
    all_pass = all(agg[k]["min"] == 1.0 for k in core) and agg["replay_deviation"]["max"] == 0
    record = {
        "experiment": "E-000001-B", "title": "Trained Mini-Transformer over the mutable knowledge layer",
        "evidence_level": "E4", "deletion_level": "F3",
        "claim": "A small transformer trained on re-sampled synthetic worlds reads, composes (multi-hop) and "
                 "traces knowledge held in the mutable layer, and reproduces the reference semantics of update, "
                 "rollback, revoke, restore and marker shredding on a fresh world, across seeds.",
        "not_claimed": "Nothing about pretrained LLMs or natural language; the facts never enter the weights by "
                       "construction (re-sampled worlds), so this does not show unlearning of weight-encoded facts.",
        "model_config": model_cfg.to_dict(), "train_config": TrainConfig(n_steps=args.steps).to_dict(),
        "eval_config": EVAL_CONFIG, "per_seed": per_seed, "aggregate": agg, "noise_aggregate": noise_agg,
        "all_pass": all_pass, "n_params": MutableKnowledgeTransformer(model_cfg).n_params(),
    }
    rows = [(k, ledger.pct(agg[k]["mean"]) if k != "replay_deviation" else f"{agg[k]['max']:.0f}",
             ledger.pct(agg[k]["min"]) if k != "replay_deviation" else "-") for k in SUITE_KEYS]
    md = "\n".join([
        "# E-000001-B — Trained Mini-Transformer over the mutable knowledge layer", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}). Deletion level demonstrated: **F3** "
        "(functional forgetting on the targeted path, verified against the mechanical reference).", "",
        f"Seeds: {args.seeds} · training steps: {args.steps} · parameters: {record['n_params']:,} · "
        f"core tests all at 100% in every seed: **{all_pass}**", "",
        ledger.table(["Measure", "Mean over seeds", "Worst seed"], rows), "",
        "Noise sweep (bank-level Gaussian perturbation, direct queries, mean over seeds):", "",
        ledger.table(["noise", "direct accuracy"], [(lvl, ledger.pct(v)) for lvl, v in noise_agg.items()]), "",
        "Per seed:", "",
        ledger.table(["seed"] + SUITE_KEYS + ["train_seconds"],
                     [[s["seed"]] + [s[k] for k in SUITE_KEYS] + [round(s["train_seconds"])] for s in per_seed]), "",
        "Interpretation: the behaviour is no longer mechanical — a trained neural core operates over the "
        "experimental knowledge structure. It is still a synthetic experiment and not proof of LLM-scale "
        "editable knowledge.",
    ])
    path = ledger.save("e000001b_mini_transformer", record, md)
    print(md)
    print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
