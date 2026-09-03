"""Experiment E-000014 — addressing at 10,000 cells.

Everything before this used banks of at most 1,000 cells over 256 entities.
Here the Mini-Transformer (with the class-balanced verified gate of E-000010)
is trained and evaluated with 7,000–10,000 cells over 2,560 entities: the
core suite, the reconstruction-attack battery after SHRED (soft and hard
gate) and a noise point.  Chance levels change with the entity count:
probe top-1 1/2560 = 0.00039, mean rank 1279.5.

Run:  python -m so.experiments.e000014_bank_10k [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.evaluation import SUITE_KEYS, run_suite
from so.experiments.attack_battery import ATTACK_ROWS, attack_battery
from so.experiments.e000001b_mini_transformer import train_or_load
from so.model import ModelConfig
from so.train import TrainConfig

N_ENT = 2560
EVAL10K: Dict[str, Any] = dict(
    n_entities=N_ENT, n_relations=4, n_synonyms=2, n_cells=10000, n_alt_structures=25,
    n_2hop=500, n_3hop=500, n_broken=100, n_rev=300, n_lifecycle=100, n_locality_updates=100,
    n_locality_revokes=50, n_locality_multihop=300, n_alt_pairs=100,
)
CORE = ["direct", "hop2", "hop3", "hop2_broken_unknown", "provenance", "reverse", "update", "rollback", "revoke",
        "restore", "shred", "resign", "locality", "alternative_path", "replay_deviation"]


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    model_cfg = ModelConfig(n_entities=N_ENT)
    suite: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []
    hard: List[Dict[str, Any]] = []
    for seed in args.seeds:
        tc = TrainConfig(seed=seed, n_steps=args.steps, n_entities=N_ENT, n_cells_min=7000, n_cells_max=10000,
                         gate_weight=5.0, gate_balanced=True)
        print(f"=== seed {seed}: training at up to 10,000 cells ===", flush=True)
        out = train_or_load("e000014", seed, model_cfg, tc, force=args.force)
        model, centre = out["model"], out["centre"]
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = run_suite(model, 1400 + seed, EVAL10K, centre, noise_levels=(0.0, 0.24), train_seed=seed)
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        suite.append(m)
        model.cfg.hard_gate = False
        soft.append(attack_battery(model, centre, seed, 1450 + seed, cfg=EVAL10K))
        model.cfg.hard_gate = True
        hard.append(attack_battery(model, centre, seed, 1450 + seed, cfg=EVAL10K))
        model.cfg.hard_gate = False
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in CORE}, flush=True)
        print("hard-gate SHRED:", {k: round(v, 4) for k, v in hard[-1].items() if k.startswith("shred/")}, flush=True)
    agg = ledger.aggregate(suite, SUITE_KEYS)
    keys_b = [k for k in soft[0] if k != "seed"]
    agg_soft, agg_hard = ledger.aggregate(soft, keys_b), ledger.aggregate(hard, keys_b)
    check = ledger.check_criteria(
        {**agg, **{f"hard/{k}": v for k, v in agg_hard.items()}},
        {"direct": (">=", 0.98), "hop2": (">=", 0.95), "hop3": (">=", 0.90), "provenance": (">=", 0.95),
         "reverse": (">=", 0.95), "revoke": (">=", 0.98), "shred": (">=", 0.95), "update": (">=", 0.98),
         "rollback": (">=", 0.98), "locality": (">=", 0.99), "alternative_path": (">=", 0.95),
         "replay_deviation": ("<=", 0), "hard/shred/direct_unknown": (">=", 0.98),
         "hard/shred/probe_top1": ("<=", 0.02), "hard/shred/forced_choice_win": ("<=", 0.6),
         "hard/shred/gated_value_contribution": ("<=", 0.1), "hard/active/direct_acc": (">=", 0.98)})
    record = {
        "experiment": "E-000014", "title": "Addressing at 10,000 cells (2,560 entities), verified gate",
        "evidence_level": "E4", "deletion_level": "F4" if check["claim_supported"] else "F3", "deletion_level_targeted": "F4",
        "claim": "At ten times the bank size the core still reads, composes and traces the right cell, reproduces every "
                 "lifecycle operation, and after SHRED with hard verification every reconstruction attack is at chance.",
        "not_claimed": "Anything about approximate retrieval or banks beyond 10,000 cells; the synthetic boundary applies.",
        "config": {"seeds": args.seeds, "steps": args.steps, "model": model_cfg.to_dict(), "eval": EVAL10K,
                   "train": {"n_entities": N_ENT, "n_cells_min": 7000, "n_cells_max": 10000, "gate_weight": 5.0, "gate_balanced": True}},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": suite, "attacks_soft": soft, "attacks_hard": hard,
        "aggregate": agg, "aggregate_attacks_soft": agg_soft, "aggregate_attacks_hard": agg_hard,
        "chance_levels": {"probe_top1": 1 / N_ENT, "mean_rank": (N_ENT - 1) / 2, "forced_choice": 0.5},
    }
    sizes = {"direct": 10000, "hop2": 500, "hop3": 500, "hop2_broken_unknown": 100, "provenance": 11000, "reverse": 300,
             "update": 100, "rollback": 100, "revoke": 100, "restore": 100, "shred": 100, "resign": 100,
             "locality": 10000 - 150 + 300, "alternative_path": 100}
    rows_b = [(a, f"{agg_soft[f'shred/{a}']['mean']:.4f}", f"{agg_hard[f'shred/{a}']['mean']:.4f}") for a in ATTACK_ROWS]
    md = "\n".join([
        "# E-000014 — Addressing at 10,000 cells", "",
        f"Evidence level: **E4**; deletion level targeted F4, recorded **{record['deletion_level']}**. Seeds: {args.seeds}; "
        f"{args.steps} steps; banks of 7,000–10,000 cells over {N_ENT} entities; class-balanced verified gate (weight 5).", "",
        ledger.table(ledger.CI_HEADERS, ledger.ci_rows(suite, SUITE_KEYS, sizes)), "",
        "Noise (bank perturbation 0.24, direct): " + ", ".join(f"seed {s['seed']}: {s['noise']['0.24']:.3f}" for s in suite), "",
        f"Attacks after SHRED (mean over seeds; chance: probe top-1 {1/N_ENT:.5f}, mean rank {(N_ENT-1)/2:.1f}, forced choice 0.5):", "",
        ledger.table(["attack after SHRED", "soft gate", "hard gate"], rows_b), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check),
    ])
    path = ledger.save("e000014_bank_10k", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
