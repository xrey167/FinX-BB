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
N_TARGETS = 500     # attack battery: at chance 1/2560, 500 targets make one hit 0.2%; thresholds below allow at most 3 hits
EVAL10K: Dict[str, Any] = dict(
    n_entities=N_ENT, n_relations=4, n_synonyms=2, n_cells=10000, n_alt_structures=25,
    n_2hop=500, n_3hop=500, n_broken=100, n_rev=300, n_lifecycle=100, n_locality_updates=100,
    n_locality_revokes=50, n_locality_multihop=300, n_alt_pairs=100,
)
CORE = ["direct", "hop2", "hop3", "hop2_broken_unknown", "provenance", "reverse", "update", "rollback", "revoke",
        "restore", "shred", "resign", "locality", "alternative_path", "replay_deviation"]


def scaling_curve(model, centre: np.ndarray, seed: int) -> Dict[str, Dict[str, float]]:
    """The same model read on fresh worlds of 1k / 3k / 10k cells: direct accuracy and routing margin."""
    from so.evaluation import build_eval_world, predict
    out: Dict[str, Dict[str, float]] = {}
    for n_cells in (1000, 3000, 10000):
        rng, world, store, kids = build_eval_world(seed + n_cells, N_ENT, 4, 2, n_cells, 10, centre)
        sample = [world.facts[int(i)] for i in rng.choice(len(world.facts), size=1000, replace=False)]
        qs = [world.make_query(rng, "fwd", f.subject, [f.relation]) for f in sample]
        p = predict(model, store, world, qs)
        top = p.routing[:, 0, :].max(axis=1)
        out[str(n_cells)] = {"direct": float(np.mean([a == f.obj for a, f in zip(p.answers, sample)])),
                             "routing_max_mass_mean": float(top.mean()), "routing_max_mass_min": float(top.min())}
    return out


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
        soft.append(attack_battery(model, centre, seed, 1450 + seed, n_targets=N_TARGETS, cfg=EVAL10K))
        model.cfg.hard_gate = True
        hard.append(attack_battery(model, centre, seed, 1450 + seed, n_targets=N_TARGETS, cfg=EVAL10K))
        model.cfg.hard_gate = False
        m["scaling"] = scaling_curve(model, centre, 1480 + seed)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in CORE}, flush=True)
        print("scaling:", m["scaling"], flush=True)
        print("hard-gate SHRED:", {k: round(v, 4) for k, v in hard[-1].items() if k.startswith("shred/")}, flush=True)
    agg = ledger.aggregate(suite, SUITE_KEYS)
    keys_b = [k for k in soft[0] if k != "seed"]
    agg_soft, agg_hard = ledger.aggregate(soft, keys_b), ledger.aggregate(hard, keys_b)
    check = ledger.check_criteria(
        {**agg, **{f"hard/{k}": v for k, v in agg_hard.items()}},
        {"direct": (">=", 0.98), "hop2": (">=", 0.95), "hop3": (">=", 0.90), "hop2_broken_unknown": (">=", 0.95),
         "provenance": (">=", 0.95), "reverse": (">=", 0.95), "revoke": (">=", 0.98), "restore": (">=", 0.98),
         "shred": (">=", 0.95), "resign": (">=", 0.98), "update": (">=", 0.98), "rollback": (">=", 0.98),
         "locality": (">=", 0.99), "alternative_path": (">=", 0.95), "replay_deviation": ("<=", 0),
         "hard/shred/direct_unknown": (">=", 0.98), "hard/shred/probe_top1": ("<=", 0.006),
         "hard/shred/true_obj_top1_among_entities": ("<=", 0.006), "hard/shred/true_obj_mean_rank": (">=", 1150.0),
         "hard/shred/forced_choice_win": ("<=", 0.56), "hard/shred/gated_value_contribution": ("<=", 0.1),
         "hard/active/direct_acc": (">=", 0.98), "hard/restored/direct_acc": (">=", 0.98)})
    # thresholds at 2,560 entities and 500 targets: chance probe / top-1 = 0.00039 (P(>=4 hits | chance) < 1e-6, so
    # <= 3 hits = 0.006 passes); chance rank 1279.5 (>= 1150 is within 10% of chance); forced choice 0.5 +- 0.022 (0.56 = 2.7 sd).
    # The core thresholds are lower than E-000001-B's (0.99 / 0.98 / 0.95) because the task is harder in two ways at once:
    # ten times the bank AND a ten times larger read-out vocabulary; this is stated in the record.
    record = {
        "experiment": "E-000014", "title": "Addressing at 10,000 cells (2,560 entities), verified gate",
        "evidence_level": "E4", "deletion_level": "F4" if check["claim_supported"] else "F3", "deletion_level_targeted": "F4",
        "claim": "At ten times the bank size (and a ten times larger entity vocabulary) the core still reads, composes and "
                 "traces the right cell, reproduces every lifecycle operation, and after SHRED with hard verification the "
                 "reconstruction attacks stay within the thresholds set for a 2,560-entity vocabulary (probe / top-1 at most "
                 "0.02 with 100 targets, mean rank at least 1100 of 1279.5, forced choice at most 0.6).",
        "not_claimed": "A same-task scale-up of E-000001-B: the entity vocabulary is 2,560 instead of 256, so the task is "
                       "harder in two ways at once (bank size and read-out classes). Nothing about approximate retrieval or "
                       "banks beyond 10,000 cells; the synthetic boundary applies.",
        "config": {"seeds": args.seeds, "steps": args.steps, "model": model_cfg.to_dict(), "eval": EVAL10K,
                   "train": {"n_entities": N_ENT, "n_cells_min": 7000, "n_cells_max": 10000, "gate_weight": 5.0, "gate_balanced": True}},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": suite, "attacks_soft": soft, "attacks_hard": hard,
        "aggregate": agg, "aggregate_attacks_soft": agg_soft, "aggregate_attacks_hard": agg_hard,
        "chance_levels": {"probe_top1": 1 / N_ENT, "mean_rank": (N_ENT - 1) / 2, "forced_choice": 0.5},
        "threshold_note": "Core thresholds (0.98 / 0.95 / 0.90) are lower than E-000001-B's (0.99 / 0.98 / 0.95) because "
                          "the task is harder in two ways at once: ten times the bank and a ten times larger read-out "
                          "vocabulary. Attack thresholds are binomially derived for 500 targets at chance 1/2560.",
    }
    sizes = {"direct": EVAL10K["n_cells"], "hop2": EVAL10K["n_2hop"], "hop3": EVAL10K["n_3hop"],
             "hop2_broken_unknown": EVAL10K["n_broken"], "hop3_broken_unknown": EVAL10K["n_broken"],
             "provenance": EVAL10K["n_cells"] + EVAL10K["n_2hop"] + EVAL10K["n_3hop"], "reverse": EVAL10K["n_rev"],
             **{k: EVAL10K["n_lifecycle"] for k in ("update", "rollback", "revoke", "restore", "shred", "resign")},
             "locality": EVAL10K["n_cells"] - EVAL10K["n_locality_updates"] - EVAL10K["n_locality_revokes"] + EVAL10K["n_locality_multihop"],
             "locality_targets_correct": EVAL10K["n_locality_updates"] + EVAL10K["n_locality_revokes"],
             "alternative_path": EVAL10K["n_alt_pairs"]}
    rows_b = [(a, f"{agg_soft[f'shred/{a}']['mean']:.4f}", f"{agg_hard[f'shred/{a}']['mean']:.4f}") for a in ATTACK_ROWS]
    md = "\n".join([
        "# E-000014 — Addressing at 10,000 cells", "",
        f"Evidence level: **E4**; deletion level targeted F4, recorded **{record['deletion_level']}**. Seeds: {args.seeds}; "
        f"{args.steps} steps; banks of 7,000–10,000 cells over {N_ENT} entities; class-balanced verified gate (weight 5).", "",
        ledger.table(ledger.CI_HEADERS, ledger.ci_rows(suite, SUITE_KEYS, sizes)), "",
        "Noise (bank perturbation 0.24, direct): " + ", ".join(f"seed {s['seed']}: {s['noise']['0.24']:.3f}" for s in suite), "",
        f"Attacks after SHRED on {N_TARGETS} targets (mean over seeds; chance: probe top-1 {1/N_ENT:.5f}, mean rank {(N_ENT-1)/2:.1f}, forced choice 0.5):", "",
        record["threshold_note"], "",
        ledger.table(["attack after SHRED", "soft gate", "hard gate"], rows_b), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "Scaling curve, same model on fresh worlds (direct accuracy / mean routing max-mass), per seed:", "",
        ledger.table(["seed", "1,000 cells", "3,000 cells", "10,000 cells"],
                     [(s["seed"], *(f"{s['scaling'][k]['direct']:.3f} / {s['scaling'][k]['routing_max_mass_mean']:.3f}" for k in ("1000", "3000", "10000"))) for s in suite]),
    ])
    path = ledger.save("e000014_bank_10k", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
