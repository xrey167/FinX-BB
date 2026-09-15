"""Experiment E-000009 — signature-verification gate: closing the SHRED residual.

E-000004 and E-000007 showed that after SHRED the target is gone behaviourally
but the learned marker gate closes only to ~9%, so a linear probe and forced
choice still recover a residual (F3 with a trace, not F4).  Two remedies are
tested on models trained from scratch (5 seeds):

    soft  the gate is trained with an explicit verification loss
          BCE(gate logits, marker validity) — the control plane knows which
          markers are signed, exactly as it knows which cell to route to
    hard  at read time the same gate is thresholded at 0.5: a payload is
          either signed or it is not (verification mode)

Both are measured with the E-000004 attack battery; the E-000001-B models
(no gate loss, soft gate) are the baseline.

Run:  python -m so.experiments.e000009_verification_gate [--seeds 0 1 2 3 4] [--steps 3000]
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.evaluation import run_suite
from so.experiments.attack_battery import ATTACK_ROWS, attack_battery
from so.experiments.common import load_base_model
from so.experiments.e000001b_mini_transformer import train_or_load
from so.experiments.e000006_ablations import SMALL_EVAL
from so.model import ModelConfig
from so.train import TrainConfig

CORE_KEYS = ["direct", "hop2", "hop3", "provenance", "reverse", "revoke", "shred", "update", "rollback", "locality"]


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--gate-weight", type=float, default=1.0)
    ap.add_argument("--balanced", action="store_true", help="class-balanced gate loss (E-000010)")
    ap.add_argument("--name", default="e000009_verification_gate", help="result / checkpoint name")
    ap.add_argument("--experiment", default="E-000009")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    ck_name = args.name.split("_")[0]
    conds = {"baseline_soft": [], "verified_soft": [], "verified_hard": []}
    core: Dict[str, List[Dict[str, Any]]] = {"verified_soft": [], "verified_hard": []}
    for seed in args.seeds:
        base = load_base_model(seed)
        conds["baseline_soft"].append(attack_battery(base["model"], base["centre"], seed, 900 + seed))
        out = train_or_load(ck_name, seed, ModelConfig(),
                            TrainConfig(seed=seed, n_steps=args.steps, gate_weight=args.gate_weight, gate_balanced=args.balanced),
                            force=args.force)
        model, centre = out["model"], out["centre"]
        model.cfg.hard_gate = False
        core["verified_soft"].append(run_suite(model, 950 + seed, SMALL_EVAL, centre, noise_levels=(0.0,), train_seed=seed))
        conds["verified_soft"].append(attack_battery(model, centre, seed, 900 + seed))
        model.cfg.hard_gate = True
        core["verified_hard"].append(run_suite(model, 950 + seed, SMALL_EVAL, centre, noise_levels=(0.0,), train_seed=seed))
        conds["verified_hard"].append(attack_battery(model, centre, seed, 900 + seed))
        model.cfg.hard_gate = False
        for c in conds:
            print(c, seed, {k: round(v, 4) for k, v in conds[c][-1].items() if k.startswith("shred/")}, flush=True)
    keys = [k for k in conds["baseline_soft"][0] if k != "seed"]
    agg = {c: ledger.aggregate(rs, keys) for c, rs in conds.items()}
    core_agg = {c: ledger.aggregate(rs, CORE_KEYS) for c, rs in core.items()}
    check = ledger.check_criteria(
        {f"{c}/{k}": v for c in conds for k, v in agg[c].items()} | {f"core_{c}/{k}": v for c in core for k, v in core_agg[c].items()},
        {"verified_hard/shred/direct_unknown": (">=", 0.98), "verified_hard/shred/probe_top1": ("<=", 0.05),
         "verified_hard/shred/forced_choice_win": ("<=", 0.6), "verified_hard/shred/true_obj_top1_among_entities": ("<=", 0.05),
         "verified_hard/shred/gated_value_contribution": ("<=", 0.1), "verified_hard/active/direct_acc": (">=", 0.98),
         "verified_hard/restored/direct_acc": (">=", 0.98), "core_verified_hard/direct": (">=", 0.98),
         "core_verified_hard/hop2": (">=", 0.98), "core_verified_hard/shred": (">=", 0.98),
         "verified_soft/shred/gated_value_contribution": ("<=", 0.5)})
    record = {
        "experiment": args.experiment, "title": "Signature-verification gate: closing the SHRED residual"
                 + (" (class-balanced loss)" if args.balanced else ""),
        "evidence_level": "E4", "deletion_level": "F4" if check["claim_supported"] else "F3", "deletion_level_targeted": "F4",
        "claim": "With an explicit verification loss the marker gate closes far more tightly on unsigned payloads, and "
                 "with hard verification at read time the SHRED residual measured in E-000004 / E-000007 disappears: "
                 "probe, forced choice and logit rank return to chance and the gated value contribution to zero, "
                 "while every other family stays intact.",
        "not_claimed": "Anything beyond the synthetic system; the hard gate is a deterministic verification step of the "
                       "control plane, not a learned property — the learned property is the soft gate's separation.",
        "by_construction_vs_learned": "The soft gate's separation of signed and unsigned markers is learned. Hard "
                                      "verification thresholds that learned score; once thresholded, a residual of exactly "
                                      "zero is by construction — the empirical content is whether thresholding at 0.5 "
                                      "misclassifies any marker (see core suite rows and gate statistics).",
        "config": {"seeds": args.seeds, "steps": args.steps, "gate_weight": args.gate_weight, "gate_balanced": args.balanced},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_condition": conds, "core_per_condition": core, "aggregate": agg, "core_aggregate": core_agg,
    }
    rows = [(a, *(f"{agg[c][f'shred/{a}']['mean']:.4f} / {agg[c][f'shred/{a}']['max' if a in ('probe_top1','probe_top5','forced_choice_win','true_obj_top1_among_entities','gated_value_contribution','gate_invalid_mean','gate_invalid_max') else 'min']:.4f}" for c in conds)) for a in ATTACK_ROWS]
    crow = [(k, *(ledger.pct(core_agg[c][k]["mean"]) + " / " + ledger.pct(core_agg[c][k]["min"]) for c in core)) for k in CORE_KEYS]
    md = "\n".join([
        f"# {args.experiment} — Signature-verification gate: closing the SHRED residual" + (" (class-balanced loss)" if args.balanced else ""), "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}); deletion level targeted for SHRED with hard verification: "
        f"F4, recorded **{record['deletion_level']}** within the synthetic system. Seeds: {args.seeds}; {args.steps} steps; gate loss weight {args.gate_weight}"
        + (", class-balanced" if args.balanced else "") + ". "
        "Baseline = the E-000001-B models (no gate loss).", "",
        "Attack battery after SHRED (mean / worst seed):", "",
        ledger.table(["attack after SHRED", "baseline (soft gate)", "verified (soft gate)", "verified (hard gate)"], rows), "",
        "Core families of the verified models (mean / worst seed), soft and hard gate:", "",
        ledger.table(["family", "verified soft", "verified hard"], crow), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        record["by_construction_vs_learned"], "",
        "Chance levels: probe top-1 0.0039, forced choice 0.5, mean rank 127.5.",
    ])
    path = ledger.save(args.name, record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
