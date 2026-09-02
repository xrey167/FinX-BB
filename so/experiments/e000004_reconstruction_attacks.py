"""Experiment E-000004 — reconstruction attacks against a deletion.

Ledger sections 23, 24: after REVOKE or SHRED of 100 target cells the deleted
objects are attacked through direct query, paraphrase, multi-hop, reverse
query, forced choice, a linear representation probe on the pre-read-out
hidden state, an activation probe (routing mass and gated value contribution
of the target cell), and dependency reconstruction (K3 derivable from
K1 + K2).  Context completion is not applicable to the symbolic query format.

Run:  python -m so.experiments.e000004_reconstruction_attacks
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.experiments.attack_battery import attack_battery
from so.experiments.common import load_base_model

N_TARGETS = 100


def run_seed(seed: int) -> Dict[str, Any]:
    base = load_base_model(seed)
    return attack_battery(base["model"], base["centre"], seed, 400 + seed, n_targets=N_TARGETS)


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    per_seed = [run_seed(s) for s in args.seeds]
    for s in per_seed: print(s, flush=True)
    keys = [k for k in per_seed[0] if k != "seed"]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {
        "active/direct_acc": (">=", 0.98), "active/probe_top1": (">=", 0.5),
        "revoke/direct_unknown": (">=", 0.98), "revoke/probe_top1": ("<=", 0.05), "revoke/forced_choice_win": ("<=", 0.6),
        "revoke/true_obj_top1_among_entities": ("<=", 0.05),
        "shred/direct_unknown": (">=", 0.95), "shred/paraphrase_unknown": (">=", 0.95), "shred/probe_top1": ("<=", 0.05),
        "shred/forced_choice_win": ("<=", 0.6), "shred/true_obj_top1_among_entities": ("<=", 0.05),
        "shred/gated_value_contribution": ("<=", 0.1), "restored/direct_acc": (">=", 0.98)})
    record = {
        "experiment": "E-000004", "title": "Reconstruction attacks against REVOKE and SHRED",
        "evidence_level": "E4", "deletion_level": "F4",
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "by_construction_vs_learned": "After REVOKE the routing mass and value contribution on the target are zero "
                                      "by the mask, not by learning — those two rows are reported for completeness "
                                      "only. After SHRED the cell is still routable, so every row is a measurement "
                                      "of learned behaviour; the SHRED column carries the F4-level evidence.",
        "claim": "After REVOKE or SHRED the deleted object is not recoverable through direct, paraphrase, multi-hop "
                 "or reverse queries, forced choice is at chance, the true object's logit rank is at chance, a "
                 "linear probe on the hidden state is at chance, and the target cell's gated value contribution is "
                 "zero — within the synthetic system. Dependency reconstruction shows that deleting a derivable "
                 "fact alone is meaningless until its dependency closure is revoked.",
        "not_claimed": "F5 in general: the probe is linear and the system synthetic; real LLM representations are "
                       "not addressed.",
        "config": {"seeds": args.seeds, "n_targets": N_TARGETS},
        "per_seed": per_seed, "aggregate": agg,
    }
    attacks = ["direct_unknown", "direct_acc", "paraphrase_unknown", "multihop_unknown", "reverse_unknown",
               "forced_choice_win", "true_obj_top1_among_entities", "true_obj_mean_rank", "probe_top1", "probe_top5",
               "routing_mass_on_target", "gated_value_contribution"]
    rows = [(a, *(f"{agg[f'{c}/{a}']['mean']:.4f}" for c in ("active", "revoke", "shred"))) for a in attacks]
    dep = [(k.split("/", 1)[1], f"{agg[k]['mean']:.4f}") for k in keys if k.startswith("dependency/")]
    md = "\n".join([
        "# E-000004 — Reconstruction attacks", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}); deletion level **F4** within the synthetic "
        "system (representation-level checks, linear probe). Seeds: " + str(args.seeds) +
        f". Probe calibration on held-out active cells: top-1 {agg['probe_calibration_top1']['mean']:.3f}, "
        f"top-5 {agg['probe_calibration_top5']['mean']:.3f}. Chance: forced choice 0.5, top-1 among entities "
        f"1/256 = 0.0039, mean rank 127.5, probe top-1 0.0039, top-5 0.0195.", "",
        ledger.table(["attack (mean over seeds)", "active", "after REVOKE (mask)", "after SHRED (learned)"], rows), "",
        record["by_construction_vs_learned"], "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        f"Sample sizes per seed: {N_TARGETS} targets (probe / forced choice / rank / direct); multi-hop and reverse "
        "subsets are smaller (only targets with an outgoing edge or a unique reverse subject).", "",
        "Dependency reconstruction (K3 derivable from K1 + K2; 'collateral' = 2-hop paths not touching the closure):", "",
        ledger.table(["measure", "mean"], dep), "",
        "Context completion: not applicable (symbolic queries, no free text).",
    ])
    path = ledger.save("e000004_reconstruction_attacks", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
