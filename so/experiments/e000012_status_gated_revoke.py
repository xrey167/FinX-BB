"""Experiment E-000012 — status-gated REVOKE in the frozen GPT-2 core.

E-000011 (seed 0) found that with the selecting gate SHRED reaches 100% ' unknown'
while REVOKE by routing mask reaches only 76%: once the cell is masked, the
routing spreads over neighbouring keys instead of the null key and the model
names another entity.  Here REVOKE keeps the cell routable and folds the status
flag into the gate (an inactive cell reads as ' unknown', exactly like an
unsigned one); only DELETE removes routing.  Everything else — training data,
templates, evaluation — is E-000011's.

Run:  python -m so.experiments.e000012_status_gated_revoke [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import torch

from so import ledger
from so.data import failing_hop_target
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000011_gpt2_v2 as E11
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, guard_recorded_checkpoint, _sha256
from so.llm_adapter import AdapterConfig
from so.world import Query


def route_targets_status_gated(queries: List[Query], bank, world, n_reads: int) -> torch.Tensor:
    B = len(queries)
    route = np.full((B, n_reads), -2, dtype=np.int64)
    for i, q in enumerate(queries):
        gt = world.answer(q, bank.index_view)
        start = n_reads - q.hops
        route[i, :start] = -1
        for t in range(q.hops):
            if t < len(gt.edges):
                route[i, start + t] = bank.kid_of_key[gt.edges[t]]
            elif t == len(gt.edges):
                route[i, start + t] = failing_hop_target(bank, q, gt, status_gated=True)
            else:
                route[i, start + t] = -2
    return torch.as_tensor(route)


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000012_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"], "loaded": True,
                "checkpoint_sha256": _sha256(path)}
    original = E8.route_targets
    E8.route_targets = route_targets_status_gated          # E-000011's trainer uses E8.route_targets
    try:
        out = E11.train_adapter_v2(gk, seed, steps)
    finally:
        E8.route_targets = original
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["loaded"] = False
    out["checkpoint_sha256"] = _sha256(path)
    return out


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
        print(f"=== seed {seed}: adapter training (status-gated revoke) ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = E11.evaluate(gk, 1200 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    # identical criteria and reporting to E-000011, under this experiment's name
    return _report(per_seed, args)


def _report(per_seed: List[Dict[str, Any]], args) -> Dict[str, Any]:
    keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups, lenient = E11.criteria_groups()      # identical bar; only the design differs
    check_lenient = ledger.check_criteria(agg, lenient)
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    level = E11.deletion_level(met, gk_status_gated=True)
    record = {
        "experiment": "E-000012", "title": "Frozen GPT-2 core: status-gated REVOKE (revoked cells stay routable and read as unknown)",
        "evidence_level": "E5", "deletion_level": level, "deletion_level_targeted": "F4",
        "deletion_level_note": "In this design REVOKE does NOT remove routing: the revoked cell stays addressed and its "
                               "status closes the gate. When the deletion groups are unmet the floor is therefore F0 "
                               "(the payload is present and routed to, the read is suppressed), not F1.",
        "evidence_level_note": "E5 names the substrate (a pretrained transformer as frozen core); support is stated per claim group.",
        "claim_groups_met": met,
        "claim_parts": [{"claim": f"{g}", "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "design_change": "REVOKE no longer removes routing: the revoked cell stays addressable and the status flag multiplies the "
                         "gate, so it reads as ' unknown' exactly like an unsigned cell. Only DELETE removes a cell from routing. "
                         "Motivation: E-000011 seed 0 — SHRED 100% but REVOKE by mask 76% ' unknown' (routing spreads over "
                         "neighbouring keys once the cell is masked).",
        "not_claimed": "LLM scale; multi-token entities; unlearning of pretrained facts.",
        "config": {"seeds": args.seeds, "steps": args.steps, "status_gated": True, "eval": E11.EVAL, "templates": E11.TEMPLATES6},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "lenient_criteria": check_lenient["criteria"], "lenient_supported": check_lenient["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, ledger.pct(agg[k]["mean"]), ledger.pct(agg[k]["min"])) for k in E11.KEYS if k in agg]
    arows = [(a, *(f"{agg[f'{c}/{a}']['mean']:.4f}" for c in ("active", "revoke", "shred_soft", "shred_hard"))) for a in E11.ATT]
    irows = [(k, ledger.pct(agg[f"interventions/{k}"]["mean"]), ledger.pct(agg[f"interventions/{k}"]["min"])) for k in E11.INT]
    md = "\n".join([
        "# E-000012 — Frozen GPT-2 core: status-gated REVOKE", "",
        f"Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **{level}**. Seeds: {args.seeds}; {args.steps} steps. "
        + record["design_change"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**") for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "Attacks on 100 targets (mean over seeds):", "",
        ledger.table(["attack", "active", "after REVOKE", "after SHRED (soft)", "after SHRED (hard)"], arows), "",
        "Causal interventions on correctly answered 2-hop questions (mean / worst seed):", "",
        ledger.table(["intervention", "mean", "worst seed"], irows), "",
        "Pre-registered criteria (worst seed; identical to E-000011):", "", ledger.criteria_table(check), "",
        "Lenient criteria (secondary):", "", ledger.criteria_table(check_lenient),
    ])
    path = ledger.save("e000012_status_gated_revoke", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
