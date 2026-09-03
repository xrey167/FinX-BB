"""Experiment E-000022 — splitting the null column: refuse a question, ignore prose.

E-000018 cut the divergence on generic sentences fivefold and still missed its bar, and the reason
was structural rather than a matter of tuning. Two requirements are routed through the same channel
and contradict each other:

  * answering ' unknown' when a cell is gone needs an INJECTION, namely the unknown direction that
    the null column carries;
  * changing nothing on text the layer has no key for needs NO injection at all.

Both are expressed by routing to that one null column. In the unknown-fallback mode the null value
is the unknown direction, so generic text that routes to it is perturbed by construction; in the
prior-fallback mode the null value is zero, so generic text is safe and a broken question falls back
to the pretrained prior and names some entity instead of refusing.

The split: the null column's contribution is multiplied by a query-relevance score read from the
model's own state, so it can fire for a question whose cell is missing and stay silent on prose. The
match gate of E-000018 stays on, because it is what suppresses the CELL contribution when nothing
matches. Together the two say: inject a payload only when a cell matches, inject the unknown
direction only when the text is a question that found none, and otherwise inject nothing.

Run:  python -m so.experiments.e000022_two_channel_null [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.llm_adapter import AdapterConfig

GENERIC_SHARE = 0.25


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000022_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": np.asarray(ck["centre"]), "history": ck["history"], "train_seconds": ck["train_seconds"],
                "checkpoint_sha256": _sha256(path)}
    out = E18.train_arm(gk, seed, steps, GENERIC_SHARE)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run. The first group is the point of the experiment; the rest
    exist so that silence on prose cannot be bought by breaking what already worked. The bars are the
    ones E-000018 pre-registered, unchanged, so the two records compare directly."""
    return {
        "no_key_no_injection": {"generic/kl_to_base": ("<=", 0.05), "broken1_unknown": (">=", 0.90)},
        "reading_not_traded_away": {"train/active_correct": (">=", 0.90), "heldout/active_correct": (">=", 0.70)},
        "refusal_not_traded_away": {"revoke_train_min": (">=", 0.95), "revoke_heldout_min": (">=", 0.85),
                                    "shred_heldout_min": (">=", 0.85)},
        "deleted_object_never_returns": {"heldout/revoked_deleted_object": ("<=", 0.02),
                                         "heldout/deleted_object_given_active_correct": ("<=", 0.02)},
    }


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(cfg)
        print(f"=== seed {seed}: match gate + two-channel null, generic share {GENERIC_SHARE:g} ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force)
        m = E17.evaluate_templates(gk, 1800 + seed, out["centre"], E18.N_TRAIN_TEMPLATES)
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: round(v, 4) for k, v in m.items() if k in E18.KEYS}, flush=True)
    agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    both = {}
    try:
        both = __import__("json").load(open("so/results/e000018_both.json"))["aggregate"]
    except Exception:
        pass
    record = {
        "experiment": "E-000022",
        "title": "Splitting the null column: inject a payload only when a cell matches, the unknown direction only "
                 "when a question found none, and otherwise nothing",
        "evidence_level": "E5", "deletion_level": "F3" if met["refusal_not_traded_away"] else "F1",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "follows_from": "E-000018 recorded that its two remedies cut injection into unrelated text fivefold and still "
                        "missed, because refusing a question and ignoring prose were routed through the same null "
                        "column. This splits them and changes nothing else.",
        "by_construction": ["the query-relevance score multiplies only the NULL column's contribution; the cell "
                            "contribution is still governed by the match gate, so the two channels cannot mask each "
                            "other",
                            "the relevance score is read from the model's own state, not from the cells, so it can "
                            "fire for a question whose cell is missing"],
        "learned": ["both channels: which text counts as a question about a cell, and how well a query has to match a "
                    "key before its payload is injected"],
        "config": {"seeds": args.seeds, "steps": args.steps, "generic_share": GENERIC_SHARE,
                   "n_train_templates": E18.N_TRAIN_TEMPLATES, "adapter": cfg.to_dict(),
                   "train_generic_prompts": E18.TRAIN_GENERIC, "eval_generic_prompts": E17.GENERIC},
        "baseline_e000017b": E18.BASELINE,
        "baseline_e000018_both": {k: both[k]["mean"] for k in E18.KEYS if k in both},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}",
             f"{both[k]['mean']:.4f}" if k in both else "-",
             f"{E18.BASELINE[k]:.4f}" if k in E18.BASELINE else "-") for k in E18.KEYS if k in agg]
    md = "\n".join([
        f"# E-000022 — {record['title']}", "", record["follows_from"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed", "E-000018 both", "E-000017-B"], rows), "",
        "Pre-registered criteria (worst seed), identical to E-000018's:", "", ledger.criteria_table(check), "",
        "By construction: " + "; ".join(record["by_construction"]) + ".", "",
        "Learned: " + "; ".join(record["learned"]) + ".",
    ])
    path = ledger.save("e000022_two_channel_null", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
