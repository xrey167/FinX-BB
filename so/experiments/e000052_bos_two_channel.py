"""E-000052 — can the BOS fix and the two-channel null fix compose?

Motivation
----------
E-000050 raised held-out reading from 0.7288 to 0.9712 on the BOS-trained arm,
but generic-text KL worsened to 4.2238 nats. E-000022 reduced generic KL with
an absolute match gate plus a separate query-relevance/null channel, but its
pre-BOS adapter did not reach the programme's reading/locality bars.

This experiment composes those two already-recorded fixes and changes nothing
else: same frozen GPT-2 small, same eight training templates, same generic
training shapes as E-000018/E-000022, same held-out/generic evaluation battery
as E-000017, and the same 3000-step budget. The only new condition is that a
BOS is present during both training and evaluation.

The experiment is deliberately falsifiable as a JOINT test. A useful external
memory must not buy locality by becoming unreadable, or buy reading by changing
unrelated text. The headline criterion therefore requires all of the following
on the worst seed:

  * trained active correctness >= 0.95
  * held-out active correctness >= 0.95
  * held-out REVOKE and SHRED propagation >= 0.95
  * broken-path UNKNOWN >= 0.90
  * deleted object return rate <= 0.02
  * generic-text KL to the frozen base <= 0.05 nats

Passing all rows is an engineering milestone for this prototype, not a novelty
claim and not evidence about knowledge already encoded in pretrained weights.

Controls
--------
1. Training generic prompts (E-000018.TRAIN_GENERIC) and evaluation generic
   prompts (E-000017.GENERIC) remain disjoint.
2. The same checkpoint is evaluated once with BOS on (claim arm) and once with
   BOS off (reverse control). If removing BOS does not materially hurt the
   subject-initial/held-out behaviour, the E-000050 causal story has not
   transported to this architecture.
3. No threshold is tuned on the evaluation set; all bars are fixed from the
   existing programme before this run.

Run:
  python -m so.experiments.e000052_bos_two_channel --seeds 0 1 2 --steps 3000
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, guard_recorded_checkpoint
from so.llm_adapter import AdapterConfig

GENERIC_SHARE = 0.25


def _bos(on: bool) -> None:
    os.environ["SO_BOS"] = "1" if on else "0"


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000052_bos_two_channel{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {
            "centre": np.asarray(ck["centre"]),
            "history": ck["history"],
            "train_seconds": float(ck["train_seconds"]),
            "checkpoint_sha256": _sha256(path),
        }

    _bos(True)
    out = E18.train_arm(gk, seed, steps, GENERIC_SHARE)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({
        "adapter": E8.adapter_state(gk.model),
        "centre": out["centre"],
        "history": out["history"],
        "train_seconds": out["train_seconds"],
        "adapter_config": gk.model.cfg.to_dict(),
        "bos": True,
        "generic_share": GENERIC_SHARE,
    }, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def measure(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, bos: bool) -> Dict[str, float]:
    _bos(bos)
    m = E17.evaluate_templates(gk, 1800 + seed, centre, E18.N_TRAIN_TEMPLATES)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))}


# Fixed before the run. These are intentionally stricter than E-000022's old
# reading bars because E-000050 established that >=0.95 is achievable after
# removing the position-0 artefact.
CRITERIA: Dict[str, Tuple[str, float]] = {
    "on/train/active_correct": (">=", 0.95),
    "on/heldout/active_correct": (">=", 0.95),
    "on/revoke_train_min": (">=", 0.95),
    "on/revoke_heldout_min": (">=", 0.95),
    "on/shred_train_min": (">=", 0.95),
    "on/shred_heldout_min": (">=", 0.95),
    "on/broken1_unknown": (">=", 0.90),
    "on/heldout/revoked_deleted_object": ("<=", 0.02),
    "on/generic/kl_to_base": ("<=", 0.05),
    # Reverse control: the same BOS-trained checkpoint should lose substantial
    # held-out capability when the BOS it was trained around is removed.
    "off/heldout/active_correct": ("<=", 0.85),
}

DECISION_RULE = (
    "JOINT PASS only if every on/* criterion passes on the worst seed and the reverse-control row also passes. "
    "That licenses only: 'this frozen-GPT-2 external memory simultaneously met the programme's reading, "
    "deletion/refusal and unrelated-text locality bars under the registered synthetic fact protocol.' It does "
    "not license a novelty claim, LLM-scale claim, or pretrained-weight unlearning claim. If generic KL remains "
    "> 0.05 while reading/deletion pass, locality is the remaining blocker. If reading/deletion regress while KL "
    "improves, the two fixes do not compose. If the reverse control fails, the E-000050 position-0 explanation "
    "did not transport cleanly and the joint result is not interpreted as a BOS composition."
)


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    # Instrument guard: generic shapes used for training and evaluation must be
    # distinct at the literal template level.
    overlap = set(E18.TRAIN_GENERIC).intersection(E17.GENERIC)
    if overlap:
        raise AssertionError(f"generic train/eval template leakage: {sorted(overlap)}")

    cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"=== E-000052 seed {seed}: BOS + match gate + two-channel null ===", flush=True)
        gk = E8.GPT2Knowledge(cfg)
        out = train_or_load(gk, seed, args.steps, args.force)
        row: Dict[str, Any] = {
            "seed": seed,
            "train_seconds": float(out["train_seconds"]),
            "checkpoint_sha256": out["checkpoint_sha256"],
        }
        mon = measure(gk, seed, out["centre"], True)
        moff = measure(gk, seed, out["centre"], False)
        row.update({f"on/{k}": v for k, v in mon.items()})
        row.update({f"off/{k}": v for k, v in moff.items()})
        per_seed.append(row)
        print({
            "on/train": round(mon["train/active_correct"], 4),
            "on/heldout": round(mon["heldout/active_correct"], 4),
            "on/generic_kl": round(mon["generic/kl_to_base"], 4),
            "on/broken_unknown": round(mon["broken1_unknown"], 4),
            "on/shred_heldout": round(mon["shred_heldout_min"], 4),
            "off/heldout": round(moff["heldout/active_correct"], 4),
        }, flush=True)

    numeric_keys = sorted(
        set.intersection(*[
            {k for k, v in row.items() if isinstance(v, (int, float)) and k != "seed"}
            for row in per_seed
        ])
    )
    agg = ledger.aggregate(per_seed, numeric_keys)
    check = ledger.check_criteria(agg, CRITERIA)

    on_keys = [k for k in CRITERIA if k.startswith("on/")]
    joint_on = all(check["criteria"][k]["pass"] for k in on_keys)
    reverse_ok = check["criteria"]["off/heldout/active_correct"]["pass"]
    breakthrough = bool(joint_on and reverse_ok)

    record = {
        "experiment": "E-000052",
        "title": "BOS plus two-channel null: joint reading/deletion/locality test",
        "evidence_level": "E5",
        "seeds": args.seeds,
        "steps": args.steps,
        "generic_share": GENERIC_SHARE,
        "adapter": cfg.to_dict(),
        "train_generic_prompts": E18.TRAIN_GENERIC,
        "eval_generic_prompts": E17.GENERIC,
        "generic_template_overlap": sorted(overlap),
        "decision_rule": DECISION_RULE,
        "criteria": check["criteria"],
        "joint_on_pass": joint_on,
        "reverse_control_pass": reverse_ok,
        "breakthrough_criterion_met": breakthrough,
        "per_seed": per_seed,
        "aggregate": agg,
        "baselines": {
            "E000050_BOS_trained": {"heldout_active_correct": 0.9712, "generic_kl": 4.2238, "shred_heldout_min": 0.9900},
            "E000022_two_channel_no_BOS": {"heldout_active_correct_worst": 0.7062, "generic_kl_worst": 0.8657},
        },
    }

    rows = []
    for k, (op, bar) in CRITERIA.items():
        worst_field = "min" if op == ">=" else "max"
        rows.append([k, f"{op} {bar:g}", f"{agg[k][worst_field]:.4f}",
                     "PASS" if check["criteria"][k]["pass"] else "FAIL"])
    md = "\n".join([
        "# E-000052 — BOS plus two-channel null: joint reading/deletion/locality test",
        "",
        "This composes E-000050's BOS training fix with E-000022's match gate + two-channel null, with no new mechanism.",
        "Training and evaluation generic templates are disjoint; overlap = 0 by runtime assertion.",
        "",
        f"**Joint breakthrough criterion: {'PASS' if breakthrough else 'FAIL'}**",
        "",
        ledger.table(["criterion", "required", "worst seed", "result"], rows),
        "",
        "Decision rule:", "", DECISION_RULE,
        "",
        "Scope: frozen GPT-2 small; synthetic, single-token facts; external mutable memory. No pretrained-weight unlearning claim.",
    ])
    path = ledger.save("e000052_bos_two_channel", record, md)
    print(md)
    print(f"\nsaved {path}")
    _bos(False)
    return record


if __name__ == "__main__":
    main()
