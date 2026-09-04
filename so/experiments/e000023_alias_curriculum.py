"""Experiment E-000023 — the reading cost of E-000020 is a learning cost, so attack the learning.

E-000020 carried the sharing and deletion contrast into a frozen GPT-2 but read direct facts at only
57% against the 85% the same adapter reaches without link cells. A five-minute diagnostic located
that cost precisely: the link-free adapter reads E-000020's own evaluation world at 82.7%, and 84.7%
with the link and dereference machinery attached. Neither the world nor the mechanism costs anything
at inference. The price is paid in LEARNING a distribution where, from the first step, a third of the
routing supervision is about aliases rather than facts.

Two ways to pay less, run as separate arms so the effect of each is attributable:

  longer      the identical distribution at twice the budget. If the gap is only optimisation time,
              this closes it and nothing else needs to change.
  curriculum  the identical budget, but aliases are phased in: link-free worlds until the reading is
              established, then the alias share ramps to full. If the gap is interference between the
              two supervisions, this closes it and the longer arm does not.

Both are measured with E-000020's evaluation and its pre-registered criteria, unchanged, so the three
records compare directly.

Run:  python -m so.experiments.e000023_alias_curriculum --arm curriculum [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.data import sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at, make_centre, routing_loss

ARMS = {"longer": dict(steps=6000, ramp_start=0, ramp_end=0),
        "curriculum": dict(steps=3000, ramp_start=600, ramp_end=1600)}


def alias_groups_at(step: int, n_groups: int, ramp_start: int, ramp_end: int) -> int:
    """How many alias groups the world at ``step`` carries.

    With ``ramp_end <= ramp_start`` the share is constant, which reproduces E-000020 exactly.
    """
    if ramp_end <= ramp_start:
        return n_groups
    if step < ramp_start:
        return 0
    if step >= ramp_end:
        return n_groups
    return int(round(n_groups * (step - ramp_start) / (ramp_end - ramp_start)))


def train_arm(gk: E8.GPT2Knowledge, seed: int, arm: str, batch_size: int = 32, route_weight: float = 1.0,
              gate_weight: float = 5.0, lr: float = 2e-3, route_only_steps: int = 400, p_revoked: float = 0.20,
              p_shred: float = 0.10, p_dangling: float = 0.05, n_groups: int = 100,
              extra_unanswerable: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
    conf = ARMS[arm]
    steps = conf["steps"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, gk.model.cfg.marker_dim)
    model = gk.model
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=50)
    mix = {"fwd1": 0.7, "fwd2": 0.3}
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    n_extra = int(round(batch_size * extra_unanswerable))
    n_reads = len(model.cfg.read_layers)
    n_skipped = 0
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_base = int(rng.integers(150, 301)) if route_only else int(rng.integers(500, 701))
        g_t = alias_groups_at(step, n_groups, conf["ramp_start"], conf["ramp_end"])
        world, spec = E15.sample_alias_world(rng, n_base, g_t, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
        bank = E15.bank_with_links(rng, world, spec, centre, p_revoked, p_shred, 0.05, p_dangling)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = E20.route_targets_slots(queries, bank, world, n_reads, model.cfg.n_deref)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid,
                                                      reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        loss = (loss_route + gate_weight * loss_gate) if route_only else \
            (loss_ans + route_weight * loss_route + gate_weight * loss_gate)
        if not torch.isfinite(loss):
            n_skipped += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 200 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "alias_groups": g_t, "batch_acc": acc,
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"route {rec['route_loss']:.4f}  groups {g_t:3d}  acc {acc:.3f}  {rec['elapsed_s']:.0f}s",
                      flush=True)
    model.eval()
    if n_skipped:
        print(f"  WARNING: {n_skipped} of {steps} steps skipped for a non-finite loss", flush=True)
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0, "skipped_steps": n_skipped}


def train_or_load(gk: E8.GPT2Knowledge, seed: int, arm: str, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000023_{arm}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": np.asarray(ck["centre"]), "history": ck["history"], "train_seconds": ck["train_seconds"],
                "skipped_steps": ck.get("skipped_steps", -1), "checkpoint_sha256": _sha256(path)}
    out = train_arm(gk, seed, arm)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "skipped_steps": out["skipped_steps"],
                "adapter_config": gk.model.cfg.to_dict(), "arm": arm}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(cfg)
        print(f"=== seed {seed}: arm {args.arm} ({ARMS[args.arm]}) ===", flush=True)
        out = train_or_load(gk, seed, args.arm, args.force)
        m = E20.evaluate(gk, 2000 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        m["skipped_steps"] = out["skipped_steps"]
        per_seed.append(m)
        print({k: round(v, 4) for k, v in m.items() if k in E20.KEYS}, flush=True)
    agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
    groups = E20.criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    level = ("F4" if met["one_shred_deletes_every_path"] and met["attacks_through_every_alias"]
             and met["attack_validity"] and met["alias_lifecycle"]
             else ("F3" if met["one_shred_deletes_every_path"] and met["attack_validity"] else "F0"))
    try:
        base = json.load(open("so/results/e000020_symlink_gpt2.json"))["aggregate"]
    except Exception:
        base = {}
    lower = {k for k, (op, _) in all_criteria.items() if op == "<="} | {
        "shred_target/alias_true_object", "shred_target/alias_probe_top1", "shred_target/alias_forced_choice",
        "shred_target/alias_top1_among_entities", "delete_target/alias_true_object",
        "duplicate_update/alias_new_object"}
    record = {
        "experiment": "E-000023", "arm": args.arm,
        "title": f"Alias reading in a frozen GPT-2: the '{args.arm}' arm against E-000020's budget",
        "evidence_level": "E5", "deletion_level": level,
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "follows_from": "E-000020 read direct facts at 57% while the same adapter without links reads this evaluation "
                        "world at 82.7%, and 84.7% with the machinery attached, so the cost is in learning rather "
                        "than in the world or the mechanism. This arm attacks the learning and changes nothing else.",
        "arm_config": ARMS[args.arm],
        "baseline_e000020": {k: base[k]["mean"] for k in E20.KEYS if k in base},
        "config": {"seeds": args.seeds, "arm": args.arm, "adapter": cfg.to_dict(), "eval": E20.EVAL,
                   "n_train_templates": E20.N_TRAIN_TEMPLATES},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{ledger.worst(agg[k], k in lower):.4f}",
             (f"{record['baseline_e000020'][k]:.4f}" if k in record["baseline_e000020"] else "-"))
            for k in E20.KEYS if k in agg]
    md = "\n".join([
        f"# E-000023 — {record['title']}", "", record["follows_from"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed", "E-000020 baseline"], rows), "",
        "Pre-registered criteria, identical to E-000020's:", "", ledger.criteria_table(check),
    ])
    path = ledger.save(f"e000023_{args.arm}", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
