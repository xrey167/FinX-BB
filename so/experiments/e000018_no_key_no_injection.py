"""Experiment E-000018 — no key, no injection.

E-000017-B closed most of the paraphrase gap and surfaced a worse problem in doing so: the adapter
perturbs the frozen model on text it has no key for, and the effect got WORSE with more trained
prompt shapes, not better. Recorded there: generic sentences move by 3.27 nats against a 0.05 bar
(E-000013: 2.27) and broken paths answer ' unknown' only 71.8% of the time against a 90% bar. This
matters more than the reading figure: a layer that changes the model's output on unrelated text
cannot be attached to a model that also has to behave normally.

The mechanism is not subtle. Routing is a softmax over the cell keys plus one null key, so the
distribution always sums to one and SOME cell always wins. The null key competes on the same scale
and loses on prompt shapes the adapter never saw. Nothing in the architecture can express "nothing
here matches".

Two remedies, pre-registered, run as separate arms so the effect of each is attributable:

  gate     an absolute match score: the injection is scaled by sigmoid((max cosine between the query
           and any REAL cell key) - tau), with tau and the temperature learned per read layer. This
           gives the model the capacity to inject nothing; the losses already want it to.
  generic  generic sentences in the training batch, with the null key as their routing target and a
           Kullback-Leibler term against the frozen model's own distribution. This trains the
           behaviour without adding capacity.
  both     both at once.

The baseline is E-000017-B: same templates, same budget, same evaluation.

Run:  python -m so.experiments.e000018_no_key_no_injection --arm both [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.data import bank_from_world, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.experiments.e000012_status_gated_revoke import route_targets_status_gated
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import World

ARMS = {"gate": (True, 0.0), "generic": (False, 0.25), "both": (True, 0.25), "baseline": (False, 0.0)}
N_TRAIN_TEMPLATES = 8
# generic sentences used in TRAINING; the evaluation set in E-000017 is a different list, so a model
# that merely memorised these shapes does not pass
TRAIN_GENERIC = ["{s} thought about it", "Later that day {s}", "The letter from {s} arrived",
                 "Nobody expected {s} to", "{s} opened the door and", "It was {s} who first",
                 "Before the meeting {s}", "Around noon, {s}"]


def train_arm(gk: E8.GPT2Knowledge, seed: int, steps: int, generic_share: float, batch_size: int = 32,
              route_weight: float = 1.0, gate_weight: float = 5.0, generic_weight: float = 1.0,
              lr: float = 2e-3, route_only_steps: int = 300, p_revoked: float = 0.20, p_shred: float = 0.10,
              extra_unanswerable: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
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
    n_gen = int(round(batch_size * generic_share))
    n_reads = len(model.cfg.read_layers)
    n_skipped = 0
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_cells = int(rng.integers(150, 301)) if route_only else int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, N_TRAIN_TEMPLATES)
        bank = bank_from_world(rng, world, centre, p_revoked, p_shred, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        texts = [E17.query_text_pc(q, gk.names, N_TRAIN_TEMPLATES) for q in queries]
        n_q = len(queries)
        gen_texts: List[str] = []
        if n_gen and not route_only:
            gen_texts = [TRAIN_GENERIC[int(rng.integers(0, len(TRAIN_GENERIC)))].format(
                s=gk.names[int(rng.integers(0, gk.n_entities))]) for _ in range(n_gen)]
        ids, am, last = E8.encode_texts(gk.tok, texts + gen_texts)
        target = E8.targets_of(queries, bank, world)
        route = route_targets_status_gated(queries, bank, world, n_reads)
        if gen_texts:                                  # generic text routes to the null key at every read
            route = torch.cat([route, torch.full((len(gen_texts), n_reads), -1, dtype=route.dtype)])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, full, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand[:n_q], target)
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid,
                                                      reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        loss_gen = cand.sum() * 0
        if gen_texts:
            with torch.no_grad():
                base_full = model(None, ids[n_q:], am[n_q:], last[n_q:])[1]
            lb, la = torch.log_softmax(base_full, -1), torch.log_softmax(full[n_q:], -1)
            loss_gen = F.kl_div(la, lb, log_target=True, reduction="batchmean")
        loss = (loss_route + gate_weight * loss_gate) if route_only else \
            (loss_ans + route_weight * loss_route + gate_weight * loss_gate + generic_weight * loss_gen)
        if not torch.isfinite(loss):
            n_skipped += 1                      # never update on a non-finite loss; the count is recorded
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand[:n_q].argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "generic_loss": float(loss_gen.item()),
                   "batch_acc": acc, "elapsed_s": time.time() - t0}
            if getattr(model, "match_tau", None) is not None:
                rec["match_tau"] = [round(float(x), 4) for x in model.match_tau.detach()]
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"route {rec['route_loss']:.4f}  gen {rec['generic_loss']:.4f}  acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    if n_skipped:
        print(f"  WARNING: {n_skipped} of {steps} steps skipped for a non-finite loss", flush=True)
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0, "skipped_steps": n_skipped}


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, arm: str, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000018_{arm}{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": np.asarray(ck["centre"]), "history": ck["history"], "train_seconds": ck["train_seconds"],
                "checkpoint_sha256": _sha256(path)}
    out = train_arm(gk, seed, steps, ARMS[arm][1])
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict(), "arm": arm}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run. The first group is what this experiment is for; the rest
    exist so that silence on unrelated text cannot be bought by breaking everything else."""
    return {
        "no_key_no_injection": {"generic/kl_to_base": ("<=", 0.05), "broken1_unknown": (">=", 0.90)},
        "reading_not_traded_away": {"train/active_correct": (">=", 0.90), "heldout/active_correct": (">=", 0.70)},
        "refusal_not_traded_away": {"revoke_train_min": (">=", 0.95), "revoke_heldout_min": (">=", 0.85),
                                    "shred_heldout_min": (">=", 0.85)},
        "deleted_object_never_returns": {"heldout/revoked_deleted_object": ("<=", 0.02),
                                         "heldout/deleted_object_given_active_correct": ("<=", 0.02)},
    }


KEYS = ["generic/kl_to_base", "broken1_unknown", "train/active_correct", "heldout/active_correct",
        "train/refusal_given_active_correct", "heldout/refusal_given_active_correct",
        "heldout/revoked_deleted_object", "heldout/deleted_object_given_active_correct",
        "revoke_train_min", "revoke_heldout_min", "shred_train_min", "shred_heldout_min"]

BASELINE = {"generic/kl_to_base": 3.2741, "broken1_unknown": 0.7183, "train/active_correct": 0.9198,
            "heldout/active_correct": 0.7400, "revoke_heldout_min": 0.8983, "shred_heldout_min": 0.8983,
            "heldout/refusal_given_active_correct": 0.9928, "heldout/revoked_deleted_object": 0.0}


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    match_gate, generic_share = ARMS[args.arm]
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True, match_gate=match_gate))
        print(f"=== seed {seed}: arm {args.arm} (match gate {match_gate}, generic share {generic_share:g}) ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.arm, args.force)
        m = E17.evaluate_templates(gk, 1800 + seed, out["centre"], N_TRAIN_TEMPLATES)
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: round(v, 4) for k, v in m.items() if k in KEYS}, flush=True)
    agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    record = {
        "experiment": "E-000018", "arm": args.arm,
        "title": f"No key, no injection — arm '{args.arm}' (match gate {match_gate}, generic text share {generic_share:g})",
        "evidence_level": "E5", "deletion_level": "F3" if met["refusal_not_traded_away"] else "F1",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "problem": "The routing softmax always sums to one, so some cell always wins and the layer injects into text "
                   "it has no key for. E-000017-B measured 3.27 nats on generic sentences against a 0.05 bar, worse "
                   "than E-000013's 2.27 with fewer templates.",
        "by_construction": ["the match gate adds the CAPACITY to inject nothing (an absolute cosine threshold against "
                            "the best real cell key); whether the model uses it is learned from the losses",
                            "the generic arm trains the behaviour on eight sentence shapes that are disjoint from the "
                            "five the evaluation uses, so passing by memorising a shape is not available"],
        "baseline_e000017b": BASELINE,
        "config": {"seeds": args.seeds, "steps": args.steps, "arm": args.arm, "match_gate": match_gate,
                   "generic_share": generic_share, "n_train_templates": N_TRAIN_TEMPLATES,
                   "train_generic_prompts": TRAIN_GENERIC, "eval_generic_prompts": E17.GENERIC,
                   "adapter": AdapterConfig(status_gated=True, match_gate=match_gate).to_dict()},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}",
             f"{BASELINE[k]:.4f}" if k in BASELINE else "-") for k in KEYS if k in agg]
    md = "\n".join([
        f"# E-000018 — {record['title']}", "", record["problem"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed", "E-000017-B baseline"], rows), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "By construction: " + "; ".join(record["by_construction"]) + ".",
    ])
    path = ledger.save(f"e000018_{args.arm}", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
