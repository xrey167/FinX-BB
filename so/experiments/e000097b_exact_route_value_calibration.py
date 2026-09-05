"""E-000097B — frozen learned address, exact routes, value-side calibration.

Baseline only. Execute only after E97A qualifies all three teachers/immutable-row
identity probes. No novelty credit for curriculum, hard routing, or calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from so.data import bank_from_store, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.experiments import e000091_real_reader_lineage_density as E91
from so.experiments import e000092_exact_support_reader as E92
from so.experiments import e000097a_teacher_identity_probe as E97A
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at


def _teacher_probe(gk, centre: np.ndarray, seed: int, groups: int) -> Dict[str, Any]:
    rng = np.random.default_rng(97000 + seed)
    world, spec = E15.sample_alias_world(rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    store, _ = E15.load_arm(world, spec, centre, 97100 + seed, symlink=True)
    bank = bank_from_store(store)
    per = {str(t): E97A._probe_template(gk, bank, spec.alias_keys, world, t) for t in range(8, 12)}
    cap = all(per[str(t)]["candidate_correct"] >= .95 and per[str(t)]["full_vocab_top1_correct"] >= .95 for t in range(8, 12))
    ident = all(per[str(t)]["immutable_alias_row_argmax_correct"] >= .95 for t in range(8, 12))
    bypass = E97A._no_memory_bypass(gk)
    return {"per_template": per, "capability": bool(cap and bypass == 0.0), "identity": bool(ident), "bypass": bypass}


def _freeze_for_calibration(model) -> List[torch.nn.Parameter]:
    # Everything is frozen first. Only payload/output calibration may move.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: List[torch.nn.Parameter] = []
    modules = [model.v_proj, *model.o_proj.values()]
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    model.inject_gain.requires_grad_(True)
    trainable.append(model.inject_gain)
    # Keep semantic address, pointer representation/deref, status/marker validity,
    # and the frozen LM untouched. null_value stays frozen as part of UNKNOWN semantics.
    return trainable


def _calibrate(gk, seed: int, steps: int, groups: int, lr: float = 8e-4, batch_size: int = 32) -> Dict[str, Any]:
    model = gk.model
    params = _freeze_for_calibration(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=50)
    rng = np.random.default_rng(197000 + seed)
    history = []
    t0 = time.time()
    mix = {"fwd1": .7, "fwd2": .3}
    for step in range(steps):
        model.train()
        n_base = int(rng.integers(500, 701))
        world, spec = E15.sample_alias_world(rng, n_base, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
        bank = E15.bank_with_links(rng, world, spec, np.asarray(gk._e97b_centre), .20, .10, .05, .05)
        queries = sample_training_queries(rng, world, bank, batch_size, mix)
        ids, am, last = E8.encode_texts(gk.tok, [E17.query_text_pc(q, gk.names, E20.N_TRAIN_TEMPLATES) for q in queries])
        target = E8.targets_of(queries, bank, world)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, tcfg)
        cand, full, _routing, _hidden = model(bank.tensors(), ids, am, last)
        loss_c = F.cross_entropy(cand, target)
        full_target = model.candidate_ids[target]
        loss_f = F.cross_entropy(full, full_target)
        loss = loss_c + .25 * loss_f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 0 or (step + 1) % 100 == 0:
            rec = {"step": step + 1, "loss": float(loss.item()), "candidate_loss": float(loss_c.item()),
                   "full_loss": float(loss_f.item()), "batch_acc": float((cand.argmax(-1) == target).float().mean().item()),
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            print(rec, flush=True)
    model.eval()
    return {"history": history, "seconds": time.time() - t0}


def run(seed: int, teacher_steps: int, calibration_steps: int, groups: int) -> Dict[str, Any]:
    E91.install_strict_contract()
    os.environ["SO_BOS"] = "1"
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = E81.train_symlink_consistent(gk, seed, teacher_steps, consistency=.15, alt_supervision=.5,
                                           n_groups=max(100, groups), verbose=True)
    centre = np.asarray(trained["centre"])
    teacher = _teacher_probe(gk, centre, seed, max(100, groups))
    if not (teacher["capability"] and teacher["identity"]):
        return {"seed": seed, "teacher": teacher, "precondition": False, "calibrated": False,
                "breakthrough": False, "novelty_claim": False, "decision": "blocked_by_E97A_equivalent_gate"}

    # Exactify execution only after teacher qualification.
    E92.install_exact_support(gk.model)
    gk._e97b_centre = centre
    before = E92._eval_world(gk, centre, seed, max(100, groups))
    cal = _calibrate(gk, seed, calibration_steps, max(100, groups))
    after = E92._eval_world(gk, centre, seed, max(100, groups))
    bypass = E92._bypass(gk)
    locality = E91._lineage_intervention(gk, centre, seed, max(100, groups))
    route_same = bool(locality.get("routing_before_vs_after", {}).get("byte_identical"))
    state_same = bool(locality["hidden_before_vs_after"]["byte_identical"] and
                      locality["full_logits_before_vs_after"]["byte_identical"] and
                      (not locality["kv_before_vs_after"]["available"] or locality["kv_before_vs_after"]["byte_identical"]))
    drops = [teacher["per_template"][str(t)]["candidate_correct"] - after["per_template"][str(t)]["candidate_correct"] for t in range(8, 12)]
    feasible = bool(after["candidate_min"] >= .95 and after["full_vocab_min"] >= .95 and max(drops) <= .02 and bypass == 0.0 and route_same and state_same)
    return {"seed": seed, "teacher": teacher, "precondition": True, "exact_before_calibration": before,
            "calibration": cal, "exact_after_calibration": after, "max_candidate_drop_from_teacher": float(max(drops)),
            "exact_no_memory_bypass_maxabs": bypass, "unrelated_b_intervention": locality,
            "exact_route_and_neural_bystander_invariance": bool(route_same and state_same),
            "calibration_feasible": feasible, "breakthrough": False, "novelty_claim": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--teacher-steps", type=int, default=3000)
    ap.add_argument("--calibration-steps", type=int, default=1200)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", type=Path, default=Path("so/results/e000097b"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [run(s, args.teacher_steps, args.calibration_steps, args.groups) for s in args.seeds]
    rec = {"experiment": "E-000097B", "rows": rows,
           "all_three_preconditions": len(rows) >= 3 and all(r.get("precondition") for r in rows),
           "all_three_calibration_feasible": len(rows) >= 3 and all(r.get("calibration_feasible", False) for r in rows),
           "breakthrough": False, "novelty_claim": False}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000097b.json"
    out.write_text(json.dumps(rec, indent=2, allow_nan=False))
    print(json.dumps(rec, indent=2, allow_nan=False), flush=True)
    if not rec["all_three_calibration_feasible"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
