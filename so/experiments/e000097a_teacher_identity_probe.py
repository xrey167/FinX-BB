"""E-000097A — does a qualified dense teacher already separate immutable alias-row identity?

This is a capability/architecture probe only. It gives ZERO novelty credit to
argmax routing, distillation, hard retrieval, semantic addressing, or pointers.

The purpose is to avoid training an exact-support student if the retained dense
teacher's semantic address representation does not itself identify the immutable
alias row on held-out paraphrases.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000081_symlink_consistency_reader as E81
from so.llm_adapter import AdapterConfig


def _probe_template(gk, bank, alias_keys, world, template: int) -> Dict[str, float]:
    candidate_ok: List[bool] = []
    full_ok: List[bool] = []
    resolve_ok: List[bool] = []
    resolve_mass: List[float] = []
    resolve_margin: List[float] = []
    positive_rows: List[int] = []
    for key in alias_keys:
        s, r = key
        text = E17.TEMPLATES12[r][template].format(s=gk.names[s])
        ids, am, last = E8.encode_texts(gk.tok, [text])
        with torch.no_grad():
            cand, full, routing, _hidden = gk.model(bank.tensors(), ids, am, last)
        truth_entity = int(world.index[key])
        candidate_ok.append(int(cand.argmax(-1)[0]) == truth_entity)
        full_ok.append(int(full.argmax(-1)[0]) == int(gk.entity_ids[truth_entity]))
        if routing is None:
            raise RuntimeError("teacher returned no routing")
        # For one-hop queries with two reads and one dereference, the final read's
        # resolve slot is -2. This slot should identify the immutable alias row;
        # the final dereference slot (-1) identifies its current target.
        p = routing[0, -2]
        real = p[:-1]
        expected = int(bank.routable_pos[key])
        pred = int(p.argmax().item())
        resolve_ok.append(pred == expected)
        resolve_mass.append(float(p[expected].item()))
        top2 = torch.topk(p, k=min(2, p.numel())).values
        resolve_margin.append(float((top2[0] - top2[1]).item()) if top2.numel() > 1 else float(top2[0].item()))
        positive_rows.append(int((real > 0).sum().item()))
    return {
        "candidate_correct": float(np.mean(candidate_ok)),
        "full_vocab_top1_correct": float(np.mean(full_ok)),
        "immutable_alias_row_argmax_correct": float(np.mean(resolve_ok)),
        "immutable_alias_row_mean_mass": float(np.mean(resolve_mass)),
        "immutable_alias_row_min_mass": float(np.min(resolve_mass)),
        "immutable_alias_row_mean_top1_margin": float(np.mean(resolve_margin)),
        "positive_real_rows_mean": float(np.mean(positive_rows)),
        "positive_real_rows_min": int(np.min(positive_rows)),
        "positive_real_rows_max": int(np.max(positive_rows)),
    }


def _no_memory_bypass(gk) -> float:
    text = "The ordinary sentence has no registered mutable knowledge key."
    ids, am, last = E8.encode_texts(gk.tok, [text])
    with torch.no_grad():
        _c0, f0, _r0, _h0 = gk.model(None, ids, am, last)
        out = gk.model.lm(input_ids=ids, attention_mask=am)
        ar = torch.arange(ids.shape[0], device=ids.device)
        f1 = out.logits[ar, last]
    return float((f0 - f1).abs().max().item())


def run(seed: int, steps: int, groups: int, consistency: float, alt_supervision: float) -> Dict[str, Any]:
    torch.manual_seed(seed)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=E20.N_DEREF)
    gk = E8.GPT2Knowledge(cfg)
    trained = E81.train_symlink_consistent(
        gk, seed, steps,
        consistency=consistency,
        alt_supervision=alt_supervision,
        n_groups=max(24, groups),
        verbose=True,
    )
    centre = np.asarray(trained["centre"])
    rng = np.random.default_rng(97000 + seed)
    world, spec = E15.sample_alias_world(
        rng, 180, groups, 2, gk.n_entities, 4, E20.N_TRAIN_TEMPLATES
    )
    store, _kids = E15.load_arm(world, spec, centre, 97100 + seed, symlink=True)
    bank = bank_from_store(store)
    per_template = {
        str(t): _probe_template(gk, bank, spec.alias_keys, world, t)
        for t in range(E20.N_TRAIN_TEMPLATES, E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT)
    }
    teacher_gate = all(
        per_template[str(t)]["candidate_correct"] >= 0.95
        and per_template[str(t)]["full_vocab_top1_correct"] >= 0.95
        for t in range(8, 12)
    ) and _no_memory_bypass(gk) == 0.0
    identity_gate = all(
        per_template[str(t)]["immutable_alias_row_argmax_correct"] >= 0.95
        for t in range(8, 12)
    )
    return {
        "seed": seed,
        "steps": steps,
        "per_template": per_template,
        "teacher_capability_gate": bool(teacher_gate),
        "identity_separability_gate": bool(identity_gate),
        "exact_no_memory_bypass_maxabs": _no_memory_bypass(gk),
        "student_attacks_interpreted": False,
        "breakthrough": False,
        "novelty_claim": False,
        "decision": (
            "eligible_for_identity_compiler_stage_B" if teacher_gate and identity_gate
            else "do_not_distill_from_this_seed"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--groups", type=int, default=100)
    ap.add_argument("--consistency", type=float, default=0.15)
    ap.add_argument("--alt-supervision", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", type=Path, default=Path("so/results/e000097a"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = [run(s, args.steps, args.groups, args.consistency, args.alt_supervision) for s in args.seeds]
    result = {
        "experiment": "E-000097A",
        "title": "qualified dense teacher immutable alias-row identity separability",
        "rows": rows,
        "all_three_teacher_qualified": len(rows) >= 3 and all(r["teacher_capability_gate"] for r in rows),
        "all_three_identity_separable": len(rows) >= 3 and all(r["identity_separability_gate"] for r in rows),
        "breakthrough": False,
        "novelty_claim": False,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / ("e000097a_" + "-".join(map(str, args.seeds)) + ".json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    # Nonzero only means the preregistered prerequisite did not hold.
    if not all(r["teacher_capability_gate"] and r["identity_separability_gate"] for r in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
