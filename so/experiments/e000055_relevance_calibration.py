"""E-000055 — directly supervise the question-vs-prose channel that E-000053 isolated.

E-000053/full + BOS already puts held-out reading and deletion in the target region, while the two
remaining failures are generic-text locality and missing-key UNKNOWN.  In the two-channel architecture
both are controlled by query_relevance, but E-000018 only supervises that head indirectly through the
final KL / answer losses.  This experiment leaves the trained memory, key geometry and payloads frozen
and calibrates ONLY query_relevance on an explicit binary task: real/broken questions = 1, unrelated
prose = 0.  It then measures the joint breakthrough bars before and after calibration, and under a
pre-declared inference hardening sweep.  Diagnostic only: a positive cell must be repeated with a
threshold selected without test data, then attacked.
"""
from __future__ import annotations

import argparse, copy, json, math, os, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from so.data import bank_from_world
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig
from so.train import make_centre
from so.world import World

BARS = {
    "train/active_correct": (">=", .95),
    "heldout/active_correct": (">=", .95),
    "revoke_train_min": (">=", .95),
    "revoke_heldout_min": (">=", .95),
    "shred_heldout_min": (">=", .95),
    "heldout/revoked_deleted_object": ("<=", .02),
    "broken1_unknown": (">=", .90),
    "generic/kl_to_base": ("<=", .05),
}
REL_THRESHOLDS = (0.30, 0.50, 0.70)
HARD_GAIN = 80.0


def checks(m: Dict[str, float]) -> Dict:
    out = {}
    for k, (op, b) in BARS.items():
        v = float(m.get(k, float("nan")))
        ok = v >= b if op == ">=" else v <= b
        out[k] = {"value": v, "op": op, "bar": b, "pass": bool(ok)}
    return out


def evaluate(gk, centre: np.ndarray, seed: int) -> Dict[str, float]:
    m = E17.evaluate_templates(gk, 7500 + seed, centre, E18.N_TRAIN_TEMPLATES)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))}


def calibrate_relevance(gk, seed: int, steps: int, batch_q: int = 24, batch_g: int = 24,
                        lr: float = 1e-3) -> List[Dict]:
    """Freeze everything except query_relevance and give that channel its own explicit labels."""
    model = gk.model
    for p in model.adapter_parameters():
        p.requires_grad_(False)
    params = []
    for head in model.query_relevance.values():
        for p in head.parameters():
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(15500 + seed)
    centre = make_centre(seed, model.cfg.marker_dim)
    captured: List[torch.Tensor] = []
    hooks = []
    for head in model.query_relevance.values():
        hooks.append(head.register_forward_hook(lambda mod, inp, out: captured.append(out)))
    hist = []
    try:
        for step in range(steps):
            model.train(); captured.clear()
            world = World.sample(rng, gk.n_entities, 4, int(rng.integers(700, 1001)), E18.N_TRAIN_TEMPLATES)
            bank = bank_from_world(rng, world, centre, 0.20, 0.10, 0.05)
            # Half answerable, half deliberately missing/broken: both are QUESTIONS and must keep relevance=1.
            q1 = world.sample_queries(rng, batch_q // 2, 1, "fwd", require_answer=True, index=bank.index_view)
            q0 = world.sample_queries(rng, batch_q - len(q1), 1, "fwd", require_answer=False, index=bank.index_view)
            qs = q1 + q0
            qtexts = [E17.query_text_pc(q, gk.names, E18.N_TRAIN_TEMPLATES) for q in qs]
            gtexts = [E18.TRAIN_GENERIC[int(rng.integers(0, len(E18.TRAIN_GENERIC)))].format(
                s=gk.names[int(rng.integers(0, gk.n_entities))]) for _ in range(batch_g)]
            ids, am, last = E8.encode_texts(gk.tok, qtexts + gtexts)
            model(bank.tensors(), ids, am, last)
            if len(captured) != len(model.cfg.read_layers):
                raise RuntimeError(f"captured {len(captured)} relevance heads, expected {len(model.cfg.read_layers)}")
            loss = captured[0].sum() * 0
            accs = []
            labels = torch.cat([torch.ones(len(qtexts)), torch.zeros(len(gtexts))])
            for z in captured:
                zz = z.squeeze(-1)
                loss = loss + F.binary_cross_entropy_with_logits(zz, labels)
                accs.append(float(((zz > 0) == labels.bool()).float().mean()))
            loss = loss / len(captured)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step == 0 or (step + 1) % 50 == 0:
                rec = {"step": step + 1, "loss": float(loss), "binary_acc": float(np.mean(accs))}
                hist.append(rec); print("  rel-cal", rec, flush=True)
    finally:
        for h in hooks: h.remove()
    model.eval()
    return hist


def harden(model, rel_prob_tau: float) -> None:
    """Preserve the learned match threshold; sharpen match and calibrated relevance decisions only."""
    with torch.no_grad():
        model.match_temp.fill_(HARD_GAIN)
        logit_tau = math.log(rel_prob_tau / (1.0 - rel_prob_tau))
        for head in model.query_relevance.values():
            last = head[-1]
            w = last.weight.detach().clone(); b = last.bias.detach().clone()
            last.weight.copy_(HARD_GAIN * w)
            last.bias.copy_(HARD_GAIN * (b - logit_tau))


def run(seed: int, train_steps: int, cal_steps: int, threads: int, outdir: str) -> Dict:
    if threads: torch.set_num_threads(threads)
    os.environ["SO_BOS"] = "1"
    cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    gk = E8.GPT2Knowledge(cfg); t0 = time.time()
    trained = E18.train_arm(gk, seed, train_steps, generic_share=0.25)
    centre = np.asarray(trained["centre"])
    base = evaluate(gk, centre, seed); base_c = checks(base)
    hist = calibrate_relevance(gk, seed, cal_steps)
    soft = evaluate(gk, centre, seed); soft_c = checks(soft)
    calibrated_state = copy.deepcopy(gk.model.state_dict())
    rows = []
    for rt in REL_THRESHOLDS:
        gk.model.load_state_dict(calibrated_state)
        harden(gk.model, rt)
        m = evaluate(gk, centre, seed); c = checks(m)
        row = {"rel_prob_tau": rt, "metrics": m, "criteria": c,
               "screening_pass": all(x["pass"] for x in c.values())}
        rows.append(row)
        print({"rel_prob_tau": rt, "screening_pass": row["screening_pass"],
               **{k: round(c[k]["value"], 4) for k in BARS}}, flush=True)
    rec = {"experiment": "E-000055", "candidate_only": True, "seed": seed,
           "train_steps": train_steps, "cal_steps": cal_steps, "adapter": cfg.to_dict(),
           "base": {"metrics": base, "criteria": base_c},
           "calibrated_soft": {"metrics": soft, "criteria": soft_c},
           "hard_rows": rows, "calibration_history": hist,
           "any_screening_pass": any(r["screening_pass"] for r in rows),
           "seconds": time.time() - t0}
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / f"e000055-seed{seed}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-steps", type=int, default=1200)
    ap.add_argument("--cal-steps", type=int, default=250)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="ci-e55")
    a = ap.parse_args(); run(a.seed, a.train_steps, a.cal_steps, a.threads, a.results_dir)
