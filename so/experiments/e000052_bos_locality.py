"""E-000052 — combine E-000050's BOS training with E-000022's locality architecture.

Goal: test the remaining breakthrough seam directly: strong held-out paraphrase reading + strong deletion
propagation + near-zero unrelated-text injection, in one frozen-GPT2 adapter.

This is a candidate experiment, not a claim. A single-seed pilot is only a screening run; a positive pilot
must be repeated on three seeds and then attacked with the existing deletion battery.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig


def run(seed: int, steps: int, threads: int, outdir: str):
    if threads:
        torch.set_num_threads(threads)
    os.environ["SO_BOS"] = "1"
    cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    gk = E8.GPT2Knowledge(cfg)
    t0 = time.time()
    trained = E18.train_arm(gk, seed, steps, generic_share=0.25)
    m = E17.evaluate_templates(gk, 5200 + seed, trained["centre"], E18.N_TRAIN_TEMPLATES)
    m = {k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))}
    m["seed"] = seed
    m["steps"] = steps
    m["seconds"] = time.time() - t0
    # Hard screening bars. These are deliberately stronger than E-000022 and use E-000050's gains.
    criteria = {
        "train/active_correct": (">=", 0.95),
        "heldout/active_correct": (">=", 0.95),
        "revoke_train_min": (">=", 0.95),
        "revoke_heldout_min": (">=", 0.95),
        "shred_heldout_min": (">=", 0.95),
        "heldout/revoked_deleted_object": ("<=", 0.02),
        "broken1_unknown": (">=", 0.90),
        "generic/kl_to_base": ("<=", 0.05),
    }
    checks = {}
    for k, (op, bar) in criteria.items():
        v = m.get(k, float("nan"))
        ok = (v >= bar) if op == ">=" else (v <= bar)
        checks[k] = {"value": v, "op": op, "bar": bar, "pass": bool(ok)}
    screening_pass = all(x["pass"] for x in checks.values())
    rec = {"experiment":"E-000052", "candidate_only": True, "adapter": cfg.to_dict(),
           "bos": True, "generic_share":0.25, "metrics":m, "criteria":checks,
           "screening_pass":screening_pass}
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / "e000052.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    lines = ["# E-000052 — BOS + locality candidate", "", f"seed={seed}, steps={steps}", "",
             "| criterion | observed | required | result |", "|---|---:|---:|---|"]
    for k, c in checks.items():
        lines.append(f"| {k} | {c['value']:.4f} | {c['op']} {c['bar']} | {'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", f"**screening_pass: {screening_pass}**"]
    (p / "e000052.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="ci-e52")
    a = ap.parse_args()
    run(a.seed, a.steps, a.threads, a.results_dir)
