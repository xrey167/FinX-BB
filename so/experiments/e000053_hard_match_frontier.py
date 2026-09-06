"""E-000053 — can a hard absolute match threshold close the E-000052 locality seam?

This is a diagnostic, not a claim. It trains the same BOS + match-gate + two-channel-null adapter as
E-000052 on a different seed, then freezes every learned parameter and sweeps a fixed hard-ish cosine
threshold at inference. The sweep uses one shared evaluation world, target set, broken-query set and
generic-text set for every threshold, so differences are attributable to the gate only.

Decision rule fixed before the run:
  * FEASIBLE if any pre-declared threshold simultaneously has train active accuracy >= .95,
    held-out active accuracy >= .95, worst held-out template >= .90, revoke UNKNOWN on every held-out
    template >= .95, deleted-object leakage after revoke <= .02, broken-key UNKNOWN >= .90, and
    generic KL to the frozen base <= .05 nats.
  * INFEASIBLE-ON-THIS-SEED if no threshold does. That falsifies "just harden the learned match gate"
    as the remaining fix on this trained adapter; it does not falsify other locality architectures.

A positive row is only a screening result: it must be repeated on three seeds with a threshold selected
without looking at held-out/generic evaluation data, then run through the existing deletion attacks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000018_no_key_no_injection as E18
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.train import make_centre
from so.world import UNKNOWN, Query, World, fill_random

THRESHOLDS = (-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
HARD_TEMP = 80.0
N_CELLS = 1000
N_TARGETS = 64
N_BROKEN = 64
N_GENERIC = 64
N_TRAIN = E18.N_TRAIN_TEMPLATES
N_TOTAL = N_TRAIN + E17.N_HELDOUT


def _read_answers(gk, bank, texts: List[str], batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        ids, am, last = E8.encode_texts(gk.tok, texts[i:i + batch])
        cand, _, _, _ = gk.model(bank.tensors(), ids, am, last)
        a = cand.argmax(-1).numpy()
        out.append(np.where(a == gk.n_entities, UNKNOWN, a))
    return np.concatenate(out)


@torch.no_grad()
def measure(gk, centre: np.ndarray, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(9300 + seed)
    world = fill_random(rng, World(gk.n_entities, 4, N_TRAIN, []), N_CELLS)
    store = MVCCStore(marker_dim=centre.shape[0], seed=9300 + seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = world.facts
    idx = rng.choice(len(facts), size=min(N_TARGETS, len(facts)), replace=False)
    targets = [facts[int(i)] for i in idx]
    truth = np.asarray([f.obj for f in targets])
    qs = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]

    def texts_for(t: int) -> List[str]:
        return [E17.TEMPLATES12[q.path[0]][t].format(s=gk.names[q.start]) for q in qs]

    bank = bank_from_store(store)
    active = {}
    for t in range(N_TOTAL):
        active[t] = float((_read_answers(gk, bank, texts_for(t)) == truth).mean())

    for f in targets:
        store.revoke(kids[f.key])
    rbank = bank_from_store(store)
    revoke_unknown = {}
    revoke_deleted = {}
    for t in range(N_TRAIN, N_TOTAL):
        ans = _read_answers(gk, rbank, texts_for(t))
        revoke_unknown[t] = float((ans == UNKNOWN).mean())
        revoke_deleted[t] = float((ans == truth).mean())
    for f in targets:
        store.restore(kids[f.key])
    bank = bank_from_store(store)

    broken = world.sample_queries(rng, N_BROKEN, 1, "fwd", require_answer=False)
    btexts = [E17.query_text_pc(q, gk.names, N_TRAIN) for q in broken]
    broken_unknown = float((_read_answers(gk, bank, btexts) == UNKNOWN).mean())

    gen = [E17.GENERIC[int(rng.integers(0, len(E17.GENERIC)))].format(
        s=gk.names[int(rng.integers(0, gk.n_entities))]) for _ in range(N_GENERIC)]
    kls = []
    for i in range(0, len(gen), 64):
        ids, am, last = E8.encode_texts(gk.tok, gen[i:i + 64])
        _, full_b, _, _ = gk.model(None, ids, am, last)
        _, full_a, _, _ = gk.model(bank.tensors(), ids, am, last)
        lb, la = torch.log_softmax(full_b, -1), torch.log_softmax(full_a, -1)
        kls.append((lb.exp() * (lb - la)).sum(-1).numpy())
    generic_kl = float(np.concatenate(kls).mean())

    held = list(range(N_TRAIN, N_TOTAL))
    train = list(range(N_TRAIN))
    return {
        "train_active_mean": float(np.mean([active[t] for t in train])),
        "heldout_active_mean": float(np.mean([active[t] for t in held])),
        "heldout_active_min": float(min(active[t] for t in held)),
        "revoke_heldout_min": float(min(revoke_unknown.values())),
        "revoked_deleted_object": float(np.mean(list(revoke_deleted.values()))),
        "broken_unknown": broken_unknown,
        "generic_kl": generic_kl,
    }


def passes(m: Dict[str, float]) -> bool:
    return (
        m["train_active_mean"] >= 0.95
        and m["heldout_active_mean"] >= 0.95
        and m["heldout_active_min"] >= 0.90
        and m["revoke_heldout_min"] >= 0.95
        and m["revoked_deleted_object"] <= 0.02
        and m["broken_unknown"] >= 0.90
        and m["generic_kl"] <= 0.05
    )


def run(seed: int, steps: int, threads: int, outdir: str) -> Dict:
    if threads:
        torch.set_num_threads(threads)
    os.environ["SO_BOS"] = "1"
    cfg = AdapterConfig(status_gated=True, match_gate=True, two_channel_null=True)
    gk = E8.GPT2Knowledge(cfg)
    t0 = time.time()
    trained = E18.train_arm(gk, seed, steps, generic_share=0.25)
    model = gk.model
    learned_tau = [float(x) for x in model.match_tau.detach()]
    learned_temp = [float(x) for x in model.match_temp.detach().abs()]

    rows = []
    for tau in THRESHOLDS:
        with torch.no_grad():
            model.match_tau.fill_(float(tau))
            model.match_temp.fill_(HARD_TEMP)
        m = measure(gk, np.asarray(trained["centre"]), seed)
        row = {"tau": tau, **m, "pass": passes(m)}
        rows.append(row)
        print(row, flush=True)

    feasible = [r for r in rows if r["pass"]]
    rec = {
        "experiment": "E-000053",
        "diagnostic_only": True,
        "seed": seed,
        "steps": steps,
        "bos": True,
        "adapter": cfg.to_dict(),
        "learned_tau_before_sweep": learned_tau,
        "learned_temp_before_sweep": learned_temp,
        "thresholds": list(THRESHOLDS),
        "hard_temp": HARD_TEMP,
        "rows": rows,
        "feasible": bool(feasible),
        "feasible_thresholds": [r["tau"] for r in feasible],
        "seconds": time.time() - t0,
    }
    p = Path(outdir); p.mkdir(parents=True, exist_ok=True)
    (p / "e000053.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    lines = [
        "# E-000053 — hard-match locality frontier", "",
        f"seed={seed}, steps={steps}, learned tau={learned_tau}, learned temp={learned_temp}", "",
        "| tau | train | held mean | held min | revoke min | deleted leak | broken UNKNOWN | generic KL | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tau']:.2f} | {r['train_active_mean']:.3f} | {r['heldout_active_mean']:.3f} | "
            f"{r['heldout_active_min']:.3f} | {r['revoke_heldout_min']:.3f} | {r['revoked_deleted_object']:.3f} | "
            f"{r['broken_unknown']:.3f} | {r['generic_kl']:.4f} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += ["", f"**feasible: {bool(feasible)}; thresholds: {[r['tau'] for r in feasible]}**"]
    (p / "e000053.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--results-dir", default="ci-e53")
    a = ap.parse_args()
    run(a.seed, a.steps, a.threads, a.results_dir)
