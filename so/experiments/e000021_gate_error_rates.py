"""Experiment E-000021 — the verification gate as a classifier: how often does it accept an unsigned marker?

The standing audit's sharpest surviving objection to the F4 label is that the deletion certificate is
a learned binary classifier with a measured false-accept rate, and that the rate is reported only as
an anecdote: "the hard gate admitted an unsigned marker in 1 of 5 seeds of E-000010 and in 2 of 3
seeds of E-000014". A worst-seed maximum says that at least one admission happened; it does not say
how often, and every deletion claim in this programme rests on how often.

This measures it directly. The gate is a small network over the marker vector alone, so it can be
evaluated on as many fresh markers as we like without touching the rest of the model. Every recorded
checkpoint of the verified-gate family is loaded and scored on markers drawn from the same
distributions the training and evaluation used:

  false accept   an UNSIGNED marker scoring above the 0.5 threshold — a shredded payload that would
                 be read out
  false reject   a SIGNED marker scoring at or below it — a live cell that would go silent

Both are reported pooled with exact binomial intervals, and per checkpoint family, so the tail is
visible rather than summarised by a maximum. Nothing is trained here.

Run:  python -m so.experiments.e000021_gate_error_rates
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from so import ledger
from so.data import invalid_markers, valid_markers
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256, checkpoint_path
from so.model import ModelConfig, MutableKnowledgeTransformer

FAMILIES = {"e000010": [0, 1, 2, 3, 4], "e000014": [0, 1, 2], "e000019": [5, 6, 7]}
N_PER_CLASS = 200_000          # markers per class per checkpoint
BATCH = 50_000
# Pre-registered before the first run. A gate that admits one unsigned marker in a thousand would
# leak roughly one shredded payload per thousand deletions, which is not a deletion guarantee; the
# bar is set an order of magnitude below the residual any recorded attack could detect.
MAX_FALSE_ACCEPT, MAX_FALSE_REJECT = 1e-3, 1e-3


def gate_scores(model: MutableKnowledgeTransformer, markers: np.ndarray, batch: int = BATCH) -> np.ndarray:
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, markers.shape[0], batch):
            m = torch.as_tensor(markers[i: i + batch], dtype=torch.float32)
            out.append(torch.sigmoid(model.gate_logits(m)).squeeze(-1).numpy())
    return np.concatenate(out)


def measure(name: str, seed: int, n_per_class: int) -> Dict[str, Any]:
    path = checkpoint_path(name, seed)
    ck = torch.load(path, weights_only=False)
    model = MutableKnowledgeTransformer(ModelConfig(**ck["model_config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    centre = np.asarray(ck["centre"], dtype=float)
    rng = np.random.default_rng(70_000 + seed)
    sv = gate_scores(model, valid_markers(rng, centre, n_per_class))
    si = gate_scores(model, invalid_markers(rng, centre, n_per_class))
    return {"family": name, "seed": seed, "n_per_class": n_per_class,
            "false_accepts": int((si > 0.5).sum()), "false_rejects": int((sv <= 0.5).sum()),
            "unsigned_max": float(si.max()), "unsigned_p999": float(np.quantile(si, 0.999)),
            "unsigned_mean": float(si.mean()), "signed_min": float(sv.min()),
            "signed_p001": float(np.quantile(sv, 0.001)), "signed_mean": float(sv.mean()),
            "checkpoint_sha256": _sha256(path)}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_CLASS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    rows: List[Dict[str, Any]] = []
    for name, seeds in FAMILIES.items():
        for seed in seeds:
            if not checkpoint_path(name, seed).exists():
                print(f"skip {name} seed {seed}: no checkpoint", flush=True)
                continue
            r = measure(name, seed, args.n)
            rows.append(r)
            print(f"{name} seed {seed}: false accepts {r['false_accepts']}/{args.n}, "
                  f"false rejects {r['false_rejects']}/{args.n}, unsigned max {r['unsigned_max']:.4f}", flush=True)
    n_total = sum(r["n_per_class"] for r in rows)
    fa, fr = sum(r["false_accepts"] for r in rows), sum(r["false_rejects"] for r in rows)
    ci_fa, ci_fr = ledger.clopper_pearson(fa, n_total), ledger.clopper_pearson(fr, n_total)
    per_family: Dict[str, Dict[str, Any]] = {}
    for name in FAMILIES:
        sub = [r for r in rows if r["family"] == name]
        if not sub:
            continue
        n = sum(r["n_per_class"] for r in sub)
        a, j = sum(r["false_accepts"] for r in sub), sum(r["false_rejects"] for r in sub)
        per_family[name] = {"n_per_class_total": n, "false_accepts": a, "false_rejects": j,
                            "false_accept_rate": a / n, "false_reject_rate": j / n,
                            "ci_false_accept": ledger.clopper_pearson(a, n),
                            "unsigned_max": max(r["unsigned_max"] for r in sub),
                            "signed_min": min(r["signed_min"] for r in sub)}
    flat = {"false_accept_rate": {"mean": fa / n_total, "min": fa / n_total, "max": fa / n_total},
            "false_reject_rate": {"mean": fr / n_total, "min": fr / n_total, "max": fr / n_total},
            "false_accept_ci_upper": {"mean": ci_fa["upper"], "min": ci_fa["upper"], "max": ci_fa["upper"]}}
    check = ledger.check_criteria(flat, {"false_accept_rate": ("<=", MAX_FALSE_ACCEPT),
                                         "false_reject_rate": ("<=", MAX_FALSE_REJECT),
                                         "false_accept_ci_upper": ("<=", 10 * MAX_FALSE_ACCEPT)})
    record = {
        "experiment": "E-000021",
        "title": "The verification gate as a classifier: false accepts and false rejects over fresh markers",
        "evidence_level": "E4", "deletion_level": None,
        "no_training": "Nothing was trained. Every recorded checkpoint of the verified-gate family is loaded and its "
                       "gate scored on freshly drawn markers; the rest of the model is not involved, because the gate "
                       "is a function of the marker alone.",
        "answers": "The standing audit's objection that the deletion certificate is a learned classifier whose "
                   "false-accept rate is reported only as a worst-seed maximum.",
        "totals": {"checkpoints": len(rows), "markers_per_class": n_total, "false_accepts": fa, "false_rejects": fr,
                   "false_accept_rate": fa / n_total, "false_reject_rate": fr / n_total,
                   "ci_false_accept": ci_fa, "ci_false_reject": ci_fr},
        "per_family": per_family, "per_checkpoint": rows,
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "interpretation_limit": "This is the gate's error rate on markers drawn from the same two distributions the "
                                "programme uses. It is not a security claim: an adversary who can choose the marker is "
                                "not modelled here, and a gate that separates two fixed distributions says nothing "
                                "about one that must resist a search for a passing vector.",
    }
    frows = [(k, f"{v['false_accepts']}/{v['n_per_class_total']}", f"{v['false_accept_rate']:.2e}",
              f"[{v['ci_false_accept']['lower']:.2e}, {v['ci_false_accept']['upper']:.2e}]",
              f"{v['false_rejects']}/{v['n_per_class_total']}", f"{v['unsigned_max']:.4f}", f"{v['signed_min']:.4f}")
             for k, v in per_family.items()]
    md = "\n".join([
        "# E-000021 — The verification gate as a classifier", "", record["answers"], "", record["no_training"], "",
        f"**Pooled over {len(rows)} checkpoints and {n_total:,} markers per class: {fa} false accepts "
        f"(rate {fa / n_total:.2e}, 95% interval [{ci_fa['lower']:.2e}, {ci_fa['upper']:.2e}]) and {fr} false "
        f"rejects (rate {fr / n_total:.2e}).**", "",
        ledger.table(["family", "false accepts", "rate", "95% interval", "false rejects",
                      "max score on an unsigned marker", "min score on a signed marker"], frows), "",
        "Pre-registered criteria:", "", ledger.criteria_table(check), "",
        record["interpretation_limit"],
    ])
    path = ledger.save("e000021_gate_error_rates", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
