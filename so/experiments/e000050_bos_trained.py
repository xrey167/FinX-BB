"""Experiment E-000050 -- the held-out paraphrase gap is the position-0 token, and what a BOS buys.

THE FINDING THAT LED HERE. E-000039-A: 88.6% of the held-out paraphrase gap is addressing, and a text
prefix lifts held-out addressing to 0.98 with no weight changed. E-000039-B: training an InfoNCE tie on
the routing query across phrasings changes nothing on held-out phrasings (ledger 31.37). A direct probe
on the E-000017-B checkpoint (scratchpad, seed 0, 100 targets, twelve templates) then showed WHY: every
template that puts the subject at position 0 reads at 0.39-0.80 and addresses at 0.54-0.89; prepend
GPT-2's own <|endoftext|> -- a token the adapter never saw -- and those same templates read at 0.95-1.00
and address at 0.97-1.00, trained and held-out alike. GPT-2's tokenizer prepends no BOS
(`add_bos_token` is False), so a subject-initial prompt makes the SUBJECT the position-0 token, the
attention-sink position whose residual is dominated by a fixed direction, and the routing query reads
the sink instead of the subject. That is the whole of the "held-out failure" on subject-initial
phrasings. It is also not free: the same probe showed the subject-MEDIAL templates, whose first word had
been the sink, FALLING under a BOS at inference (t9 "Where {s} lives is": 0.96/0.97 -> 0.79/0.64), because
an adapter trained without a BOS has learned position-0 features that a prefix moves.

THE CLAIM UNDER TEST. Train the same adapter, same trainer, same budget, with a BOS on every prompt, and
the paraphrase gap of this addressable memory closes on both halves: held-out reading and addressing
reach the trained level without any tie, deletion propagates to phrasings the memory never saw, and
nothing is paid on the trained templates or on generic text.

ARMS (three seeds each; E-000017-B's trainer via E-000039's train_arm at tie weight 0):
  A  the recorded E-000017-B checkpoint, evaluated WITHOUT a BOS         (the record, re-evaluated in-process)
  B  the same checkpoint, BOS at inference only                          (the artefact reading: subject-initial
                                                                          templates recover, medial ones fall)
  C  a NEW checkpoint trained with a BOS, evaluated with a BOS            (the claim)
  D  that BOS-trained checkpoint, evaluated WITHOUT a BOS                (the reverse control: if position 0 is
                                                                          the cause, C's subject-initial forms
                                                                          must fail here as A's do)

WHAT COULD FALSIFY IT. C failing the trained-template bar (a BOS breaks the adapter); C fixing the
subject-initial half only (the medial half is semantic, not positional); B NOT recovering the
subject-initial held-out forms on A's own weights (then the probe was a fluke of one seed); D not
degrading (then position 0 was not the cause and the gain came from somewhere else). All four are
registered below.

Prior art, stated so nothing here is claimed as a mechanism: the position-0 anomaly is attention sinks
(Xiao et al., 2023) and massive activations (Sun et al., 2024); prepending a BOS to GPT-2 is standing
mechanistic-interpretability practice (TransformerLens prepend_bos). What is measured is what those
facts cost an addressable memory's paraphrase generalisation and its deletion propagation, and what a
BOS at training time recovers of it, against the recorded control.

Run:  SO_CKPT_SUFFIX= python -m so.experiments.e000050_bos_trained [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000039_address_tying as E39
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256, guard_recorded_checkpoint
from so.experiments.e000017_paraphrase_gap import evaluate_templates
from so.llm_adapter import AdapterConfig

N_TRAIN, N_T = E39.N_TRAIN, E39.N_T
ARMS = ("A", "B", "C", "D")


def _bos(on: bool) -> None:
    os.environ["SO_BOS"] = "1" if on else "0"


def _load(seed: int, path) -> tuple:
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    ck = torch.load(path, weights_only=False)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, np.asarray(ck["centre"]), _sha256(path)


def _train_bos(seed: int, steps: int, force: bool) -> tuple:
    """E-000017-B's trainer (E-000039's train_arm at tie weight 0) with a BOS on every prompt."""
    path = CHECKPOINTS / f"e000050_bos{CKPT_SUFFIX}_seed{seed}.pt"
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return gk, np.asarray(ck["centre"]), _sha256(path), float(ck["train_seconds"])
    _bos(True)
    out = E39.train_arm(gk, seed, steps, "address", tie_weight=0.0)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict(),
                "bos": True}, path)
    return gk, np.asarray(out["centre"]), _sha256(path), float(out["train_seconds"])


def _measure(gk, seed: int, centre: np.ndarray, bos: bool, oracle: bool, n_targets: int) -> Dict[str, float]:
    """E-000017's battery plus E-000039's decomposition, under one BOS setting."""
    _bos(bos)
    m = evaluate_templates(gk, 1700 + seed, centre, N_TRAIN)
    d = E39.decompose(gk, 1700 + seed, centre, n_targets, oracle)
    initial, medial = E39.subject_initial_templates(gk.tok)
    held = range(N_TRAIN, N_T)
    hi = [t for t in held if t in initial]
    hm = [t for t in held if t in medial]
    m["heldout/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in held)
    m["heldout_initial/read_min"] = min(d[f"t{t}/heldout/read"] for t in hi)
    m["heldout_initial/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in hi)
    m["heldout_medial/read_min"] = min(d[f"t{t}/heldout/read"] for t in hm)
    m["heldout_medial/route_hit_min"] = min(d[f"t{t}/heldout/route_hit"] for t in hm)
    m["train/read_min"] = min(d[f"t{t}/train/read"] for t in range(N_TRAIN))
    m["train/route_hit_min"] = min(d[f"t{t}/train/route_hit"] for t in range(N_TRAIN))
    for k, v in d.items():
        if ("/heldout/" in k or "/train/" in k or k.startswith("heldout/") or k.startswith("prefixed/")
                or k.startswith("query_cos") or k == "address_collision") and k not in m \
                and isinstance(v, (int, float, bool)):
            m[k] = float(v)
    _bos(False)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float, bool))}


def run_seed(seed: int, steps: int, n_targets: int, force: bool, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    out: Dict[str, Any] = {"seed": seed}
    ctrl = CHECKPOINTS / f"e000017_t8_c0{CKPT_SUFFIX}_seed{seed}.pt"
    gk, centre, sha = _load(seed, ctrl)
    out["control_sha256"] = sha
    for arm, bos in (("A", False), ("B", True)):
        m = _measure(gk, seed, centre, bos, oracle=False, n_targets=n_targets)
        out.update({f"{arm}/{k}": v for k, v in m.items()})
        if verbose:
            print(f"  seed {seed} arm {arm} (BOS={bos}, recorded weights): held-out read {m['heldout/active_correct']:.4f} "
                  f"initial {m['heldout_initial/read_min']:.2f}/{m['heldout_initial/route_hit_min']:.2f} "
                  f"medial {m['heldout_medial/read_min']:.2f}/{m['heldout_medial/route_hit_min']:.2f} "
                  f"train {m['train/active_correct']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    gk, centre, sha, train_s = _train_bos(seed, steps, force)
    out["bos_sha256"] = sha
    out["train_seconds"] = train_s
    for arm, bos in (("C", True), ("D", False)):
        m = _measure(gk, seed, centre, bos, oracle=(arm == "C"), n_targets=n_targets)
        out.update({f"{arm}/{k}": v for k, v in m.items()})
        if verbose:
            print(f"  seed {seed} arm {arm} (BOS={bos}, BOS-trained weights): held-out read {m['heldout/active_correct']:.4f} "
                  f"initial {m['heldout_initial/read_min']:.2f}/{m['heldout_initial/route_hit_min']:.2f} "
                  f"medial {m['heldout_medial/read_min']:.2f}/{m['heldout_medial/route_hit_min']:.2f} "
                  f"train {m['train/active_correct']:.4f} shred_heldout {m['shred_heldout_min']:.4f} "
                  f"generic KL {m['generic/kl_to_base']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    return out


CRITERIA = {
    # the record reproduced in-process, so every other row is against the same protocol
    "A/heldout/active_correct": ("<=", 0.80),
    # the artefact reading on the recorded weights: a BOS at inference recovers the subject-initial
    # held-out forms ... and is not free on the medial ones (both are predictions; both can fail)
    "B/heldout_initial/read_min": (">=", 0.90),
    "B/heldout_medial/read_min": ("<=", 0.90),
    # THE CLAIM: trained with a BOS, the gap closes on both halves at no price
    "C/heldout/active_correct": (">=", 0.95),
    "C/heldout/route_hit_min": (">=", 0.95),
    "C/heldout_initial/read_min": (">=", 0.95),
    "C/heldout_medial/read_min": (">=", 0.95),
    "C/train/active_correct": (">=", 0.95),
    "C/shred_heldout_min": (">=", 0.95),
    "C/revoke_heldout_min": (">=", 0.95),
    "C/heldout/revoked_deleted_object": ("<=", 0.02),
    "C/broken1_unknown": (">=", 0.63),
    "C/generic/kl_to_base": ("<=", 3.65),
    # the reverse control: without its BOS, the BOS-trained adapter must lose the subject-initial forms
    "D/heldout_initial/read_min": ("<=", 0.85),
}

DECISION_RULE = (
    "C passes every row -> the held-out paraphrase gap of this addressable memory was the position-0 "
    "token, the honest held-out numbers for the memory are C's, and every held-out number in the record "
    "(E-000017's kill criterion, E-000025's bimodality, E-000026's template choice, E-000039-B) is re-scoped "
    "as measured without a BOS. C passes the subject-initial rows and fails the medial ones -> the artefact "
    "is the subject-initial half only and the remainder is semantic. C fails the trained-template or "
    "generic rows -> a BOS at training time costs capability and the finding is B's alone. B fails its "
    "initial row -> the probe was one seed's fluke and nothing here is claimed. D not degrading -> position 0 "
    "was not the cause. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-targets", type=int, default=100)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    per_seed = [run_seed(s, args.steps, args.n_targets, args.force) for s in args.seeds]
    keys = sorted(k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float)) and k != "seed")
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    rows = []
    for arm in ARMS:
        rows.append([arm, {"A": "recorded, no BOS", "B": "recorded, BOS at inference", "C": "BOS-trained, BOS",
                           "D": "BOS-trained, no BOS"}[arm],
                     f"{agg[f'{arm}/heldout/active_correct']['min']:.4f}",
                     f"{agg[f'{arm}/heldout_initial/read_min']['min']:.2f} / {agg[f'{arm}/heldout_initial/route_hit_min']['min']:.2f}",
                     f"{agg[f'{arm}/heldout_medial/read_min']['min']:.2f} / {agg[f'{arm}/heldout_medial/route_hit_min']['min']:.2f}",
                     f"{agg[f'{arm}/train/active_correct']['min']:.4f}",
                     f"{agg[f'{arm}/shred_heldout_min']['min']:.4f}",
                     f"{agg[f'{arm}/generic/kl_to_base']['max']:.3f}"])
    tbl = ledger.table(["arm", "weights, prompt", "held-out reading", "held-out subject-initial read / route",
                        "held-out subject-medial read / route", "trained reading", "SHRED reaches worst held-out",
                        "generic KL (max)"], rows)
    record = {"experiment": "E-000050", "title": "the held-out paraphrase gap is the position-0 token, and what a BOS buys",
              "evidence_level": "E5", "seeds": args.seeds, "steps": args.steps, "n_targets": args.n_targets,
              "decision_rule": DECISION_RULE, "per_seed": per_seed, "aggregate": agg, "criteria": check,
              "control": "E-000017-B's recorded checkpoints (same trainer, same budget, no BOS), re-evaluated in-process"}
    md = [f"# E-000050 — {record['title']}", "",
          f"Seeds {args.seeds}, {args.steps} steps for the BOS-trained arm, {args.n_targets} targets per seed for the",
          "decomposition. GPT-2's tokenizer prepends no BOS, so a subject-initial prompt makes the subject the",
          "position-0 token. Worst seed everywhere.", "", tbl, "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    path = ledger.save("e000050_bos_trained", record, "\n".join(md))
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
