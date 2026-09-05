"""Experiment E-000058 -- ablate the dereference slot, so the pointer claim's attribution is measured.

WHY THIS EXISTS. §31.53's twelfth finding, raised by the completeness critic after three refuters had
finished: the surviving claim credits "a trained depth-1 dereference slot" with resolving the pointer,
but that slot is DIRECTLY SUPERVISED (``e000020_symlink_gpt2.py:50-53`` assigns the target cell to the
dereference slot and ``routing_loss`` trains it, exactly as ``loss_gate`` supervises the gate at
``:117-120``), and **no arm without it was ever run**. Crediting it was architectural attribution, not
measurement. This file is the missing arm, and it costs no training.

THE ABLATION, and why it is surgical. ``KnowledgeAdapterLM``'s read hook runs the dereference loop only
``if self.cfg.n_deref > 0`` (``so/llm_adapter.py:270-285``). Setting ``cfg.n_deref = 0`` on a trained
checkpoint at inference leaves every weight untouched and removes exactly one thing: the second query
built from the value just read. What then reaches the residual stream for an alias row is that row's
own payload, which for a LINK row is ``v_link`` of the TARGET'S KEY (``so/llm_adapter.py:220-228``) --
an address, not an object. Nothing else in the read changes, and the FACT rows are untouched, which is
what the control arm checks.

ARMS (both on the same recorded checkpoints, three seeds, twelve phrasings, nothing trained):
  DEREF     the checkpoint as trained, ``n_deref = 1`` -- reproduces E-000052's alias rows
  NODEREF   the same weights with ``cfg.n_deref = 0`` at inference

ROWS: alias reading (the claim's row) and direct reading (the control). The control is what makes the
ablation surgical rather than destructive: if direct reading falls too, the arm has broken the reader
and nothing about pointers can be read from it.

WHAT EACH OUTCOME MEANS, fixed before the run. The claim's attribution is MEASURED if alias reading
collapses under the ablation while direct reading survives. It is REFUTED if alias reading survives
without the slot, because then something other than the dereference resolves the pointer and the
sentence credits the wrong component. It is UNREADABLE if direct reading collapses too.

Prior art: ablating a component to attribute a behaviour to it is the oldest move in the book, and the
dereference slot itself is E-000015's design. Nothing here is claimed as new; this is the control the
claim should have carried on the day it was made.

Run:  SO_BOS=1 python -m so.experiments.e000058_deref_ablation [--seeds 0 1 2] [--threads 1]
      SO_BOS=1 python -m so.experiments.e000058_deref_ablation --seeds 0 --templates 0 8 --quick \
          --results-dir /path/to/scratch                                        (a smoke run)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000026_lifecycle_at_a_readable_template as E26

ARMS = ("DEREF", "NODEREF")


def world_and_store(gk, seed: int, centre: np.ndarray):
    rng = np.random.default_rng(4000 + seed)
    world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"],
                                         E20.EVAL["n_alias_per_group"], gk.n_entities, 4,
                                         E20.N_TRAIN_TEMPLATES)
    store, kids = E15.load_arm(world, spec, centre, seed, symlink=True)
    return world, spec, store, kids


def run_seed(seed: int, templates: List[int], threads: int, verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    gk, centre, sha = E26.load_link_adapter(seed)
    world, spec, store, _ = world_and_store(gk, seed, centre)
    bank = bank_from_store(store)
    alias_keys = list(spec.alias_keys)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    rng = np.random.default_rng(58_000 + seed)
    pick = rng.choice(len(base_keys), size=min(E20.EVAL["n_direct"], len(base_keys)), replace=False)
    direct_keys = [base_keys[int(i)] for i in pick]
    truth_direct = np.array([world.index[k] for k in direct_keys])
    truth_alias = np.array([world.index[spec.alias_of[k]] for k in alias_keys])
    m: Dict[str, Any] = {"seed": seed, "checkpoint_sha256": sha, "n_deref_trained": int(gk.model.cfg.n_deref),
                         "n_alias": len(alias_keys), "n_direct": len(direct_keys)}
    original = gk.model.cfg.n_deref
    for arm in ARMS:
        gk.model.cfg.n_deref = original if arm == "DEREF" else 0
        for t in templates:
            a = E20._answers(gk, bank, alias_keys, gk.names, template=t)[0]
            d = E20._answers(gk, bank, direct_keys, gk.names, template=t)[0]
            m[f"{arm}/t{t}/alias"] = float((a == truth_alias).mean())
            m[f"{arm}/t{t}/direct"] = float((d == truth_direct).mean())
        m[f"{arm}/alias_min"] = min(m[f"{arm}/t{t}/alias"] for t in templates)
        m[f"{arm}/alias_max"] = max(m[f"{arm}/t{t}/alias"] for t in templates)
        m[f"{arm}/direct_min"] = min(m[f"{arm}/t{t}/direct"] for t in templates)
    gk.model.cfg.n_deref = original
    m["alias_drop_min"] = min(m[f"DEREF/t{t}/alias"] - m[f"NODEREF/t{t}/alias"] for t in templates)
    m["direct_drop_max"] = max(m[f"DEREF/t{t}/direct"] - m[f"NODEREF/t{t}/direct"] for t in templates)
    m["seconds"] = time.time() - t0
    if verbose:
        print(f"  seed {seed}: alias {m['DEREF/alias_min']:.4f}-{m['DEREF/alias_max']:.4f} with the slot, "
              f"{m['NODEREF/alias_min']:.4f}-{m['NODEREF/alias_max']:.4f} without | direct "
              f"{m['DEREF/direct_min']:.4f} -> {m['NODEREF/direct_min']:.4f} | smallest alias drop "
              f"{m['alias_drop_min']:.4f}, largest direct drop {m['direct_drop_max']:.4f}  "
              f"({m['seconds']:.0f}s)", flush=True)
    return m


# Worst seed. Fixed before the run.
CRITERIA: Dict[str, Tuple[str, float]] = {
    "DEREF/alias_min": (">=", 0.80),        # V: the trained arm reproduces the battery's alias row
    "DEREF/direct_min": (">=", 0.90),       # V: and its direct row
    "direct_drop_max": ("<=", 0.05),        # THE CONTROL: the ablation must not break ordinary reading
    "alias_drop_min": (">=", 0.50),         # THE ROW: without the slot, the pointer is not resolved
}

DECISION_RULE = (
    "Worst seed over three, every template. UNREADABLE if the trained arm does not reproduce the "
    "battery (alias below 0.80 or direct below 0.90) or if the ablation costs direct reading more than "
    "0.05 at any template -- then the arm is destructive rather than surgical and nothing about "
    "pointers can be read from it. With both holding: MEASURED if alias reading falls by at least 0.50 "
    "at every template -- the dereference slot is what resolves the pointer, and the claim's "
    "attribution stops being architectural. REFUTED if alias reading survives the ablation (drop below "
    "0.50 at any template) -- something other than the dereference hop is resolving the alias, and the "
    "claim credits the wrong component and must say so. The magnitudes are recorded whatever the "
    "reading. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--templates", type=int, nargs="*", default=list(range(12)))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, args.threads))
    if args.quick:
        os.environ["SO_RESULT_SUFFIX"] = "-smoke"
    per = [run_seed(s, args.templates, args.threads) for s in args.seeds]
    keys = sorted(k for k in per[0] if isinstance(per[0][k], (int, float)) and k != "seed")
    agg = ledger.aggregate(per, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    rows = [[f"t{t}", f"{agg[f'DEREF/t{t}/alias']['min']:.4f}", f"{agg[f'NODEREF/t{t}/alias']['max']:.4f}",
             f"{agg[f'DEREF/t{t}/direct']['min']:.4f}", f"{agg[f'NODEREF/t{t}/direct']['min']:.4f}"]
            for t in args.templates]
    tbl = ledger.table(["template", "alias, slot on (worst seed)", "alias, slot off (worst seed)",
                        "direct, slot on", "direct, slot off"], rows)
    record = {"experiment": "E-000058", "title": "ablating the dereference slot: is it what resolves the pointer?",
              "evidence_level": "E5", "seeds": args.seeds, "templates": args.templates, "quick": args.quick,
              "trains_nothing": True, "decision_rule": DECISION_RULE, "per_seed": per, "aggregate": agg,
              "criteria": check,
              "control": "the same checkpoints as E-000052, read with cfg.n_deref set to 0 at inference; "
                         "no weight is changed and the FACT rows are untouched"}
    md = [f"# E-000058 — {record['title']}", "",
          "The control §31.53 found missing from the pointer claim: the dereference slot is directly",
          "supervised and had never been ablated. Same checkpoints, same world, nothing trained; the only",
          "change is `cfg.n_deref = 0` at inference. Worst seed.", "", tbl, "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    text = "\n".join(md)
    path = None
    if not args.quick:
        path = ledger.save("e000058_deref_ablation", record, text)
    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        name = "e000058_deref_ablation" + os.environ.get("SO_RESULT_SUFFIX", "")
        record.setdefault("environment", ledger.environment())
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(ledger._to_jsonable(record), f, indent=1, sort_keys=True)
        with open(os.path.join(args.results_dir, name + ".md"), "w") as f:
            f.write(text.rstrip("\n") + "\n")
        path = path or os.path.join(args.results_dir, name + ".md")
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
