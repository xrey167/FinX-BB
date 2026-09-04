"""Experiment E-000016 — alias chains: does the mechanism extend past one indirection?

E-000015 recorded two failures. The pre-registered two-slot control did not resolve a chain of two
aliases, and the reason was in the training distribution rather than in the architecture: aliases
there always pointed at fact cells, so the second dereference slot never saw a pointer in its input
and learned to pass through. Shredding an alias (rather than the payload it points at) also reached
only 93% on the worst seed.

This experiment puts chains INTO the training distribution (30% of aliases point at another alias)
and trains two arms that differ only in how many dereference slots they have:

  * two slots  — must resolve a two-link chain;
  * one slot   — must REFUSE it (answer unknown), not invent an answer.

The one-slot arm is what makes the claim falsifiable: if it also answers chains correctly, the
slot account of the mechanism is wrong and E-000015's provenance story has to be re-examined.

Run:  python -m so.experiments.e000016_alias_chains [--seeds 0 1 2] [--steps 4000]
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments import e000015_symlink_cells as E15
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.model import ModelConfig, MutableKnowledgeTransformer

P_CHAIN = 0.30


def train_or_load(seed: int, steps: int, n_deref: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000016_deref{n_deref}{CKPT_SUFFIX}_seed{seed}.pt"
    cfg_m, cfg_t = E15.model_config(n_deref), E15.train_config(seed, steps)
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        model.eval()
        return {"model": model, "centre": np.asarray(ck["centre"]), "history": ck["history"],
                "train_seconds": ck["train_seconds"], "checkpoint_sha256": _sha256(path)}
    out = E15.train_symlink(cfg_m, cfg_t, p_chain=P_CHAIN)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out["model"].state_dict(), "model_cfg": cfg_m.to_dict(), "train_cfg": cfg_t.to_dict(),
                "centre": out["centre"], "history": out["history"], "train_seconds": out["train_seconds"],
                "p_chain": P_CHAIN}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run. ``two_slots`` carries the claim, ``one_slot_refuses``
    makes it falsifiable, and the rest guards against buying chains with a regression."""
    return {
        "two_slots_resolve_a_chain": {"two/chain2/answer_acc": (">=", 0.90)},
        "one_slot_refuses_a_chain": {"one/chain2/answer_acc": ("<=", 0.20), "one/chain2/unknown": (">=", 0.90)},
        "no_price_paid_elsewhere": {"two/direct": (">=", 0.98), "two/alias_direct": (">=", 0.95),
                                    "two/hop2": (">=", 0.95), "two/regression/direct": (">=", 0.98),
                                    "two/regression/hop2": (">=", 0.95), "two/regression/reverse": (">=", 0.95),
                                    "two/alias_provenance_pair": (">=", 0.90)},
        "sharing_still_holds": {"two/shared_update/alias_new_object": (">=", 0.95),
                                "two/duplicate_update/alias_new_object": ("<=", 0.05),
                                "two/shred_target/alias_unknown": (">=", 0.95),
                                "two/shred_target/alias_probe_top1": ("<=", 0.05),
                                "two/dup_shred/copy_direct_acc": (">=", 0.95)},
        "shredding_a_pointer": {"two/shred_alias/alias_unknown": (">=", 0.95),
                                "one/shred_alias/alias_unknown": (">=", 0.95)},
    }


KEYS = ["two/direct", "two/alias_direct", "two/chain2/answer_acc", "two/chain2/unknown", "two/chain2/depth1_acc",
        "one/direct", "one/alias_direct", "one/chain2/answer_acc", "one/chain2/unknown", "one/chain2/depth1_acc",
        "two/alias_provenance_pair", "two/hop2", "two/shared_update/alias_new_object",
        "two/duplicate_update/alias_new_object", "two/shred_target/alias_unknown", "two/shred_target/alias_probe_top1",
        "two/dup_shred/copy_direct_acc", "two/dup_shred/copy_probe_top1", "two/shred_alias/alias_unknown",
        "one/shred_alias/alias_unknown", "two/delete_target/alias_unknown", "two/deref_disabled/alias_direct",
        "two/regression/direct", "two/regression/hop2", "two/regression/hop3", "two/regression/reverse"]


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        row: Dict[str, Any] = {"seed": seed}
        for tag, n_deref in (("two", 2), ("one", 1)):
            print(f"=== seed {seed}: {n_deref} dereference slot(s), {P_CHAIN:.0%} chains in training ===", flush=True)
            out = train_or_load(seed, args.steps, n_deref, args.force)
            mm = E15.evaluate(out["model"], 1600 + seed, out["centre"])
            for k, v in mm.items():
                if k != "seed":
                    row[f"{tag}/{k}"] = v
            row[f"{tag}/checkpoint_sha256"] = out["checkpoint_sha256"]
            row[f"{tag}/train_seconds"] = out["train_seconds"]
            print(f"  chain2 {mm['chain2/answer_acc']:.3f}  unknown {mm['chain2/unknown']:.3f}  "
                  f"alias {mm['alias_direct']:.3f}  direct {mm['direct']:.3f}", flush=True)
        per_seed.append(row)
    keys = [k for k in per_seed[0] if k != "seed" and not k.endswith("checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    record = {
        "experiment": "E-000016",
        "title": "Alias chains: two dereference slots resolve a two-link chain, one slot must refuse it",
        "evidence_level": "E4",
        "deletion_level": "F3" if met["sharing_still_holds"] else "F1",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "follows_from": "E-000015 recorded that its two-slot control did not resolve chains. The cause proposed there "
                        "was the training distribution, not the architecture; this experiment tests that explanation "
                        "by putting 30% chains into training and changing nothing else.",
        "by_construction": ["the store resolves a chain by following kids with a depth limit and a cycle check; what is "
                            "measured is whether the trained model reproduces it from the pointers alone",
                            "the one-slot arm CANNOT represent a two-link chain: it has one dereference slot. Its "
                            "criterion is that it answers unknown rather than inventing an entity."],
        "learned": ["following a pointer whose target is itself a pointer, with the query for each dereference coming "
                    "from the value just read",
                    "refusing a chain that does not fit the available slots instead of naming another entity"],
        "not_claimed": "chains deeper than the number of slots; LLM scale.",
        "config": {"seeds": args.seeds, "steps": args.steps, "p_chain": P_CHAIN, "model_two": E15.model_config(2).to_dict(),
                   "model_one": E15.model_config(1).to_dict(), "train": E15.train_config(0, args.steps).to_dict(),
                   "eval": E15.EVAL},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}", f"{agg[k]['max']:.4f}") for k in KEYS if k in agg]
    md = "\n".join([
        "# E-000016 — Alias chains: how far the indirection carries", "",
        f"Evidence level: **E4** (synthetic system). Seeds: {args.seeds}; {args.steps} steps; "
        f"{P_CHAIN:.0%} of the aliases in training point at another alias.", "",
        record["follows_from"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed", "best seed"], rows), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "By construction: " + "; ".join(record["by_construction"]) + ".", "",
        "Learned: " + "; ".join(record["learned"]) + ".", "",
        "Not claimed: " + record["not_claimed"],
    ])
    path = ledger.save("e000016_alias_chains", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
