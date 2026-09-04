"""Experiment E-000026 — the symlink lifecycle battery, measured where reading works.

E-000020 ran its whole battery at template 0.  E-000025 shows that reading in this
system is bimodal by phrasing and that template 0 is one of the weak ones: on the
same checkpoints, alias reading is 0.52 at template 0, 0.88 at template 1, 0.94 at
the held-out template 9.  A deletion claim measured on a read that works half the
time is a weak claim in both directions — "the alias no longer answers" is cheap
when the alias barely answered to begin with, and "one update reached every path"
is hard when the read is noise.

So the battery is re-run unchanged at three phrasings:

  template 0   what E-000020 recorded, for comparison
  strong-train the trained template on which the LINK-FREE adapter of E-000017-B
               reads best
  strong-held  the same rule over the four held-out templates

The two strong templates are chosen from a *different* experiment's record, on a
*different* adapter — the link-free one — so the choice cannot be tuned to make
the link arm look good.  The rule is applied at run time by reading
``so/results/e000017b_templates8.json``, and the chosen indices are recorded.

Trains nothing.

Run:  python -m so.experiments.e000026_lifecycle_at_a_readable_template [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from so import ledger
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.llm_adapter import AdapterConfig


def choose_templates() -> Dict[str, Any]:
    """The selection rule, applied to E-000017-B's link-free record. No link-arm number is consulted."""
    path = ledger.RESULTS_DIR / "e000017b_templates8.json"
    if not path.exists():
        raise SystemExit(f"missing {path}: run `python -m so.experiments.e000017_paraphrase_gap --phase train` first")
    rec = json.loads(path.read_text())
    per_seed = rec["per_seed"]

    def mean_at(t: int, suffix: str) -> float:
        key = f"template{t}_{suffix}/active_correct"
        vals = [s[key] for s in per_seed if key in s]
        return float(np.mean(vals)) if vals else float("nan")

    trained = {t: mean_at(t, "train") for t in range(E20.N_TRAIN_TEMPLATES)}
    held = {t: mean_at(t, "heldout") for t in range(E20.N_TRAIN_TEMPLATES,
                                                    E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT)}
    best_train = min((t for t in trained if trained[t] == max(trained.values())))     # ties -> lowest index
    best_held = min((t for t in held if held[t] == max(held.values())))
    return {"strong_train": int(best_train), "strong_heldout": int(best_held),
            "linkfree_reading_trained": trained, "linkfree_reading_heldout": held}


def load_link_adapter(seed: int) -> Tuple[E8.GPT2Knowledge, np.ndarray, str]:
    path = CHECKPOINTS / f"e000020_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}; run: python -m so.experiments.e000020_symlink_gpt2 --seeds {seed}")
    ck = torch.load(path, weights_only=False)
    cfg = AdapterConfig(**ck["adapter_config"]) if "adapter_config" in ck else \
        AdapterConfig(status_gated=True, use_links=True, n_deref=1)
    gk = E8.GPT2Knowledge(cfg)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, np.asarray(ck["centre"]), _sha256(path)


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    choice = choose_templates()
    arms = {"template0": 0, "strong_train": choice["strong_train"], "strong_heldout": choice["strong_heldout"]}
    print(f"templates chosen from E-000017-B's link-free record: {arms}", flush=True)

    per_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    shas: Dict[int, str] = {}
    for seed in args.seeds:
        gk, centre, sha = load_link_adapter(seed)
        shas[seed] = sha
        for arm, t in arms.items():
            print(f"=== seed {seed}, {arm} (template {t}) ===", flush=True)
            m = E20.evaluate(gk, 2000 + seed, centre, template=t)
            m["template"] = t
            m["checkpoint_sha256"] = sha
            per_arm[arm].append(m)
            print(f"  direct {m['direct']:.3f}  alias {m['alias_direct']:.3f}  dup {m['dup_direct']:.3f}  "
                  f"| shred->unknown {m['shred_target/alias_unknown']:.3f}  "
                  f"update reaches alias {m['shared_update/alias_new_object']:.3f}", flush=True)

    groups = E20.criteria_groups()
    flat = {k: v for g in groups.values() for k, v in g.items()}
    agg = {a: ledger.aggregate(per_arm[a], [k for k in E20.KEYS if all(k in s for s in per_arm[a])])
           for a in arms}
    checks = {a: ledger.check_criteria(agg[a], {k: v for k, v in flat.items() if k in agg[a]}) for a in arms}

    headline = ["direct", "alias_direct", "dup_direct", "shared_update/alias_new_object",
                "duplicate_update/alias_new_object", "shred_target/alias_unknown",
                "shred_target/alias_true_object", "shred_target/alias_forced_choice",
                "shred_target/alias_probe_top1", "active/alias_probe_top1",
                "revoke_alias/sibling_readable", "delete_target/alias_unknown"]
    lower = {"duplicate_update/alias_new_object", "shred_target/alias_true_object",
             "shred_target/alias_probe_top1"}
    rows = []
    for k in headline:
        row = [k]
        for a in arms:
            row.append(f"{ledger.worst(agg[a][k], k in lower):.4f}" if k in agg[a] else "-")
        rows.append(row)
    table = ledger.table(["measure (worst seed)"] + [f"{a} (t{arms[a]})" for a in arms], rows)

    passing = {a: [g for g, crit in groups.items()
                   if all(checks[a]["criteria"].get(k, {}).get("pass") for k in crit if k in agg[a])]
               for a in arms}

    record = {"experiment": "E-000026",
              "title": "the symlink lifecycle battery, measured where reading works",
              "trains_nothing": True, "seeds": args.seeds, "arms": arms,
              "template_choice": choice, "checkpoint_sha256": shas,
              "per_arm": {a: per_arm[a] for a in arms},
              "aggregate": {a: agg[a] for a in arms},
              "criteria": {a: checks[a] for a in arms},
              "criteria_groups_passed": passing}

    md = [f"# E-000026 — {record['title']}", "",
          f"Seeds {args.seeds}. No training: E-000020's checkpoints, E-000020's battery, run three times at",
          f"three phrasings — template 0 (what that record used), template {arms['strong_train']} (the trained",
          f"template on which E-000017-B's *link-free* adapter reads best) and template {arms['strong_heldout']}",
          "(the same rule over the held-out four). The choice comes from a different experiment on a different",
          "adapter, so it cannot be tuned in the link arm's favour.", "",
          "## The battery at three phrasings", "", table, "",
          "## Pre-registered criteria (E-000020's, unchanged)", ""]
    for a in arms:
        md += [f"### {a} — template {arms[a]}", "", ledger.criteria_table(checks[a]),
               f"\nGroups passed: {', '.join(passing[a]) if passing[a] else 'none'}.", ""]
    md += ["## How the templates were chosen", "",
           ledger.table(["template", "link-free reading (E-000017-B)", "kind"],
                        [[t, f"{v:.4f}", "trained"] for t, v in choice["linkfree_reading_trained"].items()]
                        + [[t, f"{v:.4f}", "held out"] for t, v in choice["linkfree_reading_heldout"].items()]), ""]
    path = ledger.save("e000026_lifecycle_readable_template", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(table)
    for a in arms:
        print(f"{a}: groups passed -> {passing[a]}")
    return record


if __name__ == "__main__":
    main()
