"""Experiment E-000025 — re-scoring the symlink checkpoints across all twelve templates.

E-000020 recorded that reading through an alias in the frozen GPT-2 falls to
0.5067 while direct reading is 0.5667.  Both numbers are taken at template 0,
because ``E20._answers`` defaults to it.  Template 0 is not representative: on
the link-free adapter of E-000017-B the twelve templates read at 0.795, 0.992,
0.792, 1.000, 1.000, 0.998, 0.785, 0.997 (trained) and 0.565, 0.968, 1.000,
0.427 (held out).  Reading in this system is bimodal by phrasing, and template
0 is one of the weak ones.

The same E-000020 record already contains the counter-evidence it did not act
on: ``alias_template1_train`` is 0.785–0.895 and ``alias_template9_heldout`` is
0.870–0.920 on the very checkpoints whose headline alias number is 0.5067.

This experiment trains nothing.  It loads the checkpoints that are on disk and
scores five quantities at every one of the twelve templates:

  direct           base facts, link adapter, symlink store
  alias            alias keys, link adapter, symlink store   (one shared object)
  dup              alias keys, link adapter, duplication store (independent copies)
  linkfree_direct  base facts, link-free adapter, duplication store
  linkfree_dup     alias keys, link-free adapter, duplication store

``dup`` minus ``alias`` on the same adapter is the cost of *sharing*.
``linkfree_dup`` minus ``dup`` is the cost of having *trained* on links at all.
Separating the two is the whole point; the single-template record could not.

Note on provenance: a forced re-run of E-000020 overwrote the seed-0 and seed-1
checkpoints after that record was written, so only its seed-2 checkpoint still
matches the SHA-256 in ``e000020_symlink_gpt2.json``.  This record therefore
states the SHA of every checkpoint it actually scored and does not claim to
reproduce E-000020's per-seed numbers.

Run:  python -m so.experiments.e000025_template_rescoring [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.llm_adapter import AdapterConfig
from so.world import UNKNOWN

N_TEMPLATES = E20.N_TRAIN_TEMPLATES + E17.N_HELDOUT       # 8 trained + 4 held out
EVAL = dict(E20.EVAL)


def load_adapter(name: str, seed: int, fallback_cfg: AdapterConfig) -> Tuple[E8.GPT2Knowledge, Dict[str, Any]]:
    path = CHECKPOINTS / f"{name}{CKPT_SUFFIX}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}")
    ck = torch.load(path, weights_only=False)
    cfg = AdapterConfig(**ck["adapter_config"]) if "adapter_config" in ck else fallback_cfg
    gk = E8.GPT2Knowledge(cfg)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, {"centre": np.asarray(ck["centre"]), "sha256": _sha256(path), "config": cfg.to_dict()}


def score(gk: E8.GPT2Knowledge, bank: Bank, keys: Sequence[Tuple[int, int]], truth: np.ndarray,
          template: int) -> float:
    a, _, _ = E20._answers(gk, bank, list(keys), gk.names, template=template)
    return float((a == truth).mean())


def run_seed(seed: int, verbose: bool = True) -> Dict[str, Any]:
    """One alias world, two adapters, twelve templates."""
    link_gk, link_meta = load_adapter("e000020_gpt2", seed, AdapterConfig(status_gated=True, use_links=True, n_deref=1))
    free_gk, free_meta = load_adapter("e000017_t8_c0", seed, AdapterConfig(status_gated=True))
    centre = link_meta["centre"]

    # E-000020 evaluates at world seed 2000 + seed (e000020_symlink_gpt2.py:327), so the same offset is
    # used here: the template-0 column is then E-000020's own condition, differing only in the phrasing
    # loop around it, and for the one checkpoint whose SHA still matches that record it is a
    # reproduction check rather than a fresh measurement.
    world_seed = 2000 + seed
    rng = np.random.default_rng(world_seed)
    world, spec = E15.sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"],
                                         link_gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    sym_store, _ = E15.load_arm(world, spec, centre, world_seed, symlink=True)
    dup_store, _ = E15.load_arm(world, spec, centre, world_seed, symlink=False)
    sym, dup = bank_from_store(sym_store), bank_from_store(dup_store)

    alias_keys = list(spec.alias_keys)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    pick = rng.choice(len(base_keys), size=min(EVAL["n_direct"], len(base_keys)), replace=False)
    direct_keys = [base_keys[int(i)] for i in pick]
    truth_direct = np.array([world.index[k] for k in direct_keys])
    truth_alias = np.array([world.index[spec.alias_of[k]] for k in alias_keys])

    m: Dict[str, Any] = {"seed": seed, "world_seed": world_seed,
                         "link_checkpoint_sha256": link_meta["sha256"],
                         "linkfree_checkpoint_sha256": free_meta["sha256"],
                         "n_alias_queries": len(alias_keys), "n_direct_queries": len(direct_keys)}
    t0 = time.time()
    for t in range(N_TEMPLATES):
        tag = "train" if t < E20.N_TRAIN_TEMPLATES else "heldout"
        m[f"t{t}/direct"] = score(link_gk, sym, direct_keys, truth_direct, t)
        m[f"t{t}/alias"] = score(link_gk, sym, alias_keys, truth_alias, t)
        m[f"t{t}/dup"] = score(link_gk, dup, alias_keys, truth_alias, t)
        m[f"t{t}/linkfree_direct"] = score(free_gk, dup, direct_keys, truth_direct, t)
        m[f"t{t}/linkfree_dup"] = score(free_gk, dup, alias_keys, truth_alias, t)
        m[f"t{t}/kind"] = tag
        if verbose:
            print(f"  seed {seed} t{t:<2} ({tag:<7})  direct {m[f't{t}/direct']:.3f}  alias {m[f't{t}/alias']:.3f}  "
                  f"dup {m[f't{t}/dup']:.3f}  | link-free direct {m[f't{t}/linkfree_direct']:.3f}  "
                  f"dup {m[f't{t}/linkfree_dup']:.3f}  {time.time() - t0:.0f}s", flush=True)

    trained = range(E20.N_TRAIN_TEMPLATES)
    held = range(E20.N_TRAIN_TEMPLATES, N_TEMPLATES)
    for name, rng_t in (("train", trained), ("heldout", held), ("all", range(N_TEMPLATES))):
        for q in ("direct", "alias", "dup", "linkfree_direct", "linkfree_dup"):
            m[f"{name}/{q}_mean"] = float(np.mean([m[f"t{t}/{q}"] for t in rng_t]))
        m[f"{name}/alias_max"] = float(max(m[f"t{t}/alias"] for t in rng_t))
        m[f"{name}/alias_min"] = float(min(m[f"t{t}/alias"] for t in rng_t))
        # the two costs, kept apart
        m[f"{name}/cost_of_sharing"] = m[f"{name}/dup_mean"] - m[f"{name}/alias_mean"]
        m[f"{name}/cost_of_link_training"] = m[f"{name}/linkfree_dup_mean"] - m[f"{name}/dup_mean"]
    m["template0_alias"] = m["t0/alias"]
    m["template0_linkfree_dup"] = m["t0/linkfree_dup"]
    m["seconds"] = time.time() - t0
    return m


# Pre-registered before the run.  Disclosure: three of the sixty cells of this table were already
# visible in e000020_symlink_gpt2.json (alias at templates 1, 8 and 9), and the thresholds below were
# chosen knowing them.  They are therefore a confirmation of a reading of existing numbers, not an
# independent prediction, and the record says so.
CRITERIA = {
    "train/alias_max": (">=", 0.75),          # some trained phrasing reads an alias well
    "heldout/alias_mean": (">=", 0.55),       # and unseen phrasings are not a collapse
    "all/cost_of_sharing": ("<=", 0.10),      # sharing costs little against copies, same adapter
    "all/cost_of_link_training": ("<=", 0.25),  # most of the recorded loss is the price of link training
}

REPORT_KEYS = ([f"t{t}/{q}" for t in range(N_TEMPLATES)
                for q in ("direct", "alias", "dup", "linkfree_direct", "linkfree_dup")]
               + [f"{n}/{q}_mean" for n in ("train", "heldout", "all")
                  for q in ("direct", "alias", "dup", "linkfree_direct", "linkfree_dup")]
               + [f"{n}/{q}" for n in ("train", "heldout", "all")
                  for q in ("alias_max", "alias_min", "cost_of_sharing", "cost_of_link_training")])


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s) for s in args.seeds]
    keys = [k for k in REPORT_KEYS if all(k in s for s in per_seed)]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    rows = []
    for t in range(N_TEMPLATES):
        kind = "trained" if t < E20.N_TRAIN_TEMPLATES else "held out"
        rows.append([f"t{t} ({kind})"] + [f"{np.mean([s[f't{t}/{q}'] for s in per_seed]):.4f}"
                                          for q in ("direct", "alias", "dup", "linkfree_direct", "linkfree_dup")])
    per_template = ledger.table(
        ["template", "direct (link adapter)", "alias, shared object", "alias, duplicated",
         "direct (link-free)", "alias, duplicated (link-free)"], rows)

    lower = {f"{n}/{q}" for n in ("train", "heldout", "all") for q in ("cost_of_sharing", "cost_of_link_training")}
    ci = ledger.ci_rows(per_seed, [k for k in keys if k.startswith(("train/", "heldout/", "all/"))],
                        {}, lower_is_better=sorted(lower))

    record = {"experiment": "E-000025",
              "title": "re-scoring the symlink checkpoints across all twelve templates",
              "trains_nothing": True, "seeds": args.seeds, "n_templates": N_TEMPLATES,
              "n_train_templates": E20.N_TRAIN_TEMPLATES, "n_heldout_templates": E17.N_HELDOUT,
              "eval": EVAL, "per_seed": per_seed, "aggregate": agg, "criteria": check,
              "provenance_note": ("a forced re-run of E-000020 overwrote its seed-0 and seed-1 checkpoints after "
                                  "that record was written; only seed 2 still matches the SHA-256 recorded there. "
                                  "The SHA of every checkpoint scored here is in per_seed.")}

    md = [f"# E-000025 — {record['title']}", "",
          "No training. The checkpoints of E-000020 (link adapter) and E-000017-B (link-free adapter, eight",
          "trained templates) are loaded from disk and scored at every one of the twelve templates on one",
          "alias world, in both stores.", "",
          "E-000020's headline numbers — direct 0.5667, alias 0.5067 — are template 0 only, because",
          "`E20._answers` defaults to it. The table below is what the same checkpoints do everywhere else.", "",
          "## Reading, per template (mean over seeds)", "", per_template, "",
          "`alias, shared object` and `alias, duplicated` are the *same adapter* answering the *same questions*",
          "against a store that shares one object and a store that holds independent copies: their difference is",
          "the cost of sharing. The difference between `alias, duplicated (link-free)` and `alias, duplicated` is",
          "the cost of having trained on links at all.", "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "Disclosure: alias reading at templates 1, 8 and 9 was already recorded in E-000020, so these",
          "thresholds were set knowing three of the sixty cells above. This record confirms a reading of",
          "existing numbers; it is not an independent prediction.", "",
          "## Aggregates", "", ledger.table(ledger.CI_HEADERS, ci), "",
          "## Provenance", "", record["provenance_note"], ""]
    path = ledger.save("e000025_template_rescoring", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(per_template)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
