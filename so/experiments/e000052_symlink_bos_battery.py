"""Experiment E-000052 -- the pointer battery on the BOS-trained symlink adapter, narrowed.

E-000050 (ledger 31.38) showed that every held-out-phrasing number in this repository was read off an
adapter whose subject sat on GPT-2's position-0 attention sink, and that an adapter trained with a BOS
reads unseen phrasings at 0.9712 (worst seed) against 0.7288. E-000020's symlink adapter was trained
the same way. E-000052 trains it again with a BOS on every prompt (so/experiments/e000052_symlink_bos_train.py,
trainer unchanged) and runs the pointer battery on the corrected substrate: the same world written
twice -- alias keys as LINK cells, and as copies -- under one reader, at all twelve phrasings.

WHAT IS A REPRODUCTION AND WHAT IS CONTENT, said before the run (31.42). Two refuters found most of
the battery to be a predicted transport of rows the repository already holds (E-000015/20/26 through
every alias; E-000050's subject-initial recovery), and the completeness critic corrected the
pre-registration in five places. What the run carries:
  (P)  the reader's PRICE for the pointer in the BOS regime, E-000025's two costs over all twelve
       phrasings against the BOS-trained link-free adapter (e000050_bos) -- the one number whose sign
       nobody here can predict (E-000025's 0.0954 sits 0.005 under its bar);
  (N)  the SET NULL row -- a blanked alias is a self-referencing link the adapter never trained on, and
       E-000051 read one of two as a wrong entity on GPT-2 -- as the WRONG-ENTITY rate at every phrasing
       (the UNKNOWN rate is validity, not a claim);
  (T)  the subject-medial held-out residue E-000050 left at 0.91 read / 0.83 route, at t9 and t10, in
       the two rows that carry information there: alias_direct and shared_update (equal bars; the
       entity-failure rows shred/delete_target/alias_true_object are REPORTED, never scored, where a
       routing miss would pass them for free).
Everything else in the table is a reproduction of E-000026 / E-000050 and is labelled so.

ANCHOR AND CONTROL. The reproduction anchor is the trained template 3 only (E-000026's strong_train;
t10 is a claim row and cannot also be an anchor): direct within E-000026's recorded seed spread of its
worst seed, alias_direct likewise. Anchor fails -> REGRESSION reading: the BOS training changed the
adapter, nothing else is read. The reverse control D reads the BOS-trained checkpoint WITHOUT its BOS:
subject-initial held-out alias reading must collapse (E-000050 arm D: 0.00) while medial stays; if D
does not fire the checkpoint does not depend on its BOS and the substrate is VOID.

Owned and cited: SQL-92 referential actions; Raeesi and Roed name the pointer record as future work;
Yang et al. (The Fall of ROME) own the position-0 remedy; E-000025 owns the price definition. Nothing
here is a mechanism. Trains nothing (the substrate was trained by e000052_symlink_bos_train.py).

Run:  python -m so.experiments.e000052_symlink_bos_battery [--seeds 0 1 2] [--threads 4]
      python -m so.experiments.e000052_symlink_bos_battery --smoke-on-recorded --seeds 0 --templates 3 8 --threads 1
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so import ledger
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments import e000020_symlink_gpt2 as E20
from so.experiments import e000025_template_rescoring as E25
from so.experiments import e000039_address_tying as E39
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.llm_adapter import AdapterConfig
from so.world import UNKNOWN

N_TEMPLATES = 12
ANCHOR_TEMPLATE = 3
MEDIAL_HELDOUT = (9, 10)
INITIAL_HELDOUT = (8, 11)
# E-000026 strong_train (template 3), per seed: direct 0.9933/1.0/0.9933, alias 0.95/0.87/0.86
ANCHOR_DIRECT_FLOOR = 0.9933 - (1.0 - 0.9933)      # worst seed minus the recorded spread
ANCHOR_ALIAS_FLOOR = 0.86 - (0.95 - 0.86)


def set_bos(on: bool) -> None:
    os.environ["SO_BOS"] = "1" if on else "0"


def world_and_stores(gk, seed: int, centre: np.ndarray):
    """Exactly E-000020.evaluate's world and the two stores, so every row here reads the same keys."""
    rng = np.random.default_rng(seed)
    world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"], E20.EVAL["n_alias_per_group"],
                                         gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    sym_store, sym_kids = E15.load_arm(world, spec, centre, seed, symlink=True)
    return world, spec, sym_store, sym_kids


def lifecycle_extra(gk, seed: int, centre: np.ndarray, template: int) -> Dict[str, float]:
    """BLANK the first alias of every pod (SET NULL by hand), read it, its sibling and its target; then
    RELINK it back and read again. Fresh world and store per call."""
    world, spec, st, kids = world_and_stores(gk, seed, centre)
    first = [ks[0] for _, ks in spec.groups]
    second = [ks[1] for _, ks in spec.groups]
    targets = [t for t, _ in spec.groups]
    truth_first = np.array([world.index[spec.alias_of[k]] for k in first])
    truth_second = np.array([world.index[spec.alias_of[k]] for k in second])
    truth_t = np.array([world.index[k] for k in targets])
    m: Dict[str, float] = {}
    for a in first:
        st.blank(kids[a])
    b = bank_from_store(st)
    a1, _, _ = E20._answers(gk, b, first, gk.names, template=template)
    m["blank/alias_unknown"] = float((a1 == UNKNOWN).mean())
    m["blank/alias_wrong_entity"] = float((a1 != UNKNOWN).mean())
    m["blank/alias_true_object"] = float((a1 == truth_first).mean())
    m["blank/sibling_readable"] = float((E20._answers(gk, b, second, gk.names, template=template)[0] == truth_second).mean())
    m["blank/target_readable"] = float((E20._answers(gk, b, targets, gk.names, template=template)[0] == truth_t).mean())
    for a, t in zip(first, targets):
        st.relink(kids[a], kids[t])
    b2 = bank_from_store(st)
    m["relink/alias_direct"] = float((E20._answers(gk, b2, first, gk.names, template=template)[0] == truth_first).mean())
    return m


def price(link_gk, free_gk, seed: int, centre: np.ndarray, templates: Sequence[int]) -> Dict[str, float]:
    """E-000025's five quantities per template and its two costs, on the BOS-trained pair."""
    world_seed = 2000 + seed
    rng = np.random.default_rng(world_seed)
    world, spec = E15.sample_alias_world(rng, E20.EVAL["n_base"], E20.EVAL["n_groups"], E20.EVAL["n_alias_per_group"],
                                         link_gk.n_entities, 4, E20.N_TRAIN_TEMPLATES)
    sym_store, _ = E15.load_arm(world, spec, centre, world_seed, symlink=True)
    dup_store, _ = E15.load_arm(world, spec, centre, world_seed, symlink=False)
    sym, dup = bank_from_store(sym_store), bank_from_store(dup_store)
    alias_keys = list(spec.alias_keys)
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    pick = rng.choice(len(base_keys), size=min(E20.EVAL["n_direct"], len(base_keys)), replace=False)
    direct_keys = [base_keys[int(i)] for i in pick]
    truth_direct = np.array([world.index[k] for k in direct_keys])
    truth_alias = np.array([world.index[spec.alias_of[k]] for k in alias_keys])
    m: Dict[str, float] = {}
    for t in templates:
        m[f"t{t}/direct"] = E25.score(link_gk, sym, direct_keys, truth_direct, t)
        m[f"t{t}/alias"] = E25.score(link_gk, sym, alias_keys, truth_alias, t)
        m[f"t{t}/dup"] = E25.score(link_gk, dup, alias_keys, truth_alias, t)
        m[f"t{t}/linkfree_direct"] = E25.score(free_gk, dup, direct_keys, truth_direct, t)
        m[f"t{t}/linkfree_dup"] = E25.score(free_gk, dup, alias_keys, truth_alias, t)
    groups = {"train": [t for t in templates if t < E20.N_TRAIN_TEMPLATES],
              "heldout": [t for t in templates if t >= E20.N_TRAIN_TEMPLATES], "all": list(templates)}
    for name, ts in groups.items():
        if not ts:
            continue
        for q in ("direct", "alias", "dup", "linkfree_direct", "linkfree_dup"):
            m[f"{name}/{q}_mean"] = float(np.mean([m[f"t{t}/{q}"] for t in ts]))
        m[f"{name}/cost_of_sharing"] = m[f"{name}/dup_mean"] - m[f"{name}/alias_mean"]
        m[f"{name}/cost_of_link_training"] = m[f"{name}/linkfree_dup_mean"] - m[f"{name}/dup_mean"]
    return m


def run_seed(seed: int, link_name: str, free_name: str, templates: Sequence[int], verbose: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    link_gk, link_meta = E25.load_adapter(link_name, seed, AdapterConfig(status_gated=True, use_links=True, n_deref=1))
    free_gk, free_meta = E25.load_adapter(free_name, seed, AdapterConfig(status_gated=True))
    centre = link_meta["centre"]
    initial, medial = E39.subject_initial_templates(link_gk.tok)
    m: Dict[str, Any] = {"seed": seed, "link_checkpoint_sha256": link_meta["sha256"],
                         "linkfree_checkpoint_sha256": free_meta["sha256"],
                         "subject_initial_templates": list(initial), "subject_medial_templates": list(medial)}
    # ---- arm C: BOS-trained, read with a BOS
    set_bos(True)
    for t in templates:
        e = E20.evaluate(link_gk, 2000 + seed, centre, template=t)
        for k, v in e.items():
            if isinstance(v, (int, float)) and k != "seed":
                m[f"C/t{t}/{k}"] = float(v)
        for k, v in lifecycle_extra(link_gk, 2000 + seed, centre, t).items():
            m[f"C/t{t}/{k}"] = v
        if verbose:
            print(f"  seed {seed} C t{t:<2}: direct {m[f'C/t{t}/direct']:.3f} alias {m[f'C/t{t}/alias_direct']:.3f} "
                  f"dup {m[f'C/t{t}/dup_direct']:.3f} update {m[f'C/t{t}/shared_update/alias_new_object']:.3f} "
                  f"blank unk/wrong {m[f'C/t{t}/blank/alias_unknown']:.2f}/{m[f'C/t{t}/blank/alias_wrong_entity']:.2f} "
                  f"relink {m[f'C/t{t}/relink/alias_direct']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
    # ---- arm D: the same checkpoint read WITHOUT its BOS (the reverse control)
    set_bos(False)
    for t in templates:
        e = E20.evaluate(link_gk, 2000 + seed, centre, template=t)
        for k in ("direct", "alias_direct", "dup_direct", "shared_update/alias_new_object"):
            m[f"D/t{t}/{k}"] = float(e[k])
        if verbose:
            print(f"  seed {seed} D t{t:<2}: direct {m[f'D/t{t}/direct']:.3f} alias {m[f'D/t{t}/alias_direct']:.3f}", flush=True)
    # ---- arm P: the price, with a BOS, against the BOS-trained link-free adapter
    set_bos(True)
    for k, v in price(link_gk, free_gk, seed, centre, templates).items():
        m[f"P/{k}"] = v
    set_bos(False)

    # ---- summaries the criteria read
    def mn(arm, ts, key):
        vals = [m[f"{arm}/t{t}/{key}"] for t in ts if f"{arm}/t{t}/{key}" in m]
        return float(min(vals)) if vals else float("nan")

    def mx(arm, ts, key):
        vals = [m[f"{arm}/t{t}/{key}"] for t in ts if f"{arm}/t{t}/{key}" in m]
        return float(max(vals)) if vals else float("nan")

    init_h = [t for t in INITIAL_HELDOUT if t in templates]
    med_h = [t for t in MEDIAL_HELDOUT if t in templates]
    if ANCHOR_TEMPLATE in templates:
        m["C/anchor/direct"] = m[f"C/t{ANCHOR_TEMPLATE}/direct"]
        m["C/anchor/alias_direct"] = m[f"C/t{ANCHOR_TEMPLATE}/alias_direct"]
    m["C/heldout_initial/alias_direct_min"] = mn("C", init_h, "alias_direct")
    m["C/heldout_medial/alias_direct_min"] = mn("C", med_h, "alias_direct")
    m["C/heldout_medial/shared_update_min"] = mn("C", med_h, "shared_update/alias_new_object")
    m["C/heldout_medial/entity_failure_max"] = max(mx("C", med_h, "shred_target/alias_true_object"),
                                                   mx("C", med_h, "delete_target/alias_true_object"))
    m["C/blank/alias_wrong_entity_max"] = mx("C", templates, "blank/alias_wrong_entity")
    m["C/blank/alias_unknown_min"] = mn("C", templates, "blank/alias_unknown")
    m["C/blank/sibling_readable_min"] = mn("C", templates, "blank/sibling_readable")
    m["C/blank/target_readable_min"] = mn("C", templates, "blank/target_readable")
    m["C/relink/alias_direct_min"] = mn("C", templates, "relink/alias_direct")
    m["C/trained/alias_direct_min"] = mn("C", [t for t in templates if t < E20.N_TRAIN_TEMPLATES], "alias_direct")
    m["D/heldout_initial/alias_direct_max"] = mx("D", init_h, "alias_direct")
    m["D/heldout_medial/alias_direct_min"] = mn("D", med_h, "alias_direct")
    m["seconds"] = time.time() - t0
    return m


CRITERIA = {
    # ANCHOR (trained template 3, within E-000026's recorded seed spread of its worst seed)
    "C/anchor/direct": (">=", round(ANCHOR_DIRECT_FLOOR, 4)),
    "C/anchor/alias_direct": (">=", round(ANCHOR_ALIAS_FLOOR, 4)),
    # REVERSE CONTROL: without its BOS the checkpoint must lose subject-initial held-out reading and keep medial
    "D/heldout_initial/alias_direct_max": ("<=", 0.85),
    "D/heldout_medial/alias_direct_min": (">=", 0.80),
    # REPRODUCTION of E-000050's subject-initial recovery on the symlink adapter (labelled, not content)
    "C/heldout_initial/alias_direct_min": (">=", 0.80),
    # (P) THE PRICE, E-000025's bars, over all twelve phrasings in the BOS regime
    "P/all/cost_of_sharing": ("<=", 0.10),
    "P/all/cost_of_link_training": ("<=", 0.25),
    # (N) SET NULL: the claim row is the wrong-entity rate; the UNKNOWN rate and the neighbours are validity
    "C/blank/alias_wrong_entity_max": ("<=", 0.05),
    "C/blank/alias_unknown_min": (">=", 0.90),
    "C/blank/sibling_readable_min": (">=", 0.80),
    "C/blank/target_readable_min": (">=", 0.80),
    "C/relink/alias_direct_min": (">=", 0.80),
    # (T) the subject-medial held-out residue, in the two rows that carry information, equal bars
    "C/heldout_medial/alias_direct_min": (">=", 0.80),
    "C/heldout_medial/shared_update_min": (">=", 0.80),
}

DECISION_RULE = (
    "Worst seed. If the ANCHOR fails (template 3 direct or alias below E-000026's worst seed minus its "
    "recorded spread) the reading is REGRESSION: the BOS training changed the adapter and no other row is "
    "read. If the REVERSE CONTROL does not fire (subject-initial held-out alias reading above 0.85 without "
    "the BOS, or medial below 0.80) the substrate is VOID: the checkpoint does not depend on its BOS. With "
    "anchor and control holding, each content row is read on its own and named: PRICE fails if either "
    "E-000025 cost exceeds its bar in the BOS regime (E-000025's 0.0954 / 0.0688 do not transfer); SET NULL "
    "fails if a blanked alias is answered with an entity at more than 0.05 at any phrasing (the UNKNOWN and "
    "neighbour rows must hold for the row to be readable at all); MEDIAL fails if alias_direct or shared "
    "UPDATE reach at t9 or t10 is below 0.80 (the entity-failure rows there are reported and never scored, "
    "because a routing miss passes them for free). CLEAN if all three hold: the symlink adapter on the "
    "corrected substrate meets E-000020's bars at every phrasing, a SET NULL alias is never read as an "
    "entity, and the pointer costs the reader no more than it did without a BOS -- a measurement paper's "
    "table, every mechanism owned. Fixed before the run.")


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--templates", type=int, nargs="*", default=list(range(N_TEMPLATES)))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--smoke-on-recorded", action="store_true",
                    help="stand in the RECORDED no-BOS checkpoints (e000020_gpt2 / e000017_t8_c0) with reduced sizes; "
                         "written with a -smoke suffix, never a record")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if os.environ.get("SO_CKPT_SUFFIX", ""):
        raise SystemExit("run without SO_CKPT_SUFFIX: the battery names its checkpoints explicitly")
    link_name, free_name = "e000020_gpt2_bos", "e000050_bos"
    if args.smoke_on_recorded:
        link_name, free_name = "e000020_gpt2", "e000017_t8_c0"
        E20.EVAL.update(n_base=150, n_groups=20, n_direct=40, n_targets=20)
    for seed in args.seeds:
        for name in (link_name, free_name):
            p = CHECKPOINTS / f"{name}_seed{seed}.pt"
            if not p.exists():
                raise SystemExit(f"missing substrate {p}; train it with e000052_symlink_bos_train.py first")

    per = []
    for seed in args.seeds:
        print(f"=== seed {seed}: {link_name} (link) / {free_name} (link-free), templates {args.templates} ===", flush=True)
        per.append(run_seed(seed, link_name, free_name, args.templates))
    numeric = [{k: float(v) for k, v in s.items() if isinstance(v, (int, float)) and k != "seed"} for s in per]
    keys = sorted(set.intersection(*[set(s) for s in numeric])) if numeric else []
    agg = ledger.aggregate(numeric, keys)
    crit = {k: v for k, v in CRITERIA.items() if k in agg and not np.isnan(agg[k]["mean"])}
    check = ledger.check_criteria(agg, crit)

    def w(k, lower=False):
        return f"{ledger.worst(agg[k], lower):.4f}" if k in agg else "-"

    rows = []
    for t in args.templates:
        rows.append([f"t{t} ({'initial' if t in per[0]['subject_initial_templates'] else 'medial'}, "
                     f"{'trained' if t < E20.N_TRAIN_TEMPLATES else 'held out'})",
                     w(f"C/t{t}/direct"), w(f"C/t{t}/alias_direct"), w(f"C/t{t}/dup_direct"),
                     w(f"C/t{t}/shared_update/alias_new_object"), w(f"C/t{t}/shred_target/alias_unknown"),
                     w(f"C/t{t}/delete_target/alias_unknown"), w(f"C/t{t}/blank/alias_wrong_entity", lower=True),
                     w(f"C/t{t}/relink/alias_direct"),
                     # The reverse control's desired direction FLIPS with the template: low is the
                     # control firing at a subject-INITIAL template, high is it holding at a MEDIAL
                     # one. A single `lower=True` printed the most favourable seed for the medial
                     # rows (t1: 0.9100 where the worst seed is 0.7450). Ledger 31.53.
                     w(f"D/t{t}/alias_direct", lower=(t in per[0]["subject_initial_templates"]))])
    md = [f"# E-000052 — the pointer battery on the BOS-trained symlink adapter, narrowed", "",
          f"Seeds {args.seeds}, templates {args.templates}, link adapter `{link_name}`, link-free `{free_name}`"
          + (" — SMOKE ON THE RECORDED CHECKPOINTS AT REDUCED SIZES, not a record" if args.smoke_on_recorded else "")
          + ". Worst seed throughout. Arm C reads the BOS-trained checkpoint with a BOS; arm D reads it without "
          "(the reverse control); arm P is E-000025's price. NOTE (31.53): `cost_of_sharing` is dup minus "
          "alias on the SAME adapter, a within-reader contrast; only `cost_of_link_training` involves "
          "the link-free adapter. The D column takes the worst seed in the direction the control is "
          "supposed to move, which is low at subject-initial templates and high at medial ones.", "",
          ledger.table(["template", "direct", "alias", "dup", "UPDATE reaches alias", "SHRED → unknown",
                        "DELETE → unknown", "BLANK → some entity", "RELINK reads", "D: alias, no BOS"], rows), "",
          ledger.table(["price (P), BOS regime", "train", "held out", "all"],
                       [[q, w(f"P/train/{q}", True), w(f"P/heldout/{q}", True), w(f"P/all/{q}", True)]
                        for q in ("cost_of_sharing", "cost_of_link_training")]), "",
          "Reproductions, labelled: the trained-template rows against E-000026 (template 3) and the subject-initial "
          "held-out recovery against E-000050. Content: the price (P), the BLANK wrong-entity rate (N), and the "
          "subject-medial held-out rows t9/t10 (T). The entity-failure rows at t9/t10 are reported "
          f"(max {w('C/heldout_medial/entity_failure_max', True)}) and never scored.", "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    record = {"experiment": "E-000052", "title": "the pointer battery on the BOS-trained symlink adapter, narrowed",
              "evidence_level": "E5", "trains_nothing": True, "smoke_on_recorded": args.smoke_on_recorded,
              "seeds": args.seeds, "templates": args.templates, "link_adapter": link_name, "linkfree_adapter": free_name,
              "decision_rule": DECISION_RULE, "per_seed": per, "aggregate": agg, "criteria": check}
    name = "e000052_symlink_bos_battery" + ("-smoke" if args.smoke_on_recorded else "")
    if args.results_dir:
        import json
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, name + ".json"), "w") as f:
            json.dump(record, f, indent=1, default=float)
        path = os.path.join(args.results_dir, name + ".md")
        with open(path, "w") as f:
            f.write("\n".join(md))
    else:
        path = ledger.save(name, record, "\n".join(md))
    print("\n".join(md[1:])); print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
