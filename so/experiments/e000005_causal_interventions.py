"""Experiment E-000005 — causal interventions on knowledge cells.

Ledger section 25: correlation (cell C17 activates when K17 is queried) is not
causation.  For 100 targets per seed the interventions
    disable / swap / restore / replace
are applied to the cell and the answer must change *predictably*; disabling a
random other cell must change nothing.  The cell the model routes to (its
"biomarker" cell) must be the ground-truth cell, and disabling *that* routed
cell must remove the answer — C17 → K17, not merely C17 ↔ K17.

Run:  python -m so.experiments.e000005_causal_interventions
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.evaluation import predict
from so.experiments.common import answers, fresh_world, load_base_model, position_of_kid
from so.interventions import disable_mask, routed_position
from so.world import UNKNOWN

N_TARGETS = 100


def run_seed(seed: int) -> Dict[str, Any]:
    base = load_base_model(seed)
    model, centre = base["model"], base["centre"]
    rng, world, store, kids, ref = fresh_world(500 + seed, centre)
    facts = world.facts
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:N_TARGETS]]
    partners = [facts[int(i)] for i in perm[N_TARGETS:2 * N_TARGETS]]
    c = {k: 0 for k in ("disable", "disable_random_other", "swap", "swap_partner", "restore", "replace",
                        "localization", "routed_cell_causal")}
    for f, g in zip(targets, partners):
        kid, kid2 = kids[f.key], kids[g.key]
        q, q2 = world.make_query(rng, "fwd", f.subject, [f.relation]), world.make_query(rng, "fwd", g.subject, [g.relation])
        # localisation: the routed cell is the ground-truth cell
        p = predict(model, store, world, [q])
        rp = routed_position(p.routing[0])
        c["localization"] += int(rp is not None and int(store.bank()["kid"][rp]) == kid)
        # disable the routed cell (whatever it is) -> answer must vanish
        if rp is not None:
            mask = np.ones(store.bank()["kid"].shape[0], dtype=bool); mask[rp] = False
            c["routed_cell_causal"] += int(answers(model, store, world, [q], cell_mask=mask)[0] == UNKNOWN)
        # disable the ground-truth cell
        c["disable"] += int(answers(model, store, world, [q], cell_mask=disable_mask(store, kid))[0] == UNKNOWN)
        # disable a random other active cell -> unchanged
        other = kids[facts[int(rng.choice([i for i in perm[2 * N_TARGETS:2 * N_TARGETS + 50]]))].key]
        c["disable_random_other"] += int(answers(model, store, world, [q], cell_mask=disable_mask(store, other))[0] == f.obj)
        # swap payloads with the partner -> both answers exchange
        store.swap(kid, kid2)
        a = answers(model, store, world, [q, q2])
        c["swap"] += int(a[0] == g.obj); c["swap_partner"] += int(a[1] == f.obj)
        store.swap(kid, kid2)
        a = answers(model, store, world, [q, q2])
        c["restore"] += int(a[0] == f.obj and a[1] == g.obj)
        # replace in place -> new answer
        new = int((f.obj + 1 + rng.integers(0, world.n_entities - 1)) % world.n_entities)
        store.replace(kid, new)
        c["replace"] += int(answers(model, store, world, [q])[0] == new)
        store.replace(kid, f.obj)
    m = {k: v / N_TARGETS for k, v in c.items()}
    m["seed"] = seed
    m["base_checkpoint_sha256"] = base["checkpoint_sha256"]
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    per_seed = [run_seed(s) for s in args.seeds]
    for s in per_seed: print(s, flush=True)
    keys = [k for k in per_seed[0] if k not in ("seed", "base_checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: (">=", 0.98) for k in keys})
    record = {
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "by_construction_vs_learned": "That the read equation uses the cell's payload is by construction; what is "
                                      "tested is that the trained core actually routes each query to its own cell "
                                      "(localisation), does not draw the answer from anywhere else (disable -> "
                                      "UNKNOWN, random-other -> unchanged) and turns a swapped or replaced payload "
                                      "into exactly the predicted answer. Localisation is a trained objective "
                                      "(routing loss); E-000006 'no_routing_loss' reports how much of it emerges "
                                      "without that supervision.",
        "experiment": "E-000005", "title": "Causal interventions on knowledge cells",
        "evidence_level": "E4", "deletion_level": None,
        "claim": "The cell the model routes to for a query is the ground-truth cell, and intervening on it "
                 "(disable / swap / restore / replace) changes the answer exactly as predicted while intervening "
                 "on another cell changes nothing. In this architecture the cell read is the only knowledge channel, "
                 "so this establishes that the trained core uses that channel as intended (no answer from elsewhere, "
                 "exact localisation); it is a consistency result, not a discovery of localisation in a model with "
                 "competing channels.",
        "not_claimed": "Causal localisation inside a pretrained LLM.",
        "config": {"seeds": args.seeds, "n_targets": N_TARGETS}, "per_seed": per_seed, "aggregate": agg,
    }
    expected = {"disable": "UNKNOWN", "disable_random_other": "unchanged", "swap": "partner's object",
                "swap_partner": "target's object", "restore": "both original", "replace": "new object",
                "localization": "routed cell == ground-truth cell", "routed_cell_causal": "disabling routed cell -> UNKNOWN"}
    rows = [(k, expected[k], ledger.pct(agg[k]["mean"]), ledger.pct(agg[k]["min"])) for k in keys]
    md = "\n".join([
        "# E-000005 — Causal interventions", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}). Seeds: {args.seeds}, {N_TARGETS} targets per seed.", "",
        ledger.table(["intervention", "predicted outcome", "observed (mean)", "worst seed"], rows), "",
        record["by_construction_vs_learned"], "",
        f"n = {N_TARGETS} targets per seed. Pre-registered criteria (worst seed):", "", ledger.criteria_table(check),
    ])
    path = ledger.save("e000005_causal_interventions", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
