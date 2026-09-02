"""Experiment E-000003 — retention and generalisation of deletion.

Ledger sections 3, 16, 27: a deletion counts only if
    target knowledge   high -> low      (all paraphrases, multi-hop through it, reverse access)
    control knowledge  high -> high
    unrelated / general capability      unchanged
Three mechanisms are tested on the same targets: REVOKE (routing removal),
SHRED (marker destroyed, payload physically present) and UPDATE (replacement).

Run:  python -m so.experiments.e000003_retention_generalization
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.experiments.common import accuracy, all_paraphrases, answers, fresh_world, load_base_model, unknown_rate
from so.world import Query, UNKNOWN, World

N_TARGETS, N_CONTROLS, N_GENERAL = 50, 50, 300


def run_seed(seed: int) -> Dict[str, Any]:
    base = load_base_model(seed)
    model, centre = base["model"], base["centre"]
    rng, world, store, kids, ref = fresh_world(200 + seed, centre)
    facts = world.facts
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:N_TARGETS]]
    controls = [facts[int(i)] for i in perm[N_TARGETS:N_TARGETS + N_CONTROLS]]
    t_ent = {f.subject for f in targets} | {f.obj for f in targets}
    unrelated = [f for f in facts[::-1] if f.subject not in t_ent and f.obj not in t_ent and f not in targets][:N_CONTROLS]
    t_keys = {f.key for f in targets}

    def para(fs): return [q for f in fs for q in all_paraphrases(world, f.subject, f.relation)]
    def truth_para(fs): return [f.obj for f in fs for _ in range(world.n_synonyms)]

    q_t, q_c, q_u = para(targets), para(controls), para(unrelated)
    # 2-hop through a target as first edge, and 2-hop paths that avoid all targets
    q_t_hop = [world.make_query(rng, "fwd", f.subject, [f.relation, r2]) for f in targets
               for r2 in range(world.n_relations) if (f.obj, r2) in world.index][:2 * N_TARGETS]
    q_bypass = [q for q in world.sample_queries(rng, 1200, 2, "fwd", require_answer=True)
                if not (set(world.answer(q).edges) & t_keys)][:300]
    q_t_rev = [Query("rev", f.obj, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets
               if world.reverse(f.relation, f.obj).answer != UNKNOWN]
    g_rng, g_world, g_store, _, _ = fresh_world(300 + seed, centre)
    q_general = [g_world.make_query(g_rng, "fwd", f.subject, [f.relation]) for f in g_world.facts[:N_GENERAL]]
    truth_general = [f.obj for f in g_world.facts[:N_GENERAL]]

    def snapshot(tag: str) -> Dict[str, float]:
        s = {
            f"{tag}/target_para_acc": accuracy(answers(model, store, world, q_t), truth_para(targets)),
            f"{tag}/target_para_unknown": unknown_rate(answers(model, store, world, q_t)),
            f"{tag}/target_hop2_unknown": unknown_rate(answers(model, store, world, q_t_hop)),
            f"{tag}/target_hop2_ref_agree": float(np.mean([a == ref.resolve(q).answer for a, q in zip(answers(model, store, world, q_t_hop), q_t_hop)])),
            f"{tag}/target_rev_unknown": unknown_rate(answers(model, store, world, q_t_rev)) if q_t_rev else float("nan"),
            f"{tag}/bypass_hop2_acc": float(np.mean([a == ref.resolve(q).answer for a, q in zip(answers(model, store, world, q_bypass), q_bypass)])),
            f"{tag}/control_para_acc": accuracy(answers(model, store, world, q_c), truth_para(controls)),
            f"{tag}/unrelated_para_acc": accuracy(answers(model, store, world, q_u), truth_para(unrelated)),
            f"{tag}/general_fresh_world_acc": accuracy(answers(model, g_store, g_world, q_general), truth_general),
        }
        return s

    m: Dict[str, Any] = {"seed": seed, "n_target_rev": len(q_t_rev), "n_target_hop2": len(q_t_hop)}
    m.update(snapshot("before"))
    # REVOKE
    for f in targets: store.revoke(kids[f.key])
    m.update(snapshot("revoke"))
    for f in targets: store.restore(kids[f.key])
    # SHRED
    for f in targets: store.shred(kids[f.key])
    m.update(snapshot("shred"))
    for f in targets: store.resign(kids[f.key])
    # UPDATE to new objects: all paraphrases must give the new object; old reverse access must vanish
    new_obj = {f.key: int((f.obj + 1 + rng.integers(0, world.n_entities - 1)) % world.n_entities) for f in targets}
    for f in targets: store.update(kids[f.key], new_obj[f.key])
    a = answers(model, store, world, q_t)
    m["update/target_para_new_obj_acc"] = accuracy(a, [new_obj[f.key] for f in targets for _ in range(world.n_synonyms)])
    m["update/target_para_old_obj_rate"] = accuracy(a, truth_para(targets))
    m["update/target_rev_old_obj_ref_agree"] = float(np.mean([x == ref.resolve(q).answer for x, q in zip(answers(model, store, world, q_t_rev), q_t_rev)])) if q_t_rev else float("nan")
    m["update/control_para_acc"] = accuracy(answers(model, store, world, q_c), truth_para(controls))
    for f in targets: store.rollback(kids[f.key], 1)
    m["rollback/target_para_acc"] = accuracy(answers(model, store, world, q_t), truth_para(targets))
    m["after_all/identical_to_before"] = float(all(abs(m[f"before/{k}"] - v) < 1e-12 for k, v in
                                                 {kk.split('/', 1)[1]: vv for kk, vv in snapshot("x").items()}.items()))
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    per_seed = [run_seed(s) for s in args.seeds]
    for s in per_seed: print(s, flush=True)
    keys = [k for k in per_seed[0] if k not in ("seed", "n_target_rev", "n_target_hop2")]
    agg = ledger.aggregate(per_seed, keys)
    record = {
        "experiment": "E-000003", "title": "Retention and generalisation of deletion",
        "evidence_level": "E4", "deletion_level": "F3",
        "claim": "REVOKE and SHRED remove the target for every paraphrase, for multi-hop reasoning through the "
                 "target and for reverse access, while controls, unrelated cells, bypass paths and a fresh world "
                 "stay at their previous accuracy; UPDATE replaces the answer for every paraphrase and rollback "
                 "restores it exactly.",
        "not_claimed": "Reconstruction resistance beyond behaviour (see E-000004) and anything beyond the synthetic "
                       "system.",
        "config": {"seeds": args.seeds, "n_targets": N_TARGETS, "n_controls": N_CONTROLS, "n_general": N_GENERAL},
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, ledger.pct(agg[k]["mean"]), ledger.pct(agg[k]["min"]), ledger.pct(agg[k]["max"])) for k in keys]
    md = "\n".join([
        "# E-000003 — Retention and generalisation of deletion", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}); deletion level **F3** (functional forgetting "
        "generalising over paraphrases, multi-hop and reverse access). Seeds: " + str(args.seeds), "",
        ledger.table(["measure", "mean", "min", "max"], rows), "",
        "Pattern required by the ledger (section 16): target high → low, control high → high. "
        "'target_para_unknown' after revoke/shred is the deletion; 'control_para_acc', 'unrelated_para_acc', "
        "'bypass_hop2_acc' and 'general_fresh_world_acc' are the retention side.",
    ])
    path = ledger.save("e000003_retention_generalization", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
