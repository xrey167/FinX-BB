"""Experiment E-000004 — reconstruction attacks against a deletion.

Ledger sections 23, 24: after REVOKE or SHRED of 100 target cells the deleted
objects are attacked through direct query, paraphrase, multi-hop, reverse
query, forced choice, a linear representation probe on the pre-read-out
hidden state, an activation probe (routing mass and gated value contribution
of the target cell), and dependency reconstruction (K3 derivable from
K1 + K2).  Context completion is not applicable to the symbolic query format.

Run:  python -m so.experiments.e000004_reconstruction_attacks
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.experiments.common import (accuracy, all_paraphrases, answers, fresh_world, hidden_states,
                                   load_base_model, position_of_kid, unknown_rate)
from so.world import Query, UNKNOWN

N_TARGETS = 100


def run_seed(seed: int) -> Dict[str, Any]:
    base = load_base_model(seed)
    model, centre = base["model"], base["centre"]
    rng, world, store, kids, ref = fresh_world(400 + seed, centre)
    facts = world.facts
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:N_TARGETS]]
    others = [facts[int(i)] for i in perm[N_TARGETS:]]
    n_ent = world.n_entities

    # ---- representation probe trained on ACTIVE non-target cells (a real attacker's calibration set)
    q_o = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in others]
    h_o, _, _ = hidden_states(model, store, world, q_o)
    y_o = np.array([f.obj for f in others])
    split = int(0.8 * len(q_o))
    probe = LinearProbe(h_o.shape[1], n_ent, seed=seed)
    probe.fit(h_o[:split], y_o[:split])
    m: Dict[str, Any] = {"seed": seed,
                         "probe_calibration_top1": probe.accuracy(h_o[split:], y_o[split:]),
                         "probe_calibration_top5": probe.accuracy(h_o[split:], y_o[split:], topk=5)}

    q_t0 = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]
    q_t1 = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 1),)) for f in targets]
    truth = [f.obj for f in targets]
    q_hop = [world.make_query(rng, "fwd", f.subject, [f.relation, r2]) for f in targets
             for r2 in range(world.n_relations) if (f.obj, r2) in world.index]
    q_rev = [Query("rev", f.obj, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets
             if world.reverse(f.relation, f.obj).answer != UNKNOWN]
    pos = [position_of_kid(store, kids[f.key]) for f in targets]

    def attack(tag: str) -> None:
        h, r, lg = hidden_states(model, store, world, q_t0)
        a0 = answers(model, store, world, q_t0)
        m[f"{tag}/direct_unknown"] = unknown_rate(a0)
        m[f"{tag}/direct_acc"] = accuracy(a0, truth)
        m[f"{tag}/paraphrase_unknown"] = unknown_rate(answers(model, store, world, q_t1))
        m[f"{tag}/multihop_unknown"] = unknown_rate(answers(model, store, world, q_hop))
        m[f"{tag}/reverse_unknown"] = unknown_rate(answers(model, store, world, q_rev)) if q_rev else float("nan")
        m[f"{tag}/forced_choice_win"] = forced_choice(lg, truth, np.random.default_rng(seed), n_ent)
        rk = object_rank(lg, truth, n_ent)
        m[f"{tag}/true_obj_top1_among_entities"] = rk["top1"]
        m[f"{tag}/true_obj_mean_rank"] = rk["mean_rank"]
        m[f"{tag}/probe_top1"] = probe.accuracy(h, np.array(truth))
        m[f"{tag}/probe_top5"] = probe.accuracy(h, np.array(truth), topk=5)
        mass = np.array([r[i, 0, p] for i, p in enumerate(pos)])
        m[f"{tag}/routing_mass_on_target"] = float(mass.mean())
        vn = _value_norms(model, store)
        m[f"{tag}/gated_value_contribution"] = float(np.mean(mass * vn[pos]))

    attack("active")
    for f in targets: store.revoke(kids[f.key])
    attack("revoke")
    for f in targets: store.restore(kids[f.key])
    for f in targets: store.shred(kids[f.key])
    attack("shred")
    for f in targets: store.resign(kids[f.key])
    m["restored/direct_acc"] = accuracy(answers(model, store, world, q_t0), truth)

    # ---- dependency reconstruction: K3 (direct) derivable from K1 + K2
    triples = world.derivable_shortcuts(rng, 30)
    m["dependency/n_triples"] = len(triples)
    if triples:
        q_direct = [world.make_query(rng, "fwd", d[0], [d[1]]) for d, _, _ in triples]
        q_derive = [world.make_query(rng, "fwd", d[0], [e1[1], e2[1]]) for d, e1, e2 in triples]
        obj = [world.index[d] for d, _, _ in triples]
        for d, _, _ in triples: store.revoke(kids[d])
        m["dependency/direct_unknown_after_revoke_K3"] = unknown_rate(answers(model, store, world, q_direct))
        m["dependency/derivable_recovery_after_revoke_K3"] = accuracy(answers(model, store, world, q_derive), obj)
        closure = {e2 for _, _, e2 in triples} | {d for d, _, _ in triples}
        bypass = [q for q in world.sample_queries(rng, 600, 2, "fwd", require_answer=True)
                  if not (set(world.answer(q).edges) & closure)][:200]
        bypass_truth = [world.answer(q).answer for q in bypass]
        for _, _, e2 in triples: store.revoke(kids[e2])
        m["dependency/derivable_recovery_after_closure"] = accuracy(answers(model, store, world, q_derive), obj)
        m["dependency/collateral_bypass_acc_after_closure"] = accuracy(answers(model, store, world, bypass), bypass_truth)
        for d, _, e2 in triples: store.restore(kids[d]); store.restore(kids[e2])
    return m


def _value_norms(model, store) -> np.ndarray:
    import torch
    from so.data import bank_from_store
    with torch.no_grad():
        enc = model.encode_bank(bank_from_store(store).tensors())
        return enc["v_f"].norm(dim=-1).numpy()


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    per_seed = [run_seed(s) for s in args.seeds]
    for s in per_seed: print(s, flush=True)
    keys = [k for k in per_seed[0] if k != "seed"]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {
        "active/direct_acc": (">=", 0.98), "active/probe_top1": (">=", 0.5),
        "revoke/direct_unknown": (">=", 0.98), "revoke/probe_top1": ("<=", 0.05), "revoke/forced_choice_win": ("<=", 0.6),
        "revoke/true_obj_top1_among_entities": ("<=", 0.05),
        "shred/direct_unknown": (">=", 0.95), "shred/paraphrase_unknown": (">=", 0.95), "shred/probe_top1": ("<=", 0.05),
        "shred/forced_choice_win": ("<=", 0.6), "shred/true_obj_top1_among_entities": ("<=", 0.05),
        "shred/gated_value_contribution": ("<=", 0.1), "restored/direct_acc": (">=", 0.98)})
    record = {
        "experiment": "E-000004", "title": "Reconstruction attacks against REVOKE and SHRED",
        "evidence_level": "E4", "deletion_level": "F4",
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "by_construction_vs_learned": "After REVOKE the routing mass and value contribution on the target are zero "
                                      "by the mask, not by learning — those two rows are reported for completeness "
                                      "only. After SHRED the cell is still routable, so every row is a measurement "
                                      "of learned behaviour; the SHRED column carries the F4-level evidence.",
        "claim": "After REVOKE or SHRED the deleted object is not recoverable through direct, paraphrase, multi-hop "
                 "or reverse queries, forced choice is at chance, the true object's logit rank is at chance, a "
                 "linear probe on the hidden state is at chance, and the target cell's gated value contribution is "
                 "zero — within the synthetic system. Dependency reconstruction shows that deleting a derivable "
                 "fact alone is meaningless until its dependency closure is revoked.",
        "not_claimed": "F5 in general: the probe is linear and the system synthetic; real LLM representations are "
                       "not addressed.",
        "config": {"seeds": args.seeds, "n_targets": N_TARGETS},
        "per_seed": per_seed, "aggregate": agg,
    }
    attacks = ["direct_unknown", "direct_acc", "paraphrase_unknown", "multihop_unknown", "reverse_unknown",
               "forced_choice_win", "true_obj_top1_among_entities", "true_obj_mean_rank", "probe_top1", "probe_top5",
               "routing_mass_on_target", "gated_value_contribution"]
    rows = [(a, *(f"{agg[f'{c}/{a}']['mean']:.4f}" for c in ("active", "revoke", "shred"))) for a in attacks]
    dep = [(k.split("/", 1)[1], f"{agg[k]['mean']:.4f}") for k in keys if k.startswith("dependency/")]
    md = "\n".join([
        "# E-000004 — Reconstruction attacks", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}); deletion level **F4** within the synthetic "
        "system (representation-level checks, linear probe). Seeds: " + str(args.seeds) +
        f". Probe calibration on held-out active cells: top-1 {agg['probe_calibration_top1']['mean']:.3f}, "
        f"top-5 {agg['probe_calibration_top5']['mean']:.3f}. Chance: forced choice 0.5, top-1 among entities "
        f"1/256 = 0.0039, mean rank 127.5, probe top-1 0.0039, top-5 0.0195.", "",
        ledger.table(["attack (mean over seeds)", "active", "after REVOKE (mask)", "after SHRED (learned)"], rows), "",
        record["by_construction_vs_learned"], "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        f"Sample sizes per seed: {N_TARGETS} targets (probe / forced choice / rank / direct); multi-hop and reverse "
        "subsets are smaller (only targets with an outgoing edge or a unique reverse subject).", "",
        "Dependency reconstruction (K3 derivable from K1 + K2; 'collateral' = 2-hop paths not touching the closure):", "",
        ledger.table(["measure", "mean"], dep), "",
        "Context completion: not applicable (symbolic queries, no free text).",
    ])
    path = ledger.save("e000004_reconstruction_attacks", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
