"""The reconstruction-attack battery (shared by E-000004 and E-000009).

Given a trained model and a fresh world, 100 target cells are attacked before
and after REVOKE and SHRED with: direct query, paraphrase, multi-hop, reverse,
forced choice, logit rank, a linear representation probe (calibrated on active
non-target cells), and the activation probe (routing mass and gated value
contribution of the target cell).  Gate statistics for signed and unsigned
markers are recorded as well.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import bank_from_store
from so.experiments.common import accuracy, answers, fresh_world, hidden_states, position_of_kid, unknown_rate
from so.world import Query, UNKNOWN

N_TARGETS = 100


def value_norms(model, store) -> np.ndarray:
    with torch.no_grad():
        return model.encode_bank(bank_from_store(store).tensors())["v_f"].norm(dim=-1).numpy()


def gate_stats(model, store) -> Dict[str, float]:
    """Mean gate for signed and for unsigned markers in the current bank (soft gate, as trained)."""
    b = bank_from_store(store)
    with torch.no_grad():
        g = model.gate(torch.as_tensor(b.marker)).squeeze(-1).numpy()
    valid = b.marker_valid
    return {"gate_valid_mean": float(g[valid].mean()) if valid.any() else float("nan"),
            "gate_invalid_mean": float(g[~valid].mean()) if (~valid).any() else float("nan"),
            "gate_invalid_max": float(g[~valid].max()) if (~valid).any() else float("nan")}


def attack_battery(model, centre: np.ndarray, seed: int, world_seed: int, n_targets: int = N_TARGETS) -> Dict[str, Any]:
    rng, world, store, kids, ref = fresh_world(world_seed, centre)
    facts = world.facts
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:n_targets]]
    others = [facts[int(i)] for i in perm[n_targets:]]
    n_ent = world.n_entities

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
        vn = value_norms(model, store)
        m[f"{tag}/routing_mass_on_target"] = float(mass.mean())
        m[f"{tag}/gated_value_contribution"] = float(np.mean(mass * vn[pos]))
        for k, v in gate_stats(model, store).items():
            m[f"{tag}/{k}"] = v

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


ATTACK_ROWS = ["direct_unknown", "direct_acc", "paraphrase_unknown", "multihop_unknown", "reverse_unknown",
               "forced_choice_win", "true_obj_top1_among_entities", "true_obj_mean_rank", "probe_top1", "probe_top5",
               "routing_mass_on_target", "gated_value_contribution", "gate_valid_mean", "gate_invalid_mean", "gate_invalid_max"]
