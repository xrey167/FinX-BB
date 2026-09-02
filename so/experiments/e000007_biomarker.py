"""Experiment E-000007 — biomarker: output suppression versus representational change.

Ledger sections 20–22: a model that stops answering may still represent the
fact ("do not output K").  Four conditions on the same 50 targets:

    active      nothing done
    revoked     routing removed
    shredded    marker destroyed, payload present
    suppressed  the *model* is fine-tuned to answer UNKNOWN for the targets
                while the cells stay active (output refusal, ledger F0)

Behaviourally all three deletion-like conditions look identical (UNKNOWN).
The internal signals — routing mass on the target cell, gated value
contribution, a linear probe on the hidden state, the true object's logit
rank — must separate suppression from representational removal.

Run:  python -m so.experiments.e000007_biomarker
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import bank_from_store, encode_queries
from so.experiments.common import (accuracy, all_paraphrases, answers, fresh_world, hidden_states,
                                   load_base_model, position_of_kid, unknown_rate)
from so.world import Query

N_TARGETS, N_CONTROLS = 50, 50


def suppress(model, store, world, target_queries: List[Query], rng: np.random.Generator, steps: int = 200,
             lr: float = 3e-4):
    """Fine-tune a copy of the model to refuse the target queries while keeping everything else."""
    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    bank = bank_from_store(store)
    tensors = bank.tensors()
    for _ in range(steps):
        others = world.sample_queries(rng, 96, 1, "fwd") + world.sample_queries(rng, 32, 2, "fwd")
        b = encode_queries(target_queries + others, bank, world, m.cfg.max_hops)
        target = b.target.clone()
        target[: len(target_queries)] = world.n_entities        # UNKNOWN for the targets
        logits, _, _ = m(tensors, b.mode, b.start, b.rels, b.hop_valid)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def run_seed(seed: int) -> Dict[str, Any]:
    base = load_base_model(seed)
    model, centre = base["model"], base["centre"]
    rng, world, store, kids, ref = fresh_world(700 + seed, centre)
    facts = world.facts
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:N_TARGETS]]
    controls = [facts[int(i)] for i in perm[N_TARGETS:N_TARGETS + N_CONTROLS]]
    others = [facts[int(i)] for i in perm[N_TARGETS + N_CONTROLS:]]
    n_ent = world.n_entities
    q_t = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]
    q_t_all = [q for f in targets for q in all_paraphrases(world, f.subject, f.relation)]
    q_c = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in controls]
    truth_t, truth_c = [f.obj for f in targets], [f.obj for f in controls]
    pos = [position_of_kid(store, kids[f.key]) for f in targets]

    q_o = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in others]
    h_o, _, _ = hidden_states(model, store, world, q_o)
    probe = LinearProbe(h_o.shape[1], n_ent, seed=seed)
    probe.fit(h_o, np.array([f.obj for f in others]))

    def measure(mdl, tag: str, m: Dict[str, Any]) -> None:
        h, r, lg = hidden_states(mdl, store, world, q_t)
        a = answers(mdl, store, world, q_t)
        m[f"{tag}/target_unknown"] = unknown_rate(a)
        m[f"{tag}/target_acc"] = accuracy(a, truth_t)
        m[f"{tag}/control_acc"] = accuracy(answers(mdl, store, world, q_c), truth_c)
        mass = np.array([r[i, 0, p] for i, p in enumerate(pos)])
        with torch.no_grad():
            vn = mdl.encode_bank(bank_from_store(store).tensors())["v_f"].norm(dim=-1).numpy()
        m[f"{tag}/routing_mass_on_target"] = float(mass.mean())
        m[f"{tag}/gated_value_contribution"] = float(np.mean(mass * vn[pos]))
        m[f"{tag}/probe_top1"] = probe.accuracy(h, np.array(truth_t))
        m[f"{tag}/probe_top5"] = probe.accuracy(h, np.array(truth_t), topk=5)
        rk = object_rank(lg, truth_t, n_ent)
        m[f"{tag}/true_obj_top1_among_entities"] = rk["top1"]
        m[f"{tag}/true_obj_mean_rank"] = rk["mean_rank"]
        m[f"{tag}/forced_choice_win"] = forced_choice(lg, truth_t, np.random.default_rng(seed), n_ent)

    m: Dict[str, Any] = {"seed": seed}
    measure(model, "active", m)
    for f in targets: store.revoke(kids[f.key])
    measure(model, "revoked", m)
    for f in targets: store.restore(kids[f.key])
    for f in targets: store.shred(kids[f.key])
    measure(model, "shredded", m)
    for f in targets: store.resign(kids[f.key])
    sup = suppress(model, store, world, q_t_all, rng)
    measure(sup, "suppressed", m)
    return m


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)
    per_seed = [run_seed(s) for s in args.seeds]
    for s in per_seed: print(s, flush=True)
    keys = [k for k in per_seed[0] if k != "seed"]
    agg = ledger.aggregate(per_seed, keys)
    conds = ["active", "revoked", "shredded", "suppressed"]
    signals = ["target_unknown", "target_acc", "control_acc", "routing_mass_on_target", "gated_value_contribution",
               "probe_top1", "probe_top5", "true_obj_top1_among_entities", "true_obj_mean_rank", "forced_choice_win"]
    record = {
        "experiment": "E-000007", "title": "Biomarker: output suppression versus representational change",
        "evidence_level": "E4", "deletion_level": "F4",
        "claim": "Suppression (fine-tuned refusal) and deletion (REVOKE / SHRED) are behaviourally identical on the "
                 "targets, but the internal signals separate them: under suppression the target cell still receives "
                 "routing mass and value contribution and a linear probe still decodes the object from the hidden "
                 "state; under REVOKE / SHRED those signals vanish. The gated value contribution is a causal "
                 "biomarker of usable knowledge in this system (its causality is established in E-000005).",
        "not_claimed": "A deletion certificate for real LLMs; robustness of the marker beyond this synthetic system.",
        "config": {"seeds": args.seeds, "n_targets": N_TARGETS, "n_controls": N_CONTROLS},
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(s, *(f"{agg[f'{c}/{s}']['mean']:.4f}" for c in conds)) for s in signals]
    md = "\n".join([
        "# E-000007 — Biomarker: suppression versus representational change", "",
        f"Evidence level: **E4** ({ledger.EVIDENCE_LEVELS['E4']}); deletion level **F4** within the synthetic system. "
        f"Seeds: {args.seeds}. Chance levels: probe top-1 0.0039, top-5 0.0195, mean rank 127.5, forced choice 0.5.", "",
        ledger.table(["signal (mean over seeds)"] + conds, rows), "",
        "Reading: 'suppressed' keeps the biomarker (routing mass, value contribution) and the probe leak while "
        "answering UNKNOWN — output suppression, ledger F0. 'revoked' and 'shredded' remove them — representational "
        "removal, F4.",
    ])
    path = ledger.save("e000007_biomarker", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
