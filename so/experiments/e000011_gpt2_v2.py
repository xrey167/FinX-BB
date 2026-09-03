"""Experiment E-000011 — frozen GPT-2 core, version 2: verified gate, deletion behaviour, held-out paraphrases,
causal interventions inside the pretrained model.

Changes against E-000008 (whose record stands):
    * class-balanced verification loss on the adapter's marker gate (the E-000010 remedy),
    * a larger share of revoked / shredded / unanswerable queries during training,
    * 3,000 adapter steps with the routing-first curriculum,
    * two extra sentence templates per relation that are NEVER used in training and are
      evaluated as held-out paraphrases (ledger §3 "generalisation": deletion must hold on
      surface forms the model has not seen),
    * causal interventions on 2-hop questions inside the frozen model (disable the first- or
      second-hop cell, disable a random other cell, swap / replace the payload, localisation).

Run:  python -m so.experiments.e000011_gpt2_v2 [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, bank_from_store, bank_from_world, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Query, UNKNOWN, World, fill_random, inject_alternative_paths

TEMPLATES6 = {
    0: ["{s} lives in", "The home of {s} is", "{s} resides in", "The residence of {s} is",
        "Q: Where does {s} live? A: In", "According to the records, {s} lives in"],
    1: ["{s} works for", "The employer of {s} is", "{s} is employed by", "The company that employs {s} is",
        "Q: Who does {s} work for? A:", "According to the records, {s} works for"],
    2: ["{s} is married to", "The spouse of {s} is", "{s} is wed to", "The partner of {s} is",
        "Q: Who is {s} married to? A:", "According to the records, {s} is married to"],
    3: ["{s} was born in", "The birthplace of {s} is", "{s} comes from", "The place of birth of {s} is",
        "Q: Where was {s} born? A: In", "According to the records, {s} was born in"],
}
TEMPLATES4 = TEMPLATES6
TRAIN_TEMPLATES, HELDOUT_TEMPLATES = (0, 1), (2, 3, 4, 5)   # 2/3 lexical variants, 4 question form, 5 prefixed clause
E8.TEMPLATES = TEMPLATES6          # training only ever draws indices 0 and 1 (n_synonyms = 2)


def train_adapter_v2(gk: E8.GPT2Knowledge, seed: int, steps: int, batch_size: int = 32, route_weight: float = 1.0,
                     gate_weight: float = 5.0, lr: float = 2e-3, route_only_steps: int = 300, p_revoked: float = 0.20,
                     p_shred: float = 0.10, extra_unanswerable: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, gk.model.cfg.marker_dim)
    model = gk.model
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=50)
    mix = {"fwd1": 0.7, "fwd2": 0.3}
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    n_extra = int(round(batch_size * extra_unanswerable))
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_cells = int(rng.integers(150, 301)) if route_only else int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, gk.n_synonyms)
        bank = bank_from_world(rng, world, centre, p_revoked, p_shred, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [E8.query_text(q, gk.names, gk.n_synonyms) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = E8.route_targets(queries, bank, world, len(model.cfg.read_layers))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid, reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        loss = loss_route + gate_weight * loss_gate if route_only else loss_ans + route_weight * loss_route + gate_weight * loss_gate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "gate_loss": float(loss_gate.item()), "batch_acc": acc,
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {step + 1:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  route {rec['route_loss']:.4f}  "
                      f"gate {rec['gate_loss']:.4f}  acc {acc:.3f}  {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000011_gpt2_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"], "loaded": True,
                "checkpoint_sha256": _sha256(path)}
    out = train_adapter_v2(gk, seed, steps)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["loaded"] = False
    out["checkpoint_sha256"] = _sha256(path)
    return out


EVAL = dict(n_cells=1000, n_alt_structures=25, n_hop2=300, n_broken=100, n_lifecycle=100, n_locality_updates=100,
            n_locality_revokes=50, n_targets=100, n_interventions=100)


def evaluate(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    empty = World(gk.n_entities, 4, gk.n_synonyms, [])
    world = fill_random(rng, inject_alternative_paths(rng, empty, EVAL["n_alt_structures"]), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    ref = ReferenceResolver(store)
    facts = world.facts
    n_ent = gk.n_entities
    m: Dict[str, Any] = {"seed": seed}

    def bank() -> Bank:
        return bank_from_store(store)

    def q1(f) -> Query:
        return Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),))

    direct = [q1(f) for f in facts]
    truth = np.array([f.obj for f in facts])

    prior = gk.predict(None, world, direct)
    m["prior_direct_acc"] = float((prior["answers"] == truth).mean())
    masked = gk.predict(bank(), world, direct, cell_mask=np.zeros(len(facts), dtype=bool))
    m["bank_masked_direct_acc"] = float((masked["answers"] == truth).mean())
    m["bank_masked_unknown_rate"] = float((masked["answers"] == UNKNOWN).mean())

    for t in TRAIN_TEMPLATES + HELDOUT_TEMPLATES:
        p = gk.predict(bank(), world, direct, template=t)
        tag = f"template{t}_" + ("train" if t in TRAIN_TEMPLATES else "heldout")
        m[f"{tag}/direct"] = float((p["answers"] == truth).mean())
        m[f"{tag}/full_vocab_top1"] = float((p["full_top1"] == np.array(gk.entity_ids)[truth]).mean())
    m["direct"] = m["template0_train/direct"]
    m["direct_heldout_mean"] = float(np.mean([m[f"template{t}_heldout/direct"] for t in HELDOUT_TEMPLATES]))
    p0 = gk.predict(bank(), world, direct, template=0)
    prov = 0
    for i, f in enumerate(facts):
        r = p0["routing"][i, -1]; k = int(r.argmax())
        prov += int(k < len(facts) and int(store.bank()["kid"][k]) == kids[f.key] and r[k] > 0.5)
    m["provenance_direct"] = prov / len(facts)
    hop2 = world.sample_queries(rng, EVAL["n_hop2"], 2, "fwd", require_answer=True)
    ph = gk.predict(bank(), world, hop2)
    m["hop2"] = float(np.mean([a == ref.resolve(q).answer for a, q in zip(ph["answers"], hop2)]))
    for hops in (1, 2):
        broken = world.sample_queries(rng, EVAL["n_broken"], hops, "fwd", require_answer=False)
        m[f"broken{hops}_unknown"] = float((gk.predict(bank(), world, broken)["answers"] == UNKNOWN).mean())

    # ---- lifecycle, batched per operation, on train and held-out templates
    cells = [facts[int(i)] for i in rng.choice(len(facts), size=EVAL["n_lifecycle"], replace=False)]
    q_life = [q1(f) for f in cells]
    new_objs = {f.key: int((f.obj + 1 + rng.integers(0, n_ent - 1)) % n_ent) for f in cells}

    def check(name: str) -> None:
        for t in TRAIN_TEMPLATES + HELDOUT_TEMPLATES:
            a = gk.predict(bank(), world, q_life, template=t)["answers"]
            m[f"{name}/template{t}"] = float(np.mean([x == ref.resolve(q).answer for x, q in zip(a, q_life)]))
        m[name] = m[f"{name}/template0"]
        m[f"{name}_heldout"] = float(np.mean([m[f"{name}/template{t}"] for t in HELDOUT_TEMPLATES]))
        m[f"{name}_heldout_min"] = float(min(m[f"{name}/template{t}"] for t in HELDOUT_TEMPLATES))

    for f in cells: store.update(kids[f.key], new_objs[f.key])
    check("update")
    for f in cells: store.rollback(kids[f.key], 1)
    check("rollback")
    for f in cells: store.revoke(kids[f.key])
    check("revoke")
    for f in cells: store.restore(kids[f.key])
    for f in cells: store.shred(kids[f.key])
    check("shred")
    for f in cells: store.resign(kids[f.key])
    check("resign")

    # ---- locality
    snapshot = gk.predict(bank(), world, direct)["answers"]
    n_t = EVAL["n_locality_updates"] + EVAL["n_locality_revokes"]
    t_idx = rng.choice(len(facts), size=n_t, replace=False)
    t_keys = {facts[int(i)].key for i in t_idx}
    for j, i in enumerate(t_idx):
        f = facts[int(i)]
        store.update(kids[f.key], int((f.obj + 1) % n_ent)) if j < EVAL["n_locality_updates"] else store.revoke(kids[f.key])
    after = gk.predict(bank(), world, direct)["answers"]
    outside = np.array([f.key not in t_keys for f in facts])
    ref_after = np.array([ref.resolve(q).answer for q in direct])
    m["locality"] = float((snapshot[outside] == after[outside]).mean())
    m["locality_targets_correct"] = float((after[~outside] == ref_after[~outside]).mean())
    for j, i in enumerate(t_idx):
        f = facts[int(i)]
        store.rollback(kids[f.key], 1) if j < EVAL["n_locality_updates"] else store.restore(kids[f.key])
    m["locality_undo_exact"] = float(np.array_equal(gk.predict(bank(), world, direct)["answers"], snapshot))

    # ---- attacks on 100 targets: soft and hard gate; train and held-out templates
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[: EVAL["n_targets"]]]
    others = [facts[int(i)] for i in perm[EVAL["n_targets"]:]]
    q_t = [q1(f) for f in targets]
    t_truth = [f.obj for f in targets]
    q_o = [q1(f) for f in others]
    po = gk.predict(bank(), world, q_o)
    y_o = np.array([f.obj for f in others]); split = int(0.8 * len(others))
    probe = LinearProbe(po["hidden"].shape[1], n_ent, seed=seed)
    probe.fit(po["hidden"][:split], y_o[:split])
    m["probe_calibration_top1"] = probe.accuracy(po["hidden"][split:], y_o[split:])
    pos = [int(np.where(store.bank()["kid"] == kids[f.key])[0][0]) for f in targets]
    prior_t = gk.predict(None, world, q_t)["full_top1"]
    entity_id_set = set(int(i) for i in gk.entity_ids)

    def attack(tag: str) -> None:
        p = gk.predict(bank(), world, q_t)
        m[f"{tag}/direct_unknown"] = float((p["answers"] == UNKNOWN).mean())
        m[f"{tag}/direct_acc"] = float((p["answers"] == np.array(t_truth)).mean())
        # what does the model actually emit over the full vocabulary?
        ft = p["full_top1"]
        m[f"{tag}/full_vocab_is_unknown_word"] = float((ft == gk.unknown_id).mean())
        m[f"{tag}/full_vocab_is_true_object"] = float((ft == np.array(gk.entity_ids)[t_truth]).mean())
        m[f"{tag}/full_vocab_is_other_entity"] = float(np.mean([int(x) in entity_id_set and int(x) != gk.entity_ids[t] and int(x) != gk.unknown_id
                                                                for x, t in zip(ft, t_truth)]))
        m[f"{tag}/full_vocab_equals_prior"] = float((ft == prior_t).mean())
        m[f"{tag}/full_vocab_is_non_entity_token"] = float(np.mean([int(x) not in entity_id_set and int(x) != gk.unknown_id for x in ft]))
        m[f"{tag}/candidate_other_entity"] = float(np.mean([(a != UNKNOWN) and (a != t) for a, t in zip(p["answers"], t_truth)]))
        for t in HELDOUT_TEMPLATES:
            ph2 = gk.predict(bank(), world, q_t, template=t)
            m[f"{tag}/heldout{t}_unknown"] = float((ph2["answers"] == UNKNOWN).mean())
            m[f"{tag}/heldout{t}_acc"] = float((ph2["answers"] == np.array(t_truth)).mean())
        m[f"{tag}/forced_choice_win"] = forced_choice(p["logits"], t_truth, np.random.default_rng(seed), n_ent)
        rk = object_rank(p["logits"], t_truth, n_ent)
        m[f"{tag}/true_obj_top1_among_entities"] = rk["top1"]; m[f"{tag}/true_obj_mean_rank"] = rk["mean_rank"]
        m[f"{tag}/probe_top1"] = probe.accuracy(p["hidden"], np.array(t_truth))
        mass = np.array([p["routing"][i, -1, pp] for i, pp in enumerate(pos)])
        with torch.no_grad():
            enc = gk.model.encode_bank(bank().tensors())
            gate = enc["gate"].numpy()
        m[f"{tag}/routing_mass_on_target"] = float(mass.mean())
        m[f"{tag}/gate_on_target"] = float(gate[pos].mean())
        m[f"{tag}/payload_share"] = float(np.mean(mass * gate[pos]))     # biomarker: mass x gate (payload actually selected)
        m[f"{tag}/gated_value_contribution"] = m[f"{tag}/payload_share"]  # name kept for the shared criteria

    attack("active")
    for f in targets: store.revoke(kids[f.key])
    attack("revoke")
    for f in targets: store.restore(kids[f.key])
    for f in targets: store.shred(kids[f.key])
    attack("shred_soft")
    gk.model.cfg.hard_gate = True
    attack("shred_hard")
    gk.model.cfg.hard_gate = False
    for f in targets: store.resign(kids[f.key])
    m["restored/direct_acc"] = float((gk.predict(bank(), world, q_t)["answers"] == np.array(t_truth)).mean())

    # ---- in-context comparators for 2-hop (frozen model WITHOUT adapter, facts stated in the prompt)
    hop2c = [q for q in world.sample_queries(rng, 2 * EVAL["n_hop2"], 2, "fwd", require_answer=True)][: EVAL["n_hop2"]]
    def stated(sub: int, rel: int, obj: int) -> str:
        return TEMPLATES6[rel][0].format(s=gk.names[sub]) + gk.names[obj] + "."
    texts_both, texts_first, truth_c = [], [], []
    for q in hop2c:
        gt = world.answer(q)
        (s1, r1), (s2, r2) = gt.edges
        o1, o2 = world.index[(s1, r1)], world.index[(s2, r2)]
        question = E8.query_text(q, gk.names, gk.n_synonyms)
        texts_both.append(f"{stated(s1, r1, o1)} {stated(s2, r2, o2)} {question}")
        texts_first.append(f"{stated(s1, r1, o1)} {question}")
        truth_c.append(o2)
    with torch.no_grad():
        for tag, texts in (("in_context_both_facts", texts_both), ("in_context_first_fact_only", texts_first)):
            hits = 0
            for i in range(0, len(texts), 32):
                ids, am, last = E8.encode_texts(gk.tok, texts[i:i + 32])
                cand, _, _, _ = gk.model(None, ids, am, last)
                a = cand.argmax(-1).numpy()
                hits += int(sum(int(x == t) for x, t in zip(a, truth_c[i:i + 32])))
            m[f"comparator/{tag}_hop2_acc"] = hits / len(texts)
    m["comparator/adapter_no_context_hop2_acc"] = float(np.mean([a == ref.resolve(q).answer for a, q in zip(gk.predict(bank(), world, hop2c)["answers"], hop2c)]))

    # ---- alternative routes: revoking one edge breaks only the route that uses it
    pairs = world.alternative_path_pairs(rng, 50)
    alt_ok_a = alt_ok_b = 0
    for qa, qb, edge in pairs:
        target = world.answer(qa).answer
        store.revoke(kids[edge])
        a = gk.predict(bank(), world, [qa, qb])["answers"]
        alt_ok_a += int(a[0] != target); alt_ok_b += int(a[1] == target)
        store.restore(kids[edge])
    m["alt_route/n_pairs"] = len(pairs)
    m["alt_route/broken_route_changes"] = alt_ok_a / len(pairs) if pairs else float("nan")
    m["alt_route/other_route_survives"] = alt_ok_b / len(pairs) if pairs else float("nan")

    # ---- causal interventions on 2-hop questions inside the frozen model
    hop2q = list(world.sample_queries(rng, 4 * EVAL["n_interventions"], 2, "fwd", require_answer=True))   # pool
    base = gk.predict(bank(), world, hop2q)
    correct = np.array([a == ref.resolve(q).answer for a, q in zip(base["answers"], hop2q)])
    m["interventions/pool_size"] = len(hop2q)
    m["interventions/pool_correct_rate"] = float(correct.mean()) if len(hop2q) else float("nan")
    c = {k: 0 for k in ("disable_hop1_changes", "disable_hop1_unknown", "disable_hop2_changes", "disable_hop2_unknown",
                        "disable_random_unchanged", "localisation_hop1", "localisation_hop2", "swap_hop2", "replace_hop2")}
    n_used = 0
    n_swap = 0
    bank_kids = store.bank()["kid"]
    for i, q in enumerate(hop2q):
        if not correct[i]:
            continue
        if n_used >= EVAL["n_interventions"]:
            break
        n_used += 1
        gt = world.answer(q)
        k1, k2 = kids[gt.edges[0]], kids[gt.edges[1]]
        p1, p2 = int(np.where(bank_kids == k1)[0][0]), int(np.where(bank_kids == k2)[0][0])
        r = base["routing"][i]
        c["localisation_hop1"] += int(int(r[0].argmax()) == p1)
        c["localisation_hop2"] += int(int(r[1].argmax()) == p2)
        for name, p in (("disable_hop1", p1), ("disable_hop2", p2)):
            mask = np.ones(len(facts), dtype=bool); mask[p] = False
            a = int(gk.predict(bank(), world, [q], cell_mask=mask)["answers"][0])
            c[f"{name}_changes"] += int(a != gt.answer); c[f"{name}_unknown"] += int(a == UNKNOWN)
        other = int(rng.choice([j for j in range(len(facts)) if j not in (p1, p2)]))
        mask = np.ones(len(facts), dtype=bool); mask[other] = False
        c["disable_random_unchanged"] += int(int(gk.predict(bank(), world, [q], cell_mask=mask)["answers"][0]) == gt.answer)
        partner = facts[int(rng.integers(0, len(facts)))]
        if partner.key not in (gt.edges[0], gt.edges[1]) and partner.obj != gt.answer:
            n_swap += 1
            store.swap(k2, kids[partner.key])
            c["swap_hop2"] += int(int(gk.predict(bank(), world, [q])["answers"][0]) == partner.obj)
            store.swap(k2, kids[partner.key])
        new = int((gt.answer + 1 + rng.integers(0, n_ent - 1)) % n_ent)
        old = store.read(k2).obj
        store.replace(k2, new)
        c["replace_hop2"] += int(int(gk.predict(bank(), world, [q])["answers"][0]) == new)
        store.replace(k2, old)
    m["interventions/n_correct_hop2"] = n_used
    m["interventions/n_swap"] = n_swap
    for k, v in c.items():
        denom = n_swap if k == "swap_hop2" else n_used
        m[f"interventions/{k}"] = v / denom if denom else float("nan")
    return m


KEYS = ["prior_direct_acc", "bank_masked_direct_acc", "direct", "template0_train/full_vocab_top1", "template1_train/direct",
        "template2_heldout/direct", "template3_heldout/direct", "template4_heldout/direct", "template5_heldout/direct",
        "direct_heldout_mean", "provenance_direct", "hop2", "comparator/in_context_both_facts_hop2_acc",
        "comparator/in_context_first_fact_only_hop2_acc", "comparator/adapter_no_context_hop2_acc", "broken1_unknown",
        "broken2_unknown", "update", "rollback", "revoke", "shred", "resign", "update_heldout", "revoke_heldout", "revoke_heldout_min",
        "shred_heldout", "shred_heldout_min", "locality", "locality_targets_correct", "alt_route/broken_route_changes",
        "alt_route/other_route_survives", "interventions/pool_correct_rate"]
ATT = ["direct_unknown", "direct_acc", "candidate_other_entity", "full_vocab_is_unknown_word", "full_vocab_is_true_object",
       "full_vocab_is_other_entity", "full_vocab_equals_prior", "full_vocab_is_non_entity_token", "heldout2_unknown",
       "heldout4_unknown", "forced_choice_win", "true_obj_top1_among_entities", "true_obj_mean_rank", "probe_top1",
       "routing_mass_on_target", "gate_on_target", "payload_share"]
INT = ["localisation_hop1", "localisation_hop2", "disable_hop1_changes", "disable_hop1_unknown", "disable_hop2_changes",
       "disable_hop2_unknown", "disable_random_unchanged", "swap_hop2", "replace_hop2"]


def criteria_groups():
    """Pre-registered criteria, grouped per claim (shared with E-000012, which changes the design, not the bar)."""
    # STRICT thresholds = the ones pre-registered for E-000008 (kept unchanged so that nothing is relaxed after seeing E-000008)
    groups = {
        "reading": {"prior_direct_acc": ("<=", 0.05), "bank_masked_direct_acc": ("<=", 0.05), "direct": (">=", 0.95),
                    "template1_train/direct": (">=", 0.95)},
        "heldout_paraphrases": {"template2_heldout/direct": (">=", 0.80), "template3_heldout/direct": (">=", 0.80),
                                "template4_heldout/direct": (">=", 0.70), "template5_heldout/direct": (">=", 0.70)},
        "update_rollback": {"update": (">=", 0.95), "rollback": (">=", 0.95), "resign": (">=", 0.95)},
        "deletion_behaviour": {"revoke": (">=", 0.95), "shred": (">=", 0.90), "broken1_unknown": (">=", 0.90),
                               "revoke_heldout_min": (">=", 0.85), "shred_heldout_min": (">=", 0.85), "locality": (">=", 0.98),
                               "restored/direct_acc": (">=", 0.95)},
        "attacks_after_revoke": {"revoke/probe_top1": ("<=", 0.05), "revoke/forced_choice_win": ("<=", 0.6)},
        "attacks_after_shred_hard": {"shred_hard/probe_top1": ("<=", 0.05), "shred_hard/forced_choice_win": ("<=", 0.6),
                                     "shred_hard/true_obj_top1_among_entities": ("<=", 0.05),
                                     "shred_hard/payload_share": ("<=", 0.05), "shred_hard/direct_unknown": (">=", 0.90)},
        "alternative_routes": {"alt_route/broken_route_changes": (">=", 0.95), "alt_route/other_route_survives": (">=", 0.95)},
        "interventions": {"interventions/localisation_hop1": (">=", 0.90), "interventions/localisation_hop2": (">=", 0.90),
                          "interventions/disable_hop1_changes": (">=", 0.95), "interventions/disable_hop2_changes": (">=", 0.95),
                          "interventions/disable_random_unchanged": (">=", 0.95), "interventions/swap_hop2": (">=", 0.90),
                          "interventions/replace_hop2": (">=", 0.90)},
    }
    lenient = {"direct": (">=", 0.90), "template1_train/direct": (">=", 0.90), "revoke": (">=", 0.90), "shred": (">=", 0.85),
               "broken1_unknown": (">=", 0.85)}
    return groups, lenient


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(AdapterConfig())
        print(f"=== seed {seed}: adapter training v2 ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = evaluate(gk, 1100 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups, lenient = criteria_groups()
    check_lenient = ledger.check_criteria(agg, lenient)
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    level = "F4" if met["deletion_behaviour"] and met["attacks_after_shred_hard"] else ("F3" if met["deletion_behaviour"] else "F1")
    record = {
        "experiment": "E-000011", "title": "Frozen GPT-2 core v2: verified gate, deletion behaviour, held-out paraphrases, interventions",
        "evidence_level": "E5", "deletion_level": level, "deletion_level_targeted": "F4",
        "evidence_level_note": "E5 names the substrate (a pretrained transformer as frozen core); support is stated per claim group.",
        "claim_groups_met": met,
        "claim_parts": [
            {"claim": "Reading through natural-language prompts on the frozen core; copy bound holds.", "criteria": list(groups["reading"]), "supported": met["reading"]},
            {"claim": "Reading generalises to sentence templates never seen in training.", "criteria": list(groups["heldout_paraphrases"]), "supported": met["heldout_paraphrases"]},
            {"claim": "UPDATE / ROLLBACK / RESIGN reproduced against the reference.", "criteria": list(groups["update_rollback"]), "supported": met["update_rollback"]},
            {"claim": "After REVOKE / SHRED and on broken paths the model answers ' unknown', also on held-out templates (F3).", "criteria": list(groups["deletion_behaviour"]), "supported": met["deletion_behaviour"]},
            {"claim": "After SHRED with hard verification nothing is recoverable by probe, forced choice or rank (F4).", "criteria": list(groups["attacks_after_shred_hard"]), "supported": met["attacks_after_shred_hard"]},
            {"claim": "Inside the frozen model the two reads of a 2-hop question are causally the two ground-truth cells: disabling either breaks the answer, disabling another cell does not, swapping or replacing the second payload changes the answer as predicted.", "criteria": list(groups["interventions"]), "supported": met["interventions"]},
        ],
        "not_claimed": "LLM scale; multi-token entities; free-text paraphrases beyond the six templates; unlearning of pretrained facts.",
        "by_construction_vs_learned": "The frozen core cannot copy a fact; whether the adapter copies is the masked-bank row. REVOKE "
                                      "is a mask (F1). Learned: reading from prompts (including four never-trained templates), "
                                      "composition without the intermediate entity in the text, emitting ' unknown' for a masked or "
                                      "unsigned cell (a trained refusal that ledger §28 would call output suppression if it stood "
                                      "alone — here it is paired with the copy bound, the masked-bank row, the attacks and the "
                                      "answer-category rows that show what is emitted instead), and the gate's selection between "
                                      "payload and ' unknown'. The 2-hop interventions are consistency checks: the read is the only "
                                      "channel through which the adapter can inject the fact, so disabling the cell removing the "
                                      "answer is expected; the informative rows are localisation (does the frozen model's own "
                                      "residual state route to the right cell at each read) and swap / replace (does the answer "
                                      "follow the payload exactly).",
        "lenient_criteria": check_lenient["criteria"], "lenient_supported": check_lenient["claim_supported"],
        "sample_sizes": {"direct/templates": EVAL["n_cells"], "hop2": EVAL["n_hop2"], "broken": EVAL["n_broken"],
                         "lifecycle": EVAL["n_lifecycle"], "attacks": EVAL["n_targets"],
                         "interventions": f"first {EVAL['n_interventions']} correctly answered 2-hop questions from a pool of {4 * EVAL['n_interventions']} (pool correct rate recorded)",
                         "alt_routes": 50, "comparators": EVAL["n_hop2"]},
        "config": {"seeds": args.seeds, "steps": args.steps, "eval": EVAL, "templates": TEMPLATES4, "train_templates": TRAIN_TEMPLATES,
                   "heldout_templates": HELDOUT_TEMPLATES, "gate_weight": 5.0, "gate_balanced": True, "p_revoked": 0.20, "p_shred": 0.10,
                   "extra_unanswerable": 0.2},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"], "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, ledger.pct(agg[k]["mean"]), ledger.pct(agg[k]["min"])) for k in KEYS if k in agg]
    arows = [(a, *(f"{agg[f'{c}/{a}']['mean']:.4f}" for c in ("active", "revoke", "shred_soft", "shred_hard"))) for a in ATT]
    irows = [(k, ledger.pct(agg[f"interventions/{k}"]["mean"]), ledger.pct(agg[f"interventions/{k}"]["min"])) for k in INT]
    md = "\n".join([
        "# E-000011 — Frozen GPT-2 core v2", "",
        f"Evidence level: **E5** (substrate: pretrained transformer, 124M frozen). Deletion level targeted F4, recorded **{level}**. "
        f"Seeds: {args.seeds}; {args.steps} adapter steps; verified gate (class-balanced, weight 5); p_revoked 0.20, p_shred 0.10, "
        "20% extra unanswerable queries; templates 0/1 trained, 2/3 (lexical variants), 4 (question form), 5 (prefixed clause) held out. "
        "Thresholds are the ones pre-registered for E-000008 (not relaxed); a lenient set (0.90 / 0.85) is reported separately.", "",
        f"Lenient criteria met: **{check_lenient['claim_supported']}**. Sample sizes: {record['sample_sizes']}", "",
        record["by_construction_vs_learned"], "",
        "Claim parts (each on its own pre-registered criteria, worst seed):", "",
        ledger.table(["claim", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**") for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "Attacks on 100 targets (mean over seeds; chance: forced choice 0.5, top-1 0.0039, mean rank 127.5, probe 0.0039):", "",
        ledger.table(["attack", "active", "after REVOKE", "after SHRED (soft)", "after SHRED (hard)"], arows), "",
        "Causal interventions on correctly answered 2-hop questions (mean / worst seed):", "",
        ledger.table(["intervention", "mean", "worst seed"], irows), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "Lenient criteria (secondary, worst seed):", "", ledger.criteria_table(check_lenient),
    ])
    path = ledger.save("e000011_gpt2_v2", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
