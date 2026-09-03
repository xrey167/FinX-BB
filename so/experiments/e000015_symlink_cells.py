"""Experiment E-000015 — explicit symlink cells: several access keys, ONE knowledge object.

Architecture document section 10 and ledger section 7 state the Symlink hypothesis in two
directions.  The programme has so far realised only the first: the neural representation holds
no fact, it addresses a cell (copy bound 0%, provenance exact, causal interventions).  The
second direction — a cell whose payload is the ADDRESS of another cell, so that several access
keys share one knowledge object — did not exist, so every key carried its own payload and a fact
reachable under two keys was stored twice.  This experiment adds it and tests it against the
alternative it is supposed to beat.

THE COMPARISON.  Two stores are built from the SAME world with the SAME ground truth and the same
queries, and read by the SAME trained model:

  * symlink arm:   850 fact cells, and the 200 alias keys are LINK cells pointing at 100 targets;
  * duplicate arm: the same 1,050 keys are all fact cells — the shared object is COPIED.

Every claim is a difference between the arms, which is exactly how ledger section 7 states the
hypothesis (sharing versus duplicating):

  * one UPDATE on the shared object changes every access path (symlink) versus only its own key
    (duplicate);
  * one SHRED on the shared object removes it on every access path and nothing is recoverable
    through any alias (symlink) versus the object still being readable, and recoverable by probe
    and forced choice, through every remaining copy (duplicate);
  * revoking ONE alias leaves the object and its sibling aliases intact;
  * DELETE of the target leaves a dangling pointer that must resolve to UNKNOWN — the pointer is
    NOT erased by the control plane, so discovering the miss is the model's job.

WHAT IS BY CONSTRUCTION AND WHAT IS LEARNED.  The store decides which payload a row carries (the
target's key instead of an object), exactly as it decides the marker; the model is never told that
a value it has read is a pointer.  What is learned is to follow it: each hop is followed by a
dereference slot whose query comes from the pointer alone, and whose passthrough column lets the
model keep a value that was not a pointer.  Routing therefore names both cells of an alias path,
which is the provenance claim.

Run:  python -m so.experiments.e000015_symlink_cells [--seeds 0 1 2] [--steps 4000]
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, Batch, bank_from_store, failing_hop_target, reverse_target, sample_training_queries
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.model import ModelConfig, MutableKnowledgeTransformer
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Fact, Query, UNKNOWN, World, free_keys

N_ENTITIES, N_RELATIONS, N_SYNONYMS = 256, 6, 2
MAX_TRAIN_LINK_DEPTH = 3          # how deep the training bank resolves an alias chain
EVAL = dict(n_base=850, n_groups=100, n_alias_per_group=2, n_direct=600, n_hop2=300, n_broken=200,
            n_targets=100, n_chain=100)


# ---------------------------------------------------------------------------------------- worlds
@dataclass
class AliasSpec:
    """Which keys are aliases of which target key (ground truth is identical in both arms)."""

    alias_of: Dict[Tuple[int, int], Tuple[int, int]]          # alias key -> target key
    groups: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]   # target key -> its alias keys

    @property
    def alias_keys(self) -> List[Tuple[int, int]]:
        return list(self.alias_of)


def sample_alias_world(rng: np.random.Generator, n_base: int, n_groups: int, n_alias_per_group: int,
                       n_entities: int = N_ENTITIES, n_relations: int = N_RELATIONS,
                       n_synonyms: int = N_SYNONYMS) -> Tuple[World, AliasSpec]:
    """A world of ``n_base`` facts plus alias keys that resolve to the same object as their target."""
    world = World.sample(rng, n_entities, n_relations, n_base, n_synonyms)
    facts = list(world.facts)
    free = free_keys(world)
    rng.shuffle(free)
    targets_idx = rng.choice(len(facts), size=n_groups, replace=False)
    alias_of: Dict[Tuple[int, int], Tuple[int, int]] = {}
    groups: List[Tuple[Tuple[int, int], List[Tuple[int, int]]]] = []
    cur = 0
    for i in targets_idx:
        t = facts[int(i)]
        keys: List[Tuple[int, int]] = []
        for _ in range(n_alias_per_group):
            if cur >= len(free):
                raise RuntimeError("not enough free keys for aliases")
            k = free[cur]; cur += 1
            alias_of[k] = t.key
            keys.append(k)
            facts.append(Fact(k[0], k[1], t.obj))          # ground truth: the alias answers like its target
        groups.append((t.key, keys))
    return World(n_entities, n_relations, n_synonyms, facts), AliasSpec(alias_of, groups)


def load_arm(world: World, spec: AliasSpec, centre: np.ndarray, seed: int, symlink: bool
             ) -> Tuple[MVCCStore, Dict[Tuple[int, int], int]]:
    """Write the world into a store; ``symlink`` decides whether alias keys become LINK cells."""
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids: Dict[Tuple[int, int], int] = {}
    for f in world.facts:                                   # facts first: a link needs its target
        if f.key not in spec.alias_of:
            kids[f.key] = store.write(f.subject, f.relation, f.obj, provenance="fact")
    for f in world.facts:
        if f.key in spec.alias_of:
            if symlink:
                kids[f.key] = store.link(f.subject, f.relation, kids[spec.alias_of[f.key]], provenance="alias")
            else:
                kids[f.key] = store.write(f.subject, f.relation, f.obj, provenance="copy")
    return store, kids


# ------------------------------------------------------------------------------- training banks
def bank_with_links(rng: np.random.Generator, world: World, spec: AliasSpec, centre: np.ndarray,
                    p_revoked: float = 0.10, p_shred: float = 0.05, p_stale: float = 0.05,
                    p_dangling: float = 0.05, p_chain: float = 0.0) -> Bank:
    """Training bank: the world's facts, alias rows carrying their target's key, lifecycle states.

    ``p_dangling`` of the alias rows point at a key that no cell holds, so the model must learn that
    a pointer is not a promise.  ``p_chain`` of them point at ANOTHER ALIAS instead of a fact
    (E-000016): without chains in the training distribution a second dereference slot never sees a
    pointer in its input and learns to pass through.
    """
    from so.data import invalid_markers, valid_markers

    facts = list(world.facts)
    n = len(facts)
    row_of_key = {f.key: i for i, f in enumerate(facts)}
    subject = np.array([f.subject for f in facts], dtype=np.int64)
    relation = np.array([f.relation for f in facts], dtype=np.int64)
    obj = np.array([f.obj for f in facts], dtype=np.int64)
    is_link = np.zeros(n, dtype=bool)
    l_sub = np.zeros(n, dtype=np.int64)
    l_rel = np.zeros(n, dtype=np.int64)
    free = free_keys(world)
    alias_items = list(spec.alias_of.items())
    by_target: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for k, t in alias_items:
        by_target.setdefault(t, []).append(k)
    for k, t in alias_items:
        i = row_of_key[k]
        is_link[i] = True
        siblings = [x for x in by_target.get(t, []) if x != k]
        if free and rng.random() < p_dangling:
            d = free[int(rng.integers(0, len(free)))]        # points nowhere
            l_sub[i], l_rel[i] = d
        elif siblings and rng.random() < p_chain:
            l_sub[i], l_rel[i] = siblings[int(rng.integers(0, len(siblings)))]   # alias -> alias -> fact
        else:
            l_sub[i], l_rel[i] = t
        obj[i] = 0                                           # placeholder: an alias holds no object
    revoked = rng.random(n) < p_revoked
    shred = (~revoked) & (rng.random(n) < p_shred)
    marker = valid_markers(rng, centre, n)
    if shred.any():
        marker[shred] = invalid_markers(rng, centre, int(shred.sum()))
    active = ~revoked
    n_stale = int(rng.binomial(n, p_stale))
    if n_stale:
        pick = rng.choice(n, size=n_stale, replace=False)
        subject = np.concatenate([subject, subject[pick]]); relation = np.concatenate([relation, relation[pick]])
        obj = np.concatenate([obj, (obj[pick] + 1 + rng.integers(0, world.n_entities - 1, size=n_stale)) % world.n_entities])
        marker = np.concatenate([marker, valid_markers(rng, centre, n_stale)])
        active = np.concatenate([active, np.zeros(n_stale, dtype=bool)])
        shred = np.concatenate([shred, np.zeros(n_stale, dtype=bool)])
        is_link = np.concatenate([is_link, np.zeros(n_stale, dtype=bool)])
        l_sub = np.concatenate([l_sub, np.zeros(n_stale, dtype=np.int64)])
        l_rel = np.concatenate([l_rel, np.zeros(n_stale, dtype=np.int64)])
    usable = active & ~shred
    total = subject.shape[0]
    # resolution: a fact row resolves to its object; an alias row resolves to what its target holds
    fact_view: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for i in range(total):
        if usable[i] and not is_link[i]:
            fact_view.setdefault((int(subject[i]), int(relation[i])), (int(obj[i]), i))
    index_view: Dict[Tuple[int, int], int] = {k: o for k, (o, _) in fact_view.items()}
    trace_of_key: Dict[Tuple[int, int], Tuple[int, ...]] = {k: (i,) for k, (_, i) in fact_view.items()}
    link_row = {(int(subject[i]), int(relation[i])): i for i in range(total) if usable[i] and is_link[i]}
    for key, i in link_row.items():
        trace: List[int] = [i]
        cur = (int(l_sub[i]), int(l_rel[i]))
        seen = {i}
        for _ in range(MAX_TRAIN_LINK_DEPTH):
            hit = fact_view.get(cur)
            if hit is not None:
                index_view[key] = hit[0]
                trace_of_key[key] = tuple(trace + [hit[1]])
                break
            nxt = link_row.get(cur)
            if nxt is None or nxt in seen:
                break                                        # dangling, unusable or a cycle
            seen.add(nxt); trace.append(nxt)
            cur = (int(l_sub[nxt]), int(l_rel[nxt]))
    kid_of_key = {(int(s), int(r)): int(i) for i, (s, r, u) in enumerate(zip(subject, relation, usable)) if u}
    active_pos = {(int(s), int(r)): int(i) for i, (s, r, a) in enumerate(zip(subject, relation, active)) if a}
    return Bank(subject, relation, obj, marker.astype(np.float32), active, usable, np.arange(total),
                index_view, kid_of_key, active_pos, marker_valid=~shred,
                is_link=is_link, link_subject=l_sub, link_relation=l_rel, trace_of_key=trace_of_key)


def encode_slots(queries: List[Query], bank: Bank, world: World, max_hops: int, n_deref: int) -> Batch:
    """Like ``encode_queries`` but with one resolve slot and ``n_deref`` dereference slots per hop.

    The dereference target of a hop is the SECOND cell of that key's resolution path (the cell the
    alias points at); a hop that resolved directly targets the passthrough column (-1).
    """
    B = len(queries)
    S = max_hops * (1 + n_deref)
    mode = np.zeros(B, dtype=np.int64)
    start = np.zeros(B, dtype=np.int64)
    rels = np.full((B, max_hops), world.n_surface, dtype=np.int64)
    hop_valid = np.zeros((B, max_hops), dtype=bool)
    target = np.zeros(B, dtype=np.int64)
    route = np.full((B, S), -2, dtype=np.int64)
    trace = bank.trace_of_key or {}
    for i, q in enumerate(queries):
        mode[i] = 0 if q.mode == "fwd" else 1
        start[i] = q.start
        rels[i, : q.hops] = q.surface
        hop_valid[i, : q.hops] = True
        gt = world.answer(q, bank.index_view)
        target[i] = world.n_entities if gt.answer == UNKNOWN else gt.answer
        if q.mode == "fwd":
            for t, e in enumerate(gt.edges):
                tr = trace.get(e, (bank.kid_of_key[e],))
                base = t * (1 + n_deref)
                route[i, base] = tr[0]
                for d in range(n_deref):
                    route[i, base + 1 + d] = tr[1 + d] if len(tr) > 1 + d else -1
            if gt.answer == UNKNOWN and len(gt.edges) < q.hops:
                base = len(gt.edges) * (1 + n_deref)
                route[i, base] = failing_hop_target(bank, q, gt)
                for d in range(n_deref):
                    route[i, base + 1 + d] = -1
        else:
            route[i, 0] = reverse_target(bank, q, gt)
            for d in range(n_deref):
                route[i, 1 + d] = -1
    t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    return Batch(t(mode, torch.long), t(start, torch.long), t(rels, torch.long), t(hop_valid, torch.bool),
                 t(target, torch.long), t(route, torch.long), queries)


# ------------------------------------------------------------------------------------- training
def train_symlink(model_cfg: ModelConfig, cfg: TrainConfig, n_groups: int = 100, n_alias_per_group: int = 2,
                  p_dangling: float = 0.05, p_chain: float = 0.0, verbose: bool = True) -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MutableKnowledgeTransformer(model_cfg)
    centre = make_centre(cfg.seed, model_cfg.marker_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    model.train()
    for step in range(cfg.n_steps):
        n_base = int(rng.integers(cfg.n_cells_min, cfg.n_cells_max + 1))
        world, spec = sample_alias_world(rng, n_base, n_groups, n_alias_per_group,
                                         cfg.n_entities, cfg.n_relations, cfg.n_synonyms)
        bank = bank_with_links(rng, world, spec, centre, cfg.p_revoked, cfg.p_shred, cfg.p_stale, p_dangling, p_chain)
        queries = sample_training_queries(rng, world, bank, cfg.batch_size, cfg.mix)
        batch = encode_slots(queries, bank, world, model_cfg.max_hops, model_cfg.n_deref)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        tensors = bank.tensors()
        logits, routing, extras = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid,
                                        noise=cfg.train_noise)
        loss_ans = F.cross_entropy(logits, batch.target)
        loss = loss_ans + cfg.route_weight * routing_loss(routing, batch.route, model_cfg.use_null_cell)
        if model_cfg.use_marker_gate and cfg.gate_weight > 0 and extras.get("gate_logits") is not None:
            valid = tensors["marker_valid"].float()
            per_cell = F.binary_cross_entropy_with_logits(extras["gate_logits"], valid, reduction="none")
            n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
            gate_loss = (0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg) \
                if cfg.gate_balanced else per_cell.mean()
            loss = loss + cfg.gate_weight * gate_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % cfg.log_every == 0 or step == 0:
            acc = (logits.argmax(-1) == batch.target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"acc {acc:.3f}  {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"model": model, "centre": centre, "history": history, "train_seconds": time.time() - t0}


# ----------------------------------------------------------------------------------- prediction
@dataclass
class Pred:
    answers: np.ndarray
    routing: np.ndarray
    logits: np.ndarray
    hidden: np.ndarray
    traces: List[Tuple[int, ...]]


@torch.no_grad()
def predict(model, bank: Bank, world: World, queries: Sequence[Query], cell_mask: Optional[np.ndarray] = None,
            batch_size: int = 256, confident: float = 0.5) -> Pred:
    model.eval()
    tensors = bank.tensors()
    mask_t = None if cell_mask is None else torch.as_tensor(cell_mask, dtype=torch.bool)
    H, D = model.cfg.max_hops, model.cfg.n_deref
    ans, rout, lg, hid, traces = [], [], [], [], []
    for i in range(0, len(queries), batch_size):
        chunk = list(queries[i: i + batch_size])
        batch = encode_slots(chunk, bank, world, H, D)
        logits, routing, extras = model(tensors, batch.mode, batch.start, batch.rels, batch.hop_valid,
                                        cell_mask=mask_t)
        pred = logits.argmax(-1).numpy()
        ans.append(np.where(pred == world.n_entities, UNKNOWN, pred))
        r = routing.numpy()
        rout.append(r); lg.append(logits.numpy()); hid.append(extras["hidden"].numpy())
        C = bank.size
        for j, q in enumerate(chunk):
            tr: List[int] = []
            stop = False
            for t in range(q.hops):
                for sl in range(1 + D):
                    p = r[j, t * (1 + D) + sl]
                    k = int(p.argmax())
                    if k >= C or p[k] < confident:
                        stop = sl == 0            # a dereference slot may legitimately pass through
                        break
                    tr.append(int(bank.kid[k]))
                if stop:
                    break
            traces.append(tuple(tr))
    return Pred(np.concatenate(ans), np.concatenate(rout), np.concatenate(lg), np.concatenate(hid), traces)


# ----------------------------------------------------------------------------------- evaluation
def _q1(world: World, key: Tuple[int, int]) -> Query:
    return Query("fwd", key[0], (key[1],), (world.surface_of(key[1], 0),))


def evaluate(model, seed: int, centre: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    world, spec = sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"])
    sym_store, sym_kids = load_arm(world, spec, centre, seed, symlink=True)
    dup_store, dup_kids = load_arm(world, spec, centre, seed, symlink=False)
    sym_ref, dup_ref = ReferenceResolver(sym_store), ReferenceResolver(dup_store)
    m: Dict[str, Any] = {"seed": seed}
    alias_keys = spec.alias_keys
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    n_ent = world.n_entities

    def sym() -> Bank:
        return bank_from_store(sym_store)

    def dup() -> Bank:
        return bank_from_store(dup_store)

    def acc(bank: Bank, ref: ReferenceResolver, keys: Sequence[Tuple[int, int]], mask=None) -> float:
        qs = [_q1(world, k) for k in keys]
        p = predict(model, bank, world, qs, cell_mask=mask)
        return float(np.mean([a == ref.resolve(q).answer for a, q in zip(p.answers, qs)]))

    def unknown_rate(bank: Bank, keys: Sequence[Tuple[int, int]]) -> float:
        qs = [_q1(world, k) for k in keys]
        return float((predict(model, bank, world, qs).answers == UNKNOWN).mean())

    def answers_of(bank: Bank, keys: Sequence[Tuple[int, int]]) -> np.ndarray:
        return predict(model, bank, world, [_q1(world, k) for k in keys]).answers

    # ---- reading, provenance, composition
    direct_keys = [base_keys[int(i)] for i in rng.choice(len(base_keys), size=min(EVAL["n_direct"], len(base_keys)), replace=False)]
    m["direct"] = acc(sym(), sym_ref, direct_keys)
    m["alias_direct"] = acc(sym(), sym_ref, alias_keys)
    m["dup_direct"] = acc(dup(), dup_ref, alias_keys)
    m["alias_direct_minus_dup"] = m["alias_direct"] - m["dup_direct"]
    hop2 = world.sample_queries(rng, EVAL["n_hop2"], 2, "fwd", require_answer=True)
    p2 = predict(model, sym(), world, hop2)
    m["hop2"] = float(np.mean([a == sym_ref.resolve(q).answer for a, q in zip(p2.answers, hop2)]))
    broken = world.sample_queries(rng, EVAL["n_broken"], 1, "fwd", require_answer=False)
    m["broken1_unknown"] = float((predict(model, sym(), world, broken).answers == UNKNOWN).mean())
    pd_ = predict(model, sym(), world, [_q1(world, k) for k in direct_keys])
    m["provenance_direct"] = float(np.mean([tr == sym_ref.resolve(_q1(world, k)).trace
                                            for tr, k in zip(pd_.traces, direct_keys)]))
    pa = predict(model, sym(), world, [_q1(world, k) for k in alias_keys])
    m["alias_provenance_pair"] = float(np.mean([tr == sym_ref.resolve(_q1(world, k)).trace
                                                for tr, k in zip(pa.traces, alias_keys)]))
    m["alias_provenance_len2"] = float(np.mean([len(tr) == 2 for tr in pa.traces]))

    # ---- the dereference slot is what reads an alias (inference-time ablation, no retraining)
    model.cfg.disable_deref = True
    m["deref_disabled/alias_direct"] = acc(sym(), sym_ref, alias_keys)
    m["deref_disabled/direct"] = acc(sym(), sym_ref, direct_keys)
    model.cfg.disable_deref = False

    # ---- SHARING: one UPDATE on the shared object versus one UPDATE on one copy
    targets = [t for t, _ in spec.groups]
    new_obj = {k: int((world.index[k] + 1 + rng.integers(0, n_ent - 1)) % n_ent) for k in targets}
    for k in targets:
        sym_store.update(sym_kids[k], new_obj[k]); dup_store.update(dup_kids[k], new_obj[k])
    want = np.array([new_obj[spec.alias_of[a]] for a in alias_keys])
    m["shared_update/alias_new_object"] = float((answers_of(sym(), alias_keys) == want).mean())
    m["duplicate_update/alias_new_object"] = float((answers_of(dup(), alias_keys) == want).mean())
    m["shared_update/target_new_object"] = float((answers_of(sym(), targets) == np.array([new_obj[k] for k in targets])).mean())
    m["duplicate_update/ops_to_propagate"] = 1 + EVAL["n_alias_per_group"]     # by construction, recorded
    m["shared_update/ops_to_propagate"] = 1
    for k in targets:
        sym_store.rollback(sym_kids[k], 1); dup_store.rollback(dup_kids[k], 1)
    m["rollback/alias_direct"] = acc(sym(), sym_ref, alias_keys)

    # ---- probe calibration on fact cells (as in E-000004 / E-000010)
    probe_keys = [k for k in base_keys if k not in set(targets)]
    pp = predict(model, sym(), world, [_q1(world, k) for k in probe_keys])
    y = np.array([world.index[k] for k in probe_keys])
    split = int(0.8 * len(probe_keys))
    probe = LinearProbe(pp.hidden.shape[1], n_ent, seed=seed)
    probe.fit(pp.hidden[:split], y[:split])
    m["probe_calibration_top1"] = probe.accuracy(pp.hidden[split:], y[split:])

    # ---- ONE-OP DELETION: shred the shared object once, attack through EVERY access path
    t_truth = np.array([world.index[spec.alias_of[a]] for a in alias_keys])
    for k in targets:
        sym_store.shred(sym_kids[k]); dup_store.shred(dup_kids[k])
    pa_s = predict(model, sym(), world, [_q1(world, k) for k in alias_keys])
    m["shred_target/alias_unknown"] = float((pa_s.answers == UNKNOWN).mean())
    m["shred_target/alias_true_object"] = float((pa_s.answers == t_truth).mean())
    m["shred_target/alias_probe_top1"] = probe.accuracy(pa_s.hidden, t_truth)
    m["shred_target/alias_forced_choice"] = forced_choice(pa_s.logits, t_truth, np.random.default_rng(seed), n_ent)
    rk = object_rank(pa_s.logits, t_truth, n_ent)
    m["shred_target/alias_top1_among_entities"] = rk["top1"]
    m["shred_target/alias_mean_rank"] = rk["mean_rank"]
    m["shred_target/chance_mean_rank"] = rk["chance_mean_rank"]
    m["shred_target/target_unknown"] = float((answers_of(sym(), targets) == UNKNOWN).mean())
    # the duplicate arm: shredding ONE copy leaves the object readable and recoverable through the others
    pa_d = predict(model, dup(), world, [_q1(world, k) for k in alias_keys])
    m["dup_shred/copy_direct_acc"] = float((pa_d.answers == t_truth).mean())
    m["dup_shred/copy_probe_top1"] = probe.accuracy(pa_d.hidden, t_truth)
    m["dup_shred/copy_forced_choice"] = forced_choice(pa_d.logits, t_truth, np.random.default_rng(seed), n_ent)
    m["dup_shred/ops_to_delete"] = 1 + EVAL["n_alias_per_group"]               # by construction, recorded
    m["shred_target/ops_to_delete"] = 1
    for k in targets:
        sym_store.resign(sym_kids[k]); dup_store.resign(dup_kids[k])
    m["resign_target/alias_direct"] = acc(sym(), sym_ref, alias_keys)

    # ---- alias lifecycle: one access path at a time
    first = [ks[0] for _, ks in spec.groups]
    second = [ks[1] for _, ks in spec.groups]
    for a in first:
        sym_store.revoke(sym_kids[a])
    m["revoke_alias/alias_unknown"] = float((answers_of(sym(), first) == UNKNOWN).mean())
    m["revoke_alias/sibling_readable"] = acc(sym(), sym_ref, second)
    m["revoke_alias/target_readable"] = acc(sym(), sym_ref, targets)
    for a in first:
        sym_store.restore(sym_kids[a])
    for a in first:
        sym_store.shred(sym_kids[a])
    m["shred_alias/alias_unknown"] = float((answers_of(sym(), first) == UNKNOWN).mean())
    m["shred_alias/sibling_readable"] = acc(sym(), sym_ref, second)
    m["shred_alias/target_readable"] = acc(sym(), sym_ref, targets)
    for a in first:
        sym_store.resign(sym_kids[a])
    # RELINK: point the first alias of each group at another group's target
    others = targets[1:] + targets[:1]
    for a, o in zip(first, others):
        sym_store.relink(sym_kids[a], sym_kids[o])
    m["relink/alias_new_object"] = float((answers_of(sym(), first) == np.array([world.index[o] for o in others])).mean())
    m["relink/sibling_unchanged"] = acc(sym(), sym_ref, second)
    for a in first:
        sym_store.rollback(sym_kids[a], 1)
    m["relink_rollback/alias_direct"] = acc(sym(), sym_ref, first)
    # DELETE the shared object: the pointer stays, the referent is gone
    m["refcount_before_delete"] = float(np.mean([sym_store.refcount(sym_kids[k]) for k in targets]))
    for k in targets:
        sym_store.delete(sym_kids[k])
    m["delete_target/alias_unknown"] = float((answers_of(sym(), alias_keys) == UNKNOWN).mean())
    m["delete_target/alias_true_object"] = float((answers_of(sym(), alias_keys) == t_truth).mean())
    m["delete_target/pointer_still_in_bank"] = float(np.mean(sym().is_link))

    # ---- chains of two aliases: out of reach for one dereference slot (pre-registered limit)
    m.update(_chain_metrics(model, rng, centre, seed))
    # ---- regression: a link-free world must still behave like E-000001-B
    m.update(_regression(model, rng, centre, seed))
    return m


def _chain_metrics(model, rng: np.random.Generator, centre: np.ndarray, seed: int) -> Dict[str, float]:
    world, spec = sample_alias_world(rng, 400, EVAL["n_chain"], 1, N_ENTITIES, N_RELATIONS, N_SYNONYMS)
    store, kids = load_arm(world, spec, centre, seed + 7, symlink=True)
    free = free_keys(world)
    rng.shuffle(free)
    chain_keys: List[Tuple[int, int]] = []
    truth: List[int] = []
    for i, (t, ks) in enumerate(spec.groups):
        if i >= len(free):
            break
        k2 = free[i]
        store.link(k2[0], k2[1], kids[ks[0]], provenance="chain")     # k2 -> alias -> fact
        chain_keys.append(k2)
        truth.append(world.index[t])
    bank = bank_from_store(store)
    qs = [_q1(world, k) for k in chain_keys]
    p = predict(model, bank, world, qs)
    depth = {"chain2/n": float(len(chain_keys)),
             "chain2/answer_acc": float(np.mean([a == t for a, t in zip(p.answers, truth)])),
             "chain2/unknown": float((p.answers == UNKNOWN).mean()),
             "chain2/depth1_acc": float(np.mean([a == world.index[spec.alias_of[k]] for a, k in
                                                 zip(predict(model, bank, world,
                                                             [_q1(world, k) for k in spec.alias_keys]).answers,
                                                     spec.alias_keys)]))}
    return depth


def _regression(model, rng: np.random.Generator, centre: np.ndarray, seed: int) -> Dict[str, float]:
    world = World.sample(rng, N_ENTITIES, N_RELATIONS, 1000, N_SYNONYMS)
    spec = AliasSpec({}, [])
    store, kids = load_arm(world, spec, centre, seed + 13, symlink=True)
    ref = ReferenceResolver(store)
    bank = bank_from_store(store)
    out: Dict[str, float] = {}
    for name, hops, mode in (("direct", 1, "fwd"), ("hop2", 2, "fwd"), ("hop3", 3, "fwd"), ("reverse", 1, "rev")):
        qs = world.sample_queries(rng, 300, hops, mode, require_answer=True)
        p = predict(model, bank, world, qs)
        out[f"regression/{name}"] = float(np.mean([a == ref.resolve(q).answer for a, q in zip(p.answers, qs)]))
        if name == "direct":
            out["regression/provenance"] = float(np.mean([tr == ref.resolve(q).trace for tr, q in zip(p.traces, qs)]))
    qs = world.sample_queries(rng, 200, 2, "fwd", require_answer=False)
    out["regression/broken2_unknown"] = float((predict(model, bank, world, qs).answers == UNKNOWN).mean())
    return out


# --------------------------------------------------------------------------------------- record
def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run.  Every claim about sharing is a DIFFERENCE between the
    two arms; the duplicate arm is the alternative the symlink is supposed to beat, not a strawman:
    it holds the identical ground truth and is read by the identical model."""
    return {
        "reading": {"direct": (">=", 0.98), "alias_direct": (">=", 0.95), "dup_direct": (">=", 0.98),
                    "hop2": (">=", 0.95), "broken1_unknown": (">=", 0.95)},
        "provenance_through_the_alias": {"provenance_direct": (">=", 0.95), "alias_provenance_pair": (">=", 0.90)},
        "dereference_is_what_reads_an_alias": {"deref_disabled/alias_direct": ("<=", 0.20),
                                               "deref_disabled/direct": (">=", 0.90)},
        "one_update_reaches_every_path": {"shared_update/alias_new_object": (">=", 0.95),
                                          "duplicate_update/alias_new_object": ("<=", 0.05),
                                          "shared_update/target_new_object": (">=", 0.95),
                                          "rollback/alias_direct": (">=", 0.95)},
        "one_shred_deletes_every_path": {"shred_target/alias_unknown": (">=", 0.95),
                                         "shred_target/alias_true_object": ("<=", 0.05),
                                         "dup_shred/copy_direct_acc": (">=", 0.95),
                                         "resign_target/alias_direct": (">=", 0.95)},
        "attacks_through_every_alias": {"shred_target/alias_probe_top1": ("<=", 0.05),
                                        "shred_target/alias_forced_choice": ("<=", 0.60),
                                        "shred_target/alias_top1_among_entities": ("<=", 0.05)},
        "alias_lifecycle": {"revoke_alias/alias_unknown": (">=", 0.95), "revoke_alias/sibling_readable": (">=", 0.95),
                            "revoke_alias/target_readable": (">=", 0.95), "shred_alias/alias_unknown": (">=", 0.95),
                            "shred_alias/target_readable": (">=", 0.95), "relink/alias_new_object": (">=", 0.90),
                            "relink_rollback/alias_direct": (">=", 0.90), "delete_target/alias_unknown": (">=", 0.95),
                            "delete_target/alias_true_object": ("<=", 0.05)},
        "capability_limit_of_one_slot": {"chain2/answer_acc": ("<=", 0.20)},
        "no_regression_without_links": {"regression/direct": (">=", 0.98), "regression/hop2": (">=", 0.95),
                                        "regression/hop3": (">=", 0.90), "regression/reverse": (">=", 0.95),
                                        "regression/provenance": (">=", 0.95), "regression/broken2_unknown": (">=", 0.95)},
    }


def deletion_level(met: Dict[str, bool]) -> str:
    if met["one_shred_deletes_every_path"] and met["attacks_through_every_alias"] and met["alias_lifecycle"]:
        return "F4"
    if met["one_shred_deletes_every_path"]:
        return "F3"
    return "F1"


KEYS = ["direct", "alias_direct", "dup_direct", "hop2", "broken1_unknown", "provenance_direct",
        "alias_provenance_pair", "alias_provenance_len2", "deref_disabled/alias_direct", "deref_disabled/direct",
        "shared_update/alias_new_object", "duplicate_update/alias_new_object", "shared_update/target_new_object",
        "rollback/alias_direct", "probe_calibration_top1",
        "shred_target/alias_unknown", "shred_target/alias_true_object", "shred_target/alias_probe_top1",
        "shred_target/alias_forced_choice", "shred_target/alias_top1_among_entities", "shred_target/alias_mean_rank",
        "dup_shred/copy_direct_acc", "dup_shred/copy_probe_top1", "dup_shred/copy_forced_choice",
        "resign_target/alias_direct", "revoke_alias/alias_unknown", "revoke_alias/sibling_readable",
        "revoke_alias/target_readable", "shred_alias/alias_unknown", "shred_alias/target_readable",
        "relink/alias_new_object", "relink/sibling_unchanged", "relink_rollback/alias_direct",
        "delete_target/alias_unknown", "delete_target/alias_true_object", "refcount_before_delete",
        "chain2/answer_acc", "chain2/unknown", "chain2/depth1_acc",
        "regression/direct", "regression/hop2", "regression/hop3", "regression/reverse", "regression/provenance",
        "regression/broken2_unknown"]


def model_config(n_deref: int = 1) -> ModelConfig:
    return ModelConfig(n_entities=N_ENTITIES, n_relations=N_RELATIONS, n_surface=N_RELATIONS * N_SYNONYMS,
                       max_hops=3, use_links=True, n_deref=n_deref)


def train_config(seed: int, steps: int) -> TrainConfig:
    return TrainConfig(seed=seed, n_steps=steps, n_entities=N_ENTITIES, n_relations=N_RELATIONS,
                       n_synonyms=N_SYNONYMS, n_cells_min=600, n_cells_max=850, route_weight=0.5,
                       gate_weight=5.0, gate_balanced=True, log_every=250)


def train_or_load(seed: int, steps: int, n_deref: int = 1, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000015_deref{n_deref}_seed{seed}.pt"
    cfg_m, cfg_t = model_config(n_deref), train_config(seed, steps)
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        model = MutableKnowledgeTransformer(ModelConfig(**ck["model_cfg"]))
        model.load_state_dict(ck["model"])
        model.eval()
        return {"model": model, "centre": np.asarray(ck["centre"]), "history": ck["history"],
                "train_seconds": ck["train_seconds"], "checkpoint_sha256": _sha256(path)}
    out = train_symlink(cfg_m, cfg_t)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"model": out["model"].state_dict(), "model_cfg": cfg_m.to_dict(), "train_cfg": cfg_t.to_dict(),
                "centre": out["centre"], "history": out["history"], "train_seconds": out["train_seconds"]}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--deref2-seed", type=int, default=0, help="seed of the single-seed two-slot control")
    ap.add_argument("--skip-deref2", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        print(f"=== seed {seed}: training (1 dereference slot) ===", flush=True)
        out = train_or_load(seed, args.steps, 1, args.force)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        mm = evaluate(out["model"], 1500 + seed, out["centre"])
        mm["train_seconds"] = out["train_seconds"]; mm["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(mm)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()}, flush=True)
    control: Dict[str, Any] = {}
    if not args.skip_deref2:
        print("=== control: two dereference slots (single seed) ===", flush=True)
        c = train_or_load(args.deref2_seed, args.steps, 2, args.force)
        cm = evaluate(c["model"], 1500 + args.deref2_seed, c["centre"])
        control = {"seed": args.deref2_seed, "chain2/answer_acc": cm["chain2/answer_acc"],
                   "chain2/depth1_acc": cm["chain2/depth1_acc"], "alias_direct": cm["alias_direct"],
                   "direct": cm["direct"], "checkpoint_sha256": c["checkpoint_sha256"], "full": cm}
        print("two-slot control:", {k: v for k, v in control.items() if k != "full"}, flush=True)

    keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    level = deletion_level(met)
    n_alias = EVAL["n_groups"] * EVAL["n_alias_per_group"]
    sizes = {"direct": EVAL["n_direct"], "alias_direct": n_alias, "dup_direct": n_alias, "hop2": EVAL["n_hop2"],
             "broken1_unknown": EVAL["n_broken"], "provenance_direct": EVAL["n_direct"],
             "alias_provenance_pair": n_alias, "shared_update/alias_new_object": n_alias,
             "duplicate_update/alias_new_object": n_alias, "shred_target/alias_unknown": n_alias,
             "shred_target/alias_true_object": n_alias, "shred_target/alias_probe_top1": n_alias,
             "shred_target/alias_top1_among_entities": n_alias, "dup_shred/copy_direct_acc": n_alias,
             "revoke_alias/alias_unknown": EVAL["n_groups"], "shred_alias/alias_unknown": EVAL["n_groups"],
             "relink/alias_new_object": EVAL["n_groups"], "delete_target/alias_unknown": n_alias,
             "chain2/answer_acc": EVAL["n_chain"], "regression/direct": 300, "regression/hop2": 300,
             "regression/hop3": 300, "regression/reverse": 300}
    record = {
        "experiment": "E-000015",
        "title": "Explicit symlink cells: several access keys share one knowledge object (symlink arm versus duplication arm)",
        "evidence_level": "E4", "deletion_level": level, "deletion_level_targeted": "F4",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "design": "Two stores hold the SAME world with the SAME ground truth and are read by the SAME trained model: "
                  "in the symlink arm the 200 alias keys are LINK cells pointing at 100 target cells, in the duplication "
                  "arm the same keys are ordinary fact cells holding a copy of the object. Every sharing claim is the "
                  "difference between the arms.",
        "by_construction": ["the store decides which payload a row carries (an alias row carries its target's KEY, a "
                            "fact row its object), exactly as it decides the marker; the bank never exports the target's "
                            "payload, its status, its signature or the chain depth",
                            "that ONE update or ONE shred on a shared object reaches every alias is a property of the "
                            "store; what is measured is whether the trained model reports it, and whether the SAME model "
                            "reports the duplication arm (where it does not) correctly",
                            "a deleted target keeps its key as a tombstone, so a dangling pointer stays a pointer and the "
                            "miss is not pre-resolved by the control plane"],
        "learned": ["following a pointer: the dereference slot's query comes from the value just read, not from the "
                    "question, and the model is never told that a value is a pointer",
                    "keeping a value that was not a pointer (the passthrough column) so that fact cells still read "
                    "correctly, measured as deref_disabled/direct versus deref_disabled/alias_direct",
                    "answering UNKNOWN for a dangling pointer, for a revoked or shredded alias and for a shredded target",
                    "provenance across the indirection: the routing names the alias AND the cell it points at"],
        "not_claimed": "LLM scale (the frozen-GPT-2 chain does not yet carry links); chains deeper than the number of "
                       "dereference slots; reference counting as a garbage-collection policy.",
        "config": {"seeds": args.seeds, "steps": args.steps, "model": model_config(1).to_dict(),
                   "train": train_config(0, args.steps).to_dict(), "eval": EVAL,
                   "n_entities": N_ENTITIES, "n_relations": N_RELATIONS, "n_synonyms": N_SYNONYMS},
        "two_slot_control": {k: v for k, v in control.items() if k != "full"} if control else None,
        "sample_sizes": sizes,
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}", f"{agg[k]['max']:.4f}") for k in KEYS if k in agg]
    contrast = [
        ("one UPDATE on the shared object reaches every access path", ledger.pct(agg["shared_update/alias_new_object"]["mean"]),
         ledger.pct(agg["duplicate_update/alias_new_object"]["mean"])),
        ("one SHRED on the shared object leaves nothing readable", ledger.pct(agg["shred_target/alias_unknown"]["mean"]),
         ledger.pct(1 - agg["dup_shred/copy_direct_acc"]["mean"])),
        ("object recoverable by probe after that one operation", ledger.pct(agg["shred_target/alias_probe_top1"]["mean"]),
         ledger.pct(agg["dup_shred/copy_probe_top1"]["mean"])),
        ("operations needed to reach every access path", "1", str(1 + EVAL["n_alias_per_group"])),
    ]
    md = "\n".join([
        "# E-000015 — Explicit symlink cells: several access keys, one knowledge object", "",
        f"Evidence level: **E4** (synthetic system). Deletion level targeted F4, recorded **{level}**. "
        f"Seeds: {args.seeds}; {args.steps} steps.", "",
        record["design"], "",
        "Symlink versus duplication (mean over seeds), the two arms holding identical ground truth:", "",
        ledger.table(["what is measured", "symlink arm", "duplication arm"], contrast), "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed", "best seed"], rows), "",
        "Exact binomial intervals (pooled over seeds):", "",
        ledger.table(ledger.CI_HEADERS, ledger.ci_rows(per_seed, [k for k in KEYS if k in sizes], sizes,
                                                       lower_is_better=[k for k in KEYS if "true_object" in k or "probe" in k
                                                                        or "forced_choice" in k or "top1_among" in k
                                                                        or k == "duplicate_update/alias_new_object"
                                                                        or k == "chain2/answer_acc"
                                                                        or k.startswith("deref_disabled/alias")])), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        ("Two-slot control (single seed): " + str({k: round(v, 4) if isinstance(v, float) else v
                                                   for k, v in control.items() if k != "full"}) if control else ""), "",
        "By construction: " + "; ".join(record["by_construction"]) + ".", "",
        "Learned: " + "; ".join(record["learned"]) + ".", "",
        "Not claimed: " + record["not_claimed"],
    ])
    path = ledger.save("e000015_symlink_cells", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
