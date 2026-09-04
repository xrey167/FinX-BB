"""Experiment E-000020 — shared knowledge objects in a frozen GPT-2.

E-000015 and E-000016 established, inside the synthetic system, that a cell whose payload is the
ADDRESS of another cell lets several access keys share ONE knowledge object, and that this beats
duplicating the object on every deletion measure. Both records live in a toy transformer trained
from scratch, which is exactly the objection an external reviewer raises first.

This experiment carries the mechanism into the frozen GPT-2 adapter, with the same two-arm protocol
that made the synthetic result falsifiable: the SAME world is written twice, once with link cells
over shared targets and once with the alias keys as ordinary fact cells holding a copy, and both
are read by the SAME trained adapter from natural-language prompts. Every sharing claim is the
difference between the arms, so nothing rests on the mechanism looking good in isolation.

The adapter gains what the mini transformer already had: an alias row's value is its target's key
through a separate projection, and each read layer is followed by a dereference read whose query
comes from the value just read, with the null column carrying the incoming value so that a fact
read passes through unchanged. Both are off by default, so every earlier configuration is
unaffected.

Run:  python -m so.experiments.e000020_symlink_gpt2 [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, bank_from_store, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000015_symlink_cells as E15
from so.experiments import e000017_paraphrase_gap as E17
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, guard_recorded_checkpoint, _sha256
from so.llm_adapter import AdapterConfig
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Query, UNKNOWN, World

N_TRAIN_TEMPLATES = 8
N_DEREF = 1
EVAL = dict(n_base=700, n_groups=100, n_alias_per_group=2, n_direct=300, n_targets=100)


def route_targets_slots(queries: List[Query], bank: Bank, world: World, n_reads: int, n_deref: int) -> torch.Tensor:
    """(B, n_reads * (1 + n_deref)). A hop that resolves through an alias targets the alias at its
    resolve slot and the cell it points at in the dereference slot; a direct hop dereferences
    nothing, which is the passthrough column (-1)."""
    B, D = len(queries), n_deref
    S = n_reads * (1 + D)
    route = np.full((B, S), -2, dtype=np.int64)
    trace = bank.trace_of_key or {}
    for i, q in enumerate(queries):
        gt = world.answer(q, bank.index_view)
        start = n_reads - q.hops
        for t in range(start):
            for sl in range(1 + D):
                route[i, t * (1 + D) + sl] = -1
        for t in range(q.hops):
            base = (start + t) * (1 + D)
            if t < len(gt.edges):
                tr = trace.get(gt.edges[t], (bank.kid_of_key[gt.edges[t]],))
                route[i, base] = tr[0]
                for dd in range(D):
                    route[i, base + 1 + dd] = tr[1 + dd] if len(tr) > 1 + dd else -1
            elif t == len(gt.edges):
                cur = q.start
                for e in gt.edges:
                    cur = bank.index_view[e]
                key = (cur, q.path[len(gt.edges)])
                pos = bank.routable_pos.get(key, -1) if bank.routable_pos is not None else bank.active_pos.get(key, -1)
                route[i, base] = pos
                for dd in range(D):
                    route[i, base + 1 + dd] = -1
    return torch.as_tensor(route)


def train_adapter_links(gk: E8.GPT2Knowledge, seed: int, steps: int, batch_size: int = 32, route_weight: float = 1.0,
                        gate_weight: float = 5.0, lr: float = 2e-3, route_only_steps: int = 400,
                        p_revoked: float = 0.20, p_shred: float = 0.10, p_dangling: float = 0.05,
                        n_groups: int = 100, extra_unanswerable: float = 0.2, verbose: bool = True) -> Dict[str, Any]:
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
    n_reads = len(model.cfg.read_layers)
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_base = int(rng.integers(150, 301)) if route_only else int(rng.integers(500, 701))
        world, spec = E15.sample_alias_world(rng, n_base, n_groups, 2, gk.n_entities, 4, N_TRAIN_TEMPLATES)
        bank = E15.bank_with_links(rng, world, spec, centre, p_revoked, p_shred, 0.05, p_dangling)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [E17.query_text_pc(q, gk.names, N_TRAIN_TEMPLATES) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = route_targets_slots(queries, bank, world, n_reads, model.cfg.n_deref)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        cand, _, routing, _ = model(tensors, ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid,
                                                      reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        loss = (loss_route + gate_weight * loss_gate) if route_only else \
            (loss_ans + route_weight * loss_route + gate_weight * loss_gate)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"route {rec['route_loss']:.4f}  acc {acc:.3f}  {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000020_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": np.asarray(ck["centre"]), "history": ck["history"], "train_seconds": ck["train_seconds"],
                "checkpoint_sha256": _sha256(path)}
    out = train_adapter_links(gk, seed, steps)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def _answers(gk: E8.GPT2Knowledge, bank: Bank, keys: Sequence[Tuple[int, int]], names, template: int = 0):
    texts = [E17.TEMPLATES12[r][template].format(s=names[s]) for s, r in keys]
    out, hid, lg = [], [], []
    for i in range(0, len(texts), 64):
        ids, am, last = E8.encode_texts(gk.tok, texts[i: i + 64])
        with torch.no_grad():
            cand, _, _, h = gk.model(bank.tensors(), ids, am, last)
        a = cand.argmax(-1).numpy()
        out.append(np.where(a == gk.n_entities, UNKNOWN, a)); hid.append(h.numpy()); lg.append(cand.numpy())
    return np.concatenate(out), np.concatenate(hid), np.concatenate(lg)


def evaluate(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    world, spec = E15.sample_alias_world(rng, EVAL["n_base"], EVAL["n_groups"], EVAL["n_alias_per_group"],
                                         gk.n_entities, 4, N_TRAIN_TEMPLATES)
    sym_store, sym_kids = E15.load_arm(world, spec, centre, seed, symlink=True)
    dup_store, dup_kids = E15.load_arm(world, spec, centre, seed, symlink=False)
    m: Dict[str, Any] = {"seed": seed}
    alias_keys = spec.alias_keys
    base_keys = [f.key for f in world.facts if f.key not in spec.alias_of]
    targets = [t for t, _ in spec.groups]
    n_ent = gk.n_entities

    def sym() -> Bank:
        return bank_from_store(sym_store)

    def dup() -> Bank:
        return bank_from_store(dup_store)

    direct_keys = [base_keys[int(i)] for i in rng.choice(len(base_keys), size=min(EVAL["n_direct"], len(base_keys)),
                                                         replace=False)]
    truth_d = np.array([world.index[k] for k in direct_keys])
    a_d, _, _ = _answers(gk, sym(), direct_keys, gk.names)
    m["direct"] = float((a_d == truth_d).mean())
    truth_a = np.array([world.index[spec.alias_of[k]] for k in alias_keys])
    a_a, hid_a, lg_a = _answers(gk, sym(), alias_keys, gk.names)
    m["alias_direct"] = float((a_a == truth_a).mean())
    a_dup, _, _ = _answers(gk, dup(), alias_keys, gk.names)
    m["dup_direct"] = float((a_dup == truth_a).mean())
    heldout = tuple(range(N_TRAIN_TEMPLATES, N_TRAIN_TEMPLATES + E17.N_HELDOUT))
    for t in (1, *heldout):
        tag = "train" if t < N_TRAIN_TEMPLATES else "heldout"
        aa, _, _ = _answers(gk, sym(), alias_keys, gk.names, template=t)
        m[f"alias_template{t}_{tag}"] = float((aa == truth_a).mean())
    m["alias_heldout_min"] = float(min(m[f"alias_template{t}_heldout"] for t in heldout))
    m["alias_heldout_mean"] = float(np.mean([m[f"alias_template{t}_heldout"] for t in heldout]))

    # ---- SHARING: one UPDATE on the shared object
    new_obj = {k: int((world.index[k] + 1 + rng.integers(0, n_ent - 1)) % n_ent) for k in targets}
    for k in targets:
        sym_store.update(sym_kids[k], new_obj[k]); dup_store.update(dup_kids[k], new_obj[k])
    want = np.array([new_obj[spec.alias_of[a]] for a in alias_keys])
    a_sym_u = _answers(gk, sym(), alias_keys, gk.names)[0]
    a_dup_u = _answers(gk, dup(), alias_keys, gk.names)[0]
    m["shared_update/alias_new_object"] = float((a_sym_u == want).mean())
    m["duplicate_update/alias_new_object"] = float((a_dup_u == want).mean())
    # the complement: in the duplication arm the copies must still hold the OLD object, which is what
    # makes "one update did not reach them" a fact about the store rather than about a broken read
    m["shared_update/alias_old_object"] = float((a_sym_u == truth_a).mean())
    m["duplicate_update/alias_old_object"] = float((a_dup_u == truth_a).mean())
    for k in targets:
        sym_store.rollback(sym_kids[k], 1); dup_store.rollback(dup_kids[k], 1)
    m["rollback/alias_direct"] = float((_answers(gk, sym(), alias_keys, gk.names)[0] == truth_a).mean())

    # ---- probe on fact cells, then attacks through every alias after ONE shred of the shared object
    probe_keys = [k for k in base_keys if k not in set(targets)]
    _, hid_p, _ = _answers(gk, sym(), probe_keys, gk.names)
    y = np.array([world.index[k] for k in probe_keys])
    split = int(0.8 * len(probe_keys))
    probe = LinearProbe(hid_p.shape[1], n_ent, seed=seed)
    probe.fit(hid_p[:split], y[:split])
    m["probe_calibration_top1"] = probe.accuracy(hid_p[split:], y[split:])
    m["active/alias_probe_top1"] = probe.accuracy(hid_a, truth_a)          # positive control
    m["active/alias_forced_choice"] = forced_choice(lg_a, truth_a, np.random.default_rng(seed), n_ent)
    for k in targets:
        sym_store.shred(sym_kids[k]); dup_store.shred(dup_kids[k])
    a_s, hid_s, lg_s = _answers(gk, sym(), alias_keys, gk.names)
    m["shred_target/alias_unknown"] = float((a_s == UNKNOWN).mean())
    m["shred_target/alias_true_object"] = float((a_s == truth_a).mean())
    m["shred_target/alias_probe_top1"] = probe.accuracy(hid_s, truth_a)
    m["shred_target/alias_forced_choice"] = forced_choice(lg_s, truth_a, np.random.default_rng(seed), n_ent)
    rk = object_rank(lg_s, truth_a, n_ent)
    m["shred_target/alias_top1_among_entities"] = rk["top1"]; m["shred_target/alias_mean_rank"] = rk["mean_rank"]
    a_ds, hid_ds, lg_ds = _answers(gk, dup(), alias_keys, gk.names)
    m["dup_shred/copy_direct_acc"] = float((a_ds == truth_a).mean())
    m["dup_shred/copy_probe_top1"] = probe.accuracy(hid_ds, truth_a)
    m["dup_shred/copy_forced_choice"] = forced_choice(lg_ds, truth_a, np.random.default_rng(seed), n_ent)
    for k in targets:
        sym_store.resign(sym_kids[k]); dup_store.resign(dup_kids[k])
    m["resign_target/alias_direct"] = float((_answers(gk, sym(), alias_keys, gk.names)[0] == truth_a).mean())

    # ---- one alias at a time, and the dangling pointer after DELETE
    first = [ks[0] for _, ks in spec.groups]
    second = [ks[1] for _, ks in spec.groups]
    truth_second = np.array([world.index[spec.alias_of[k]] for k in second])
    truth_t = np.array([world.index[k] for k in targets])
    for a in first:
        sym_store.revoke(sym_kids[a])
    m["revoke_alias/alias_unknown"] = float((_answers(gk, sym(), first, gk.names)[0] == UNKNOWN).mean())
    m["revoke_alias/sibling_readable"] = float((_answers(gk, sym(), second, gk.names)[0] == truth_second).mean())
    m["revoke_alias/target_readable"] = float((_answers(gk, sym(), targets, gk.names)[0] == truth_t).mean())
    for a in first:
        sym_store.restore(sym_kids[a])
    for k in targets:
        sym_store.delete(sym_kids[k])
    a_del, _, _ = _answers(gk, sym(), alias_keys, gk.names)
    m["delete_target/alias_unknown"] = float((a_del == UNKNOWN).mean())
    m["delete_target/alias_true_object"] = float((a_del == truth_a).mean())
    return m


def criteria_groups() -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run. The bars are lower than the synthetic system's because a
    frozen 124M core reads at about 0.9, not 1.0; the SHARING claims are differences between arms and
    are therefore not softened."""
    return {
        "reading_through_an_alias": {"direct": (">=", 0.85), "alias_direct": (">=", 0.80),
                                     "dup_direct": (">=", 0.85), "alias_heldout_min": (">=", 0.50)},
        "one_update_reaches_every_path": {"shared_update/alias_new_object": (">=", 0.90),
                                          "duplicate_update/alias_new_object": ("<=", 0.05),
                                          "rollback/alias_direct": (">=", 0.80),
                                          "duplicate_update/alias_old_object": (">=", 0.85)},
        "one_shred_deletes_every_path": {"shred_target/alias_unknown": (">=", 0.90),
                                         "shred_target/alias_true_object": ("<=", 0.05),
                                         "dup_shred/copy_direct_acc": (">=", 0.85),
                                         "resign_target/alias_direct": (">=", 0.80)},
        "attacks_through_every_alias": {"shred_target/alias_probe_top1": ("<=", 0.05),
                                        "shred_target/alias_forced_choice": ("<=", 0.60),
                                        "shred_target/alias_top1_among_entities": ("<=", 0.05)},
        # the duplication arm's probe number is the right-hand side of the headline contrast; without a
        # floor a merely weak probe would print "3% vs 4%" and read as a deletion result
        "attack_validity": {"active/alias_probe_top1": (">=", 0.25), "probe_calibration_top1": (">=", 0.20),
                            "dup_shred/copy_probe_top1": (">=", 0.20)},
        "alias_lifecycle": {"revoke_alias/alias_unknown": (">=", 0.90), "revoke_alias/sibling_readable": (">=", 0.80),
                            "revoke_alias/target_readable": (">=", 0.80), "delete_target/alias_unknown": (">=", 0.90),
                            "delete_target/alias_true_object": ("<=", 0.05)},
    }


KEYS = ["direct", "alias_direct", "dup_direct", "alias_heldout_min", "probe_calibration_top1",
        "active/alias_probe_top1", "active/alias_forced_choice",
        "shared_update/alias_new_object", "duplicate_update/alias_new_object", "rollback/alias_direct",
        "shred_target/alias_unknown", "shred_target/alias_true_object", "shred_target/alias_probe_top1",
        "shred_target/alias_forced_choice", "shred_target/alias_top1_among_entities", "shred_target/alias_mean_rank",
        "dup_shred/copy_direct_acc", "dup_shred/copy_probe_top1", "dup_shred/copy_forced_choice",
        "resign_target/alias_direct", "revoke_alias/alias_unknown", "revoke_alias/sibling_readable",
        "revoke_alias/target_readable", "delete_target/alias_unknown", "delete_target/alias_true_object"]


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AdapterConfig(status_gated=True, use_links=True, n_deref=N_DEREF)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(cfg)
        print(f"=== seed {seed}: frozen GPT-2 with link cells ({N_DEREF} dereference read per layer) ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force)
        m = evaluate(gk, 2000 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: round(v, 4) for k, v in m.items() if k in KEYS}, flush=True)
    agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    # the same rule as E-000015: F4 also requires the alias lifecycle (a revoked alias, a dangling
    # pointer after DELETE). The floor is F0, not F1: this design keeps a revoked cell addressed and
    # closes the gate instead of removing it from routing, so F1 would claim routing removal.
    level = ("F4" if met["one_shred_deletes_every_path"] and met["attacks_through_every_alias"]
             and met["attack_validity"] and met["alias_lifecycle"]
             else ("F3" if met["one_shred_deletes_every_path"] and met["attack_validity"] else "F0"))
    n_alias = EVAL["n_groups"] * EVAL["n_alias_per_group"]
    record = {
        "experiment": "E-000020",
        "title": "Shared knowledge objects in a frozen GPT-2: link cells against duplication, natural-language prompts",
        "evidence_level": "E5", "deletion_level": level, "deletion_level_targeted": "F4",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "design": "The same world is written twice and read by the same trained adapter from natural-language prompts: "
                  "in the symlink arm the alias keys are LINK cells over shared targets, in the duplication arm they "
                  "are ordinary fact cells holding a copy. Every sharing claim is the difference between the arms.",
        "by_construction": ["the store decides which payload a row carries; the bank never exports the target's "
                            "payload, status or signature, and the model is never told that a value is a pointer",
                            "that one operation on a shared object reaches every alias is a property of the store; "
                            "what is measured is whether the frozen model reports it, and whether the SAME model "
                            "reports the duplication arm, where it does not, correctly"],
        "learned": ["following a pointer inside a frozen pretrained transformer: the dereference query comes from the "
                    "value just read, and the passthrough column keeps a value that was not a pointer",
                    "answering unknown for a dangling pointer after DELETE and for a revoked alias"],
        "not_claimed": "chains deeper than one dereference; multi-token entities; anything above 124M parameters.",
        "sample_sizes": {"alias_direct": n_alias, "dup_direct": n_alias, "direct": EVAL["n_direct"],
                         "shared_update/alias_new_object": n_alias, "duplicate_update/alias_new_object": n_alias,
                         "shred_target/alias_unknown": n_alias, "shred_target/alias_true_object": n_alias,
                         "shred_target/alias_probe_top1": n_alias, "dup_shred/copy_direct_acc": n_alias,
                         "delete_target/alias_unknown": n_alias, "revoke_alias/alias_unknown": EVAL["n_groups"]},
        "config": {"seeds": args.seeds, "steps": args.steps, "adapter": cfg.to_dict(), "eval": EVAL,
                   "n_train_templates": N_TRAIN_TEMPLATES, "n_deref": N_DEREF},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    lower = {k for k, (op, _) in all_criteria.items() if op == "<="} | {
        "shred_target/alias_true_object", "shred_target/alias_probe_top1", "shred_target/alias_forced_choice",
        "shred_target/alias_top1_among_entities", "delete_target/alias_true_object",
        "duplicate_update/alias_new_object"}
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{ledger.worst(agg[k], k in lower):.4f}") for k in KEYS if k in agg]
    contrast = [
        ("one UPDATE on the shared object reaches every access path",
         ledger.pct(agg["shared_update/alias_new_object"]["mean"]), ledger.pct(agg["duplicate_update/alias_new_object"]["mean"])),
        ("after one SHRED the object is still readable",
         ledger.pct(1 - agg["shred_target/alias_unknown"]["mean"]), ledger.pct(agg["dup_shred/copy_direct_acc"]["mean"])),
        ("after one SHRED a probe recovers the object",
         ledger.pct(agg["shred_target/alias_probe_top1"]["mean"]), ledger.pct(agg["dup_shred/copy_probe_top1"]["mean"])),
        ("operations needed to reach every access path", "1", str(1 + EVAL["n_alias_per_group"])),
    ]
    md = "\n".join([
        "# E-000020 — Shared knowledge objects in a frozen GPT-2", "",
        f"Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **{level}**. "
        f"Seeds: {args.seeds}; {args.steps} steps.", "", record["design"], "",
        ledger.table(["what is measured", "symlink arm", "duplication arm"], contrast), "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "Exact binomial intervals (pooled over seeds):", "",
        ledger.table(ledger.CI_HEADERS, ledger.ci_rows(per_seed, [k for k in KEYS if k in record["sample_sizes"]],
                                                       record["sample_sizes"],
                                                       lower_is_better=sorted(lower))), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "By construction: " + "; ".join(record["by_construction"]) + ".", "",
        "Learned: " + "; ".join(record["learned"]) + ".", "",
        "Not claimed: " + record["not_claimed"],
    ])
    path = ledger.save("e000020_symlink_gpt2", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
