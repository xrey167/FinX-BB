"""Experiment E-000017 — reading versus refusal on phrasings the adapter never saw.

Roadmap kill criterion 5 has fired: in the frozen GPT-2, answering ' unknown' after REVOKE reaches
98% on the trained template and 52% on the weakest held-out one (E-000012; E-000011: 70% / 16%),
and E-000013 shows the same failure a third time, together with injection into generic text where
there should be none. The standing audit corrected the diagnosis: the deleted object is NEVER the
top-1 answer on a held-out template (0.0 in every seed) — the model names some other entity — and
reading on those templates is already unreliable while the cell is ACTIVE. So the measured failure
is one of REFUSAL on top of a reading failure, not recovery of deleted knowledge.

This experiment has two phases, run as separate processes so no template table is ever mutated
under a running evaluation:

  --phase diagnose   No training. From E-000012's checkpoints, decompose each held-out template
                     into the joint distribution of (active: correct / other entity / unknown) x
                     (after REVOKE: unknown / other entity / the deleted object). The quantity the
                     audit asked for is the refusal rate CONDITIONED on the model having read the
                     fact correctly while the cell was active.

  --phase train      The remedy the roadmap prescribes for stage 2: eight trained templates per
                     relation instead of two, four held out, optionally with a paraphrase
                     consistency loss that ties the routing distribution of one fact across two
                     surface forms. The control arm is the same budget without that loss, so any
                     gain is attributable.

Run:  python -m so.experiments.e000017_paraphrase_gap --phase diagnose [--seeds 0 1 2]
      python -m so.experiments.e000017_paraphrase_gap --phase train --n-templates 8 [--consistency 0.0]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.data import Bank, bank_from_store, bank_from_world, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, guard_recorded_checkpoint, _sha256
from so.experiments.e000012_status_gated_revoke import route_targets_status_gated
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Query, UNKNOWN, World, fill_random

# Twelve surface forms per relation: eight for training, four held out.  The first six are exactly
# E-000011's table so that the trained-template numbers stay comparable.
TEMPLATES12 = {
    0: ["{s} lives in", "The home of {s} is", "{s} resides in", "The residence of {s} is",
        "Q: Where does {s} live? A: In", "According to the records, {s} lives in",
        "{s} has settled in", "The town of {s} is",
        "{s} currently lives in", "Where {s} lives is", "It is known that {s} lives in", "{s}, who lives in"],
    1: ["{s} works for", "The employer of {s} is", "{s} is employed by", "The company that employs {s} is",
        "Q: Who does {s} work for? A:", "According to the records, {s} works for",
        "{s} has a job at", "The firm of {s} is",
        "{s} currently works for", "Where {s} works is", "It is known that {s} works for", "{s}, who works for"],
    2: ["{s} is married to", "The spouse of {s} is", "{s} is wed to", "The partner of {s} is",
        "Q: Who is {s} married to? A:", "According to the records, {s} is married to",
        "{s} has married", "The husband or wife of {s} is",
        "{s} is currently married to", "Who {s} married is", "It is known that {s} is married to", "{s}, who married"],
    3: ["{s} was born in", "The birthplace of {s} is", "{s} comes from", "The place of birth of {s} is",
        "Q: Where was {s} born? A: In", "According to the records, {s} was born in",
        "{s} originates from", "The origin of {s} is",
        "{s} was originally born in", "Where {s} was born is", "It is known that {s} was born in", "{s}, who was born in"],
}
NOUN12 = {0: "home", 1: "employer", 2: "spouse", 3: "birthplace"}
N_TRAIN_DEFAULT, N_HELDOUT = 8, 4
EVAL = dict(n_cells=1000, n_targets=200, n_broken=200, n_generic=200)
GENERIC = ["{s} said that", "The story of {s} begins", "In the morning, {s}", "Everyone knows that {s}",
           "{s} walked into the"]


def query_text_pc(q: Query, names: List[str], n_synonyms: int, template: Optional[int] = None) -> str:
    """This experiment's own prompt builder: E-000008's module globals are never rebound."""
    s = names[q.start]
    if q.hops == 1:
        r, k = q.path[0], (q.surface[0] % n_synonyms if template is None else template)
        return TEMPLATES12[r][k].format(s=s)
    inner = " of the ".join(NOUN12[r] for r in reversed(q.path))
    return f"The {inner} of {s} is"


# ---------------------------------------------------------------------------- phase 1: diagnosis
def diagnose(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, templates: Dict[int, List[str]],
             heldout: Tuple[int, ...], n_synonyms: int) -> Dict[str, Any]:
    """Decompose reading and refusal per template, jointly, on one set of target cells."""
    rng = np.random.default_rng(seed)
    world = fill_random(rng, World(gk.n_entities, 4, n_synonyms, []), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = world.facts
    idx = rng.choice(len(facts), size=min(EVAL["n_targets"], len(facts)), replace=False)
    targets = [facts[int(i)] for i in idx]
    truth = np.array([f.obj for f in targets])
    qs = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]

    def answers(template: int) -> np.ndarray:
        texts = [templates[q.path[0]][template].format(s=gk.names[q.start]) for q in qs]
        out = []
        for i in range(0, len(texts), 64):
            ids, am, last = E8.encode_texts(gk.tok, texts[i: i + 64])
            cand, _, _, _ = gk.model(bank_from_store(store).tensors(), ids, am, last)
            a = cand.argmax(-1).numpy()
            out.append(np.where(a == gk.n_entities, UNKNOWN, a))
        return np.concatenate(out)

    m: Dict[str, Any] = {"seed": seed, "n_targets": len(targets)}
    n_templates = len(templates[0])
    active = {t: answers(t) for t in range(n_templates)}
    for f in targets:
        store.revoke(kids[f.key])
    revoked = {t: answers(t) for t in range(n_templates)}
    for f in targets:
        store.restore(kids[f.key])

    for t in range(n_templates):
        a, r = active[t], revoked[t]
        a_ok, a_unk = a == truth, a == UNKNOWN
        a_wrong = ~a_ok & ~a_unk
        r_unk, r_obj = r == UNKNOWN, r == truth
        r_wrong = ~r_unk & ~r_obj
        tag = f"template{t}_" + ("train" if t < n_templates - len(heldout) else "heldout")
        m[f"{tag}/active_correct"] = float(a_ok.mean())
        m[f"{tag}/active_other_entity"] = float(a_wrong.mean())
        m[f"{tag}/active_unknown"] = float(a_unk.mean())
        m[f"{tag}/revoked_unknown"] = float(r_unk.mean())
        m[f"{tag}/revoked_other_entity"] = float(r_wrong.mean())
        m[f"{tag}/revoked_deleted_object"] = float(r_obj.mean())
        # the quantity the audit asked for: refusal AMONG the cases the model could read at all
        m[f"{tag}/refusal_given_active_correct"] = float(r_unk[a_ok].mean()) if a_ok.any() else float("nan")
        m[f"{tag}/deleted_object_given_active_correct"] = float(r_obj[a_ok].mean()) if a_ok.any() else float("nan")
        m[f"{tag}/n_active_correct"] = int(a_ok.sum())
    tr = [t for t in range(n_templates) if t not in heldout]
    m["train/refusal_given_active_correct"] = float(np.mean([m[f"template{t}_train/refusal_given_active_correct"] for t in tr]))
    m["heldout/refusal_given_active_correct"] = float(np.nanmean(
        [m[f"template{t}_heldout/refusal_given_active_correct"] for t in heldout]))
    m["train/active_correct"] = float(np.mean([m[f"template{t}_train/active_correct"] for t in tr]))
    m["heldout/active_correct"] = float(np.mean([m[f"template{t}_heldout/active_correct"] for t in heldout]))
    m["heldout/deleted_object_given_active_correct"] = float(np.nanmean(
        [m[f"template{t}_heldout/deleted_object_given_active_correct"] for t in heldout]))
    m["heldout/revoked_deleted_object"] = float(np.mean([m[f"template{t}_heldout/revoked_deleted_object"] for t in heldout]))
    return m


# -------------------------------------------------------------------------- phase 2: the remedy
def train_adapter_templates(gk: E8.GPT2Knowledge, seed: int, steps: int, n_train_templates: int,
                            consistency: float = 0.0, batch_size: int = 32, route_weight: float = 1.0,
                            gate_weight: float = 5.0, lr: float = 2e-3, route_only_steps: int = 300,
                            p_revoked: float = 0.20, p_shred: float = 0.10, extra_unanswerable: float = 0.2,
                            verbose: bool = True) -> Dict[str, Any]:
    """E-000011's trainer with the stage-2 template budget and an optional consistency term.

    The consistency term renders the SAME queries under a second trained template and ties the two
    routing distributions and the two answer distributions together, so the routing has to depend on
    the fact rather than on the wording.
    """
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
    n_cons = batch_size // 4
    for step in range(steps):
        model.train()
        route_only = step < route_only_steps
        n_cells = int(rng.integers(150, 301)) if route_only else int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, n_train_templates)
        bank = bank_from_world(rng, world, centre, p_revoked, p_shred, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [query_text_pc(q, gk.names, n_train_templates) for q in queries])
        target = E8.targets_of(queries, bank, world)
        route = route_targets_status_gated(queries, bank, world, n_reads)
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
        loss_cons = cand.sum() * 0
        if consistency > 0 and not route_only and n_cons > 0:
            sub = [q for q in queries[:n_cons] if q.hops == 1]
            if sub:
                t_alt = [int(rng.integers(0, n_train_templates)) for _ in sub]
                ids2, am2, last2 = E8.encode_texts(gk.tok, [query_text_pc(q, gk.names, n_train_templates, t)
                                                            for q, t in zip(sub, t_alt)])
                cand2, _, routing2, _ = model(tensors, ids2, am2, last2)
                k = len(sub)
                lp1, lp2 = torch.log_softmax(cand[:k], -1), torch.log_softmax(cand2, -1)
                ans_kl = 0.5 * (F.kl_div(lp1, lp2, log_target=True, reduction="batchmean")
                                + F.kl_div(lp2, lp1, log_target=True, reduction="batchmean"))
                r1 = torch.log(routing[:k, -1].clamp_min(1e-9))
                r2 = torch.log(routing2[:, -1].clamp_min(1e-9))
                route_kl = 0.5 * (F.kl_div(r1, r2, log_target=True, reduction="batchmean")
                                  + F.kl_div(r2, r1, log_target=True, reduction="batchmean"))
                loss_cons = ans_kl + route_kl
        loss = (loss_route + gate_weight * loss_gate) if route_only else \
            (loss_ans + route_weight * loss_route + gate_weight * loss_gate + consistency * loss_cons)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "route_loss": float(loss_route.item()), "gate_loss": float(loss_gate.item()),
                   "consistency": float(loss_cons.item()) if consistency > 0 else 0.0,
                   "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {rec['step']:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  "
                      f"route {rec['route_loss']:.4f}  cons {rec['consistency']:.4f}  acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def train_or_load(gk: E8.GPT2Knowledge, seed: int, steps: int, n_train: int, consistency: float,
                  force: bool = False) -> Dict[str, Any]:
    tag = f"t{n_train}_c{consistency:g}"
    path = CHECKPOINTS / f"e000017_{tag}{CKPT_SUFFIX}_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"],
                "checkpoint_sha256": _sha256(path)}
    out = train_adapter_templates(gk, seed, steps, n_train, consistency)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    guard_recorded_checkpoint(path)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict(),
                "n_train_templates": n_train, "consistency": consistency}, path)
    out["checkpoint_sha256"] = _sha256(path)
    return out


def evaluate_templates(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, n_train: int) -> Dict[str, Any]:
    """Reading, refusal after REVOKE and after SHRED, per template, plus injection where there is no key."""
    heldout = tuple(range(n_train, n_train + N_HELDOUT))
    m = diagnose(gk, seed, centre, TEMPLATES12, heldout, n_train)
    rng = np.random.default_rng(seed + 5000)
    world = fill_random(rng, World(gk.n_entities, 4, n_train, []), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    facts = world.facts
    idx = rng.choice(len(facts), size=min(EVAL["n_targets"], len(facts)), replace=False)
    targets = [facts[int(i)] for i in idx]
    qs = [Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),)) for f in targets]

    def unknown_rate(template: int) -> float:
        texts = [TEMPLATES12[q.path[0]][template].format(s=gk.names[q.start]) for q in qs]
        out = []
        for i in range(0, len(texts), 64):
            ids, am, last = E8.encode_texts(gk.tok, texts[i: i + 64])
            cand, _, _, _ = gk.model(bank_from_store(store).tensors(), ids, am, last)
            out.append(cand.argmax(-1).numpy() == gk.n_entities)
        return float(np.concatenate(out).mean())

    for f in targets:
        store.revoke(kids[f.key])
    rev = {t: unknown_rate(t) for t in range(n_train + N_HELDOUT)}
    for f in targets:
        store.restore(kids[f.key])
        store.shred(kids[f.key])
    gk.model.cfg.hard_gate = True
    shr = {t: unknown_rate(t) for t in range(n_train + N_HELDOUT)}
    gk.model.cfg.hard_gate = False
    for f in targets:
        store.resign(kids[f.key])
    m["revoke_train_min"] = float(min(rev[t] for t in range(n_train)))
    m["revoke_heldout_min"] = float(min(rev[t] for t in heldout))
    m["shred_train_min"] = float(min(shr[t] for t in range(n_train)))
    m["shred_heldout_min"] = float(min(shr[t] for t in heldout))
    for t in range(n_train + N_HELDOUT):
        m[f"revoke/template{t}_unknown"] = rev[t]
        m[f"shred_hard/template{t}_unknown"] = shr[t]
    # injection where there is no key: broken paths and generic text
    bank = bank_from_store(store)
    broken = world.sample_queries(rng, EVAL["n_broken"], 1, "fwd", require_answer=False)
    unk = []
    for i in range(0, len(broken), 64):
        chunk = broken[i: i + 64]
        ids, am, last = E8.encode_texts(gk.tok, [query_text_pc(q, gk.names, n_train) for q in chunk])
        cand, _, _, _ = gk.model(bank.tensors(), ids, am, last)
        unk.append(cand.argmax(-1).numpy() == gk.n_entities)
    m["broken1_unknown"] = float(np.concatenate(unk).mean())
    gen = [GENERIC[int(rng.integers(0, len(GENERIC)))].format(s=gk.names[int(rng.integers(0, gk.n_entities))])
           for _ in range(EVAL["n_generic"])]
    kls = []
    for i in range(0, len(gen), 64):
        ids, am, last = E8.encode_texts(gk.tok, gen[i: i + 64])
        with torch.no_grad():
            _, full_b, _, _ = gk.model(None, ids, am, last)
            _, full_a, _, _ = gk.model(bank.tensors(), ids, am, last)
        lb, la = torch.log_softmax(full_b, -1), torch.log_softmax(full_a, -1)
        kls.append((lb.exp() * (lb - la)).sum(-1).numpy())
    m["generic/kl_to_base"] = float(np.concatenate(kls).mean())
    return m


# ---------------------------------------------------------------------------------------- record
def criteria_groups(n_train: int) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """Pre-registered before the first run of the training phase.

    The bars for the held-out families are the ones roadmap stage 2 prescribes; the trained-template
    bars guard against buying generalisation with a regression on what already worked.
    """
    return {
        "reading_generalises": {"heldout/active_correct": (">=", 0.90), "train/active_correct": (">=", 0.95)},
        "refusal_generalises": {"revoke_heldout_min": (">=", 0.85), "shred_heldout_min": (">=", 0.85)},
        "refusal_on_trained_templates_holds": {"revoke_train_min": (">=", 0.95), "shred_train_min": (">=", 0.90)},
        "deleted_object_never_returns": {"heldout/revoked_deleted_object": ("<=", 0.02),
                                         "heldout/deleted_object_given_active_correct": ("<=", 0.02)},
        "no_key_no_injection": {"broken1_unknown": (">=", 0.90), "generic/kl_to_base": ("<=", 0.05)},
    }


DIAG_KEYS = ["train/active_correct", "heldout/active_correct", "train/refusal_given_active_correct",
             "heldout/refusal_given_active_correct", "heldout/revoked_deleted_object",
             "heldout/deleted_object_given_active_correct"]
TRAIN_KEYS = DIAG_KEYS + ["revoke_train_min", "revoke_heldout_min", "shred_train_min", "shred_heldout_min",
                          "broken1_unknown", "generic/kl_to_base"]


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["diagnose", "train"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-templates", type=int, default=N_TRAIN_DEFAULT)
    ap.add_argument("--consistency", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.phase == "diagnose":
        from so.experiments import e000011_gpt2_v2 as E11          # its table is the one E-000012 trained on
        per_seed = []
        for seed in args.seeds:
            gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
            path = CHECKPOINTS / f"e000012_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
            ck = torch.load(path, weights_only=False)
            gk.model.load_state_dict(ck["adapter"], strict=False)
            gk.model.eval()
            print(f"=== seed {seed}: diagnosing E-000012's checkpoint ===", flush=True)
            m = diagnose(gk, 1700 + seed, np.asarray(ck["centre"]), E11.TEMPLATES6, E11.HELDOUT_TEMPLATES, 2)
            m["checkpoint_sha256"] = _sha256(path)
            per_seed.append(m)
            print({k: round(v, 4) for k, v in m.items() if k in DIAG_KEYS}, flush=True)
        agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
        record = {
            "experiment": "E-000017-A", "title": "Diagnosis: reading versus refusal on held-out phrasings",
            "evidence_level": "E5", "deletion_level": None,
            "question": "Is the held-out failure of E-000011/E-000012 a deletion failure or a reading failure that "
                        "deletion inherits?",
            "no_training": "No model was trained for this record: E-000012's three checkpoints are evaluated as they "
                           "were recorded, so this is a decomposition of an existing result, not a new one.",
            "per_seed": per_seed, "aggregate": agg,
        }
        rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}") for k in DIAG_KEYS if k in agg]
        md = "\n".join([
            "# E-000017-A — Reading versus refusal on held-out phrasings (diagnosis, no training)", "",
            record["question"], "", record["no_training"], "",
            ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
            "Read the two conditional rows together: `refusal_given_active_correct` is how often the model answers "
            "' unknown' after REVOKE among exactly those targets it read correctly while the cell was ACTIVE, and "
            "`deleted_object_given_active_correct` is how often it returns the deleted object instead.",
        ])
        path = ledger.save("e000017a_paraphrase_diagnosis", record, md)
        print(md); print(f"\nsaved {path}")
        return record

    per_seed = []
    for seed in args.seeds:
        gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
        print(f"=== seed {seed}: {args.n_templates} trained templates, consistency {args.consistency:g} ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.n_templates, args.consistency, args.force)
        m = evaluate_templates(gk, 1700 + seed, out["centre"], args.n_templates)
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: round(v, 4) for k, v in m.items() if k in TRAIN_KEYS}, flush=True)
    agg = ledger.aggregate(per_seed, [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")])
    groups = criteria_groups(args.n_templates)
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    name = f"e000017b_templates{args.n_templates}" + (f"_consistency{args.consistency:g}" if args.consistency else "")
    record = {
        "experiment": "E-000017-B",
        "title": f"Stage-2 template budget: {args.n_templates} trained, {N_HELDOUT} held out"
                 + (f", paraphrase consistency {args.consistency:g}" if args.consistency else ", no consistency loss"),
        "evidence_level": "E5", "deletion_level": "F3" if met["refusal_generalises"] else "F1",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "answers": "Roadmap kill criterion 5 fired on a two-template budget. This run gives the stage the budget it "
                   "prescribes and reports whether the held-out failure survives it.",
        "config": {"seeds": args.seeds, "steps": args.steps, "n_train_templates": args.n_templates,
                   "n_heldout_templates": N_HELDOUT, "consistency": args.consistency, "templates": TEMPLATES12,
                   "eval": EVAL, "adapter": AdapterConfig(status_gated=True).to_dict()},
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}") for k in TRAIN_KEYS if k in agg]
    md = "\n".join([
        f"# E-000017-B — {record['title']}", "",
        record["answers"], "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**")
                                                    for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check),
    ])
    path = ledger.save(name, record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
