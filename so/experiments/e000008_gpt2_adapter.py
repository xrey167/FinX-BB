"""Experiment E-000008 — pretrained transformer core (GPT-2 small) with the mutable knowledge layer.

The neural core is frozen GPT-2 (124M parameters, pretrained on natural
language).  The SO knowledge layer is attached as a symlink adapter that reads
cells at blocks 8 and 10 and writes the gated value into the residual stream of
the last token; the unchanged LM head then emits the object token.  Queries
are natural-language sentences with two paraphrase templates per relation and
noun-phrase composition for 2-hop questions ("The employer of the spouse of
Anna is").  Worlds are re-sampled every step, so neither the frozen core nor
the adapter can memorise a fact.

This is the CPU-feasible analogue of the ledger's outstanding C55–C57
real-model chain.  Evidence level claimed: E5 (transformer evidence) and a
*partial* E6 (a real pretrained LM, but a small one, on CPU) — not LLM scale.

Run:  python -m so.experiments.e000008_gpt2_adapter [--seeds 0 1 2] [--steps 1500]
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, bank_from_store, bank_from_world, failing_hop_target, sample_training_queries
from so.experiments.e000001b_mini_transformer import CHECKPOINTS
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.train import lr_at, make_centre, routing_loss, TrainConfig
from so.world import Query, UNKNOWN, World, fill_random, inject_alternative_paths

TEMPLATES = {
    0: ["{s} lives in", "The home of {s} is"],
    1: ["{s} works for", "The employer of {s} is"],
    2: ["{s} is married to", "The spouse of {s} is"],
    3: ["{s} was born in", "The birthplace of {s} is"],
}
NOUN = {0: "home", 1: "employer", 2: "spouse", 3: "birthplace"}
UNKNOWN_WORD = " unknown"
STOP = {"The", "This", "That", "There", "These", "Those", "They", "When", "What", "Where", "Which", "While", "With",
        "Then", "Than", "Also", "After", "Before", "Some", "Such", "Many", "More", "Most", "Only", "Over", "Into",
        "From", "Just", "Like", "Here", "Have", "Been", "Were", "Will", "Would", "Could", "Should", "About", "Because",
        "However", "Although", "Since", "Until", "Under", "Through", "Between", "During", "Without", "Within", "Both",
        "Each", "Every", "Other", "Another", "Even", "Still", "Very", "Much", "Well", "Being", "Does", "Doing", "Their",
        "Your", "Ours", "Mine", "Yours", "Whom", "Whose", "Yeah", "Okay", "Please", "Thanks", "Thank"}


def select_entities(tok, n: int = 256) -> List[int]:
    ids = []
    for tid in range(len(tok)):
        s = tok.convert_ids_to_tokens(tid)
        if re.fullmatch(r"Ġ[A-Z][a-z]{3,}", s) and s[1:] not in STOP:
            ids.append(tid)
        if len(ids) >= n:
            break
    if len(ids) < n:
        raise RuntimeError("not enough single-token entity names")
    return ids


def query_text(q: Query, names: List[str], n_synonyms: int) -> str:
    s = names[q.start]
    if q.hops == 1:
        r, k = q.path[0], q.surface[0] % n_synonyms
        return TEMPLATES[r][k].format(s=s)
    inner = " of the ".join(NOUN[r] for r in reversed(q.path))
    return f"The {inner} of {s} is"


def encode_texts(tok, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc = tok(texts, return_tensors="pt", padding=True)
    last_idx = enc["attention_mask"].sum(1) - 1
    return enc["input_ids"], enc["attention_mask"], last_idx


def route_targets(queries: List[Query], bank: Bank, world: World, n_reads: int) -> torch.Tensor:
    """(B, n_reads): a query with h hops reads at the LAST h read layers; earlier reads take the null cell."""
    B = len(queries)
    route = np.full((B, n_reads), -2, dtype=np.int64)
    for i, q in enumerate(queries):
        gt = world.answer(q, bank.index_view)
        start = n_reads - q.hops
        route[i, :start] = -1
        for t in range(q.hops):
            if t < len(gt.edges):
                route[i, start + t] = bank.kid_of_key[gt.edges[t]]
            elif t == len(gt.edges):
                route[i, start + t] = failing_hop_target(bank, q, gt)   # shredded cell -> attend, find gate closed
            else:
                route[i, start + t] = -2
    return torch.as_tensor(route)


def targets_of(queries: List[Query], bank: Bank, world: World) -> torch.Tensor:
    t = [world.answer(q, bank.index_view).answer for q in queries]
    return torch.as_tensor([world.n_entities if a == UNKNOWN else a for a in t], dtype=torch.long)


class GPT2Knowledge:
    """Everything needed to run the adapter: tokenizer, entity names, model."""

    def __init__(self, cfg: AdapterConfig, n_entities: int = 256):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        lm = GPT2LMHeadModel.from_pretrained("gpt2")
        lm.eval()
        self.entity_ids = select_entities(self.tok, n_entities)
        self.names = [self.tok.decode([i]) for i in self.entity_ids]
        unk = self.tok.encode(UNKNOWN_WORD)
        assert len(unk) == 1, unk
        self.unknown_id = unk[0]
        self.model = KnowledgeAdapterLM(lm, cfg, self.entity_ids, self.unknown_id)
        self.n_entities = n_entities
        self.n_synonyms = 2

    @torch.no_grad()
    def predict(self, bank: Optional[Bank], world: World, queries: Sequence[Query], batch_size: int = 64,
                cell_mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        self.model.eval()
        tensors = bank.tensors() if bank is not None else None
        mask_t = None if cell_mask is None else torch.as_tensor(cell_mask, dtype=torch.bool)
        answers, full_top1, routing, hidden, cand_logits = [], [], [], [], []
        for i in range(0, len(queries), batch_size):
            chunk = list(queries[i: i + batch_size])
            ids, am, last = encode_texts(self.tok, [query_text(q, self.names, self.n_synonyms) for q in chunk])
            cand, full, r, h = self.model(tensors, ids, am, last, cell_mask=mask_t)
            a = cand.argmax(-1).numpy()
            answers.append(np.where(a == self.n_entities, UNKNOWN, a))
            full_top1.append(full.argmax(-1).numpy())
            hidden.append(h.numpy()); cand_logits.append(cand.numpy())
            if r is not None:
                routing.append(r.numpy())
        return {"answers": np.concatenate(answers), "full_top1": np.concatenate(full_top1),
                "routing": np.concatenate(routing) if routing else None, "hidden": np.concatenate(hidden),
                "logits": np.concatenate(cand_logits)}


def train_adapter(gk: GPT2Knowledge, seed: int, steps: int, batch_size: int = 32, route_weight: float = 0.5,
                  lr: float = 1e-3, verbose: bool = True) -> Dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    centre = make_centre(seed, gk.model.cfg.marker_dim)
    model = gk.model
    params = model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    tcfg = TrainConfig(seed=seed, n_steps=steps, lr=lr, warmup=100)
    mix = {"fwd1": 0.7, "fwd2": 0.3}
    history = []
    t0 = time.time()
    for step in range(steps):
        model.train()
        n_cells = int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, gk.n_synonyms)
        bank = bank_from_world(rng, world, centre, 0.10, 0.05, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size, mix)
        ids, am, last = encode_texts(gk.tok, [query_text(q, gk.names, gk.n_synonyms) for q in queries])
        target = targets_of(queries, bank, world)
        route = route_targets(queries, bank, world, len(model.cfg.read_layers))
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        cand, _, routing, _ = model(bank.tensors(), ids, am, last)
        loss_ans = F.cross_entropy(cand, target)
        loss = loss_ans + route_weight * routing_loss(routing, route)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1) == target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()), "batch_acc": acc,
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {step + 1:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  acc {acc:.3f}  {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def adapter_state(model: KnowledgeAdapterLM) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if not k.startswith("lm.")}


def train_or_load(gk: GPT2Knowledge, seed: int, steps: int, force: bool = False, batch_size: int = 32) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000008_gpt2_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"], "loaded": True}
    out = train_adapter(gk, seed, steps, batch_size=batch_size)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["loaded"] = False
    return out


EVAL = dict(n_cells=1000, n_alt_structures=25, n_hop2=300, n_broken=100, n_lifecycle=100, n_locality_updates=100,
            n_locality_revokes=50, n_targets=100)


def evaluate(gk: GPT2Knowledge, seed: int, centre: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    empty = World(gk.n_entities, 4, gk.n_synonyms, [])
    world = fill_random(rng, inject_alternative_paths(rng, empty, EVAL["n_alt_structures"]), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    ref = ReferenceResolver(store)
    facts = world.facts
    m: Dict[str, Any] = {"seed": seed}

    def bank() -> Bank:
        return bank_from_store(store)

    def q1(f, k=0) -> Query:
        return Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, k),))

    direct0 = [q1(f, 0) for f in facts]
    direct1 = [q1(f, 1) for f in facts]
    truth = np.array([f.obj for f in facts])

    # ---- pretrained prior (no knowledge layer at all)
    prior = gk.predict(None, world, direct0)
    m["prior_direct_acc"] = float((prior["answers"] == truth).mean())
    m["prior_unknown_rate"] = float((prior["answers"] == UNKNOWN).mean())
    prior_full = prior["full_top1"]

    # ---- copy bound: with EVERY cell masked the adapter must add nothing -> back to the prior (E-000002 analogue)
    masked = gk.predict(bank(), world, direct0, cell_mask=np.zeros(len(facts), dtype=bool))
    m["bank_masked_direct_acc"] = float((masked["answers"] == truth).mean())
    m["bank_masked_full_vocab_top1_equals_prior"] = float((masked["full_top1"] == prior_full).mean())
    m["bank_masked_unknown_rate"] = float((masked["answers"] == UNKNOWN).mean())

    # ---- direct, paraphrase, full-vocabulary win, 2-hop, broken
    p0 = gk.predict(bank(), world, direct0)
    m["direct"] = float((p0["answers"] == truth).mean())
    m["direct_full_vocab_top1"] = float((p0["full_top1"] == np.array(gk.entity_ids)[truth]).mean())
    p1 = gk.predict(bank(), world, direct1)
    m["paraphrase"] = float((p1["answers"] == truth).mean())
    prov = 0
    for i, f in enumerate(facts):
        r = p0["routing"][i, -1]
        k = int(r.argmax())
        prov += int(k < len(facts) and int(store.bank()["kid"][k]) == kids[f.key] and r[k] > 0.5)
    m["provenance_direct"] = prov / len(facts)
    hop2 = world.sample_queries(rng, EVAL["n_hop2"], 2, "fwd", require_answer=True)
    ph = gk.predict(bank(), world, hop2)
    m["hop2"] = float(np.mean([a == ref.resolve(q).answer for a, q in zip(ph["answers"], hop2)]))
    for hops in (1, 2):
        broken = world.sample_queries(rng, EVAL["n_broken"], hops, "fwd", require_answer=False)
        m[f"broken{hops}_unknown"] = float((gk.predict(bank(), world, broken)["answers"] == UNKNOWN).mean())

    # ---- lifecycle: each operation applied to 100 cells at once, then all 100 queries compared with the reference
    cells = [facts[int(i)] for i in rng.choice(len(facts), size=EVAL["n_lifecycle"], replace=False)]
    q_life = [q1(f, int(rng.integers(0, 2))) for f in cells]
    new_objs = {f.key: int((f.obj + 1 + rng.integers(0, world.n_entities - 1)) % world.n_entities) for f in cells}

    def check(name: str) -> None:
        a = gk.predict(bank(), world, q_life)["answers"]
        m[name] = float(np.mean([x == ref.resolve(q).answer for x, q in zip(a, q_life)]))

    for f in cells: store.update(kids[f.key], new_objs[f.key])
    check("update")
    for f in cells: store.rollback(kids[f.key], 1)
    check("rollback")
    for f in cells: store.revoke(kids[f.key])
    check("revoke")
    for f in cells: store.restore(kids[f.key])
    check("restore")
    for f in cells: store.shred(kids[f.key])
    check("shred")
    for f in cells: store.resign(kids[f.key])
    check("resign")
    m["lifecycle_all"] = float(np.mean([m[k] for k in ("update", "rollback", "revoke", "restore", "shred", "resign")]))

    # ---- locality
    snapshot = gk.predict(bank(), world, direct0)["answers"]
    n_t = EVAL["n_locality_updates"] + EVAL["n_locality_revokes"]
    t_idx = rng.choice(len(facts), size=n_t, replace=False)
    t_keys = {facts[int(i)].key for i in t_idx}
    for j, i in enumerate(t_idx):
        f = facts[int(i)]
        if j < EVAL["n_locality_updates"]:
            store.update(kids[f.key], int((f.obj + 1) % world.n_entities))
        else:
            store.revoke(kids[f.key])
    after = gk.predict(bank(), world, direct0)["answers"]
    outside = np.array([f.key not in t_keys for f in facts])
    ref_after = np.array([ref.resolve(q).answer for q in direct0])
    m["locality"] = float((snapshot[outside] == after[outside]).mean())
    m["locality_targets_correct"] = float((after[~outside] == ref_after[~outside]).mean())
    for j, i in enumerate(t_idx):
        f = facts[int(i)]
        if j < EVAL["n_locality_updates"]:
            store.rollback(kids[f.key], 1)
        else:
            store.restore(kids[f.key])
    m["locality_undo_exact"] = float(np.array_equal(gk.predict(bank(), world, direct0)["answers"], snapshot))

    # ---- attacks on 100 targets after REVOKE and SHRED (probe trained on active non-targets)
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[: EVAL["n_targets"]]]
    others = [facts[int(i)] for i in perm[EVAL["n_targets"]:]]
    q_t = [q1(f, 0) for f in targets]; q_tp = [q1(f, 1) for f in targets]
    t_truth = [f.obj for f in targets]
    q_o = [q1(f, 0) for f in others]
    po = gk.predict(bank(), world, q_o)
    y_o = np.array([f.obj for f in others]); split = int(0.8 * len(others))
    probe = LinearProbe(po["hidden"].shape[1], gk.n_entities, seed=seed)
    probe.fit(po["hidden"][:split], y_o[:split])
    m["probe_calibration_top1"] = probe.accuracy(po["hidden"][split:], y_o[split:])
    m["probe_calibration_top5"] = probe.accuracy(po["hidden"][split:], y_o[split:], topk=5)
    pos = [int(np.where(store.bank()["kid"] == kids[f.key])[0][0]) for f in targets]
    prior_t = gk.predict(None, world, q_t)["full_top1"]

    def attack(tag: str) -> None:
        p = gk.predict(bank(), world, q_t)
        m[f"{tag}/direct_unknown"] = float((p["answers"] == UNKNOWN).mean())
        m[f"{tag}/direct_acc"] = float((p["answers"] == np.array(t_truth)).mean())
        m[f"{tag}/paraphrase_unknown"] = float((gk.predict(bank(), world, q_tp)["answers"] == UNKNOWN).mean())
        m[f"{tag}/forced_choice_win"] = forced_choice(p["logits"], t_truth, np.random.default_rng(seed), gk.n_entities)
        rk = object_rank(p["logits"], t_truth, gk.n_entities)
        m[f"{tag}/true_obj_top1_among_entities"] = rk["top1"]; m[f"{tag}/true_obj_mean_rank"] = rk["mean_rank"]
        m[f"{tag}/probe_top1"] = probe.accuracy(p["hidden"], np.array(t_truth))
        m[f"{tag}/probe_top5"] = probe.accuracy(p["hidden"], np.array(t_truth), topk=5)
        mass = np.array([p["routing"][i, -1, pp] for i, pp in enumerate(pos)])
        with torch.no_grad():
            vn = gk.model.encode_bank(bank().tensors())["values"].norm(dim=-1).numpy()
        m[f"{tag}/routing_mass_on_target"] = float(mass.mean())
        m[f"{tag}/gated_value_contribution"] = float(np.mean(mass * vn[pos]))
        m[f"{tag}/full_vocab_top1_equals_prior"] = float((p["full_top1"] == prior_t).mean())
        m[f"{tag}/full_vocab_top1_is_unknown_word"] = float((p["full_top1"] == gk.unknown_id).mean())

    attack("active")
    for f in targets: store.revoke(kids[f.key])
    attack("revoke")
    for f in targets: store.restore(kids[f.key])
    for f in targets: store.shred(kids[f.key])
    attack("shred")
    for f in targets: store.resign(kids[f.key])
    m["restored/direct_acc"] = float((gk.predict(bank(), world, q_t)["answers"] == np.array(t_truth)).mean())
    return m


KEYS = ["prior_direct_acc", "bank_masked_direct_acc", "bank_masked_unknown_rate", "direct", "direct_full_vocab_top1", "paraphrase", "provenance_direct", "hop2",
        "broken1_unknown", "broken2_unknown", "update", "rollback", "revoke", "restore", "shred", "resign",
        "lifecycle_all", "locality", "locality_targets_correct", "locality_undo_exact"]
ATTACKS = ["direct_unknown", "direct_acc", "paraphrase_unknown", "forced_choice_win", "true_obj_top1_among_entities",
           "true_obj_mean_rank", "probe_top1", "probe_top5", "routing_mass_on_target", "gated_value_contribution",
           "full_vocab_top1_equals_prior", "full_vocab_top1_is_unknown_word"]


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = AdapterConfig()
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        gk = GPT2Knowledge(cfg)
        print(f"=== seed {seed}: adapter training (frozen GPT-2 core, {sum(p.numel() for p in gk.model.adapter_parameters()):,} adapter params) ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force, batch_size=args.batch)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = evaluate(gk, 800 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]
        per_seed.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    keys = [k for k in per_seed[0] if k != "seed"]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {
        "prior_direct_acc": ("<=", 0.05), "bank_masked_direct_acc": ("<=", 0.05), "direct": (">=", 0.95),
        "paraphrase": (">=", 0.95), "broken1_unknown": (">=", 0.90), "revoke": (">=", 0.95), "restore": (">=", 0.95),
        "update": (">=", 0.95), "rollback": (">=", 0.95), "shred": (">=", 0.90), "resign": (">=", 0.95),
        "locality": (">=", 0.98), "revoke/probe_top1": ("<=", 0.05), "revoke/forced_choice_win": ("<=", 0.6),
        "shred/probe_top1": ("<=", 0.05), "shred/forced_choice_win": ("<=", 0.6), "restored/direct_acc": (">=", 0.95)})
    record = {
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "by_construction_vs_learned": "The frozen core cannot copy a fact by construction; whether the ADAPTER copies is "
                                      "measured by the masked-bank rows (must equal the prior). REVOKE is a mask (F1); "
                                      "what is learned is reading the right cell from natural-language prompts, "
                                      "turning the value into the object token through the unchanged LM head, "
                                      "answering ' unknown' for null reads, and refusing a shredded payload.",
        "experiment": "E-000008", "title": "Frozen pretrained GPT-2 core with the mutable knowledge layer (symlink adapter)",
        "evidence_level": "E5", "deletion_level": "F4",
        "claim": "With a frozen pretrained transformer as neural core and natural-language queries, an adapter learns "
                 "to read the mutable knowledge layer so that the model's own LM head emits the object; UPDATE / "
                 "ROLLBACK / REVOKE / RESTORE / SHRED / RESIGN are reproduced against the reference, deletion "
                 "generalises across paraphrases, unrelated cells are unaffected, and after REVOKE / SHRED the object "
                 "is not recoverable by forced choice, logit rank or a linear probe on the final hidden state.",
        "not_claimed": "E6 at LLM scale: GPT-2 small (124M) on CPU; entities are single tokens; the core is frozen, "
                       "so nothing is shown about facts already encoded in pretrained weights.",
        "config": {"seeds": args.seeds, "steps": args.steps, "batch_size": args.batch, "adapter": cfg.to_dict(), "eval": EVAL,
                   "entity_names_note": "entities are the first 256 capitalised single BPE tokens of GPT-2 (some are word fragments)",
                   "templates": TEMPLATES, "nouns": NOUN},
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, ledger.pct(agg[k]["mean"]), ledger.pct(agg[k]["min"])) for k in KEYS]
    arows = [(a, *(f"{agg[f'{c}/{a}']['mean']:.4f}" for c in ("active", "revoke", "shred"))) for a in ATTACKS]
    md = "\n".join([
        "# E-000008 — Frozen pretrained GPT-2 core with the mutable knowledge layer", "",
        f"Evidence level: **E5** ({ledger.EVIDENCE_LEVELS['E5']}), partial E6 (a real pretrained LM, GPT-2 small, "
        f"on CPU). Deletion level within this system: **F4**. Seeds: {args.seeds}; adapter steps: {args.steps}; "
        "the 124M pretrained weights are frozen.", "",
        ledger.table(["measure", "mean over seeds", "worst seed"], rows), "",
        "Attacks on 100 targets (mean over seeds; chance: forced choice 0.5, top-1 among entities 0.0039, "
        "mean rank 127.5, probe top-1 0.0039 / top-5 0.0195):", "",
        ledger.table(["attack", "active", "after REVOKE", "after SHRED"], arows), "",
        f"Probe calibration on held-out active cells: top-1 {agg['probe_calibration_top1']['mean']:.3f}, "
        f"top-5 {agg['probe_calibration_top5']['mean']:.3f}.", "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        record["by_construction_vs_learned"], "",
        "Reading: 'prior_direct_acc' is what frozen GPT-2 answers without the layer (chance); 'bank_masked_direct_acc' "
        "is the adapter with every cell masked — the copy bound: it must not exceed the prior. "
        "'direct_full_vocab_top1' is the fraction of direct queries where the object token wins over the entire "
        "50,257-token vocabulary, not only among the 257 candidates. 'full_vocab_top1_equals_prior' after REVOKE "
        "shows whether the model falls back to its pretrained prior once the cell is gone.",
    ])
    path = ledger.save("e000008_gpt2_adapter", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
