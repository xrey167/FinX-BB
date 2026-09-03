"""Experiment E-000013 — prior conflict in the frozen GPT-2 core.

Roadmap stage 2, item 4.  Real countries whose capitals GPT-2 small already knows
(restricted prior 48/50 on "The capital of {s} is") receive COUNTERFACTUAL capital cells.
Claims under test:

    * override: while the cell is ACTIVE the frozen model names the counterfactual capital;
    * fallback: after REVOKE / SHRED the model's distribution returns to the pretrained one
      (KL to the base model, top-1 agreement), the counterfactual object is not recoverable
      (probe, forced choice), and the pretrained fact — untouched by construction — is what
      the model answers again;
    * no key, no injection: prompts without a matching cell (broken paths, generic text)
      leave the pretrained distribution unchanged while a full bank is attached.

Design change against E-000011 / E-000012: the adapter runs in ``fallback="prior"`` mode —
the payload is unit-RMS, the injection is scaled statically, the null value is a fixed zero
and an unsigned or revoked cell injects nothing — so "nothing found" is not a trained
' unknown' token but the absence of any intervention.  Training therefore uses a full-vocabulary
cross-entropy on answerable queries and KL(base || adapter) on unanswerable ones.

The copy bound (masked bank == base model) is BY CONSTRUCTION in this design: the adapter's
parameters act only through the injection.  It is recorded, not claimed as learned.

Run:  python -m so.experiments.e000013_prior_conflict [--seeds 0 1 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, bank_from_store, bank_from_world, sample_training_queries
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, _sha256
from so.experiments.e000012_status_gated_revoke import route_targets_status_gated
from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.train import TrainConfig, lr_at, make_centre, routing_loss
from so.world import Fact, Query, UNKNOWN, World, fill_random

# 50 country / capital pairs whose names are single GPT-2 tokens (with a leading space); measured before the
# experiment: restricted prior (top-1 among the 50 capital tokens) 48/50 on template 0, 50/50 on template 2.
PAIRS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
         ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("England", "London"),
         ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Ireland", "Dublin"), ("Sweden", "Stockholm"),
         ("Norway", "Oslo"), ("Poland", "Warsaw"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
         ("India", "Delhi"), ("Peru", "Lima"), ("Chile", "Santiago"), ("Cuba", "Havana"), ("Hungary", "Budapest"),
         ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Denmark", "Copenhagen"), ("Finland", "Helsinki"),
         ("Scotland", "Edinburgh"), ("Wales", "Cardiff"), ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Israel", "Jerusalem"),
         ("Libya", "Tripoli"), ("Thailand", "Bangkok"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
         ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"), ("Australia", "Canberra"), ("Argentina", "Buenos"),
         ("Ukraine", "Kiev"), ("Czech", "Prague"), ("Switzerland", "Bern"), ("Korea", "Seoul"), ("Tunisia", "Tunis"),
         ("Jamaica", "Kingston"), ("Haiti", "Port"), ("Malaysia", "Kuala")]

# relation 0 is the prior-laden one; 1-3 are the prior-free relations of E-000008/E-000011.  Templates 0/1 are used in
# training (n_synonyms = 2), 2/3 are held-out paraphrases.
TEMPLATES_PC = {
    0: ["The capital of {s} is", "The capital city of {s} is", "Q: What is the capital of {s}? A:", "{s}'s capital city is"],
    1: ["{s} lives in", "The home of {s} is", "{s} resides in", "Q: Where does {s} live? A: In"],
    2: ["{s} works for", "The employer of {s} is", "{s} is employed by", "Q: Who does {s} work for? A:"],
    3: ["{s} was born in", "The birthplace of {s} is", "{s} comes from", "Q: Where was {s} born? A: In"],
}
NOUN_PC = {0: "capital", 1: "home", 2: "employer", 3: "birthplace"}   # must match TEMPLATES_PC: used for 2-hop prompts
TRAIN_TEMPLATES, HELDOUT_TEMPLATES = (0, 1), (2, 3)
GENERIC = ["{s} said that", "The story of {s} begins", "In the morning, {s}", "Everyone knows that {s}", "{s} walked into the"]
N_NAMES = 256

EVAL = dict(n_cells=1000, n_hop2=200, n_broken=100, n_generic=200, n_locality_updates=100, n_locality_revokes=50)


def query_text_pc(q: Query, names: List[str], n_synonyms: int, template: Optional[int] = None) -> str:
    """E-000008's prompt builder over THIS experiment's templates and relation nouns.

    E-000013 remaps the relations (0 is the prior-laden "capital"), so E-000008's module-level
    TEMPLATES / NOUN must not be used here — and must not be monkeypatched either, because
    E-000011 and E-000012 index templates 4 and 5 of their own table in the same process.
    """
    s = names[q.start]
    if q.hops == 1:
        r, k = q.path[0], (q.surface[0] % n_synonyms if template is None else template)
        return TEMPLATES_PC[r][k].format(s=s)
    inner = " of the ".join(NOUN_PC[r] for r in reversed(q.path))
    return f"The {inner} of {s} is"


class GPT2KnowledgePrior(E8.GPT2Knowledge):
    """Entity table = 256 prior-free names + 50 countries + 50 capitals (all single tokens)."""

    def __init__(self, cfg: AdapterConfig):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        lm = GPT2LMHeadModel.from_pretrained("gpt2")
        lm.eval()
        country_ids, capital_ids = [], []
        for c, k in PAIRS:
            ci, ki = self.tok.encode(" " + c), self.tok.encode(" " + k)
            assert len(ci) == 1 and len(ki) == 1, (c, k)
            country_ids.append(ci[0]); capital_ids.append(ki[0])
        taken = set(country_ids) | set(capital_ids)
        names = [i for i in E8.select_entities(self.tok, N_NAMES + 64) if i not in taken][:N_NAMES]
        assert len(names) == N_NAMES
        self.entity_ids = names + country_ids + capital_ids
        self.names = [self.tok.decode([i]) for i in self.entity_ids]
        self.n_names = N_NAMES
        self.country_idx = list(range(N_NAMES, N_NAMES + len(PAIRS)))
        self.capital_idx = list(range(N_NAMES + len(PAIRS), N_NAMES + 2 * len(PAIRS)))
        unk = self.tok.encode(E8.UNKNOWN_WORD)
        self.unknown_id = unk[0]
        self.model = KnowledgeAdapterLM(lm, cfg, self.entity_ids, self.unknown_id)
        self.n_entities = len(self.entity_ids)
        self.n_synonyms = 2

    @torch.no_grad()
    def predict_full(self, bank: Optional[Bank], queries: Sequence[Query], template: Optional[int] = None,
                     cell_mask: Optional[np.ndarray] = None, texts: Optional[List[str]] = None, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """Like predict, but also returns the full-vocabulary logits at the last token."""
        self.model.eval()
        tensors = bank.tensors() if bank is not None else None
        mask_t = None if cell_mask is None else torch.as_tensor(cell_mask, dtype=torch.bool)
        if texts is None:
            texts = [query_text_pc(q, self.names, self.n_synonyms, template) for q in queries]
        out: Dict[str, List[np.ndarray]] = {"answers": [], "full": [], "cand": [], "hidden": [], "routing": []}
        for i in range(0, len(texts), batch_size):
            ids, am, last = E8.encode_texts(self.tok, texts[i: i + batch_size])
            cand, full, r, h = self.model(tensors, ids, am, last, cell_mask=mask_t)
            a = cand.argmax(-1).numpy()
            out["answers"].append(np.where(a == self.n_entities, UNKNOWN, a))
            out["full"].append(full.numpy()); out["cand"].append(cand.numpy()); out["hidden"].append(h.numpy())
            if r is not None:
                out["routing"].append(r.numpy())
        res = {k: np.concatenate(v) for k, v in out.items() if v}
        res.setdefault("routing", None)
        return res


def kl_rows(base_full: np.ndarray, adapter_full: np.ndarray) -> np.ndarray:
    """KL(base || adapter) per row, in nats, over the full vocabulary."""
    b = torch.log_softmax(torch.as_tensor(base_full), -1)
    a = torch.log_softmax(torch.as_tensor(adapter_full), -1)
    return (b.exp() * (b - a)).sum(-1).numpy()


# ---------------------------------------------------------------------------------------------- training
def train_adapter_prior(gk: GPT2KnowledgePrior, seed: int, steps: int, batch_size: int = 32, route_weight: float = 1.0,
                        gate_weight: float = 5.0, fallback_weight: float = 1.0, lr: float = 2e-3, route_only_steps: int = 300,
                        p_revoked: float = 0.20, p_shred: float = 0.10, extra_unanswerable: float = 0.2,
                        verbose: bool = True) -> Dict[str, Any]:
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
    model.eval()          # the frozen core keeps dropout OFF: base and adapter forward must be comparable
    for step in range(steps):
        route_only = step < route_only_steps
        n_cells = int(rng.integers(150, 301)) if route_only else int(rng.integers(700, 1001))
        world = World.sample(rng, gk.n_entities, 4, n_cells, gk.n_synonyms)
        bank = bank_from_world(rng, world, centre, p_revoked, p_shred, 0.05)
        queries = sample_training_queries(rng, world, bank, batch_size - n_extra, mix)
        queries += world.sample_queries(rng, n_extra, 1, "fwd", require_answer=False, index=bank.index_view)
        ids, am, last = E8.encode_texts(gk.tok, [query_text_pc(q, gk.names, gk.n_synonyms) for q in queries])
        target = E8.targets_of(queries, bank, world)
        answerable = target != gk.n_entities
        n_edges = np.array([len(world.answer(q, bank.index_view).edges) for q in queries])
        fb_mask = torch.as_tensor((~answerable.numpy()) & (n_edges == 0))
        route = route_targets_status_gated(queries, bank, world, n_reads)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tcfg)
        tensors = bank.tensors()
        with torch.no_grad():
            base_full = model(None, ids, am, last)[1]
        cand, full, routing, _ = model(tensors, ids, am, last)
        if answerable.any():
            loss_ans = F.cross_entropy(full[answerable], model.entity_token_ids[target[answerable]])
        else:
            loss_ans = full.sum() * 0
        if fb_mask.any():
            loss_fb = F.kl_div(F.log_softmax(full[fb_mask], -1), F.log_softmax(base_full[fb_mask], -1),
                               log_target=True, reduction="batchmean")
        else:
            loss_fb = full.sum() * 0
        loss_route = routing_loss(routing, route)
        valid = tensors["marker_valid"].float()
        per_cell = F.binary_cross_entropy_with_logits(model.gate_logits(tensors["marker"]).squeeze(-1), valid, reduction="none")
        n_pos, n_neg = valid.sum().clamp_min(1), (1 - valid).sum().clamp_min(1)
        loss_gate = 0.5 * (per_cell * valid).sum() / n_pos + 0.5 * (per_cell * (1 - valid)).sum() / n_neg
        if route_only:
            loss = loss_route + gate_weight * loss_gate
        else:
            loss = loss_ans + fallback_weight * loss_fb + route_weight * loss_route + gate_weight * loss_gate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            acc = (cand.argmax(-1)[answerable] == target[answerable]).float().mean().item() if answerable.any() else float("nan")
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "fallback_kl": float(loss_fb.item()), "route_loss": float(loss_route.item()),
                   "gate_loss": float(loss_gate.item()), "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {step + 1:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  kl {rec['fallback_kl']:.4f}  "
                      f"route {rec['route_loss']:.4f}  gate {rec['gate_loss']:.4f}  acc {acc:.3f}  {rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"centre": centre, "history": history, "train_seconds": time.time() - t0}


def train_or_load(gk: GPT2KnowledgePrior, seed: int, steps: int, force: bool = False) -> Dict[str, Any]:
    path = CHECKPOINTS / f"e000013_gpt2_seed{seed}.pt"
    if path.exists() and not force:
        ck = torch.load(path, weights_only=False)
        gk.model.load_state_dict(ck["adapter"], strict=False)
        gk.model.eval()
        return {"centre": ck["centre"], "history": ck["history"], "train_seconds": ck["train_seconds"], "loaded": True,
                "checkpoint_sha256": _sha256(path)}
    out = train_adapter_prior(gk, seed, steps)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": E8.adapter_state(gk.model), "centre": out["centre"], "history": out["history"],
                "train_seconds": out["train_seconds"], "adapter_config": gk.model.cfg.to_dict()}, path)
    out["loaded"] = False
    out["checkpoint_sha256"] = _sha256(path)
    return out


# ---------------------------------------------------------------------------------------------- evaluation
def evaluate(gk: GPT2KnowledgePrior, seed: int, centre: np.ndarray) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_ent, n_c = gk.n_entities, len(PAIRS)
    country, capital = np.array(gk.country_idx), np.array(gk.capital_idx)
    # a derangement: every country gets somebody else's capital
    while True:
        perm = rng.permutation(n_c)
        if not np.any(perm == np.arange(n_c)):
            break
    cf_facts = [Fact(int(country[i]), 0, int(capital[perm[i]])) for i in range(n_c)]
    world = fill_random(rng, World(n_ent, 4, gk.n_synonyms, cf_facts), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    kids = load_world(store, world)
    ref = ReferenceResolver(store)
    facts = world.facts
    filler = [f for f in facts if f.key not in {f2.key for f2 in cf_facts}]
    m: Dict[str, Any] = {"seed": seed}

    def bank() -> Bank:
        return bank_from_store(store)

    def q1(f) -> Query:
        return Query("fwd", f.subject, (f.relation,), (world.surface_of(f.relation, 0),))

    q_c = [q1(f) for f in cf_facts]                       # the 50 prior-laden queries
    cf_obj = np.array([f.obj for f in cf_facts])          # counterfactual capitals (entity ids)
    true_obj = capital                                    # the pretrained truth (entity ids)
    cap_tok = np.array(gk.entity_ids)[capital]            # capital token ids
    q_f = [q1(f) for f in filler]
    f_truth = np.array([f.obj for f in filler])

    # ---- the pretrained prior (no adapter) on every template
    base = {t: gk.predict_full(None, q_c, template=t) for t in range(4)}
    for t in range(4):
        sub = base[t]["full"][:, cap_tok]
        m[f"prior/template{t}_restricted_top1"] = float((sub.argmax(-1) == np.arange(n_c)).mean())
        m[f"prior/template{t}_full_top1_is_true_capital"] = float((base[t]["full"].argmax(-1) == cap_tok).mean())
    m["prior/restricted_top1"] = m["prior/template0_restricted_top1"]
    p = torch.softmax(torch.as_tensor(base[0]["full"]), -1).numpy()
    m["prior/true_capital_prob"] = float(p[np.arange(n_c), cap_tok].mean())
    m["prior/counterfactual_prob"] = float(p[np.arange(n_c), np.array(gk.entity_ids)[cf_obj]].mean())
    m["prior/counterfactual_top1"] = float((base[0]["answers"] == cf_obj).mean())
    m["prior/counterfactual_top1_pooled"] = float(np.mean([(base[t]["answers"] == cf_obj).mean() for t in range(4)]))
    m["prior/forced_choice_win"] = forced_choice(base[0]["cand"], cf_obj, np.random.default_rng(seed), n_ent)
    rk0 = object_rank(base[0]["cand"], cf_obj, n_ent)
    m["prior/counterfactual_top1_among_entities"] = rk0["top1"]
    m["prior/counterfactual_mean_rank"] = rk0["mean_rank"]
    base_f = gk.predict_full(None, q_f)

    # ---- copy bound (by construction in this design): every cell masked == base model
    masked_c = gk.predict_full(bank(), q_c, cell_mask=np.zeros(len(facts), dtype=bool))
    masked_f = gk.predict_full(bank(), q_f, cell_mask=np.zeros(len(facts), dtype=bool))
    kl = np.concatenate([kl_rows(base[0]["full"], masked_c["full"]), kl_rows(base_f["full"], masked_f["full"])])
    m["masked/kl_to_base"] = float(kl.mean()); m["masked/kl_to_base_max"] = float(kl.max())
    m["masked/top1_matches_base"] = float(np.concatenate([masked_c["full"].argmax(-1) == base[0]["full"].argmax(-1),
                                                          masked_f["full"].argmax(-1) == base_f["full"].argmax(-1)]).mean())

    # ---- prior-free reading with the full bank attached
    for t in range(4):
        pf = gk.predict_full(bank(), q_f, template=t)
        tag = f"template{t}_" + ("train" if t in TRAIN_TEMPLATES else "heldout")
        m[f"{tag}/direct"] = float((pf["answers"] == f_truth).mean())
        m[f"{tag}/full_vocab_top1"] = float((pf["full"].argmax(-1) == np.array(gk.entity_ids)[f_truth]).mean())
        if t == 0:
            pos = [int(np.where(store.bank()["kid"] == kids[f.key])[0][0]) for f in filler]
            r = pf["routing"][:, -1]
            m["provenance_direct"] = float(np.mean([int(r[i].argmax()) == pos[i] and r[i, pos[i]] > 0.5 for i in range(len(filler))]))
    m["direct"] = m["template0_train/direct"]
    m["direct_heldout_min"] = float(min(m[f"template{t}_heldout/direct"] for t in HELDOUT_TEMPLATES))
    hop2 = world.sample_queries(rng, EVAL["n_hop2"], 2, "fwd", require_answer=True)
    m["hop2"] = float(np.mean([a == ref.resolve(q).answer for a, q in zip(gk.predict_full(bank(), hop2)["answers"], hop2)]))

    # ---- no key, no injection: broken paths and generic text with the full bank attached
    broken = world.sample_queries(rng, EVAL["n_broken"], 1, "fwd", require_answer=False)
    pb, bb = gk.predict_full(bank(), broken), gk.predict_full(None, broken)
    kl = kl_rows(bb["full"], pb["full"])
    m["broken1/kl_to_base"] = float(kl.mean()); m["broken1/kl_to_base_max"] = float(kl.max())
    m["broken1/top1_matches_base"] = float((pb["full"].argmax(-1) == bb["full"].argmax(-1)).mean())
    m["broken1/routing_mass_on_null"] = float(pb["routing"][:, -1, -1].mean())
    m["broken1/routing_mass_on_null_first_read"] = float(pb["routing"][:, 0, -1].mean())
    gen_idx = [int(rng.integers(0, len(GENERIC))) for _ in range(EVAL["n_generic"])]
    gen_texts = [GENERIC[i].format(s=gk.names[int(rng.integers(0, gk.n_names))]) for i in gen_idx]
    pg, bg = gk.predict_full(bank(), [], texts=gen_texts), gk.predict_full(None, [], texts=gen_texts)
    kl = kl_rows(bg["full"], pg["full"])
    m["generic/kl_to_base"] = float(kl.mean()); m["generic/kl_to_base_max"] = float(kl.max())
    m["generic/top1_matches_base"] = float((pg["full"].argmax(-1) == bg["full"].argmax(-1)).mean())
    m["generic/routing_mass_on_null"] = float(pg["routing"][:, -1, -1].mean())
    m["generic/routing_mass_on_null_first_read"] = float(pg["routing"][:, 0, -1].mean())
    gi = np.array(gen_idx)
    per_prompt = []
    for k in range(len(GENERIC)):
        sel = gi == k
        if not sel.any():
            continue
        m[f"generic/prompt{k}_kl_to_base"] = float(kl[sel].mean())
        m[f"generic/prompt{k}_top1_matches_base"] = float((pg["full"].argmax(-1)[sel] == bg["full"].argmax(-1)[sel]).mean())
        per_prompt.append(m[f"generic/prompt{k}_kl_to_base"])
    # two of the prompts end on the subject token, i.e. a half key the model never saw in training:
    # the worst single prompt is reported separately so a pooled mean cannot hide it
    m["generic/kl_to_base_worst_prompt"] = float(max(per_prompt)) if per_prompt else float("nan")

    # ---- override while ACTIVE (counterfactual cells), on every template
    def override(tag: str, objs: np.ndarray) -> None:
        for t in range(4):
            po = gk.predict_full(bank(), q_c, template=t)
            m[f"{tag}/template{t}_direct"] = float((po["answers"] == objs).mean())
            m[f"{tag}/template{t}_full_vocab_top1"] = float((po["full"].argmax(-1) == np.array(gk.entity_ids)[objs]).mean())
            sub = po["full"][:, cap_tok]
            m[f"{tag}/template{t}_true_capital_restricted_top1"] = float((sub.argmax(-1) == np.arange(n_c)).mean())
        m[f"{tag}/direct"] = m[f"{tag}/template0_direct"]
        m[f"{tag}/full_vocab_top1"] = m[f"{tag}/template0_full_vocab_top1"]
        m[f"{tag}_heldout_min"] = float(min(m[f"{tag}/template{t}_direct"] for t in HELDOUT_TEMPLATES))
        m[f"{tag}/true_capital_restricted_top1"] = m[f"{tag}/template0_true_capital_restricted_top1"]

    override("override", cf_obj)
    for f in cf_facts: store.update(kids[f.key], int(true_obj[gk.country_idx.index(f.subject)]))
    override("agree", true_obj)                           # cells that agree with the prior
    for f in cf_facts: store.rollback(kids[f.key], 1)
    m["rollback/direct"] = float((gk.predict_full(bank(), q_c)["answers"] == cf_obj).mean())

    # ---- probe calibration on filler facts (hidden state -> object), as in E-000011
    pf0 = gk.predict_full(bank(), q_f)
    split = int(0.8 * len(filler))
    probe = LinearProbe(pf0["hidden"].shape[1], n_ent, seed=seed)
    probe.fit(pf0["hidden"][:split], f_truth[:split])
    m["probe_calibration_top1"] = probe.accuracy(pf0["hidden"][split:], f_truth[split:])
    m["prior/probe_top1"] = probe.accuracy(base[0]["hidden"], cf_obj)      # what the probe reads without any injection
    pos_c = [int(np.where(store.bank()["kid"] == kids[f.key])[0][0]) for f in cf_facts]

    # ---- fallback after REVOKE / SHRED: the pretrained distribution returns, nothing counterfactual remains
    def fallback(tag: str) -> None:
        kls = []
        for t in range(4):
            pa = gk.predict_full(bank(), q_c, template=t)
            kl = kl_rows(base[t]["full"], pa["full"])
            kls.append(float(kl.mean()))
            m[f"{tag}/template{t}_kl_to_base"] = float(kl.mean())
            m[f"{tag}/template{t}_top1_matches_base"] = float((pa["full"].argmax(-1) == base[t]["full"].argmax(-1)).mean())
            m[f"{tag}/template{t}_counterfactual_top1"] = float((pa["answers"] == cf_obj).mean())
            sub = pa["full"][:, cap_tok]
            m[f"{tag}/template{t}_true_capital_restricted_top1"] = float((sub.argmax(-1) == np.arange(n_c)).mean())
            m[f"{tag}/template{t}_restricted_matches_base"] = float((sub.argmax(-1) == base[t]["full"][:, cap_tok].argmax(-1)).mean())
            if t == 0:
                m[f"{tag}/kl_to_base"] = float(kl.mean()); m[f"{tag}/kl_to_base_max"] = float(kl.max())
                m[f"{tag}/top1_matches_base"] = m[f"{tag}/template0_top1_matches_base"]
                m[f"{tag}/restricted_matches_base"] = m[f"{tag}/template0_restricted_matches_base"]
                m[f"{tag}/counterfactual_top1"] = m[f"{tag}/template0_counterfactual_top1"]
                m[f"{tag}/counterfactual_full_top1"] = float((pa["full"].argmax(-1) == np.array(gk.entity_ids)[cf_obj]).mean())
                m[f"{tag}/true_capital_restricted_top1"] = m[f"{tag}/template0_true_capital_restricted_top1"]
                m[f"{tag}/forced_choice_win"] = forced_choice(pa["cand"], cf_obj, np.random.default_rng(seed), n_ent)
                rk = object_rank(pa["cand"], cf_obj, n_ent)
                m[f"{tag}/counterfactual_top1_among_entities"] = rk["top1"]; m[f"{tag}/counterfactual_mean_rank"] = rk["mean_rank"]
                m[f"{tag}/probe_top1"] = probe.accuracy(pa["hidden"], cf_obj)
                m[f"{tag}/routing_mass_on_target"] = float(np.mean([pa["routing"][i, -1, pp] for i, pp in enumerate(pos_c)]))
                with torch.no_grad():
                    enc = gk.model.encode_bank(bank().tensors())
                m[f"{tag}/gate_on_target"] = float(enc["gate"].numpy()[pos_c].mean())
                m[f"{tag}/injection_rms_share"] = float(np.mean([pa["routing"][i, -1, pp] * enc["gate"].numpy()[pp] for i, pp in enumerate(pos_c)]))
        m[f"{tag}/heldout_kl_max"] = float(max(kls[t] for t in HELDOUT_TEMPLATES))
        m[f"{tag}/kl_to_base_pooled"] = float(np.mean(kls))
        # pooled over item x template (n = 4 x 50 per seed): 50 items cannot resolve a 0.05 bar
        m[f"{tag}/counterfactual_top1_pooled"] = float(np.mean([m[f"{tag}/template{t}_counterfactual_top1"] for t in range(4)]))
        m[f"{tag}/top1_matches_base_pooled"] = float(np.mean([m[f"{tag}/template{t}_top1_matches_base"] for t in range(4)]))
        # PAIRED excess over the frozen model itself: the counterfactual object is a real capital token that the
        # pretrained prior already favours, so only the excess over the base model can show leakage
        m[f"{tag}/forced_choice_excess"] = m[f"{tag}/forced_choice_win"] - m["prior/forced_choice_win"]
        m[f"{tag}/probe_excess"] = m[f"{tag}/probe_top1"] - m["prior/probe_top1"]
        m[f"{tag}/counterfactual_top1_excess"] = m[f"{tag}/counterfactual_top1_pooled"] - m["prior/counterfactual_top1_pooled"]
        m[f"{tag}/counterfactual_rank_excess"] = m["prior/counterfactual_mean_rank"] - m[f"{tag}/counterfactual_mean_rank"]
        # retention: what happens to the counterfactual cells must not touch the 950 prior-free filler cells
        pfx = gk.predict_full(bank(), q_f)
        m[f"{tag}/filler_direct"] = float((pfx["answers"] == f_truth).mean())
        m[f"{tag}/filler_kl_to_active"] = float(kl_rows(pf0["full"], pfx["full"]).mean())
        with torch.no_grad():
            tb = bank().tensors()
            gate_all = gk.model.encode_bank(tb)["gate"].numpy()
        mv = tb["marker_valid"].numpy()
        m[f"{tag}/gate_on_signed_cells"] = float(gate_all[mv].mean()) if mv.any() else float("nan")
        m[f"{tag}/gate_on_unsigned_cells"] = float(gate_all[~mv].mean()) if (~mv).any() else float("nan")

    # positive control: with the counterfactual cells ACTIVE every attack must SUCCEED, otherwise
    # "nothing is recoverable after deletion" is vacuous (ledger section 28)
    fallback("active")
    for f in cf_facts: store.revoke(kids[f.key])
    fallback("revoke")
    for f in cf_facts: store.restore(kids[f.key])
    m["restored/direct"] = float((gk.predict_full(bank(), q_c)["answers"] == cf_obj).mean())
    for f in cf_facts: store.shred(kids[f.key])
    fallback("shred_soft")
    gk.model.cfg.hard_gate = True
    fallback("shred_hard")
    gk.model.cfg.hard_gate = False
    for f in cf_facts: store.resign(kids[f.key])
    m["resigned/direct"] = float((gk.predict_full(bank(), q_c)["answers"] == cf_obj).mean())

    # ---- locality: updates / revokes of filler cells leave every other answer unchanged
    snapshot = gk.predict_full(bank(), q_f)["answers"]
    n_t = EVAL["n_locality_updates"] + EVAL["n_locality_revokes"]
    t_idx = rng.choice(len(filler), size=n_t, replace=False)
    t_keys = {filler[int(i)].key for i in t_idx}
    for j, i in enumerate(t_idx):
        f = filler[int(i)]
        store.update(kids[f.key], int((f.obj + 1) % n_ent)) if j < EVAL["n_locality_updates"] else store.revoke(kids[f.key])
    after = gk.predict_full(bank(), q_f)["answers"]
    outside = np.array([f.key not in t_keys for f in filler])
    m["locality"] = float((snapshot[outside] == after[outside]).mean())
    m["locality_counterfactual_unchanged"] = float((gk.predict_full(bank(), q_c)["answers"] == cf_obj).mean())
    for j, i in enumerate(t_idx):
        f = filler[int(i)]
        store.rollback(kids[f.key], 1) if j < EVAL["n_locality_updates"] else store.restore(kids[f.key])
    m["locality_undo_exact"] = float(np.array_equal(gk.predict_full(bank(), q_f)["answers"], snapshot))
    return m


def criteria_groups():
    """Pre-registered before the first run (no checkpoint existed when these thresholds were written).

    Two conventions that the review of this protocol forced:
      * every attack bar is a PAIRED EXCESS over the frozen model itself.  The counterfactual object is a
        real capital token, so an absolute forced-choice or probe bar would measure GPT-2's prior and not
        leakage from the cell.
      * ``attack_validity`` is a validity condition, not a claim: if the same attacks do not succeed while
        the cell is ACTIVE, then their failure after deletion says nothing.
    """
    groups = {
        "copy_bound_by_construction": {"masked/kl_to_base": ("<=", 0.05), "masked/top1_matches_base": (">=", 0.95)},
        "reading_prior_free": {"direct": (">=", 0.95), "template1_train/direct": (">=", 0.95), "direct_heldout_min": (">=", 0.70)},
        "override": {"override/direct": (">=", 0.90), "override/full_vocab_top1": (">=", 0.80), "override_heldout_min": (">=", 0.70),
                     "agree/direct": (">=", 0.95)},
        "attack_validity": {"probe_calibration_top1": (">=", 0.20), "active/probe_top1": (">=", 0.25),
                            "active/counterfactual_top1_excess": (">=", 0.50), "active/forced_choice_excess": (">=", 0.10)},
        "fallback_after_revoke_by_construction": {"revoke/kl_to_base": ("<=", 0.05), "revoke/top1_matches_base_pooled": (">=", 0.95),
                                                  "revoke/counterfactual_top1_excess": ("<=", 0.05), "revoke/probe_excess": ("<=", 0.05),
                                                  "revoke/forced_choice_excess": ("<=", 0.05), "revoke/heldout_kl_max": ("<=", 0.10),
                                                  "revoke/routing_mass_on_target": (">=", 0.90)},
        "fallback_after_shred_soft": {"shred_soft/kl_to_base": ("<=", 0.05), "shred_soft/top1_matches_base_pooled": (">=", 0.95),
                                      "shred_soft/counterfactual_top1_excess": ("<=", 0.05), "shred_soft/probe_excess": ("<=", 0.05),
                                      "shred_soft/forced_choice_excess": ("<=", 0.05), "shred_soft/heldout_kl_max": ("<=", 0.10)},
        "fallback_after_shred_hard": {"shred_hard/kl_to_base": ("<=", 0.05), "shred_hard/top1_matches_base_pooled": (">=", 0.95),
                                      "shred_hard/counterfactual_top1_excess": ("<=", 0.05), "shred_hard/probe_excess": ("<=", 0.05),
                                      "shred_hard/forced_choice_excess": ("<=", 0.05), "shred_hard/heldout_kl_max": ("<=", 0.10)},
        "no_key_no_injection": {"broken1/kl_to_base": ("<=", 0.05), "generic/kl_to_base": ("<=", 0.05),
                                "generic/kl_to_base_worst_prompt": ("<=", 0.10)},
        "retention_under_deletion": {"revoke/filler_direct": (">=", 0.95), "shred_soft/filler_direct": (">=", 0.95),
                                     "shred_hard/filler_direct": (">=", 0.95), "shred_hard/filler_kl_to_active": ("<=", 0.05)},
        "locality_restore": {"locality": (">=", 0.98), "restored/direct": (">=", 0.90), "resigned/direct": (">=", 0.90),
                             "rollback/direct": (">=", 0.90)},
    }
    return groups


def deletion_level(met: Dict[str, bool]) -> str:
    """F3/F4 rest on the LEARNED gate, never on the revoke group.

    In this design a revoked cell's value is exactly zero (the status flag multiplies the gate) and the
    null read is a fixed zero, so "the base distribution returns after REVOKE" is by construction; only
    the routing residue is learned.  SHRED goes through the learned verification gate, so the soft-gate
    group carries F3 and the hard-gate group F4 — and both only if the attacks were valid at all.
    """
    if not met["attack_validity"]:
        return "F1"
    if met["fallback_after_shred_soft"] and met["fallback_after_shred_hard"] and met["retention_under_deletion"]:
        return "F4"
    if met["fallback_after_shred_soft"] and met["retention_under_deletion"]:
        return "F3"
    return "F1"


KEYS = ["prior/restricted_top1", "prior/true_capital_prob", "prior/counterfactual_top1_pooled", "prior/forced_choice_win",
        "prior/probe_top1", "prior/counterfactual_mean_rank", "probe_calibration_top1",
        "masked/kl_to_base", "direct", "template1_train/direct", "direct_heldout_min", "hop2", "provenance_direct",
        "broken1/kl_to_base", "broken1/routing_mass_on_null", "generic/kl_to_base", "generic/kl_to_base_worst_prompt",
        "generic/routing_mass_on_null",
        "override/direct", "override/full_vocab_top1", "override_heldout_min", "override/true_capital_restricted_top1",
        "agree/direct", "rollback/direct",
        "active/probe_top1", "active/forced_choice_excess", "active/counterfactual_top1_excess", "active/kl_to_base",
        "active/routing_mass_on_target", "active/gate_on_target",
        "revoke/kl_to_base", "revoke/top1_matches_base_pooled", "revoke/restricted_matches_base",
        "revoke/true_capital_restricted_top1", "revoke/counterfactual_top1_pooled", "revoke/counterfactual_top1_excess",
        "revoke/probe_top1", "revoke/probe_excess", "revoke/forced_choice_win", "revoke/forced_choice_excess",
        "revoke/heldout_kl_max", "revoke/routing_mass_on_target", "revoke/gate_on_target", "revoke/filler_direct",
        "shred_soft/kl_to_base", "shred_soft/top1_matches_base_pooled", "shred_soft/counterfactual_top1_excess",
        "shred_soft/probe_excess", "shred_soft/forced_choice_excess", "shred_soft/heldout_kl_max",
        "shred_soft/injection_rms_share", "shred_soft/gate_on_unsigned_cells", "shred_soft/filler_direct",
        "shred_hard/kl_to_base", "shred_hard/top1_matches_base_pooled", "shred_hard/counterfactual_top1_excess",
        "shred_hard/probe_excess", "shred_hard/forced_choice_excess", "shred_hard/heldout_kl_max",
        "shred_hard/filler_direct", "shred_hard/filler_kl_to_active",
        "restored/direct", "resigned/direct", "locality", "locality_counterfactual_unchanged"]


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
        gk = GPT2KnowledgePrior(AdapterConfig(status_gated=True, fallback="prior"))
        print(f"=== seed {seed}: adapter training (fallback to prior) ===", flush=True)
        out = train_or_load(gk, seed, args.steps, args.force)
        print(f"=== seed {seed}: evaluating ===", flush=True)
        m = evaluate(gk, 1300 + seed, out["centre"])
        m["train_seconds"] = out["train_seconds"]; m["checkpoint_sha256"] = out["checkpoint_sha256"]
        per_seed.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    keys = [k for k in per_seed[0] if k not in ("seed", "checkpoint_sha256")]
    agg = ledger.aggregate(per_seed, keys)
    groups = criteria_groups()
    all_criteria = {k: v for g in groups.values() for k, v in g.items()}
    check = ledger.check_criteria(agg, all_criteria)
    met = {g: all(check["criteria"][k]["pass"] for k in ks) for g, ks in groups.items()}
    level = deletion_level(met)
    record = {
        "experiment": "E-000013", "title": "Frozen GPT-2 core: prior conflict — counterfactual cells override a pretrained fact, "
                                           "the pretrained distribution returns after REVOKE / SHRED",
        "evidence_level": "E5", "deletion_level": level, "deletion_level_targeted": "F4",
        "evidence_level_note": "E5 names the substrate (a pretrained transformer as frozen core); support is stated per claim group.",
        "claim_groups_met": met,
        "claim_parts": [{"claim": g, "criteria": list(ks), "supported": met[g]} for g, ks in groups.items()],
        "by_construction": ["copy_bound_by_construction: the adapter acts only through the injection; with every cell masked the "
                            "null read is a fixed zero, so the base distribution is returned exactly (recorded, not learned)",
                            "fallback_after_revoke_by_construction: with status_gated the status flag multiplies the gate, so a "
                            "revoked cell's value is exactly zero and the injection vanishes. Exact equality to the base model "
                            "after REVOKE is therefore arithmetic, not a learned behaviour; the LEARNED residue is that the "
                            "routing does not spill onto neighbouring ACTIVE cells (kl_to_base, heldout_kl_max) while the "
                            "revoked cell itself stays addressed (routing_mass_on_target). This group is recorded and does NOT "
                            "grant a deletion level.",
                            "the pretrained fact is never deleted: the weights are frozen; what is measured is that the model "
                            "answers with it again after REVOKE / SHRED"],
        "learned": ["override (the injected direction wins over a pretrained fact the model knows)",
                    "fallback to the base distribution after SHRED, through the class-balanced verification gate (soft gate "
                    "carries F3, hard gate F4); the SHRED payload is physically present and routed to",
                    "routing precision after REVOKE: no spill onto neighbouring active cells, on train and held-out templates",
                    "no key, no injection on broken paths and generic text (routing to the null key)"],
        "attack_convention": "Every attack bar is a PAIRED EXCESS over the frozen model itself, because the counterfactual "
                             "object is a real capital token the pretrained prior already favours: an absolute forced-choice "
                             "or probe threshold would measure GPT-2's prior, not leakage from the cell. The floors are "
                             "recorded as prior/forced_choice_win, prior/probe_top1 and prior/counterfactual_top1_pooled, "
                             "measured on the same rows with the same distractor draws.",
        "validity_condition": "attack_validity: with the cell ACTIVE the same attacks must succeed. If they do not, "
                              "their failure after deletion is uninformative and the record reports F1 regardless of the "
                              "fallback groups.",
        "not_claimed": "unlearning of pretrained facts; LLM scale; multi-token entities.",
        "config": {"seeds": args.seeds, "steps": args.steps, "adapter": AdapterConfig(status_gated=True, fallback="prior").to_dict(),
                   "eval": EVAL, "templates": TEMPLATES_PC, "train_templates": TRAIN_TEMPLATES, "heldout_templates": HELDOUT_TEMPLATES,
                   "pairs": PAIRS, "n_names": N_NAMES, "generic_prompts": GENERIC,
                   "nouns": NOUN_PC,
                   "training": {"loss": "full-vocabulary CE on answerable + KL(base||adapter) on queries with NO resolved edge "
                                        "(a partially resolved 2-hop query legitimately injects at the first read layer) + "
                                        "routing CE + class-balanced gate BCE (weight 5); routing-first curriculum 300 steps",
                                "core_mode": "the frozen GPT-2 stays in eval mode during training, so the KL target and the "
                                             "adapter forward see the same (dropout-free) core state",
                                "p_revoked": 0.20, "p_shred": 0.10, "extra_unanswerable": 0.2, "lr": 2e-3, "batch": 32}},
        "sample_sizes": {"direct": EVAL["n_cells"] - len(PAIRS), "template1_train/direct": EVAL["n_cells"] - len(PAIRS),
                         "direct_heldout_min": EVAL["n_cells"] - len(PAIRS), "hop2": EVAL["n_hop2"],
                         "broken1/top1_matches_base": EVAL["n_broken"], "generic/top1_matches_base": EVAL["n_generic"],
                         "override/direct": len(PAIRS), "agree/direct": len(PAIRS), "restored/direct": len(PAIRS),
                         "resigned/direct": len(PAIRS), "rollback/direct": len(PAIRS),
                         "prior/counterfactual_top1_pooled": 4 * len(PAIRS),
                         "revoke/counterfactual_top1_pooled": 4 * len(PAIRS), "revoke/top1_matches_base_pooled": 4 * len(PAIRS),
                         "shred_soft/counterfactual_top1_pooled": 4 * len(PAIRS), "shred_soft/top1_matches_base_pooled": 4 * len(PAIRS),
                         "shred_hard/counterfactual_top1_pooled": 4 * len(PAIRS), "shred_hard/top1_matches_base_pooled": 4 * len(PAIRS),
                         "revoke/probe_top1": len(PAIRS), "shred_hard/probe_top1": len(PAIRS), "active/probe_top1": len(PAIRS),
                         "revoke/forced_choice_win": len(PAIRS), "shred_hard/forced_choice_win": len(PAIRS),
                         "revoke/filler_direct": EVAL["n_cells"] - len(PAIRS), "shred_hard/filler_direct": EVAL["n_cells"] - len(PAIRS),
                         "locality": EVAL["n_cells"] - len(PAIRS) - EVAL["n_locality_updates"] - EVAL["n_locality_revokes"]},
        "sample_size_note": "50 counterfactual items per seed cannot resolve a 0.05 bar on their own; the gating rate "
                            "criteria are therefore pooled over item x template (200 per seed, 600 over three seeds) and "
                            "the exact binomial intervals below are reported for the pooled counts.",
        "criteria": check["criteria"], "claim_supported": check["claim_supported"],
        "per_seed": per_seed, "aggregate": agg,
    }
    rows = [(k, f"{agg[k]['mean']:.4f}", f"{agg[k]['min']:.4f}", f"{agg[k]['max']:.4f}") for k in KEYS if k in agg]
    md = "\n".join([
        "# E-000013 — Frozen GPT-2 core: prior conflict (override while ACTIVE, fallback to the pretrained distribution after REVOKE / SHRED)", "",
        f"Evidence level: **E5** (substrate). Deletion level targeted F4, recorded **{level}**. Seeds: {args.seeds}; {args.steps} steps.", "",
        "50 real countries whose capitals GPT-2 small knows receive counterfactual capital cells; 950 prior-free filler facts. "
        "The adapter runs in fallback-to-prior mode: an unsigned or revoked cell injects nothing, the null read is a fixed zero.", "",
        ledger.table(["claim group", "supported"], [(c["claim"], "yes" if c["supported"] else "**no**") for c in record["claim_parts"]]), "",
        ledger.table(["measure", "mean over seeds", "min", "max"], rows), "",
        "Pre-registered criteria (worst seed):", "", ledger.criteria_table(check), "",
        "Exact binomial intervals (pooled over seeds):", "",
        ledger.table(ledger.CI_HEADERS, ledger.ci_rows(per_seed, [k for k in KEYS if k in record["sample_sizes"]],
                                                       record["sample_sizes"],
                                                       lower_is_better=[k for k in KEYS if "counterfactual" in k or "kl_to" in k])), "",
        "Sample sizes: " + record["sample_size_note"], "",
        "Attack convention: " + record["attack_convention"], "",
        "Validity condition: " + record["validity_condition"], "",
        "By construction: " + "; ".join(record["by_construction"]) + ".", "",
        "Not claimed: " + record["not_claimed"],
    ])
    path = ledger.save("e000013_prior_conflict", record, md)
    print(md); print(f"\nsaved {path}")
    return record


if __name__ == "__main__":
    main()
