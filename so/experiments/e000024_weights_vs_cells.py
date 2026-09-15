"""Experiment E-000024 — deleting a fact from weights versus deleting it from cells.

Every result so far says what the SO architecture *does*.  None of them says
what it *buys*, because nothing was measured against the ordinary way of
putting knowledge into a language model and taking it out again.  This
experiment runs that comparison under one protocol.

One frozen GPT-2.  One world of 400 facts.  One set of 50 deletion targets and
50 untouched bystanders.  Three ways of holding the facts and removing them:

  cells    the facts live in addressable cells read by the trained adapter of
           E-000012; removal is one SHRED per cell — no gradient, no
           optimisation, no hyper-parameter, and not one weight changes.
  ga       the facts are fine-tuned into a LoRA over the pretrained weights
           (the ordinary way to put knowledge into a model); removal is
           gradient ascent on the targets with a retain loss (the classic
           unlearning recipe).
  relabel  same LoRA; removal is a supervised fine-tune that relabels the
           targets to ' unknown' (the strongest practical baseline: it deletes
           the *behaviour* directly).

All three are driven to the same surface criterion — the direct question no
longer returns the object — and then attacked identically:

  * the direct question and a held-out paraphrase template,
  * forced choice between the true object and a random distractor (chance 1/2),
  * the rank and top-1 rate of the true object among all 256 entities,
  * a linear probe on the last hidden state, calibrated on live facts,
  * a relearning attack: the attacker fine-tunes on HALF the deleted facts and
    is scored on the OTHER half, which they never supplied,
  * collateral damage: bystander accuracy, and the perplexity of the frozen
    core on a fixed text.

The question the experiment answers is not "does deletion work" — every arm
passes the surface criterion by construction.  It is "what is left behind, and
what did it cost".

Run:  python -m so.experiments.e000024_weights_vs_cells [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from so import ledger
from so.attacks import LinearProbe, forced_choice, object_rank
from so.data import Bank, bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import (CHECKPOINTS, CKPT_SUFFIX, _sha256,
                                                      guard_recorded_checkpoint)
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.world import Query, UNKNOWN, World, fill_random

N_CELLS = 400
N_TARGETS = 50
N_BYSTANDERS = 50
# Rank 16 over the four projections of all twelve blocks is 2,359,296 trainable parameters against the
# SO adapter's 2,370,692 -- a 1.00x match, so neither arm is given more capacity than the other. Rank 8
# was half the adapter's budget and could not learn the world at all: after 400 steps it was still
# emitting one entity for every question.
LORA_RANK = 16
LORA_STEPS = 6000
LORA_LR = 3e-4
UNLEARN_STEPS = 400
UNLEARN_LR = 1e-4
# The relearning attacker is run until it demonstrably works on the half of the deleted facts it was
# HANDED, then scored on the half it was not. Equal step budgets are not equal attacker strength: the
# weights arms only have to re-fine-tune what was already in their LoRA and reach 1.000 almost at once,
# while the cells attacker has to memorise into an adapter whose cells are shredded and reached only
# 0.28 at 60 steps -- which made its 0.04 on the held-out half a weak attacker's number, not a result.
RELEARN_STEPS = 400
RELEARN_LR = 1e-4
RELEARN_TARGET_SUPPLIED = 0.60
RELEARN_EVAL_EVERY = 40
PPL_TEXTS = [
    "The history of the city is long and its buildings show every period of it. A river runs through "
    "the middle, and the bridges over it were rebuilt twice. Most of the population lives on the "
    "eastern bank, where the railway station is.",
    "Water expands when it freezes, which is why pipes burst in winter and why ice floats. The same "
    "property keeps lakes from freezing solid, so fish survive under the surface until spring.",
    "She opened the letter twice before reading it, first at the door and then again at the kitchen "
    "table, as though the second time the words might have arranged themselves differently.",
    "A compiler reads the source text, builds a tree from it, checks that the names and types agree, "
    "and only then emits instructions. Each stage can reject the program with its own kind of error.",
    "Trade along the coast collapsed after the harbour silted up. Ships that once anchored in the bay "
    "went further north, and within a generation the warehouses stood empty.",
    "The rule is simple enough to state and hard to apply: whoever holds the key is responsible for "
    "what is done with it, whether or not they were the one who turned it.",
]
# the same shapes E-000013 measured, including the two that end on a name -- a state that looks
# maximally like a routing key, and where the adapter is most likely to inject into ordinary text
GENERIC = ["{s} said that", "The story of {s} begins", "In the morning, {s}",
           "Everyone knows that {s}", "{s} walked into the", "According to {s}",
           "The meeting ended when {s}", "Nobody expected {s} to"]
N_GENERIC = 200


# --------------------------------------------------------------------------- LoRA

class LoRAConv1D(nn.Module):
    """Low-rank additive update on a GPT-2 Conv1D layer (which computes x @ W + b)."""

    def __init__(self, base, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        nx = base.weight.shape[0]
        nf = base.weight.shape[1]
        self.a = nn.Parameter(torch.zeros(nx, rank))
        self.b = nn.Parameter(torch.zeros(rank, nf))
        nn.init.normal_(self.a, std=0.02)

    def forward(self, x):
        return self.base(x) + (x @ self.a) @ self.b


LORA_TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def attach_lora(lm, rank: int = LORA_RANK) -> List[nn.Parameter]:
    """Replace the projections of every block with LoRA-wrapped copies; return the trainable parameters."""
    params: List[nn.Parameter] = []
    for block in lm.transformer.h:
        for path in LORA_TARGETS:
            parent, attr = path.split(".")
            mod = getattr(getattr(block, parent), attr)
            wrapped = LoRAConv1D(mod, rank)
            setattr(getattr(block, parent), attr, wrapped)
            params += [wrapped.a, wrapped.b]
    return params


def lora_state(lm) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in lm.named_parameters() if n.endswith((".a", ".b"))}


def load_lora_state(lm, state: Dict[str, torch.Tensor]) -> None:
    own = dict(lm.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            own[n].copy_(v)


def lora_delta_norm(lm) -> float:
    """Frobenius norm of the total weight change the LoRA represents, over all wrapped layers."""
    total = 0.0
    for block in lm.transformer.h:
        for path in LORA_TARGETS:
            parent, attr = path.split(".")
            mod = getattr(getattr(block, parent), attr)
            if isinstance(mod, LoRAConv1D):
                total += float((mod.a @ mod.b).pow(2).sum().item())
    return float(np.sqrt(total))


# --------------------------------------------------------------------------- the shared world

@dataclass
class Setup:
    world: World
    facts: List[Any]
    targets: List[Any]
    bystanders: List[Any]
    retain: List[Any]
    centre: np.ndarray


def build_setup(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, n_targets: int = N_TARGETS,
                n_cells: int = N_CELLS) -> Setup:
    rng = np.random.default_rng(4000 + seed)
    world = fill_random(rng, World(gk.n_entities, 4, gk.n_synonyms, []), n_cells)
    facts = list(world.facts)
    perm = rng.permutation(len(facts))
    targets = [facts[int(i)] for i in perm[:n_targets]]
    bystanders = [facts[int(i)] for i in perm[n_targets:n_targets + N_BYSTANDERS]]
    retain = [facts[int(i)] for i in perm[n_targets:]]
    return Setup(world, facts, targets, bystanders, retain, centre)


def q_of(fact, template: int) -> Query:
    return Query("fwd", fact.subject, (fact.relation,), (template,))


def unknown_queries(setup: Setup, rng: np.random.Generator, n: int) -> List[Query]:
    """Subject/relation pairs the world does not contain — the ' unknown' half of the output convention."""
    out: List[Query] = []
    seen = {(f.subject, f.relation) for f in setup.facts}
    while len(out) < n:
        s = int(rng.integers(setup.world.n_entities))
        r = int(rng.integers(setup.world.n_relations))
        if (s, r) not in seen:
            out.append(Query("fwd", s, (r,), (int(rng.integers(2)),)))
    return out


# --------------------------------------------------------------------------- scorers

Scorer = Callable[[Sequence[Query], int], Dict[str, np.ndarray]]


def cells_scorer(gk: E8.GPT2Knowledge, store: MVCCStore, world: World) -> Scorer:
    def score(queries: Sequence[Query], template: int) -> Dict[str, np.ndarray]:
        bank = bank_from_store(store, respect_markers=True)
        out = gk.predict(bank, world, list(queries), template=template)
        return {"logits": out["logits"], "hidden": out["hidden"], "answers": out["answers"]}
    return score


def weights_scorer(gk: E8.GPT2Knowledge) -> Scorer:
    """Candidate logits from the LoRA'd LM itself — the adapter is bypassed entirely."""
    lm = gk.model.lm
    cand = gk.model.candidate_ids

    @torch.no_grad()
    def score(queries: Sequence[Query], template: int) -> Dict[str, np.ndarray]:
        lm.eval()
        logits, hidden = [], []
        for i in range(0, len(queries), 64):
            chunk = list(queries[i: i + 64])
            texts = [E8.query_text(q, gk.names, gk.n_synonyms, template) for q in chunk]
            ids, am, last = E8.encode_texts(gk.tok, texts)
            out = lm(input_ids=ids, attention_mask=am, output_hidden_states=True)
            row = torch.arange(ids.shape[0])
            lg = out.logits[row, last][:, cand]
            h = out.hidden_states[-1][row, last]
            logits.append(lg.numpy()); hidden.append(h.numpy())
        lg = np.concatenate(logits)
        a = lg.argmax(-1)
        return {"logits": lg, "hidden": np.concatenate(hidden),
                "answers": np.where(a == gk.n_entities, UNKNOWN, a)}
    return score


# --------------------------------------------------------------------------- weights arm: train and unlearn

def fact_batch(gk: E8.GPT2Knowledge, items: Sequence[Tuple[Query, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """items: (query, target token id).  Returns ids, attention mask, last index, target ids."""
    texts = [E8.query_text(q, gk.names, gk.n_synonyms, int(q.surface[0])) for q, _ in items]
    ids, am, last = E8.encode_texts(gk.tok, texts)
    tgt = torch.as_tensor([t for _, t in items], dtype=torch.long)
    return ids, am, last, tgt


def token_of_entity(gk: E8.GPT2Knowledge, obj: int) -> int:
    return int(gk.entity_ids[obj])


def lm_nll(gk: E8.GPT2Knowledge, items: Sequence[Tuple[Query, int]]) -> torch.Tensor:
    ids, am, last, tgt = fact_batch(gk, items)
    out = gk.model.lm(input_ids=ids, attention_mask=am)
    row = torch.arange(ids.shape[0])
    return F.cross_entropy(out.logits[row, last], tgt)


def train_lora(gk: E8.GPT2Knowledge, setup: Setup, seed: int, steps: int, lr: float,
               batch_size: int = 16, verbose: bool = True, target_acc: float = 0.95,
               eval_every: int = 200, rank: int = LORA_RANK) -> Dict[str, Any]:
    """Fine-tune the facts into the pretrained weights, plus the ' unknown' convention for absent pairs.

    Training stops as soon as the model answers ``target_acc`` of the deletion targets, so the arm is
    as strong as the comparison needs it to be and no stronger: an under-trained weights arm would be
    trivially "unlearnable" and the head-to-head would prove nothing.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(9000 + seed)
    params = attach_lora(gk.model.lm, rank)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    pool: List[Tuple[Query, int]] = []
    for f in setup.facts:
        for t in (0, 1):
            pool.append((q_of(f, t), token_of_entity(gk, f.obj)))
    unk = [(q, gk.unknown_id) for q in unknown_queries(setup, rng, 200)]
    pool += unk
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    for step in range(steps):
        gk.model.lm.train()
        idx = rng.integers(0, len(pool), batch_size)
        items = [pool[int(i)] for i in idx]
        loss = lm_nll(gk, items)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == 0:
            gk.model.lm.eval()
            score = weights_scorer(gk)
            a = score([q_of(f, 0) for f in setup.targets], 0)["answers"]
            acc = float(np.mean([int(x == f.obj) for x, f in zip(a, setup.targets)]))
            rec = {"step": step + 1, "loss": float(loss.item()), "target_acc": acc,
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  lora step {step + 1:5d}  loss {rec['loss']:.4f}  target_acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
            if acc >= target_acc and step + 1 >= eval_every:
                break
    gk.model.lm.eval()
    return {"history": history, "train_seconds": time.time() - t0, "steps_used": len(history) and history[-1]["step"],
            "n_lora_params": int(sum(p.numel() for p in params)), "params": params}


def base_logprobs(gk: E8.GPT2Knowledge, texts: Sequence[str]) -> torch.Tensor:
    """Reference distribution of the model as it stands (used as the KL anchor during unlearning)."""
    with torch.no_grad():
        ids, am, last = E8.encode_texts(gk.tok, list(texts))
        out = gk.model.lm(input_ids=ids, attention_mask=am)
        row = torch.arange(ids.shape[0])
        return F.log_softmax(out.logits[row, last], dim=-1)


def unlearn(gk: E8.GPT2Knowledge, setup: Setup, params: List[nn.Parameter], mode: str, seed: int,
            steps: int, lr: float, stop_at: float = 0.02, verbose: bool = True) -> Dict[str, Any]:
    """mode 'ga': ascent on the targets + retain loss.  mode 'relabel': supervised relabel to ' unknown'."""
    rng = np.random.default_rng(11000 + seed)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    tgt_pool = [(q_of(f, t), token_of_entity(gk, f.obj)) for f in setup.targets for t in (0, 1)]
    unk_pool = [(q_of(f, t), gk.unknown_id) for f in setup.targets for t in (0, 1)]
    ret_pool = [(q_of(f, t), token_of_entity(gk, f.obj)) for f in setup.retain for t in (0, 1)]
    score = weights_scorer(gk)
    truth = [f.obj for f in setup.targets]
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    used = steps
    for step in range(steps):
        gk.model.lm.train()
        ti = rng.integers(0, len(tgt_pool), 8)
        ri = rng.integers(0, len(ret_pool), 8)
        retain = lm_nll(gk, [ret_pool[int(i)] for i in ri])
        if mode == "ga":
            forget = -lm_nll(gk, [tgt_pool[int(i)] for i in ti])
            loss = forget + retain
        elif mode == "relabel":
            forget = lm_nll(gk, [unk_pool[int(i)] for i in ti])
            loss = forget + retain
        else:
            raise ValueError(mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 20 == 0:
            gk.model.lm.eval()
            a = score([q_of(f, 0) for f in setup.targets], 0)["answers"]
            acc = float(np.mean([int(x == y) for x, y in zip(a, truth)]))
            rec = {"step": step + 1, "loss": float(loss.item()), "target_acc": acc,
                   "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  unlearn[{mode}] step {step + 1:4d}  loss {rec['loss']:+.4f}  target_acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
            if acc <= stop_at:
                used = step + 1
                break
    gk.model.lm.eval()
    return {"history": history, "seconds": time.time() - t0, "steps_used": used}


# --------------------------------------------------------------------------- measurement

def perplexity(gk: E8.GPT2Knowledge, texts: Sequence[str] = PPL_TEXTS) -> float:
    """Token-weighted perplexity of the LM itself over several paragraphs of ordinary prose."""
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for text in texts:
            ids = gk.tok(text, return_tensors="pt")["input_ids"]
            out = gk.model.lm(input_ids=ids, labels=ids)
            k = ids.shape[1] - 1
            nll += float(out.loss.item()) * k
            n_tok += k
    return float(np.exp(nll / max(n_tok, 1)))


def generic_prompts(gk: E8.GPT2Knowledge, seed: int, n: int = N_GENERIC) -> List[str]:
    rng = np.random.default_rng(21000 + seed)
    return [GENERIC[int(rng.integers(len(GENERIC)))].format(s=gk.names[int(rng.integers(gk.n_entities))])
            for _ in range(n)]


@torch.no_grad()
def full_logits(gk: E8.GPT2Knowledge, texts: Sequence[str], bank: Optional[Bank] = None) -> np.ndarray:
    """Full-vocabulary logits at the last token, with the knowledge layer attached or bypassed."""
    tensors = bank.tensors() if bank is not None else None
    out = []
    for i in range(0, len(texts), 32):
        ids, am, last = E8.encode_texts(gk.tok, list(texts[i: i + 32]))
        if tensors is None:
            lg = gk.model.lm(input_ids=ids, attention_mask=am).logits
            out.append(lg[torch.arange(ids.shape[0]), last].numpy())
        else:
            _, full, _, _ = gk.model(tensors, ids, am, last)
            out.append(full.numpy())
    return np.concatenate(out)


def kl_rows(base: np.ndarray, other: np.ndarray) -> np.ndarray:
    """KL(base || other) per row, in nats, over the full vocabulary (E-000013's measure)."""
    b = torch.log_softmax(torch.as_tensor(base), -1)
    a = torch.log_softmax(torch.as_tensor(other), -1)
    return (b.exp() * (b - a)).sum(-1).numpy()


def collateral(base: np.ndarray, current: np.ndarray, tag: str) -> Dict[str, float]:
    kl = kl_rows(base, current)
    return {f"{tag}/generic_kl_mean": float(kl.mean()),
            f"{tag}/generic_kl_max": float(kl.max()),
            f"{tag}/generic_top1_matches_base": float((current.argmax(-1) == base.argmax(-1)).mean())}


def measure(score: Scorer, setup: Setup, gk: E8.GPT2Knowledge, seed: int, tag: str) -> Dict[str, float]:
    """The identical attack battery, applied to whatever produces candidate logits."""
    n_ent = gk.n_entities
    tq = [q_of(f, 0) for f in setup.targets]
    tq1 = [q_of(f, 1) for f in setup.targets]
    truth = [f.obj for f in setup.targets]
    bq = [q_of(f, 0) for f in setup.bystanders]
    btruth = [f.obj for f in setup.bystanders]

    out0 = score(tq, 0)
    out1 = score(tq1, 1)
    outb = score(bq, 0)
    lg = out0["logits"]
    m: Dict[str, float] = {}
    m[f"{tag}/direct_acc"] = float(np.mean([int(a == t) for a, t in zip(out0["answers"], truth)]))
    m[f"{tag}/direct_unknown"] = float(np.mean(out0["answers"] == UNKNOWN))
    m[f"{tag}/paraphrase_acc"] = float(np.mean([int(a == t) for a, t in zip(out1["answers"], truth)]))
    m[f"{tag}/bystander_acc"] = float(np.mean([int(a == t) for a, t in zip(outb["answers"], btruth)]))
    m[f"{tag}/forced_choice"] = forced_choice(lg, truth, np.random.default_rng(seed), n_ent)
    rk = object_rank(lg, truth, n_ent)
    m[f"{tag}/true_obj_top1"] = rk["top1"]
    m[f"{tag}/true_obj_mean_rank"] = rk["mean_rank"]

    # linear probe calibrated on live (non-target, non-bystander) facts, applied to the deleted ones
    live = [f for f in setup.retain if f not in setup.bystanders][:250]
    lq = [q_of(f, 0) for f in live]
    hl = score(lq, 0)["hidden"]
    yl = np.array([f.obj for f in live])
    cut = int(0.8 * len(live))
    probe = LinearProbe(hl.shape[1], n_ent, seed=seed)
    probe.fit(hl[:cut], yl[:cut])
    m[f"{tag}/probe_calibration_top1"] = probe.accuracy(hl[cut:], yl[cut:])
    m[f"{tag}/probe_top1"] = probe.accuracy(out0["hidden"], np.array(truth))
    m[f"{tag}/probe_top5"] = probe.accuracy(out0["hidden"], np.array(truth), topk=5)
    return m


def relearn_attack_weights(gk: E8.GPT2Knowledge, setup: Setup, params: List[nn.Parameter], seed: int) -> Dict[str, float]:
    """The attacker holds half the deleted facts, fine-tunes on them, and is scored on the other half."""
    rng = np.random.default_rng(13000 + seed)
    half = len(setup.targets) // 2
    known, held = setup.targets[:half], setup.targets[half:]
    pool = [(q_of(f, t), token_of_entity(gk, f.obj)) for f in known for t in (0, 1)]
    opt = torch.optim.AdamW(params, lr=RELEARN_LR, weight_decay=0.0)
    steps = 0
    for step in range(RELEARN_STEPS):
        gk.model.lm.train()
        idx = rng.integers(0, len(pool), 8)
        loss = lm_nll(gk, [pool[int(i)] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        steps = step + 1
        if steps % RELEARN_EVAL_EVERY == 0:
            gk.model.lm.eval()
            a = weights_scorer(gk)([q_of(f, 0) for f in known], 0)["answers"]
            if float(np.mean([int(x == f.obj) for x, f in zip(a, known)])) >= RELEARN_TARGET_SUPPLIED:
                break
    gk.model.lm.eval()
    out = _relearn_metrics(weights_scorer(gk), gk, known, held, seed)
    out["relearn/steps_used"] = float(steps)
    return out


def relearn_attack_cells(gk: E8.GPT2Knowledge, setup: Setup, store: MVCCStore, seed: int) -> Dict[str, float]:
    """Same attacker, same budget, against the adapter: the shredded payloads are in no parameter to recover."""
    rng = np.random.default_rng(13000 + seed)
    half = len(setup.targets) // 2
    known, held = setup.targets[:half], setup.targets[half:]
    params = gk.model.adapter_parameters()
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
    bank = bank_from_store(store, respect_markers=True)
    tensors = bank.tensors()
    pool = [(q_of(f, t), f.obj) for f in known for t in (0, 1)]
    steps = 0
    for step in range(RELEARN_STEPS):
        gk.model.train()
        idx = rng.integers(0, len(pool), 8)
        items = [pool[int(i)] for i in idx]
        texts = [E8.query_text(q, gk.names, gk.n_synonyms, int(q.surface[0])) for q, _ in items]
        ids, am, last = E8.encode_texts(gk.tok, texts)
        target = torch.as_tensor([o for _, o in items], dtype=torch.long)
        cand, _, _, _ = gk.model(tensors, ids, am, last)
        loss = F.cross_entropy(cand, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        steps = step + 1
        if steps % RELEARN_EVAL_EVERY == 0:
            gk.model.eval()
            a = cells_scorer(gk, store, setup.world)([q_of(f, 0) for f in known], 0)["answers"]
            if float(np.mean([int(x == f.obj) for x, f in zip(a, known)])) >= RELEARN_TARGET_SUPPLIED:
                break
    gk.model.eval()
    out = _relearn_metrics(cells_scorer(gk, store, setup.world), gk, known, held, seed)
    out["relearn/steps_used"] = float(steps)
    return out


def _relearn_metrics(score: Scorer, gk: E8.GPT2Knowledge, known, held, seed: int) -> Dict[str, float]:
    n_ent = gk.n_entities
    ok = score([q_of(f, 0) for f in known], 0)
    oh = score([q_of(f, 0) for f in held], 0)
    tk = [f.obj for f in known]
    th = [f.obj for f in held]
    rk = object_rank(oh["logits"], th, n_ent)
    return {"relearn/supplied_acc": float(np.mean([int(a == t) for a, t in zip(ok["answers"], tk)])),
            "relearn/heldout_acc": float(np.mean([int(a == t) for a, t in zip(oh["answers"], th)])),
            "relearn/heldout_forced_choice": forced_choice(oh["logits"], th, np.random.default_rng(seed), n_ent),
            "relearn/heldout_top1": rk["top1"],
            "relearn/heldout_mean_rank": rk["mean_rank"]}


# --------------------------------------------------------------------------- the three arms

def run_cells_arm(seed: int, n_targets: int, n_cells: int = N_CELLS, verbose: bool = True) -> Tuple[Dict[str, float], Setup, E8.GPT2Knowledge]:
    """Frozen GPT-2 + the E-000012 adapter.  Deletion is one SHRED per cell."""
    path = CHECKPOINTS / f"e000012_gpt2{CKPT_SUFFIX}_seed{seed}.pt"
    if not path.exists():
        raise SystemExit(f"missing checkpoint {path}; run: python -m so.experiments.e000012_status_gated_revoke --seeds {seed}")
    ck = torch.load(path, weights_only=False)
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    setup = build_setup(gk, seed, np.asarray(ck["centre"]), n_targets, n_cells)
    store = MVCCStore(marker_dim=setup.centre.shape[0], seed=seed, marker_centre=setup.centre)
    kids = load_world(store, setup.world)

    weights_before = torch.cat([p.detach().reshape(-1) for p in gk.model.lm.parameters()])
    generic = generic_prompts(gk, seed)
    base_generic = full_logits(gk, generic)              # the pretrained model, knowledge layer bypassed
    m: Dict[str, float] = {"cells/ppl_base": perplexity(gk)}
    score = cells_scorer(gk, store, setup.world)
    m.update(measure(score, setup, gk, seed, "cells/before"))
    m["cells/ppl_before"] = m["cells/ppl_base"]          # the frozen core is the same model either way
    # the honest deployment condition for this arm: the store is attached while ordinary text is read,
    # so the adapter can perturb it. E-000013 measured exactly this and found it is not zero.
    m.update(collateral(base_generic, full_logits(gk, generic, bank_from_store(store, respect_markers=True)),
                        "cells/before"))

    t0 = time.time()
    for f in setup.targets:
        store.shred(kids[f.key])
    m["cells/delete_seconds"] = time.time() - t0
    m["cells/delete_seconds_per_fact"] = m["cells/delete_seconds"] / max(len(setup.targets), 1)

    m.update(measure(score, setup, gk, seed, "cells/after"))
    weights_after = torch.cat([p.detach().reshape(-1) for p in gk.model.lm.parameters()])
    m["cells/weight_delta_l2"] = float((weights_after - weights_before).norm().item())
    m["cells/ppl_after"] = perplexity(gk)
    m["cells/ppl_delta"] = abs(m["cells/ppl_after"] - m["cells/ppl_base"])
    m["cells/n_params_changed"] = 0.0
    m.update(collateral(base_generic, full_logits(gk, generic, bank_from_store(store, respect_markers=True)),
                        "cells/after"))
    if verbose:
        print(f"  cells: direct {m['cells/before/direct_acc']:.3f} -> {m['cells/after/direct_acc']:.3f}, "
              f"forced choice {m['cells/after/forced_choice']:.3f}, delete {m['cells/delete_seconds']*1e3:.1f} ms",
              flush=True)

    adapter_before = {k: v.detach().clone() for k, v in gk.model.state_dict().items() if not k.startswith("lm.")}
    m.update({f"cells/{k}": v for k, v in relearn_attack_cells(gk, setup, store, seed).items()})
    gk.model.load_state_dict(adapter_before, strict=False)   # the attack is a probe, not a state change
    gk.model.eval()
    return m, setup, gk


def run_weights_arms(setup: Setup, seed: int, steps: int, unlearn_steps: int, verbose: bool = True,
                     rank: int = LORA_RANK) -> Dict[str, float]:
    """A second, independent GPT-2 whose weights DO carry the facts, unlearned two ways."""
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))   # config unused: the adapter is bypassed
    ppl_base = perplexity(gk)
    generic = generic_prompts(gk, seed)
    base_generic = full_logits(gk, generic)                  # captured BEFORE any LoRA exists
    # The LoRA is cached, because it is the only expensive thing here and everything downstream of it
    # -- the unlearning recipes, the attacks, the collateral measurements -- is cheap and gets edited.
    ck = CHECKPOINTS / f"e000024_lora_r{rank}{CKPT_SUFFIX}_seed{seed}.pt"
    if ck.exists():
        state = torch.load(ck, weights_only=False)
        if int(state.get("rank", -1)) == rank and int(state.get("n_cells", -1)) == len(setup.facts):
            params = attach_lora(gk.model.lm, rank)
            for p in params:
                p.requires_grad_(True)
            load_lora_state(gk.model.lm, state["lora"])
            out = {"history": state["history"], "train_seconds": state["train_seconds"],
                   "steps_used": state["steps_used"], "n_lora_params": state["n_lora_params"],
                   "params": params}
            if verbose:
                print(f"  loaded the trained LoRA from {ck.name} "
                      f"({out['steps_used']} steps, {out['train_seconds']:.0f}s)", flush=True)
        else:
            ck = None
    if not ck or not ck.exists():
        out = train_lora(gk, setup, seed, steps, LORA_LR, verbose=verbose, rank=rank)
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        path = CHECKPOINTS / f"e000024_lora_r{rank}{CKPT_SUFFIX}_seed{seed}.pt"
        guard_recorded_checkpoint(path)
        torch.save({"lora": lora_state(gk.model.lm), "rank": rank, "n_cells": len(setup.facts),
                    "history": out["history"], "train_seconds": out["train_seconds"],
                    "steps_used": out["steps_used"], "n_lora_params": out["n_lora_params"]}, path)
    params = out["params"]
    trained = lora_state(gk.model.lm)
    m: Dict[str, float] = {"weights/n_lora_params": float(out["n_lora_params"]),
                           "weights/train_seconds": out["train_seconds"],
                           "weights/train_steps_used": float(out["steps_used"] or 0),
                           "weights/ppl_base": ppl_base,
                           "weights/ppl_after_learning": perplexity(gk)}
    score = weights_scorer(gk)
    m.update(measure(score, setup, gk, seed, "weights/before"))
    m.update(collateral(base_generic, full_logits(gk, generic), "weights/before"))
    m["weights/before_delta_l2"] = lora_delta_norm(gk.model.lm)

    for mode, tag in (("ga", "ga"), ("relabel", "relabel")):
        load_lora_state(gk.model.lm, trained)
        u = unlearn(gk, setup, params, mode, seed, unlearn_steps, UNLEARN_LR, verbose=verbose)
        m[f"{tag}/delete_seconds"] = u["seconds"]
        m[f"{tag}/delete_seconds_per_fact"] = u["seconds"] / max(len(setup.targets), 1)
        m[f"{tag}/unlearn_steps"] = float(u["steps_used"])
        m.update(measure(weights_scorer(gk), setup, gk, seed, f"{tag}/after"))
        m[f"{tag}/ppl_after"] = perplexity(gk)
        m.update(collateral(base_generic, full_logits(gk, generic), f"{tag}/after"))
        m[f"{tag}/n_params_changed"] = float(out["n_lora_params"])
        m[f"{tag}/weight_delta_l2"] = lora_delta_norm(gk.model.lm)
        after = lora_state(gk.model.lm)
        m.update({f"{tag}/{k}": v for k, v in relearn_attack_weights(gk, setup, params, seed).items()})
        load_lora_state(gk.model.lm, after)
        if verbose:
            print(f"  {tag}: direct {m[f'{tag}/after/direct_acc']:.3f}, forced choice "
                  f"{m[f'{tag}/after/forced_choice']:.3f}, relearn held-out "
                  f"{m[f'{tag}/relearn/heldout_acc']:.3f}, ppl {ppl_base:.1f} -> {m[f'{tag}/ppl_after']:.1f}",
                  flush=True)
    return m


# --------------------------------------------------------------------------- driver

CHANCE_TOP1 = 1.0 / 256
CHANCE_RANK = (256 - 1) / 2.0

REPORT_KEYS = [
    "cells/before/direct_acc", "cells/after/direct_acc", "cells/after/paraphrase_acc",
    "cells/after/forced_choice", "cells/after/true_obj_top1", "cells/after/true_obj_mean_rank",
    "cells/after/probe_top1", "cells/after/probe_top5", "cells/after/bystander_acc",
    "cells/ppl_base", "cells/ppl_after", "cells/ppl_delta", "cells/after/generic_kl_mean", "cells/after/generic_top1_matches_base",
    "cells/relearn/supplied_acc", "cells/relearn/heldout_acc", "cells/relearn/heldout_forced_choice",
    "cells/relearn/heldout_top1", "cells/relearn/steps_used", "cells/weight_delta_l2", "cells/delete_seconds",
    "weights/before/direct_acc",
    "ga/after/direct_acc", "ga/after/paraphrase_acc", "ga/after/forced_choice", "ga/after/true_obj_top1",
    "ga/after/true_obj_mean_rank", "ga/after/probe_top1", "ga/after/probe_top5", "ga/after/bystander_acc",
    "ga/relearn/supplied_acc", "ga/relearn/heldout_acc", "ga/relearn/heldout_forced_choice",
    "ga/relearn/heldout_top1", "ga/relearn/steps_used", "ga/weight_delta_l2", "ga/delete_seconds",
    "ga/ppl_after", "ga/after/generic_kl_mean", "ga/after/generic_top1_matches_base",
    "relabel/after/direct_acc", "relabel/after/paraphrase_acc", "relabel/after/forced_choice",
    "relabel/after/true_obj_top1", "relabel/after/true_obj_mean_rank", "relabel/after/probe_top1",
    "relabel/after/probe_top5", "relabel/after/bystander_acc",
    "relabel/relearn/supplied_acc", "relabel/relearn/heldout_acc", "relabel/relearn/heldout_forced_choice",
    "relabel/relearn/heldout_top1", "relabel/relearn/steps_used", "relabel/weight_delta_l2", "relabel/delete_seconds",
    "relabel/ppl_after", "relabel/after/generic_kl_mean", "relabel/after/generic_top1_matches_base",
    "weights/ppl_base", "weights/ppl_after_learning", "weights/before/generic_kl_mean",
    "weights/n_lora_params", "weights/train_steps_used",
]

CRITERIA = {
    # every arm must actually reach the surface criterion, or its "after" numbers mean nothing
    "cells/after/direct_acc": ("<=", 0.02),
    "ga/after/direct_acc": ("<=", 0.02),
    "relabel/after/direct_acc": ("<=", 0.02),
    # the weights arm must have learned the facts in the first place
    "weights/before/direct_acc": (">=", 0.80),
    # what is left behind, on the worst seed
    "cells/after/forced_choice": ("<=", 0.60),
    "cells/after/true_obj_top1": ("<=", 0.02),
    "cells/relearn/heldout_acc": ("<=", 0.05),
    # validity: the relearning attack must demonstrably work on the facts the attacker supplied,
    # or "the others did not come back" is a statement about a weak attacker, not about the system
    "cells/relearn/supplied_acc": (">=", 0.50),   # each attacker runs until it clears this or exhausts
    "ga/relearn/supplied_acc": (">=", 0.50),
    "relabel/relearn/supplied_acc": (">=", 0.50),
    # the frozen core is untouched, bit for bit
    "cells/weight_delta_l2": ("<=", 0.0),
    "cells/ppl_delta": ("<=", 0.0),          # and the frozen core reads ordinary prose exactly as before
    "cells/after/bystander_acc": (">=", 0.70),
}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--lora-steps", type=int, default=LORA_STEPS)
    ap.add_argument("--unlearn-steps", type=int, default=UNLEARN_STEPS)
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--n-cells", type=int, default=N_CELLS)
    ap.add_argument("--lora-rank", type=int, default=LORA_RANK)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--tag", default="", help="suffix for the record name, so parallel seed runs do not "
                                              "overwrite each other; combine them later with --combine")
    ap.add_argument("--combine", nargs="*", default=None,
                    help="do not run anything: merge these tagged records into the canonical one")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.combine is not None:
        return combine(args.combine, args.n_targets, args.n_cells, args.lora_steps, args.unlearn_steps)

    per_seed: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"=== seed {seed}: cells arm (E-000012 checkpoint, SHRED) ===", flush=True)
        m, setup, _ = run_cells_arm(seed, args.n_targets, args.n_cells)
        print(f"=== seed {seed}: weights arms (LoRA fine-tune, then unlearning) ===", flush=True)
        m.update(run_weights_arms(setup, seed, args.lora_steps, args.unlearn_steps,
                                  rank=args.lora_rank))
        m["seed"] = seed
        per_seed.append(m)

    return report(per_seed, args.seeds, args.n_targets, args.n_cells, args.lora_steps,
                  args.unlearn_steps, args.tag, args.lora_rank)


def combine(tags: Sequence[str], n_targets: int, n_cells: int, lora_steps: int, unlearn_steps: int) -> Dict[str, Any]:
    """Merge the per-seed records written by parallel runs into the canonical record."""
    import json
    per_seed: List[Dict[str, float]] = []
    for tag in tags:
        path = ledger.RESULTS_DIR / f"e000024_weights_vs_cells{tag}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        per_seed += json.loads(path.read_text())["per_seed"]
    per_seed.sort(key=lambda s: s["seed"])
    seeds = [int(s["seed"]) for s in per_seed]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"the records overlap on seeds {sorted(seeds)}; each seed may appear once")
    return report(per_seed, seeds, n_targets, n_cells, lora_steps, unlearn_steps, "")


def report(per_seed: List[Dict[str, float]], seeds: Sequence[int], n_targets: int, n_cells: int,
           lora_steps: int, unlearn_steps: int, tag: str, rank: int = LORA_RANK) -> Dict[str, Any]:
    keys = [k for k in REPORT_KEYS if all(k in s for s in per_seed)]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})

    lower = {"cells/after/direct_acc", "ga/after/direct_acc", "relabel/after/direct_acc",
             "cells/after/paraphrase_acc", "ga/after/paraphrase_acc", "relabel/after/paraphrase_acc",
             "cells/after/true_obj_top1", "ga/after/true_obj_top1", "relabel/after/true_obj_top1",
             "cells/after/probe_top1", "ga/after/probe_top1", "relabel/after/probe_top1",
             "cells/after/probe_top5", "ga/after/probe_top5", "relabel/after/probe_top5",
             "cells/relearn/heldout_acc", "ga/relearn/heldout_acc", "relabel/relearn/heldout_acc",
             "cells/relearn/heldout_top1", "ga/relearn/heldout_top1", "relabel/relearn/heldout_top1",
             "cells/weight_delta_l2", "ga/weight_delta_l2", "relabel/weight_delta_l2",
             "cells/delete_seconds", "ga/delete_seconds", "relabel/delete_seconds",
             "cells/ppl_after", "ga/ppl_after", "relabel/ppl_after",
             "cells/after/generic_kl_mean", "ga/after/generic_kl_mean", "relabel/after/generic_kl_mean"}
    sizes = {k: n_targets for k in keys if k.endswith(("direct_acc", "paraphrase_acc", "forced_choice",
                                                            "true_obj_top1", "probe_top1", "probe_top5",
                                                            "bystander_acc"))}
    for k in list(sizes):
        if "relearn/" in k:
            sizes[k] = n_targets // 2
    rows = ledger.ci_rows(per_seed, keys, sizes, lower_is_better=sorted(lower))

    head = ["measure", "cells (SHRED)", "weights, ascent", "weights, relabel", "chance"]
    def w(key: str, lib: bool) -> str:
        return f"{ledger.worst(agg[key], lib):.4f}" if key in agg else "-"
    compare = ledger.table(head, [
        ["direct question answered", w("cells/after/direct_acc", True), w("ga/after/direct_acc", True),
         w("relabel/after/direct_acc", True), "-"],
        ["held-out paraphrase answered", w("cells/after/paraphrase_acc", True), w("ga/after/paraphrase_acc", True),
         w("relabel/after/paraphrase_acc", True), "-"],
        ["forced choice, true vs random", w("cells/after/forced_choice", False), w("ga/after/forced_choice", False),
         w("relabel/after/forced_choice", False), "0.5000"],
        ["true object top-1 of 256", w("cells/after/true_obj_top1", True), w("ga/after/true_obj_top1", True),
         w("relabel/after/true_obj_top1", True), f"{CHANCE_TOP1:.4f}"],
        ["mean rank of true object", w("cells/after/true_obj_mean_rank", False), w("ga/after/true_obj_mean_rank", False),
         w("relabel/after/true_obj_mean_rank", False), f"{CHANCE_RANK:.1f}"],
        ["linear probe top-1", w("cells/after/probe_top1", True), w("ga/after/probe_top1", True),
         w("relabel/after/probe_top1", True), f"{CHANCE_TOP1:.4f}"],
        ["relearn attack: held-out recovered", w("cells/relearn/heldout_acc", True), w("ga/relearn/heldout_acc", True),
         w("relabel/relearn/heldout_acc", True), "-"],
        ["relearn attack: supplied recovered", w("cells/relearn/supplied_acc", False), w("ga/relearn/supplied_acc", False),
         w("relabel/relearn/supplied_acc", False), "-"],
        ["bystander facts still answered", w("cells/after/bystander_acc", False), w("ga/after/bystander_acc", False),
         w("relabel/after/bystander_acc", False), "-"],
        ["L2 change of model weights", w("cells/weight_delta_l2", True), w("ga/weight_delta_l2", True),
         w("relabel/weight_delta_l2", True), "-"],
        ["perplexity on ordinary prose", w("cells/ppl_after", True), w("ga/ppl_after", True),
         w("relabel/ppl_after", True), f"{'-'}"],
        ["KL to the pretrained model, generic text", w("cells/after/generic_kl_mean", True),
         w("ga/after/generic_kl_mean", True), w("relabel/after/generic_kl_mean", True), "0.0000"],
        ["generic top-1 still the pretrained one", w("cells/after/generic_top1_matches_base", False),
         w("ga/after/generic_top1_matches_base", False), w("relabel/after/generic_top1_matches_base", False),
         "1.0000"],
        ["seconds to delete 50 facts", w("cells/delete_seconds", True), w("ga/delete_seconds", True),
         w("relabel/delete_seconds", True), "-"],
    ])

    record = {"experiment": "E-000024", "title": "deleting a fact from weights versus deleting it from cells",
              "seeds": list(seeds), "n_cells": n_cells, "n_targets": n_targets,
              "n_bystanders": N_BYSTANDERS, "lora_rank": rank, "lora_steps": lora_steps,
              "unlearn_steps_budget": unlearn_steps, "relearn_steps": RELEARN_STEPS,
              "chance_top1": CHANCE_TOP1, "chance_mean_rank": CHANCE_RANK,
              "per_seed": per_seed, "aggregate": agg, "criteria": check}

    md = [f"# E-000024 — {record['title']}", "",
          f"Seeds {list(seeds)}; {n_cells} facts, {n_targets} deletion targets, {N_BYSTANDERS} bystanders.",
          "The cells arm is the frozen GPT-2 of E-000012 with its trained adapter; the weights arms are the same",
          f"frozen GPT-2 with a rank-{rank} LoRA fine-tuned on the identical facts and then unlearned two ways.",
          "All three arms are driven to the same surface criterion and attacked identically.", "",
          "## The comparison (worst seed)", "", compare, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## All measures", "", ledger.table(ledger.CI_HEADERS, rows), ""]
    path = ledger.save(f"e000024_weights_vs_cells{tag}", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(compare)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
