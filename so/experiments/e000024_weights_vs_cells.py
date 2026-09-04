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
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.llm_adapter import AdapterConfig
from so.mvcc import MVCCStore
from so.reference import load_world
from so.world import Query, UNKNOWN, World, fill_random

N_CELLS = 400
N_TARGETS = 50
N_BYSTANDERS = 50
LORA_RANK = 8
LORA_STEPS = 1200
LORA_LR = 3e-4
UNLEARN_STEPS = 400
UNLEARN_LR = 1e-4
RELEARN_STEPS = 60
RELEARN_LR = 1e-4
PPL_TEXT = (
    "The history of the city is long and its buildings show every period of it. "
    "A river runs through the middle, and the bridges over it were rebuilt twice. "
    "Most of the population lives on the eastern bank, where the railway station is."
)


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
               eval_every: int = 200) -> Dict[str, Any]:
    """Fine-tune the facts into the pretrained weights, plus the ' unknown' convention for absent pairs.

    Training stops as soon as the model answers ``target_acc`` of the deletion targets, so the arm is
    as strong as the comparison needs it to be and no stronger: an under-trained weights arm would be
    trivially "unlearnable" and the head-to-head would prove nothing.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(9000 + seed)
    params = attach_lora(gk.model.lm, LORA_RANK)
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

def perplexity(gk: E8.GPT2Knowledge, text: str = PPL_TEXT) -> float:
    with torch.no_grad():
        ids = gk.tok(text, return_tensors="pt")["input_ids"]
        out = gk.model.lm(input_ids=ids, labels=ids)
        return float(torch.exp(out.loss).item())


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
    for _ in range(RELEARN_STEPS):
        gk.model.lm.train()
        idx = rng.integers(0, len(pool), 8)
        loss = lm_nll(gk, [pool[int(i)] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    gk.model.lm.eval()
    score = weights_scorer(gk)
    return _relearn_metrics(score, gk, known, held, seed)


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
    for _ in range(RELEARN_STEPS):
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
    gk.model.eval()
    score = cells_scorer(gk, store, setup.world)
    return _relearn_metrics(score, gk, known, held, seed)


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
    ppl_before = perplexity(gk)
    score = cells_scorer(gk, store, setup.world)
    m = measure(score, setup, gk, seed, "cells/before")
    m["cells/ppl_before"] = ppl_before

    t0 = time.time()
    for f in setup.targets:
        store.shred(kids[f.key])
    m["cells/delete_seconds"] = time.time() - t0
    m["cells/delete_seconds_per_fact"] = m["cells/delete_seconds"] / max(len(setup.targets), 1)

    m.update(measure(score, setup, gk, seed, "cells/after"))
    weights_after = torch.cat([p.detach().reshape(-1) for p in gk.model.lm.parameters()])
    m["cells/weight_delta_l2"] = float((weights_after - weights_before).norm().item())
    m["cells/ppl_after"] = perplexity(gk)
    m["cells/n_params_changed"] = 0.0
    if verbose:
        print(f"  cells: direct {m['cells/before/direct_acc']:.3f} -> {m['cells/after/direct_acc']:.3f}, "
              f"forced choice {m['cells/after/forced_choice']:.3f}, delete {m['cells/delete_seconds']*1e3:.1f} ms",
              flush=True)

    adapter_before = {k: v.detach().clone() for k, v in gk.model.state_dict().items() if not k.startswith("lm.")}
    m.update({f"cells/{k}": v for k, v in relearn_attack_cells(gk, setup, store, seed).items()})
    gk.model.load_state_dict(adapter_before, strict=False)   # the attack is a probe, not a state change
    gk.model.eval()
    return m, setup, gk


def run_weights_arms(setup: Setup, seed: int, steps: int, unlearn_steps: int, verbose: bool = True) -> Dict[str, float]:
    """A second, independent GPT-2 whose weights DO carry the facts, unlearned two ways."""
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))   # config unused: the adapter is bypassed
    ppl_base = perplexity(gk)
    out = train_lora(gk, setup, seed, steps, LORA_LR, verbose=verbose)
    params = out["params"]
    trained = lora_state(gk.model.lm)
    m: Dict[str, float] = {"weights/n_lora_params": float(out["n_lora_params"]),
                           "weights/train_seconds": out["train_seconds"],
                           "weights/train_steps_used": float(out["steps_used"] or 0),
                           "weights/ppl_base": ppl_base,
                           "weights/ppl_after_learning": perplexity(gk)}
    score = weights_scorer(gk)
    m.update(measure(score, setup, gk, seed, "weights/before"))
    m["weights/before_delta_l2"] = lora_delta_norm(gk.model.lm)

    for mode, tag in (("ga", "ga"), ("relabel", "relabel")):
        load_lora_state(gk.model.lm, trained)
        u = unlearn(gk, setup, params, mode, seed, unlearn_steps, UNLEARN_LR, verbose=verbose)
        m[f"{tag}/delete_seconds"] = u["seconds"]
        m[f"{tag}/delete_seconds_per_fact"] = u["seconds"] / max(len(setup.targets), 1)
        m[f"{tag}/unlearn_steps"] = float(u["steps_used"])
        m.update(measure(weights_scorer(gk), setup, gk, seed, f"{tag}/after"))
        m[f"{tag}/ppl_after"] = perplexity(gk)
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
    "cells/relearn/supplied_acc", "cells/relearn/heldout_acc", "cells/relearn/heldout_forced_choice",
    "cells/relearn/heldout_top1", "cells/weight_delta_l2", "cells/delete_seconds",
    "weights/before/direct_acc",
    "ga/after/direct_acc", "ga/after/paraphrase_acc", "ga/after/forced_choice", "ga/after/true_obj_top1",
    "ga/after/true_obj_mean_rank", "ga/after/probe_top1", "ga/after/probe_top5", "ga/after/bystander_acc",
    "ga/relearn/supplied_acc", "ga/relearn/heldout_acc", "ga/relearn/heldout_forced_choice",
    "ga/relearn/heldout_top1", "ga/weight_delta_l2", "ga/delete_seconds",
    "relabel/after/direct_acc", "relabel/after/paraphrase_acc", "relabel/after/forced_choice",
    "relabel/after/true_obj_top1", "relabel/after/true_obj_mean_rank", "relabel/after/probe_top1",
    "relabel/after/probe_top5", "relabel/after/bystander_acc",
    "relabel/relearn/supplied_acc", "relabel/relearn/heldout_acc", "relabel/relearn/heldout_forced_choice",
    "relabel/relearn/heldout_top1", "relabel/weight_delta_l2", "relabel/delete_seconds",
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
    # the frozen core is untouched, bit for bit
    "cells/weight_delta_l2": ("<=", 0.0),
    "cells/after/bystander_acc": (">=", 0.70),
}


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--lora-steps", type=int, default=LORA_STEPS)
    ap.add_argument("--unlearn-steps", type=int, default=UNLEARN_STEPS)
    ap.add_argument("--n-targets", type=int, default=N_TARGETS)
    ap.add_argument("--n-cells", type=int, default=N_CELLS)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed: List[Dict[str, float]] = []
    for seed in args.seeds:
        print(f"=== seed {seed}: cells arm (E-000012 checkpoint, SHRED) ===", flush=True)
        m, setup, _ = run_cells_arm(seed, args.n_targets, args.n_cells)
        print(f"=== seed {seed}: weights arms (LoRA fine-tune, then unlearning) ===", flush=True)
        m.update(run_weights_arms(setup, seed, args.lora_steps, args.unlearn_steps))
        m["seed"] = seed
        per_seed.append(m)

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
             "cells/delete_seconds", "ga/delete_seconds", "relabel/delete_seconds"}
    sizes = {k: args.n_targets for k in keys if k.endswith(("direct_acc", "paraphrase_acc", "forced_choice",
                                                            "true_obj_top1", "probe_top1", "probe_top5",
                                                            "bystander_acc"))}
    for k in list(sizes):
        if "relearn/" in k:
            sizes[k] = args.n_targets // 2
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
        ["seconds to delete 50 facts", w("cells/delete_seconds", True), w("ga/delete_seconds", True),
         w("relabel/delete_seconds", True), "-"],
    ])

    record = {"experiment": "E-000024", "title": "deleting a fact from weights versus deleting it from cells",
              "seeds": args.seeds, "n_cells": args.n_cells, "n_targets": args.n_targets,
              "n_bystanders": N_BYSTANDERS, "lora_rank": LORA_RANK, "lora_steps": args.lora_steps,
              "unlearn_steps_budget": args.unlearn_steps, "relearn_steps": RELEARN_STEPS,
              "chance_top1": CHANCE_TOP1, "chance_mean_rank": CHANCE_RANK,
              "per_seed": per_seed, "aggregate": agg, "criteria": check}

    md = [f"# E-000024 — {record['title']}", "",
          f"Seeds {args.seeds}; {args.n_cells} facts, {args.n_targets} deletion targets, {N_BYSTANDERS} bystanders.",
          "The cells arm is the frozen GPT-2 of E-000012 with its trained adapter; the weights arms are the same",
          f"frozen GPT-2 with a rank-{LORA_RANK} LoRA fine-tuned on the identical facts and then unlearned two ways.",
          "All three arms are driven to the same surface criterion and attacked identically.", "",
          "## The comparison (worst seed)", "", compare, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), "",
          "## All measures", "", ledger.table(ledger.CI_HEADERS, rows), ""]
    path = ledger.save("e000024_weights_vs_cells", record, "\n".join(md))
    print(f"\nwritten: {path}")
    print(compare)
    print(ledger.criteria_table(check))
    return record


if __name__ == "__main__":
    main()
