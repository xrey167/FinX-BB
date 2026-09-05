"""Experiment E-000049 -- is the held-out addressing gap a linear-erasure problem?

THE CLAIM UNDER TEST. E-000039-A (ledger 31.21, 31.24) found that 88.6% of the held-out paraphrase gap
of E-000017-B's adapter is ADDRESSING -- the routing query ``q_l = q_proj[l](q_ln[l](h_l[last]))``,
the only phrasing-dependent tensor in the read path -- and that a neutral prefix (``PREFIX``) lifts
held-out addressing to 0.98-1.00 with no weight changed. E-000039-B (ledger 31.37) found that training
a tie on q across the trained phrasings does not move the held-out ones. This experiment asks the
training-free question in between: is the component of ``h_l[last]`` that the held-out subject-initial
forms (t8 ``'{s} currently lives in'``, t11 ``'{s}, who lives in'``) carry and the query head misreads
an ADDITIVE, LOW-RANK TEMPLATE COMPONENT that a linear eraser, fitted on the eight TRAINED templates
over facts disjoint from the evaluation targets, removes? The eraser sits on ``hl`` immediately before
``q_ln`` (``so/llm_adapter.py``, ``_make_hook``: ``q = q_proj(q_ln(hl))``) and on nothing else: the
residual, ``rms_h``, the injection and the frozen core are untouched. The number reported is the
RECOVERED FRACTION of a ceiling the same weights demonstrably reach,

    F_route(t) = (route_hit_erased(t) - route_hit_clean(t)) / (route_hit_prefixed(t) - route_hit_clean(t))

on t8 and t11, worst of the two, worst seed, with the prefixed ceiling RE-MEASURED in the same run.

WHAT COULD FALSIFY IT, EACH REGISTERED BELOW.  (i) The wrapper changes the forward at all: the clean
arm (erasers installed, mode 'off') must be bit-identical to the pristine model and must reproduce
E-000039-A's recorded per-seed t8/t11 numbers through ``decompose()`` -- else VOID.  (ii) The ceiling
is not reproduced (prefixed route_hit / read on t8, t11 below 0.95) -- F is undefined, VOID.  (iii) A
matched-rank RANDOM subspace erased at the same point moves addressing (a LayerNorm-statistics
artefact: ``q_ln`` re-normalises after the erasure) -- no sentence.  (iv) The design-matched
PERMUTATION null (E-000043's pattern: template labels shuffled within each fit fact, identical
construction) moves addressing -- the effect is variance removal, not template removal -- no sentence.
(v) The TRANSPORT control: the same subspace projected out of the residual at the last token AFTER
the query is taken (a forward hook on ``blocks[l]`` registered after the adapter's, so ``q_l`` is
computed from the unerased state and only what is carried downstream is erased) moves READING -- the
template component the head misreads is downstream, E-000039-A's addressing share was wrong, no
sentence.  (vi) Collapse: the query arm buys its recovery with an address collision rise, a between-
fact query cosine rise, a trained-template reading loss, a medial held-out addressing loss, or a
first-read-null loss -- PARTIAL, not positive.

TWO RESIDUALS THE REFUTER RAISED, CARRIED INTO THE SCOPE RATHER THAN SMOOTHED OVER.  (1) The rank-7
orthogonal arm may be pre-empted by training: ``q_proj . q_ln`` was trained to give one routing query
across the eight trained templates (within-fact cosine 0.78-0.80 at read layer 10), and a near-linear
map that agrees at eight template means approximately annihilates their difference span, which is
T_l^7 itself.  So the erasure may change q by little for ANY prompt and F_route ~ 0 would restate that
the head is already blind to the trained span.  This is measured, not assumed: ``sensitivity/<arm>/l<l>``
reports ||q(x) - q(erase(x))|| / ||q(x)|| on the fit states; below ~0.05 the rank-7 arm is
uninformative a priori and the PCA-32 interaction arm carries the reading.  (2) E-000050 (ledger
31.38) says the subject-initial failure is GPT-2's position-0 attention sink -- SUBJECT information is
lost at token 0, and a projection at ``h_l[last]`` removes, never adds -- so the NEGATIVE is the likelier
outcome and its closing clause ('the prefix acts through something other than an additive template
component') is predicted by a known mechanism.  What survives of a negative is the numeric bound: a
rank <= 32 linear erasure of the query input recovers < 0.25 of a ceiling the weights reach, which
bounds any training-free linear query normaliser in this adapter.  E-000050's BOS-at-inference ceiling
is measured on the same targets as a second denominator (``bos/``, ``*/F_route_bos``), diagnostic only.

OWNED, NOT CLAIMED.  Linear concept erasure as a mechanism: INLP (Ravfogel et al., 2020), LEACE
(Belrose et al., 2023 -- the whitened oblique eraser used verbatim as the LEACE-7 arm), amnesic
probing, mean-difference / format-direction projections; the routing-query decomposition and the
prefix ceiling are E-000039-A; the permutation null is E-000043; paraphrase-consistency training is
owned and is not used; the pod / expand-rule reading of 'phrasing as symlink' is GRACE / MELO /
MEMOIR (ledger 31.36) and the ``route_agreement_correct_all12`` line is reported as a number, not as a
mechanism.  What is unowned is the measurement: the pair (F_route, F_read) on a frozen LM's
addressable memory under a training-free template-subspace erasure of the retrieval query, against a
re-measured prompt-level ceiling, with the transport-side application of the identical subspace as
the control that separates 'the query input carries a template component' from 'the residual
downstream does'.

ONE DEVIATION FROM THE DESIGN, RECORDED.  The design asked for a permutation null at PCA-32 as well
as at rank 7.  PCA of the within-fact-centred states does not depend on the template labels at all --
the rows of the centred matrix are the same multiset under any within-fact relabelling -- so that
'null' is the PCA-32 arm itself, an identity, and a criterion on it could not fail.  It is not
computed; the rank-32 null is the matched-rank random subspace (3 draws), which the design already
carries, and the permutation null is computed at rank 7 where it is a genuine control.

Trains nothing.  Runs on the three recorded E-000017-B checkpoints in minutes.

Run:  python -m so.experiments.e000049_template_nullspace_addressing [--seeds 0 1 2] [--threads 1]
      smoke: --seeds 0 --n-targets 8 --n-fit 12 --pca-ranks 16 --null-ranks 7 16 --draws 1 --record-targets 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from so import ledger
from so.capacity import subspace_overlap
from so.data import bank_from_store
from so.experiments import e000008_gpt2_adapter as E8
from so.experiments import e000039_address_tying as E39
from so.experiments.e000001b_mini_transformer import CHECKPOINTS, CKPT_SUFFIX, _sha256
from so.experiments.e000017_paraphrase_gap import EVAL, TEMPLATES12
from so.llm_adapter import AdapterConfig, transformer_blocks
from so.mvcc import MVCCStore
from so.reference import load_world
from so.workspace import project_out
from so.world import UNKNOWN, World, fill_random

N_TRAIN, N_T, PREFIX = E39.N_TRAIN, E39.N_T, E39.PREFIX
RECORD_TARGETS = 100                  # E-000039-A's n_targets; the evaluation targets here are a PREFIX of that draw
HELD_INITIAL = (8, 11)                # the two held-out subject-initial forms the prediction is about
HELD_MEDIAL = (9, 10)
RECORD_JSON = ledger.RESULTS_DIR / "e000039a_address_decision.json"
PRIMARY = "query_T7"
QUERY_ARMS_FIXED = ("query_T7", "query_LEACE7")


# ------------------------------------------------------------------------------------ the eraser
class Eraser(nn.Module):
    """Sits between ``hl`` and ``q_ln``. ``off``: identity. ``record``: identity, keeps ``x``.
    ``erase``: ``x - (x Q) Q^T`` for an orthonormal ``Q`` (d, k), or LEACE's ``(x - mu) P^T + mu``."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.mode = "off"
        self.Q: Optional[torch.Tensor] = None
        self.P: Optional[torch.Tensor] = None
        self.mu: Optional[torch.Tensor] = None
        self.records: List[torch.Tensor] = []

    def set_orthogonal(self, Q: torch.Tensor) -> "Eraser":
        self.Q, self.P, self.mu = Q.to(torch.float32), None, None
        return self

    def set_leace(self, P: torch.Tensor, mu: torch.Tensor) -> "Eraser":
        self.Q, self.P, self.mu = None, P.to(torch.float32), mu.to(torch.float32)
        return self

    def erase(self, x: torch.Tensor) -> torch.Tensor:
        if self.P is not None:
            return (x - self.mu) @ self.P.t() + self.mu
        if self.Q is None or self.Q.numel() == 0:
            return x
        return x - (x @ self.Q) @ self.Q.t()

    def erase_linear(self, x: torch.Tensor) -> torch.Tensor:
        """The linear part, for variance accounting on centred data."""
        return x @ self.P.t() if self.P is not None else self.erase(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "record":
            self.records.append(x.detach().clone())
            return x
        if self.mode == "erase":
            return self.erase(x)
        return x


def wrap_query(model) -> Tuple[Dict[int, Eraser], Dict[int, nn.Module]]:
    """``model.q_ln[l] -> Sequential(Eraser, q_ln[l])`` after ``load_state_dict``; nothing else moves."""
    erasers, lns = {}, {}
    for l in model.cfg.read_layers:
        ln = model.q_ln[str(l)]
        if isinstance(ln, nn.Sequential):
            raise RuntimeError("q_ln is already wrapped")
        erasers[l] = Eraser(model.d)
        lns[l] = ln
        model.q_ln[str(l)] = nn.Sequential(erasers[l], ln)
    return erasers, lns


def set_mode(erasers: Dict[int, Eraser], mode: str) -> None:
    for e in erasers.values():
        e.mode = mode
        if mode == "record":
            e.records = []


def transport_hooks(model, Q: Dict[int, torch.Tensor]) -> List[Any]:
    """Project ``Q[l]`` out of the residual at the last token AFTER the adapter's hook on ``blocks[l]``.

    PyTorch runs forward hooks in registration order and hands each the previous one's return, so
    this hook sees ``h + delta``: the query at layer l was taken from the unerased state, and only what
    the block passes upward is erased."""
    blocks = transformer_blocks(model.lm)
    handles = []
    for l, q in Q.items():
        dirs = q.t().contiguous()

        def hook(module, inputs, output, dirs=dirs):
            if model._ctx is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            idx = model._ctx["last_idx"]
            ar = torch.arange(h.shape[0], device=h.device)
            h2 = h.clone()
            h2[ar, idx] = project_out(h[ar, idx], dirs)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

        handles.append(blocks[l].register_forward_hook(hook))
    return handles


# -------------------------------------------------------------------------------- the world
def build_world(gk: E8.GPT2Knowledge, seed: int, centre: np.ndarray, n_targets: int, n_fit: int) -> Dict[str, Any]:
    """Exactly ``decompose()``'s world, store and bank (e000039:255-260); the targets are the first
    ``n_targets`` of its RECORD_TARGETS-draw, the fit facts are drawn from what that draw left."""
    rng = np.random.default_rng(seed)
    world = fill_random(rng, World(gk.n_entities, 4, N_TRAIN, []), EVAL["n_cells"])
    store = MVCCStore(marker_dim=centre.shape[0], seed=seed, marker_centre=centre)
    load_world(store, world)
    bank = bank_from_store(store)
    n_draw = min(max(RECORD_TARGETS, n_targets), len(world.facts))
    idx = rng.choice(len(world.facts), size=n_draw, replace=False)
    targets = [world.facts[int(i)] for i in idx[:n_targets]]
    rest = np.setdiff1d(np.arange(len(world.facts)), idx)
    fit_idx = rng.choice(rest, size=min(n_fit, len(rest)), replace=False)
    fit = [world.facts[int(i)] for i in fit_idx]
    return {"world": world, "bank": bank, "tensors": bank.tensors(), "targets": targets, "fit": fit,
            "truth": np.array([f.obj for f in targets]),
            "cstar": np.array([bank.kid_of_key[f.key] for f in targets])}


def prompts(gk: E8.GPT2Knowledge, facts, t: int, prefix: str = "") -> List[str]:
    return [prefix + TEMPLATES12[f.relation][t].format(s=gk.names[f.subject]) for f in facts]


# --------------------------------------------------------------------------- the measurement
@torch.no_grad()
def measure(gk: E8.GPT2Knowledge, tensors, targets, cstar: np.ndarray, template_ids: Sequence[int],
            prefix: str = "", batch: int = 32) -> Dict[str, Any]:
    """``decompose()``'s per-template loop (e000039:273-290), returning the per-fact matrices.

    read[t, j]: candidate argmax == object; cell[t, j]: argmax of the LAST read slot's routing;
    hit = cell == cstar; null0: the layer-8 slot routed to the null column; q: ``last_query``."""
    gk.model.eval()
    N = len(targets)
    out = {"read": {}, "hit": {}, "cell": {}, "null0": {}, "q": {}}
    for t in template_ids:
        texts = prompts(gk, targets, t, prefix)
        a_all, cell, null0, qs = [], [], [], []
        for i in range(0, N, batch):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i + batch])
            cand, _, routing, _ = gk.model(tensors, ids, am, last)
            a = cand.argmax(-1).numpy()
            a_all.append(np.where(a == gk.n_entities, UNKNOWN, a))
            cell.append(routing[:, -1].numpy().argmax(-1))
            null0.append(routing[:, 0].numpy().argmax(-1) == routing.shape[-1] - 1)
            qs.append(gk.model.last_query.numpy())
        out["read"][t] = np.concatenate(a_all)
        out["cell"][t] = np.concatenate(cell)
        out["hit"][t] = out["cell"][t] == cstar
        out["null0"][t] = np.concatenate(null0)
        out["q"][t] = np.concatenate(qs)                      # (N, R, d_key)
    return out


def query_cosines(meas: Dict[str, Any], template_ids: Sequence[int]) -> Dict[str, float]:
    """decompose():293-301 -- mean cosine of the routing query within a fact across phrasings, and
    between different facts, per read slot."""
    Q = np.stack([meas["q"][t] for t in template_ids])                          # (T, N, R, d)
    U = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-9)
    T, N = U.shape[0], U.shape[1]
    m = {}
    for r in range(U.shape[2]):
        M = U[:, :, r]
        pairs = [(M[i] * M[j]).sum(-1).mean() for i in range(T) for j in range(i + 1, T)]
        m[f"query_cos_within_fact/read{r}"] = float(np.mean(pairs)) if pairs else float("nan")
        G = M.reshape(-1, M.shape[-1]) @ M.reshape(-1, M.shape[-1]).T
        lab = np.repeat(np.arange(N)[None], T, 0).reshape(-1)
        m[f"query_cos_between_fact/read{r}"] = float(G[lab[:, None] != lab[None, :]].mean())
    return m


def arm_metrics(arm: str, meas: Dict[str, Any], pref: Dict[str, Any], truth: np.ndarray, cstar: np.ndarray,
                clean: Optional[Dict[str, float]] = None, clean_meas: Optional[Dict[str, Any]] = None,
                bos: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Flat per-arm record. With ``clean`` given, the recovered fractions against the CLEAN arm's
    re-measured prefixed ceiling, and the deltas the collapse bars are read from."""
    N = len(truth)
    m: Dict[str, float] = {}
    for t in range(N_T):
        m[f"{arm}/t{t}/read"] = float((meas["read"][t] == truth).mean())
        m[f"{arm}/t{t}/route_hit"] = float(meas["hit"][t].mean())
        m[f"{arm}/t{t}/first_read_null"] = float(meas["null0"][t].mean())
    for t in HELD_INITIAL:
        m[f"{arm}/prefixed/t{t}/read"] = float((pref["read"][t] == truth).mean())
        m[f"{arm}/prefixed/t{t}/route_hit"] = float(pref["hit"][t].mean())
    m[f"{arm}/train_read_mean"] = float(np.mean([m[f"{arm}/t{t}/read"] for t in range(N_TRAIN)]))
    m[f"{arm}/heldout_route_min"] = min(m[f"{arm}/t{t}/route_hit"] for t in range(N_TRAIN, N_T))
    m[f"{arm}/heldout_read_min"] = min(m[f"{arm}/t{t}/read"] for t in range(N_TRAIN, N_T))
    m[f"{arm}/initial_heldout_route_min"] = min(m[f"{arm}/t{t}/route_hit"] for t in HELD_INITIAL)
    m[f"{arm}/medial_heldout_route_min"] = min(m[f"{arm}/t{t}/route_hit"] for t in HELD_MEDIAL)
    m[f"{arm}/first_read_null_mean"] = float(np.mean([m[f"{arm}/t{t}/first_read_null"] for t in range(N_T)]))
    m[f"{arm}/address_collision"] = float(np.mean([1.0 - len(set(meas["cell"][t].tolist())) / N
                                                   for t in range(N_TRAIN, N_T)]))
    for k, v in query_cosines(meas, range(N_T)).items():
        m[f"{arm}/{k}"] = v
    C = np.stack([meas["cell"][t] for t in range(N_T)])                       # (12, N)
    agree = (C == C[0][None]).all(0)
    m[f"{arm}/route_agreement_all12"] = float(agree.mean())
    m[f"{arm}/route_agreement_correct_all12"] = float((agree & (C[0] == cstar)).mean())
    if clean is None:
        return m
    for what, key in (("route", "route_hit"), ("read", "read")):
        vals = []
        for t in HELD_INITIAL:
            lo, hi, x = clean[f"clean/t{t}/{key}"], clean[f"clean/prefixed/t{t}/{key}"], m[f"{arm}/t{t}/{key}"]
            f = (x - lo) / (hi - lo) if hi - lo > 0.02 else float("nan")   # no ceiling above the clean arm: undefined
            m[f"{arm}/F_{what}/t{t}"] = f
            vals.append(f)
        m[f"{arm}/F_{what}_min"] = float(np.min(vals)) if not any(np.isnan(vals)) else float("nan")
        m[f"{arm}/F_{what}_max"] = float(np.max(vals)) if not any(np.isnan(vals)) else float("nan")
    if bos is not None:
        vals = []
        for t in HELD_INITIAL:
            lo, hi, x = clean[f"clean/t{t}/route_hit"], bos[f"bos/t{t}/route_hit"], m[f"{arm}/t{t}/route_hit"]
            vals.append((x - lo) / (hi - lo) if hi - lo > 0.02 else float("nan"))
        m[f"{arm}/F_route_bos_min"] = float(np.min(vals)) if not any(np.isnan(vals)) else float("nan")
    m[f"{arm}/medial_abs_change_mean"] = float(np.mean([abs(m[f"{arm}/t{t}/route_hit"] - clean[f"clean/t{t}/route_hit"])
                                                        for t in HELD_MEDIAL]))
    m[f"{arm}/address_collision_delta"] = m[f"{arm}/address_collision"] - clean["clean/address_collision"]
    m[f"{arm}/train_read_drop"] = clean["clean/train_read_mean"] - m[f"{arm}/train_read_mean"]
    m[f"{arm}/medial_route_drop"] = clean["clean/medial_heldout_route_min"] - m[f"{arm}/medial_heldout_route_min"]
    m[f"{arm}/first_read_null_drop"] = clean["clean/first_read_null_mean"] - m[f"{arm}/first_read_null_mean"]
    if clean_meas is not None:
        # how far the arm moved the query itself on the subject-initial held-out prompts, per read slot
        for r in range(meas["q"][HELD_INITIAL[0]].shape[1]):
            cs = [F.cosine_similarity(torch.as_tensor(meas["q"][t][:, r]), torch.as_tensor(clean_meas["q"][t][:, r]),
                                      dim=-1).mean().item() for t in HELD_INITIAL]
            m[f"{arm}/q_cos_to_clean/read{r}"] = float(np.mean(cs))
    return m


# ------------------------------------------------------------------------ the fit states
@torch.no_grad()
def template_states(gk: E8.GPT2Knowledge, tensors, fit, erasers: Dict[int, Eraser],
                    batch: int = 32) -> Dict[int, torch.Tensor]:
    """``h_l[last]`` for every fit fact x trained template, {l: (n_fit, N_TRAIN, d)}, recorded through
    the eraser in record mode with the bank attached (layer 10 sees layer 8's injection)."""
    set_mode(erasers, "record")
    for t in range(N_TRAIN):
        texts = prompts(gk, fit, t)
        for i in range(0, len(texts), batch):
            ids, am, last = E8.encode_texts(gk.tok, texts[i:i + batch])
            gk.model(tensors, ids, am, last)
    out = {}
    for l, e in erasers.items():
        X = torch.cat(e.records)                                             # (N_TRAIN * n_fit, d), template-major
        out[l] = X.reshape(N_TRAIN, len(fit), -1).permute(1, 0, 2).contiguous()
    set_mode(erasers, "off")
    return out


def top_right_singular(a: torch.Tensor, k: int) -> torch.Tensor:
    """Top-``k`` right singular vectors of ``a`` (n, d) as an orthonormal (d, k)."""
    _, _, vh = torch.linalg.svd(a.to(torch.float64), full_matrices=False)
    return vh[:k].t().contiguous()


def within_fact_centred(states: torch.Tensor) -> torch.Tensor:
    return states.to(torch.float64) - states.to(torch.float64).mean(1, keepdim=True)


def between_template_span(states: torch.Tensor) -> torch.Tensor:
    """(a) the rank-(N_TRAIN - 1) span of the centred template means: orthonormal Q (d, N_TRAIN - 1)."""
    means = within_fact_centred(states).mean(0)                              # (N_TRAIN, d), rows sum to 0
    return top_right_singular(means, N_TRAIN - 1)


def pca_span(states: torch.Tensor, k: int) -> torch.Tensor:
    """(c) top-k right singular vectors of the within-fact-centred states: between-template span PLUS
    the fact x template interaction."""
    X = within_fact_centred(states).reshape(-1, states.shape[-1])
    return top_right_singular(X, k)


def leace(states: torch.Tensor, lam: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """(b) LEACE (Belrose et al. 2023, Thm 4.1): ``P = I - W^+ Proj(W Sigma_XZ) W`` with ``W =
    Sigma_XX^{-1/2}`` on the raw fit states (shrinkage ``lam * tr/d`` on the diagonal), Z the one-hot
    template.  In this balanced design Sigma_XZ's column span is the centred-template-mean span of (a),
    so the arm removes the same concept span through the whitened, oblique projection."""
    n_fit, T, d = states.shape
    X = states.to(torch.float64).reshape(-1, d)
    Z = torch.eye(T, dtype=torch.float64).repeat(n_fit, 1)
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    n = X.shape[0]
    Sxx = Xc.t() @ Xc / n
    Sxx = Sxx + lam * (torch.trace(Sxx) / d) * torch.eye(d, dtype=torch.float64)
    Sxz = Xc.t() @ Zc / n                                                     # (d, T), rank T - 1
    e, V = torch.linalg.eigh(Sxx)
    W = (V * e.rsqrt()) @ V.t()
    W_pinv = (V * e.sqrt()) @ V.t()
    U, s, _ = torch.linalg.svd(W @ Sxz, full_matrices=False)
    keep = min(T - 1, int((s > s.max() * 1e-8).sum()))
    U = U[:, :keep]
    P = torch.eye(d, dtype=torch.float64) - W_pinv @ (U @ U.t()) @ W
    return P, mu


def permuted_span(states: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """E-000043's design-matched null: template labels shuffled WITHIN each fit fact, then (a) again.
    The marginal spread of the states survives; only the template structure is destroyed."""
    S = states.clone()
    for j in range(S.shape[0]):
        S[j] = S[j][torch.as_tensor(rng.permutation(S.shape[1]))]
    return between_template_span(S)


def random_span(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return q


def removed_fraction(er: Eraser, X: torch.Tensor) -> float:
    """Share of ||X||_F^2 the eraser's linear part removes from centred rows ``X`` (n, d)."""
    Xf = X.to(torch.float32)
    return float(((Xf - er.erase_linear(Xf)).pow(2).sum() / Xf.pow(2).sum().clamp_min(1e-12)).item())


@torch.no_grad()
def query_sensitivity(model, ln: nn.Module, l: int, er: Eraser, X: torch.Tensor) -> Tuple[float, float]:
    """The refuter's a-priori check: how much the routing query moves when the subspace is erased from
    the fit states themselves.  Returns (mean ||q - q'|| / ||q||, mean cos(q, q'))."""
    Xf = X.to(torch.float32)
    q = model.q_proj[str(l)](ln(Xf))
    qe = model.q_proj[str(l)](ln(er.erase(Xf)))
    rel = ((q - qe).norm(dim=-1) / q.norm(dim=-1).clamp_min(1e-9)).mean().item()
    return float(rel), float(F.cosine_similarity(q, qe, dim=-1).mean().item())


# ------------------------------------------------------------------------------ one seed
def load_checkpoint(seed: int) -> Tuple[E8.GPT2Knowledge, np.ndarray, str]:
    """e000039:402-406 -- E-000017-B's recorded checkpoint, adapter loaded strict=False, eval()."""
    gk = E8.GPT2Knowledge(AdapterConfig(status_gated=True))
    path = CHECKPOINTS / f"e000017_t8_c0{CKPT_SUFFIX}_seed{seed}.pt"
    ck = torch.load(path, weights_only=False)
    gk.model.load_state_dict(ck["adapter"], strict=False)
    gk.model.eval()
    return gk, np.asarray(ck["centre"]), _sha256(path)


def recorded_e39a(seed: int) -> Optional[Dict[str, float]]:
    if not RECORD_JSON.exists():
        return None
    rec = json.loads(RECORD_JSON.read_text())
    for s in rec.get("per_seed", []):
        if int(s.get("seed", -1)) == 1700 + seed:
            return s
    return None


@torch.no_grad()
def run_seed(seed: int, n_targets: int, n_fit: int, pca_ranks: Sequence[int], null_ranks: Sequence[int],
             draws: int, record_targets: int, batch: int) -> Dict[str, Any]:
    t0 = time.time()
    gk, centre, sha = load_checkpoint(seed)
    model = gk.model
    W = build_world(gk, 1700 + seed, centre, n_targets, n_fit)
    tensors, targets, truth, cstar = W["tensors"], W["targets"], W["truth"], W["cstar"]
    m: Dict[str, Any] = {"seed": seed, "checkpoint_sha256": sha, "n_targets": len(targets), "n_fit": len(W["fit"])}
    print(f"=== seed {seed}: {len(targets)} targets, {len(W['fit'])} fit facts ===", flush=True)

    # INSTRUMENT (i): the wrapper must not change the forward. Pristine forward, wrap, forward again.
    ids, am, last = E8.encode_texts(gk.tok, prompts(gk, targets, HELD_INITIAL[0])[:batch])
    c0, _, r0, _ = model(tensors, ids, am, last)
    q0 = model.last_query.clone()
    erasers, lns = wrap_query(model)
    set_mode(erasers, "off")
    c1, _, r1, _ = model(tensors, ids, am, last)
    q1 = model.last_query
    m["instrument/wrapper_max_abs_diff"] = float(max((c0 - c1).abs().max(), (r0 - r1).abs().max(), (q0 - q1).abs().max()))

    # INSTRUMENT (ii): decompose() itself, through the wrapped model, against E-000039-A's record.
    m["instrument/clean_matches_e39a_abs"] = float("nan")
    if record_targets > 0:
        d = E39.decompose(gk, 1700 + seed, centre, record_targets, oracle=False)
        keys = [f"t{t}/heldout/{k}" for t in HELD_INITIAL for k in ("route_hit", "read")] + \
               [f"t{t}/prefixed_{k}" for t in HELD_INITIAL for k in ("route_hit", "read")]
        for k in keys:
            m[f"record/{k}"] = float(d[k])
        ref = recorded_e39a(seed)
        m["record/n_targets"] = record_targets
        m["record/at_record_size"] = float(record_targets == RECORD_TARGETS and ref is not None)
        if ref is not None:
            m["instrument/clean_matches_e39a_abs"] = float(max(abs(d[k] - ref[k]) for k in keys if k in ref))
    print(f"  instrument: wrapper diff {m['instrument/wrapper_max_abs_diff']:.2e}, "
          f"record diff {m['instrument/clean_matches_e39a_abs']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # CLEAN arm and the two ceilings (prefix, re-measured; BOS, E-000050 arm B, diagnostic).
    clean_meas = measure(gk, tensors, targets, cstar, range(N_T), batch=batch)
    clean_pref = measure(gk, tensors, targets, cstar, HELD_INITIAL, PREFIX, batch=batch)
    clean = arm_metrics("clean", clean_meas, clean_pref, truth, cstar)
    m.update(clean)
    m["prefix/route_hit_min"] = min(m[f"clean/prefixed/t{t}/route_hit"] for t in HELD_INITIAL)
    m["prefix/read_min"] = min(m[f"clean/prefixed/t{t}/read"] for t in HELD_INITIAL)
    prev = os.environ.get("SO_BOS")
    os.environ["SO_BOS"] = "1"
    try:
        bos_meas = measure(gk, tensors, targets, cstar, HELD_INITIAL, batch=batch)
    finally:
        if prev is None:
            os.environ.pop("SO_BOS", None)
        else:
            os.environ["SO_BOS"] = prev
    bos = {}
    for t in HELD_INITIAL:
        bos[f"bos/t{t}/route_hit"] = float(bos_meas["hit"][t].mean())
        bos[f"bos/t{t}/read"] = float((bos_meas["read"][t] == truth).mean())
    m.update(bos)
    print(f"  clean: t8 route {m['clean/t8/route_hit']:.4f} read {m['clean/t8/read']:.4f} | "
          f"t11 route {m['clean/t11/route_hit']:.4f} read {m['clean/t11/read']:.4f} | "
          f"prefix route_min {m['prefix/route_hit_min']:.4f} read_min {m['prefix/read_min']:.4f} | "
          f"bos t8/t11 route {bos['bos/t8/route_hit']:.4f}/{bos['bos/t11/route_hit']:.4f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    # FIT STATES and the subspaces, per read layer.
    states = template_states(gk, tensors, W["fit"], erasers, batch)
    layers = list(states)
    d = model.d
    Xw = {l: within_fact_centred(states[l]).reshape(-1, d) for l in layers}
    Xt = {l: (states[l].to(torch.float64).reshape(-1, d) - states[l].to(torch.float64).reshape(-1, d).mean(0)) for l in layers}
    subspaces: Dict[str, Dict[int, Eraser]] = {}
    Q_T7 = {l: between_template_span(states[l]) for l in layers}
    subspaces["query_T7"] = {l: Eraser(d).set_orthogonal(Q_T7[l]) for l in layers}
    subspaces["query_LEACE7"] = {l: Eraser(d).set_leace(*leace(states[l])) for l in layers}
    Q_pca = {k: {l: pca_span(states[l], k) for l in layers} for k in pca_ranks}
    for k in pca_ranks:
        subspaces[f"query_PCA{k}"] = {l: Eraser(d).set_orthogonal(Q_pca[k][l]) for l in layers}
    rng = np.random.default_rng(seed * 100 + 49)
    Q_perm = {l: permuted_span(states[l], rng) for l in layers}
    subspaces["null_perm_T7"] = {l: Eraser(d).set_orthogonal(Q_perm[l]) for l in layers}
    for k in null_ranks:
        for dr in range(draws):
            subspaces[f"null_random_k{k}_d{dr}"] = {l: Eraser(d).set_orthogonal(random_span(d, k, seed * 100 + dr * 10 + li))
                                                   for li, l in enumerate(layers)}
    for name, ers in subspaces.items():
        for l in layers:
            m[f"subspace/{name}/l{l}/removed_within"] = removed_fraction(ers[l], Xw[l])
            m[f"subspace/{name}/l{l}/removed_total"] = removed_fraction(ers[l], Xt[l])
            rel, cos = query_sensitivity(model, lns[l], l, ers[l], states[l].reshape(-1, d))
            m[f"sensitivity/{name}/l{l}/rel_change"] = rel
            m[f"sensitivity/{name}/l{l}/cos"] = cos
    if len(layers) == 2:
        a, b = layers
        m["subspace/query_T7/overlap_8_10"] = subspace_overlap(Q_T7[a].t().float(), Q_T7[b].t().float())
        for k in pca_ranks:
            m[f"subspace/query_PCA{k}/overlap_8_10"] = subspace_overlap(Q_pca[k][a].t().float(), Q_pca[k][b].t().float())
    print("  subspaces: " + ", ".join(f"{n} within {m[f'subspace/{n}/l{layers[-1]}/removed_within']:.3f} "
                                       f"sens {m[f'sensitivity/{n}/l{layers[-1]}/rel_change']:.3f}"
                                       for n in subspaces), flush=True)

    # QUERY arms and NULLS: erase before q_ln at both read layers, nothing else.
    def run_arm(name: str, ers: Dict[int, Eraser]) -> None:
        for l in layers:
            erasers[l].Q, erasers[l].P, erasers[l].mu = ers[l].Q, ers[l].P, ers[l].mu
        set_mode(erasers, "erase")
        meas = measure(gk, tensors, targets, cstar, range(N_T), batch=batch)
        pref = measure(gk, tensors, targets, cstar, HELD_INITIAL, PREFIX, batch=batch)
        set_mode(erasers, "off")
        m.update(arm_metrics(name, meas, pref, truth, cstar, clean, clean_meas, bos))
        print(f"  {name:22s} t8 route {m[f'{name}/t8/route_hit']:.4f} read {m[f'{name}/t8/read']:.4f} | "
              f"t11 route {m[f'{name}/t11/route_hit']:.4f} read {m[f'{name}/t11/read']:.4f} | "
              f"F_route_min {m[f'{name}/F_route_min']:.4f} F_read_min {m[f'{name}/F_read_min']:.4f} | "
              f"coll {m[f'{name}/address_collision']:.4f} train {m[f'{name}/train_read_mean']:.4f} "
              f"agree {m[f'{name}/route_agreement_correct_all12']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    for name, ers in subspaces.items():
        run_arm(name, ers)
    for l in layers:
        erasers[l].Q = erasers[l].P = erasers[l].mu = None

    # TRANSPORT: the same T7 and PCA-32 (the largest PCA rank asked for) out of the residual AFTER the query.
    transport_sets = {"transport_T7": Q_T7}
    if pca_ranks:
        kmax = max(pca_ranks)
        transport_sets[f"transport_PCA{kmax}"] = Q_pca[kmax]
    for name, Qs in transport_sets.items():
        handles = transport_hooks(model, {l: Qs[l].to(torch.float32) for l in layers})
        try:
            meas = measure(gk, tensors, targets, cstar, range(N_T), batch=batch)
            pref = measure(gk, tensors, targets, cstar, HELD_INITIAL, PREFIX, batch=batch)
        finally:
            for h in handles:
                h.remove()
        m.update(arm_metrics(name, meas, pref, truth, cstar, clean, clean_meas, bos))
        print(f"  {name:22s} t8 route {m[f'{name}/t8/route_hit']:.4f} read {m[f'{name}/t8/read']:.4f} | "
              f"t11 route {m[f'{name}/t11/route_hit']:.4f} read {m[f'{name}/t11/read']:.4f} | "
              f"F_route_min {m[f'{name}/F_route_min']:.4f} F_read_max {m[f'{name}/F_read_max']:.4f} | "
              f"q_cos_to_clean/read1 {m[f'{name}/q_cos_to_clean/read1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # SUMMARIES the criteria read.
    rand = [n for n in subspaces if n.startswith("null_random")]
    m["null_random/F_route_max"] = float(np.max([m[f"{n}/F_route_max"] for n in rand])) if rand else float("nan")
    m["null_random/medial_abs_change_max"] = float(np.max([m[f"{n}/medial_abs_change_mean"] for n in rand])) if rand else float("nan")
    m["null_perm/F_route_max"] = m["null_perm_T7/F_route_max"]
    m["transport/F_read_max"] = float(np.max([m[f"{n}/F_read_max"] for n in transport_sets]))
    m["transport/F_route_max"] = float(np.max([m[f"{n}/F_route_max"] for n in transport_sets]))
    query_arms = [n for n in subspaces if n.startswith("query_")]
    m["query_best/F_route_min"] = float(np.max([m[f"{n}/F_route_min"] for n in query_arms]))
    m["query_best/F_read_min"] = float(np.max([m[f"{n}/F_read_min"] for n in query_arms]))
    m["query_best/arm"] = max(query_arms, key=lambda n: (m[f"{n}/F_route_min"] if m[f"{n}/F_route_min"] == m[f"{n}/F_route_min"] else -9))
    if pca_ranks:
        m["rank_monotone/PCA_minus_T7"] = m[f"query_PCA{max(pca_ranks)}/F_route_min"] - m["query_T7/F_route_min"]
    m["seconds"] = time.time() - t0
    return m


# ---------------------------------------------------------------------- pre-registered bars
CRITERIA = {
    # INSTRUMENT: the wrapper is neutral (bit-identical forward), and decompose() through it reproduces
    # E-000039-A's recorded per-seed t8/t11 route_hit and read and the prefixed ceiling within 0.03.
    "instrument/wrapper_max_abs_diff": ("<=", 1e-6),
    "instrument/clean_matches_e39a_abs": ("<=", 0.03),
    # CEILING EXISTS, re-measured: without it F is undefined and the run is VOID.
    "prefix/route_hit_min": (">=", 0.95),
    "prefix/read_min": (">=", 0.95),
    # PRIMARY: the rank-7 orthogonal erasure at both read layers recovers half the ceiling on both.
    "query_T7/F_route_min": (">=", 0.50),
    "query_T7/F_read_min": (">=", 0.50),
    # NULLS: erasing k directions that are not the template subspace must not move addressing.
    "null_random/F_route_max": ("<=", 0.10),
    "null_random/medial_abs_change_max": ("<=", 0.03),
    "null_perm/F_route_max": ("<=", 0.10),
    # TRANSPORT: the same subspace erased AFTER the query must not move reading.
    "transport/F_read_max": ("<=", 0.10),
    # COLLAPSE, each relative to the measured clean arm (the absolute 0.02 collision bar of E-000039-B
    # fails at baseline, ledger 31.37, and is not reused).
    "query_T7/address_collision_delta": ("<=", 0.02),
    "query_T7/query_cos_between_fact/read1": ("<=", 0.40),
    "query_T7/train_read_drop": ("<=", 0.02),
    "query_T7/medial_route_drop": ("<=", 0.03),
    "query_T7/first_read_null_drop": ("<=", 0.03),
    # POD: the phrasing-as-symlink line, a separate claim reported in every branch.
    "query_T7/route_agreement_correct_all12": (">=", 0.90),
}
NEGATIVE_BAR = 0.25          # max over the four query arms of F_route_min below this -> NEGATIVE
COLLAPSE_KEYS = ("query_T7/address_collision_delta", "query_T7/query_cos_between_fact/read1",
                 "query_T7/train_read_drop", "query_T7/medial_route_drop", "query_T7/first_read_null_drop")
VOID_KEYS = ("instrument/wrapper_max_abs_diff", "instrument/clean_matches_e39a_abs", "prefix/route_hit_min", "prefix/read_min")
NULL_KEYS = ("null_random/F_route_max", "null_random/medial_abs_change_max", "null_perm/F_route_max", "transport/F_read_max")

DECISION_RULE = (
    "Fixed before the run, worst seed over {0, 1, 2} on the 64 targets of decompose()'s world (seed 1700+seed). "
    "(1) If the instrument check or the ceiling check fails, VOID, no other number is read. "
    "(2) If any null exceeds 0.10 (random, k in {7, 32}, 3 draws; permutation, k = 7) or the transport control "
    "exceeds 0.10 on reading, the run records 'erasure moves addressing for a reason that is not the template "
    "subspace' and neither result sentence is claimed. "
    "(3) With (1)-(2) clean: query_T7/F_route_min >= 0.50 and query_T7/F_read_min >= 0.50 and all five collapse "
    "bars pass -> POSITIVE with the rank-7 numbers; max over the four query arms (T7, LEACE-7, PCA-16, PCA-32) of "
    "F_route_min < 0.25 -> NEGATIVE; anything between (0.25 <= F < 0.50, or >= 0.50 only at rank 16/32, or >= 0.50 "
    "with a collapse bar failing) -> PARTIAL, recorded with the arm at which F first exceeds 0.25 and the collapse "
    "bar that failed, no result sentence claimed. "
    "(4) The pod line (query_T7/route_agreement_correct_all12 >= 0.90) is reported under the primary arm in every "
    "branch and is a separate claim from (3). Rank monotonicity (F_route at PCA-32 >= F_route at T7 - 0.05) is a "
    "sanity check that flags, not decides. The refuter's a-priori check, sensitivity/query_T7 below 0.05, flags the "
    "rank-7 arm as uninformative before its F is read; it does not change the rule.")


def decide(check: Dict[str, Any], agg: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    c = check["criteria"]
    ok = lambda k: bool(c.get(k, {}).get("pass", False))     # noqa: E731
    out: Dict[str, Any] = {}
    if not all(ok(k) for k in VOID_KEYS):
        out["reading"] = "VOID"
        out["why"] = "instrument or ceiling check failed: " + ", ".join(k for k in VOID_KEYS if not ok(k))
    elif not all(ok(k) for k in NULL_KEYS):
        out["reading"] = "NO SENTENCE"
        out["why"] = ("erasure moves addressing for a reason that is not the template subspace: "
                      + ", ".join(k for k in NULL_KEYS if not ok(k)))
    else:
        primary = ok("query_T7/F_route_min") and ok("query_T7/F_read_min")
        collapse_failed = [k for k in COLLAPSE_KEYS if not ok(k)]
        best = agg.get("query_best/F_route_min", {}).get("min", float("nan"))
        if primary and not collapse_failed:
            out["reading"] = "POSITIVE"
            out["why"] = "rank-7 query erasure recovers >= 0.50 of the prefix ceiling on addressing and reading, no collapse"
        elif best == best and best < NEGATIVE_BAR:
            out["reading"] = "NEGATIVE"
            out["why"] = f"max over the query arms of F_route_min (worst seed) = {best:.4f} < {NEGATIVE_BAR}"
        else:
            out["reading"] = "PARTIAL"
            out["why"] = (f"best query arm F_route_min {best:.4f}; primary passes: {primary}; "
                          f"collapse bars failed: {collapse_failed or 'none'}")
    out["pod_line"] = ok("query_T7/route_agreement_correct_all12")
    rm = agg.get("rank_monotone/PCA_minus_T7", {}).get("min", float("nan"))
    out["rank_monotone_flag"] = bool(rm == rm and rm < -0.05)
    sens = agg.get("sensitivity/query_T7/l10/rel_change", agg.get("sensitivity/query_T7/l8/rel_change", {})).get("max", float("nan"))
    out["rank7_uninformative_a_priori"] = bool(sens == sens and sens < 0.05)
    return out


# ------------------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-targets", type=int, default=64)
    ap.add_argument("--n-fit", type=int, default=128)
    ap.add_argument("--pca-ranks", type=int, nargs="*", default=[16, 32])
    ap.add_argument("--null-ranks", type=int, nargs="*", default=[7, 32])
    ap.add_argument("--draws", type=int, default=3, help="random-subspace null draws per rank")
    ap.add_argument("--record-targets", type=int, default=RECORD_TARGETS,
                    help="targets for the in-process decompose() instrument check (100 = the record; 0 skips)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SO_THREADS", "0")))
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    per_seed = [run_seed(s, args.n_targets, args.n_fit, args.pca_ranks, args.null_ranks, args.draws,
                         args.record_targets, args.batch) for s in args.seeds]
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float, bool, np.floating))
            and k not in ("seed",) and all(k in s for s in per_seed)]
    agg = ledger.aggregate(per_seed, keys)
    check = ledger.check_criteria(agg, {k: v for k, v in CRITERIA.items() if k in agg})
    verdict = decide(check, agg)

    arms = ["clean"] + [k.split("/")[0] for k in per_seed[0] if k.endswith("/F_route_min") and "/" in k
                        and not k.startswith("query_best")]
    arms = list(dict.fromkeys(arms))

    def worst(k: str, lower: bool) -> str:
        return "-" if k not in agg else f"{ledger.worst(agg[k], lower):.4f}"

    rows = []
    for a in arms:
        rows.append([a, worst(f"{a}/t8/route_hit", False), worst(f"{a}/t11/route_hit", False),
                     worst(f"{a}/t8/read", False), worst(f"{a}/t11/read", False),
                     worst(f"{a}/F_route_min", False) if a != "clean" else "-",
                     worst(f"{a}/F_read_min", False) if a != "clean" else "-",
                     worst(f"{a}/address_collision", True), worst(f"{a}/train_read_mean", False),
                     worst(f"{a}/medial_heldout_route_min", False),
                     worst(f"{a}/route_agreement_correct_all12", False)])
    rows.insert(1, ["prefix (clean, re-measured)", worst("clean/prefixed/t8/route_hit", False),
                    worst("clean/prefixed/t11/route_hit", False), worst("clean/prefixed/t8/read", False),
                    worst("clean/prefixed/t11/read", False), "1 by definition", "1 by definition", "-", "-", "-", "-"])
    rows.insert(2, ["BOS at inference (E-000050 arm B, diagnostic)", worst("bos/t8/route_hit", False),
                    worst("bos/t11/route_hit", False), worst("bos/t8/read", False), worst("bos/t11/read", False),
                    "-", "-", "-", "-", "-", "-"])
    sub_rows = []
    for n in [a for a in arms if a.startswith(("query_", "null_"))]:
        for l in (8, 10):
            if f"subspace/{n}/l{l}/removed_within" in agg:
                sub_rows.append([n, l, f"{agg[f'subspace/{n}/l{l}/removed_within']['mean']:.4f}",
                                 f"{agg[f'subspace/{n}/l{l}/removed_total']['mean']:.4f}",
                                 f"{agg[f'sensitivity/{n}/l{l}/rel_change']['mean']:.4f}",
                                 f"{agg[f'sensitivity/{n}/l{l}/cos']['mean']:.4f}",
                                 worst(f"{n}/q_cos_to_clean/read{0 if l == 8 else 1}", False)])

    record = {"experiment": "E-000049",
              "title": "is the held-out addressing gap a linear-erasure problem? training-free erasure of the "
                       "trained-template subspace from the routing query, against the prefix ceiling and a "
                       "transport-side control",
              "evidence_level": "E5", "trains_nothing": True,
              "no_training": "E-000017-B's three recorded checkpoints, frozen GPT-2 small, adapter unchanged; the "
                             "eraser is a projection fitted on 128 facts disjoint from the evaluation targets.",
              "decision_rule": DECISION_RULE, "verdict": verdict, "criteria": check["criteria"],
              "claim_supported": verdict["reading"] == "POSITIVE",
              "claim": "the held-out addressing gap is an additive rank-7 template component of the query input "
                       "that a linear erasure removes without training",
              "args": vars(args), "per_seed": per_seed, "aggregate": agg}
    md = [f"# E-000049 -- {record['title']}", "", record["no_training"], "",
          f"**Reading: {verdict['reading']}** -- {verdict['why']}. Pod line (12 phrasings address one correct cell "
          f"at >= 0.90 under the primary arm): {'PASS' if verdict['pod_line'] else 'FAIL'}. "
          f"Rank-monotonicity flag: {verdict['rank_monotone_flag']}. Rank-7 arm uninformative a priori "
          f"(query sensitivity < 0.05): {verdict['rank7_uninformative_a_priori']}.", "",
          "## Arms, worst seed (lower-is-better columns show the worst = max)", "",
          ledger.table(["arm", "t8 route_hit", "t11 route_hit", "t8 read", "t11 read", "F_route_min", "F_read_min",
                        "address collision", "trained read (mean t0-t7)", "medial held-out route_min",
                        "12-way agreement, correct cell"], rows), "",
          "F is the recovered fraction of the clean arm's re-measured prefix ceiling on the two held-out "
          "subject-initial forms (t8, t11), worst of the two. Transport arms erase the same subspace from the residual "
          "AFTER the query is taken, so their F_route is diagnostic (layer 8's erasure reaches q_10 through blocks 9-10) "
          "and their F_read is the control.", "",
          "## The subspaces, mean over seeds", "",
          ledger.table(["arm", "layer", "share of within-fact variance removed", "share of total variance removed",
                        "query rel. change on fit states", "query cos on fit states", "q cos to clean, t8/t11 (worst seed)"],
                       sub_rows), "",
          "## The rule, fixed before the run", "", DECISION_RULE, "",
          "## Pre-registered criteria", "", ledger.criteria_table(check), ""]
    name = "e000049_template_nullspace_addressing" + (f"-{args.tag}" if args.tag else "")
    path = ledger.save(name, record, "\n".join(md))
    print("\n".join(md[1:]))
    print(f"\nwritten: {path}")
    return record


if __name__ == "__main__":
    main()
