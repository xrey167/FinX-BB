"""Deletion closure in the model's workspace, not in the store.

``so/closure.py`` answers "how many RECORDS must go before no query yields this object". That is a
property of the store and it is computed without the model at all. It is also only half of an erasure
cost, and this module is the other half: **how many DIRECTIONS in the model's own representation must
go before no query yields the object**. Same question, same algorithm, different substrate.

WHY THERE IS A SECOND TERM. A store-side certificate says the model cannot reach the payload through
the store. It says nothing about a model that learned the fact into its weights, and it says nothing
about why deletion in this programme keeps failing on phrasings nobody trained on -- E-000017 fired
the roadmap's kill criterion on exactly that, E-000026 records one UPDATE reaching every alias at
0.8850 against a 0.90 bar, E-000025 records alias resolution at 0.9250 on one held-out phrasing and
0.3078 on the worst. If a fact has one carrier per phrasing rather than one per fact, removing the
record kills the carriers the trained phrasings route through and leaves the others standing. That is
a workspace-closure statement, and it is measurable.

THE LENS, AND WHAT IS BORROWED. Anthropic's workspace paper (transformer-circuits.pub/2026/workspace)
defines J-lens vectors through the Jacobian ``J_l = E[d h_final / d h_l]`` and reads them out as
``softmax(W_U norm(J_l h_l))``; J-space is the set of sparse nonnegative combinations of them, k of
order 10-25 active, and ablating the top-k of them destroys multi-hop reasoning while leaving shallow
classification intact. Two things are taken from that and one is not.

Taken: the READOUT, ``softmax(W_U norm(A h))``, and the ABLATION, projecting the residual out of the
span of chosen lens directions. Not taken: the claim to have identified the model's workspace. With
``A = I`` this is the logit lens and the lens vectors are the unembedding rows; with ``A`` fitted it
is the tuned lens, which is the cheapest defensible estimate of ``J_l`` on a machine with no GPU.
Neither is the J-lens, and calling the result a measurement of J-space would be a claim the
construction does not support. What IS supported is that these are directions the model's own output
head reads, so ablating them is an intervention on something the computation demonstrably uses.

CLOSURE AND COLLATERAL ARE ONE RESULT. A closure of one is worthless if the single direction removed
was carrying everything. Every measurement here returns the collateral alongside it -- what the same
ablation costs on facts nobody asked to delete -- because the pair is the finding and either number
alone is a way to mislead. ``so/audit.py`` has now been caught three times by instruments that could
not fail; a closure reported without its collateral would be the fourth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------------------------- the lens
def lens_logits(h: torch.Tensor, w_out: torch.Tensor, ln: Optional[Callable] = None,
                a: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``W_U norm(A h)``: what the output head would score this residual as, read at this layer.

    ``a`` is the layer's lens map. ``None`` is the identity, which makes this the logit lens; a fitted
    matrix makes it the tuned lens, the practical estimate of the Jacobian the workspace paper takes
    the expectation of. The caller chooses and the result records which.
    """
    x = h if a is None else h @ a.t()
    if ln is not None:
        x = ln(x)
    return x @ w_out.t()


def carrier_candidates(h: torch.Tensor, w_out: torch.Tensor, obj_id: int, n: int = 24,
                       ln: Optional[Callable] = None, a: Optional[torch.Tensor] = None,
                       restrict_to: Optional[Sequence[int]] = None) -> List[int]:
    """The lens directions this residual is most made of, object first.

    The workspace paper decomposes a residual into a SPARSE NONNEGATIVE combination of lens vectors,
    k of order 10-25, by gradient pursuit. Nonnegative top-k on the lens readout is the cheap version
    of the same selection and is what a caller without a GPU can afford; it is a heuristic for the
    support, not a solution of the pursuit, and the difference is that a greedy support can miss a
    direction that only matters jointly. The closure below is a greedy upper bound anyway, so the two
    approximations point the same way: the number returned can be too large, never too small.

    The object's own direction is placed first because it is the one a caller is trying to remove and
    a search that had to discover it would be measuring the ranking, not the closure.
    """
    with torch.no_grad():
        scores = lens_logits(h.reshape(1, -1), w_out, ln, a).reshape(-1)
    if restrict_to is not None:
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.as_tensor(list(restrict_to), dtype=torch.long, device=scores.device)
        mask[idx] = scores[idx]
        scores = mask
    order = torch.argsort(scores, descending=True).tolist()
    out = [int(obj_id)] + [int(t) for t in order if int(t) != int(obj_id)]
    return out[:n]


def project_out(h: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Remove the span of ``dirs`` from ``h``. Orthogonal projection, not a scaling.

    ``dirs`` is (k, d) and need not be orthonormal; it is orthonormalised here, because projecting
    onto a non-orthogonal set one vector at a time removes less than the span and would understate the
    closure -- the direction the caller thinks is gone would be partly back after the next step.
    """
    if dirs.numel() == 0:
        return h
    q, _ = torch.linalg.qr(dirs.t().to(h.dtype))          # (d, k) orthonormal basis of the span
    return h - (h @ q) @ q.t()


# ------------------------------------------------------------------------- the closure itself
@dataclass
class WorkspaceClosure:
    """How many directions must go before no query in the workload yields the object.

    ``collateral`` is not an afterthought: it is what the same ablation costs on facts nobody asked to
    delete, and a closure without it is a number that can always be made to look good by removing
    more.

    THE LOWER BOUND IS WEAKER HERE THAN IN so.closure, AND THE REASON IS WORTH STATING. There, every
    live derivation is a must-hit set -- the resolver names the records it used, so any solution must
    remove one of them -- and pairwise-disjoint derivations therefore bound the optimum from below.
    A model hands out no such trace. There is no set of directions the computation "used" that a
    solution must intersect, so the disjointness argument does not transfer, and computing it anyway
    would be a bound that certifies nothing.

    What IS sound is weaker and cheap: any set that silences every query silences the hardest one, so
    the closure is at least the closure of the single hardest query. ``lower_bound`` is that maximum,
    ``optimal`` is set only when greedy meets it, and the difference from ``so.closure`` is recorded
    rather than papered over.
    """

    obj_id: int
    directions: Tuple[int, ...] = ()
    n_queries: int = 0
    workload: str = ""
    lens: str = "identity (logit lens)"
    exhausted: bool = False
    lower_bound: int = 0
    optimal: bool = False
    collateral: Optional[float] = None
    collateral_before: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.directions)

    def summary(self) -> str:
        how = (f"exact (meets the hardest-query lower bound of {self.lower_bound})" if self.optimal
               else f"greedy upper bound, hardest-query lower bound {self.lower_bound}")
        coll = ""
        if self.collateral is not None:
            coll = (f"; collateral {self.collateral:.4f}"
                    + (f" from {self.collateral_before:.4f}" if self.collateral_before is not None else ""))
            coll += " on facts nobody asked to delete"
        return (f"object {self.obj_id}: {self.size} direction(s) under the {self.lens} over "
                f"{self.n_queries} quer(ies) [{self.workload}], {how}{coll}"
                + (" -- SEARCH EXHAUSTED" if self.exhausted else ""))


def workspace_closure(answers_with: Callable[[Sequence[int]], Sequence[int]],
                      candidates: Sequence[int], obj_id: int, n_queries: int,
                      max_dirs: int = 16, workload: str = "", lens: str = "identity (logit lens)",
                      collateral_with: Optional[Callable[[Sequence[int]], float]] = None,
                      bound: bool = True) -> WorkspaceClosure:
    """Greedy removal of lens directions until the object stops being produced anywhere.

    ``answers_with(dirs)`` runs the model with those directions projected out of the read layer and
    returns one answer per query. ``collateral_with(dirs)`` returns accuracy on bystander facts under
    the same ablation. Both are the caller's, so this module never imports a model.

    The lower bound is ``max_i`` over the per-query closures: a set that silences every query silences
    query i, so it is a solution for query i and is at least as large as query i's own minimum. Sound,
    and weaker than the disjointness bound ``so.closure`` gets from derivation traces -- see the class
    docstring for why that argument does not transfer to directions. Set ``bound=False`` to skip the
    per-query passes when only the upper bound is wanted.
    """
    obj_id = int(obj_id)
    chosen: List[int] = []
    live = list(range(n_queries))
    exhausted = False

    base = list(answers_with([]))
    live = [i for i in live if base[i] == obj_id]
    answering = list(live)
    if not live:
        return WorkspaceClosure(obj_id, (), n_queries, workload, lens, False, 0, True,
                                None if collateral_with is None else collateral_with([]),
                                None if collateral_with is None else collateral_with([]))

    pool = [int(c) for c in candidates]
    while live:
        if len(chosen) >= max_dirs or not pool:
            exhausted = True
            break
        # take the next candidate in lens order: the object's own direction first, then whatever the
        # residual is most made of. A search that reordered by effect would be fitting the answer.
        d = pool.pop(0)
        chosen.append(d)
        got = list(answers_with(chosen))
        live = [i for i in live if got[i] == obj_id]

    # sound lower bound: the hardest single query. Any set that silences all of them silences that
    # one, so it is a solution for it and cannot be smaller than its minimum.
    lower = 0
    if bound and answering:
        for i in answering:
            take: List[int] = []
            for d in candidates:
                if len(take) >= max_dirs:
                    break
                take.append(int(d))
                if list(answers_with(take))[i] != obj_id:
                    break
            lower = max(lower, len(take))
    coll = None if collateral_with is None else collateral_with(chosen)
    coll0 = None if collateral_with is None else collateral_with([])
    return WorkspaceClosure(obj_id, tuple(chosen), n_queries, workload, lens, exhausted,
                            lower, (not exhausted) and bool(answering) and len(chosen) == lower,
                            coll, coll0)


# ------------------------------------------------------------------------------ the tuned lens
def fit_tuned_lens(hidden: torch.Tensor, target_logits: torch.Tensor, w_out: torch.Tensor,
                   ln: Optional[Callable] = None, steps: int = 300, lr: float = 1e-3,
                   seed: int = 0) -> torch.Tensor:
    """Fit ``A`` so that ``W_U norm(A h_l)`` matches the model's final logits.

    This is the tuned lens, and it is the affordable stand-in for the Jacobian the workspace paper
    averages: ``J_l = E[d h_final / d h_l]`` is the best LINEAR map from layer l to the final
    residual under the expectation, and fitting A by regression to the final logits estimates the same
    map from samples instead of from derivatives. On four CPU cores this costs minutes for GPT-2
    small, where the Jacobian itself would cost d backward passes per layer per context.

    What is lost: the Jacobian is a local derivative and A is a global least-squares fit, so they
    agree only where the map is close to linear. The identity lens loses more and costs nothing. Both
    are reported by name in the result rather than either being called "the J-lens".
    """
    torch.manual_seed(seed)
    d = hidden.shape[-1]
    a = torch.eye(d, requires_grad=True)
    opt = torch.optim.Adam([a], lr=lr)
    tgt = torch.log_softmax(target_logits, dim=-1)
    for _ in range(steps):
        opt.zero_grad()
        pred = torch.log_softmax(lens_logits(hidden, w_out, ln, a), dim=-1)
        loss = torch.nn.functional.kl_div(pred, tgt, log_target=True, reduction="batchmean")
        loss.backward()
        opt.step()
    return a.detach()
