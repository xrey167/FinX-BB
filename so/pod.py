"""The pod objective: one carrier per fact, private to that fact. "Pod in, private out."

WHAT E-000043 MEASURED AND WHY THIS EXISTS. On frozen GPT-2 the deletion subspaces of seventeen facts
overlap 0.4118 more than a matched null while 92% of the dimension budget sits unused. The clean-
deletion capacity bound ``n <= d/s`` is nowhere near binding, so the failure is not capacity. It is
ALLOCATION -- the model had room to give each fact a private subspace and did not -- and allocation is
a training objective rather than a law of dimension. This module is that objective.

It also measured WHERE the sharing is, which decides what the objective has to do:

    content direction only    overlap 0.2232   null 0.0638   excess +0.1594
    addressing rows only      overlap 0.5954   null 0.1930   excess +0.4024

A fact's own content direction is nearly private; what is shared is the machinery that says WHICH
PHRASING asked for it. So the objective has two halves and they pull in opposite directions:

    POD      the access paths of ONE fact should run through ONE carrier, so the phrasing spread
             collapses onto the shared direction and the deletion closure is 1 rather than the number
             of ways the fact can be asked. This is the symlink: many keys, one object.
    PRIVATE  the carriers of DIFFERENT facts should be mutually near-orthogonal, so that removing one
             leaves the others standing. This is what makes the collateral zero.

WHY THE PRIVACY TERM IS HINGED. No set of n unit vectors in d dimensions can have mutual coherence
below the Welch bound ``sqrt((n - d) / (d(n - 1)))``, so a penalty that keeps pushing past it is
fighting the answer loss over a margin that provably does not exist. The hinge is not a tuning
convenience; it is the point at which the objective is asking for something impossible.

WHY THE LOSSES ARE ON HIDDEN STATES AND NOT ON A WEIGHT TENSOR. An earlier attempt shaped
``so/carrier.py``'s injection-space carrier, and a parallel measurement found that tensor holds 0.0091
of the state's energy -- sixteen directions removed, zero of forty facts silenced. The readout-space
direction holds 0.5427. Shaping a tensor the readout barely sees changes the geometry of nothing that
matters, so these losses are computed on ``extras["hidden"]``, which is the state the readout actually
consumes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so.carrier import welch_bound
from so.world import Query, World

__all__ = ["pod_queries", "fact_directions", "pod_loss", "private_loss", "pod_private"]


def pod_queries(world: World, facts: Sequence[Tuple[int, int]], rng: np.random.Generator,
                n_surface: Optional[int] = None) -> Tuple[List[Query], torch.Tensor]:
    """Every surface form of each fact, and the fact each row belongs to.

    A pod loss needs several ACCESS PATHS of the same fact in one batch, and ordinary training batches
    almost never contain two -- with a few hundred facts and a batch of 128 the collision rate is
    negligible. So the pod batch is built deliberately: one row per (fact, surface form).
    """
    n_syn = world.n_synonyms if n_surface is None else int(n_surface)
    qs: List[Query] = []
    ids: List[int] = []
    for f, (subj, rel) in enumerate(facts):
        for k in range(n_syn):
            qs.append(Query("fwd", int(subj), (int(rel),), (int(world.surface_of(int(rel), k)),)))
            ids.append(f)
    return qs, torch.as_tensor(ids, dtype=torch.long)


def fact_directions(hidden: torch.Tensor, fact_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(centred states, unit direction per fact). The states are centred over the batch first.

    Centring matters for the same reason it mattered in E-000043: a residual is dominated by what
    every state at this point has in common, and an uncentred cosine between two facts measures that
    common mode rather than the facts.
    """
    h = hidden - hidden.mean(0, keepdim=True)
    n = int(fact_ids.max()) + 1
    sums = torch.zeros(n, h.shape[1], dtype=h.dtype, device=h.device)
    sums.index_add_(0, fact_ids, h)
    counts = torch.zeros(n, dtype=h.dtype, device=h.device)
    counts.index_add_(0, fact_ids, torch.ones_like(fact_ids, dtype=h.dtype))
    means = sums / counts.clamp(min=1).unsqueeze(1)
    return h, means / means.norm(dim=1, keepdim=True).clamp(min=1e-8)


def pod_loss(h_centred: torch.Tensor, dirs: torch.Tensor, fact_ids: torch.Tensor) -> torch.Tensor:
    """Pull every access path of a fact onto that fact's shared direction.

    ``1 - cos`` rather than a distance, so the term cares about the ANGLE and not the magnitude: a
    phrasing that reads the fact strongly and one that reads it weakly should still run through the
    same carrier, and penalising the length difference would be asking for something else.
    """
    cos = (h_centred / h_centred.norm(dim=1, keepdim=True).clamp(min=1e-8) * dirs[fact_ids]).sum(1)
    return (1.0 - cos).mean()


def private_loss(dirs: torch.Tensor, centred: bool = True) -> Tuple[torch.Tensor, float]:
    """Mean squared off-diagonal cosine between fact directions, hinged at the achievable floor.

    TWO FLOORS, and the second was found by a test rather than by reasoning. The Welch bound
    ``sqrt((n - d) / (d(n - 1)))`` is the usual one: no n unit vectors in d dimensions do better. But
    these directions are CENTRED -- each is a fact's mean minus the batch mean -- and n centred vectors
    sum to zero, which forces a mean pairwise cosine of exactly ``-1/(n - 1)``. At n = 2 that is -1:
    two centred directions are always antipodal and no objective can change it. So the hinge is the
    larger of the two floors, and a privacy term below it would be fighting arithmetic rather than the
    model. This is the same artefact that demoted sigma_min in ``so/capacity.py``, arriving from the
    other side.

    ``centred`` says whether the caller's directions carry that second floor. Everything in this module
    passes centred directions, so it defaults to true; a caller with raw directions should say so
    rather than have a floor applied that does not bind them.
    """
    n, d = dirs.shape
    if n < 2:
        return dirs.new_zeros(()), 0.0
    bound = max(welch_bound(n, d), 1.0 / (n - 1) if centred else 0.0)
    g = dirs @ dirs.t()
    off = g[~torch.eye(n, dtype=torch.bool, device=g.device)]
    return (off.abs() - bound).clamp(min=0.0).pow(2).mean(), float(bound)


def pod_private(hidden: torch.Tensor, fact_ids: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Both halves, and the statistics E-000043 reports, from one set of states."""
    h, dirs = fact_directions(hidden, fact_ids)
    lp = pod_loss(h, dirs, fact_ids)
    lpr, bound = private_loss(dirs)
    with torch.no_grad():
        n = dirs.shape[0]
        g = (dirs @ dirs.t()).abs()
        off = g[~torch.eye(n, dtype=torch.bool, device=g.device)] if n > 1 else g.new_zeros(1)
        cos = (h / h.norm(dim=1, keepdim=True).clamp(min=1e-8) * dirs[fact_ids]).sum(1)
        stats = {"pod/spread": float(1.0 - cos.mean()), "pod/coherence_mean": float(off.mean()),
                 "pod/coherence_max": float(off.max()), "pod/welch_bound": bound,
                 "pod/excess_coherence": float(off.mean()) - bound}
    return lp, lpr, stats
