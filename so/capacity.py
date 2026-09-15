"""Clean-deletion capacity: why a dimension budget bounds it, and whether a model is using the budget.

THE ARGUMENT THIS MODULE MEASURES. A clean deletion of fact i is an orthogonal projection removing a
subspace ``A_i`` such that no access path of fact i still yields the object -- UNREACHABLE -- and every
access path of every other fact still does -- ZERO COLLATERAL. Take ``A_i`` minimal, so every direction
in it is load-bearing for at least one of fact i's paths. Let ``V_j`` be the subspace fact j's readout
depends on, so ``A_j`` is inside ``V_j``. Zero collateral for the deletion of i requires the projection
to fix ``V_j``, hence ``A_i`` orthogonal to ``A_j`` for every j != i. Mutually orthogonal subspaces
satisfy ``sum_i dim A_i <= d``, so with each of dimension at least ``s``

    n <= d / s          -- CLEAN-DELETION CAPACITY IS LINEAR IN THE DIMENSION.

Representation capacity is not. Johnson-Lindenstrauss gives exponentially many almost-orthogonal
directions in d dimensions, and superposition is the observation that models use them. So superposition
buys representation capacity and does not buy deletion capacity, and the gap between exponential and
linear is a cost of superposition that is paid at deletion time.

WHY THE MODULE MEASURES RATHER THAN ASSERTS. The bound says what is POSSIBLE. It says nothing about
what a particular model did, and the difference is the whole question. Two numbers separate them:

    pressure      = sum_i dim A_i / d   -- how much of the budget the deletion subspaces demand
    mean_overlap  = mean pairwise principal cosine, WHICH MUST BE READ AGAINST A MATCHED NULL
    orthogonality = sigma_min of the stacked orthonormal bases -- reported, but see the warning below

TWO WARNINGS, both found by running the instrument rather than by reading it.

FIRST, ``sigma_min`` IS FRAGILE AND IS NOT THE PRIMARY. It is a minimum over the whole collection, so
one linear dependency zeroes it however well the rest is allocated -- and a dependency is easy to
manufacture by accident. A basis built as "this fact's state minus the mean over the others" is a
CENTRED vector, and n centred vectors span at most n-1 dimensions, so their sigma_min is exactly 0
whatever the angles are. Measured: 0.0000 for seventeen real facts AND 0.0000 for seventeen random
ones. A number that cannot distinguish the data from noise is not a summary. Use ``mean_overlap`` and
``max_overlap``, and read ``orthogonality`` only as a dependency check.

SECOND, OVERLAP HAS A NON-ZERO NULL AND MUST BE REPORTED AGAINST IT. Putting random states through
the identical basis construction gives a mean pairwise overlap of 0.2048 on seventeen subspaces in
768 dimensions, not 0. Any overlap quoted against a baseline of zero overstates the effect by that
much; the quantity that means something is the EXCESS over a matched null built by the same code path
on states with no shared structure.

A NOTE ON WHY THE SECOND IS NOT A RANK. The first version of this module used
``rank(union) / sum_i dim A_i``, which measures LINEAR INDEPENDENCE, and the theorem needs
ORTHOGONALITY. The two are not the same and the difference is not academic: measured on GPT-2 (
E-000043) twelve deletion subspaces totalling 58 directions had rank exactly 58 -- a direct sum,
"efficiency" 1.0000 -- while their pairwise principal cosines were 0.5566 on average and 0.8559 at
worst. Independent and nowhere near orthogonal. ``sigma_min`` of the stacked bases is 1.0 only for a
mutually orthonormal collection and degrades smoothly, which is the quantity the argument runs on.

A model at LOW pressure and LOW efficiency had room for private subspaces and did not take it: the
failure is allocation, and allocation is a training objective. A model at pressure near 1 is against
the bound: the failure is capacity, and no objective fixes it without more dimensions. Reporting the
pair is the point; either number alone is quotable in the wrong direction.

An addressable store has no such bound, because its addresses are records rather than directions in a
fixed-dimensional space, and ``pressure`` for it is undefined rather than small.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = ["capacity_bound", "orthonormalise", "subspace_overlap", "Allocation", "allocation"]


def capacity_bound(d: int, s: float) -> float:
    """``d / s``: the most facts that can have mutually orthogonal deletion subspaces of size ``s``."""
    return float(d) / max(float(s), 1e-9)


def orthonormalise(a: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis of the row space of ``a`` (k, d), returned as (d, r).

    Rank-deficient input is handled by dropping the directions QR could not separate, so a caller that
    passes a nearly collinear pair gets a rank-1 subspace and not a rank-2 one with a tiny singular
    value pretending to be a dimension.
    """
    if a.numel() == 0:
        return torch.zeros(a.shape[-1], 0, dtype=torch.float32)
    u, sv, _ = torch.linalg.svd(a.to(torch.float32), full_matrices=False)
    keep = int((sv > sv.max().clamp(min=1e-12) * 1e-6).sum())
    q, _ = torch.linalg.qr(a.to(torch.float32)[:keep].t())
    return q[:, :keep]


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest principal cosine between two subspaces: 0 is orthogonal, 1 shares a direction.

    This is what the theorem's orthogonality requirement looks like as a measurement. It is the
    quantity that decides whether deleting one fact must damage another, and it is reported instead of
    a binary because real subspaces are never exactly orthogonal and a threshold hides the margin.
    """
    qa, qb = orthonormalise(a), orthonormalise(b)
    if qa.shape[1] == 0 or qb.shape[1] == 0:
        return 0.0
    return float(torch.linalg.svdvals(qa.t() @ qb).max().clamp(max=1.0))


@dataclass
class Allocation:
    """How much of the dimension budget the deletion subspaces demand, and how well they use it."""

    d: int = 0
    n_facts: int = 0
    dims: Tuple[int, ...] = ()
    union_rank: int = 0
    pressure: float = 0.0
    orthogonality: float = 1.0     # sigma_min of the stacked bases: the quantity the theorem needs
    rank_efficiency: float = 1.0   # rank / sum of dims: independence only, reported for contrast
    max_overlap: float = 0.0
    mean_overlap: float = 0.0
    bound: float = 0.0
    null_overlap: Optional[float] = None    # the same construction on states with no shared structure

    @property
    def excess(self) -> Optional[float]:
        """Overlap above a matched null. This is the effect; the raw overlap is not."""
        return None if self.null_overlap is None else self.mean_overlap - self.null_overlap

    @property
    def headroom(self) -> float:
        """Fraction of the dimension budget left unused. Large headroom plus low efficiency is the
        allocation failure; small headroom is the capacity limit."""
        return 1.0 - self.pressure

    def verdict(self) -> str:
        if not self.dims:
            return "no deletion subspaces given"
        if self.pressure > 0.8:
            why = ("AGAINST THE BOUND: the deletion subspaces demand most of the dimension budget, so "
                   "overlap is forced and no training objective removes it without more dimensions")
        elif self.excess is not None and self.excess <= 0.10:
            why = ("ALLOCATED: the subspaces are no more overlapping than a matched null and there is "
                   "budget to spare, so clean deletion is available here")
        elif self.excess is not None:
            why = (f"ALLOCATION, NOT CAPACITY: the subspaces overlap {self.excess:.4f} more than a "
                   f"matched null while {100 * self.headroom:.0f}% of the budget is unused -- the "
                   "model had room to give each fact a private subspace and did not, which is a "
                   "training objective and not a law of dimension")
        else:
            why = ("NO NULL SUPPLIED: overlap has a non-zero null under this construction, so the "
                   "raw number below cannot be read as an effect")
        null = "" if self.null_overlap is None else f" against a matched null of {self.null_overlap:.4f}"
        return (f"{self.n_facts} fact(s), {sum(self.dims)} direction(s) in d={self.d}: pressure "
                f"{self.pressure:.4f} against a bound of {self.bound:.0f} facts, mean pairwise overlap "
                f"{self.mean_overlap:.4f}{null} (max {self.max_overlap:.4f}) -- {why}")


def allocation(subspaces: Sequence[torch.Tensor], d: Optional[int] = None,
               null_overlap: Optional[float] = None) -> Allocation:
    """Read pressure and efficiency off a set of per-fact deletion subspaces.

    ``subspaces[i]`` is (k_i, d): the directions whose removal deletes fact i. Efficiency is the rank
    of their union over the sum of their dimensions, which is 1.0 exactly when the theorem's
    orthogonality requirement is met and falls as they share directions.
    """
    mats = [s for s in subspaces if s is not None and s.numel()]
    if not mats:
        return Allocation(d=int(d or 0))
    dim = int(d if d is not None else mats[0].shape[-1])
    qs = [orthonormalise(m) for m in mats]
    dims = tuple(int(q.shape[1]) for q in qs)
    total = sum(dims)
    union = torch.cat([q for q in qs if q.shape[1]], dim=1)
    rank = int(torch.linalg.matrix_rank(union, tol=1e-5)) if union.numel() else 0
    # sigma_min of the stacked orthonormal bases: 1.0 iff the collection is mutually orthonormal.
    # Rank alone cannot see the difference between a direct sum and an orthogonal one, and it is the
    # orthogonal one the argument needs.
    smin = float(torch.linalg.svdvals(union).min()) if union.numel() and union.shape[1] > 1 else 1.0
    pairs = [subspace_overlap(mats[i], mats[j]) for i in range(len(mats)) for j in range(i + 1, len(mats))]
    s_mean = total / max(len(mats), 1)
    return Allocation(d=dim, n_facts=len(mats), dims=dims, union_rank=rank,
                      pressure=total / max(dim, 1), orthogonality=smin,
                      rank_efficiency=rank / max(total, 1),
                      max_overlap=max(pairs) if pairs else 0.0,
                      mean_overlap=float(sum(pairs) / len(pairs)) if pairs else 0.0,
                      bound=capacity_bound(dim, s_mean), null_overlap=null_overlap)
