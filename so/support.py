"""Sparse supports as must-hit sets: the derivation trace a model was said not to emit.

``so.closure`` gets a CERTIFIED LOWER BOUND on a store's deletion closure from one argument. Every
live derivation the mechanical resolver produces is a **must-hit set** -- the resolver names the
records it used, so any solution must remove one of them -- and a family of pairwise-disjoint
derivations therefore bounds the optimum from below by the size of that family. Greedy meeting that
bound proves optimality, and E-000032 reports ``proved optimal`` at 1.00 in every arm.

``so.workspace`` implements the same closure over DIRECTIONS in a representation, and its docstring
says the bound does not transfer:

    "A model hands out no such trace. There is no set of directions the computation 'used' that a
     solution must intersect, so the disjointness argument does not transfer, and computing it anyway
     would be a bound that certifies nothing."

**This module is the claim that a J-lens decomposition IS that trace.** Anthropic's workspace work
decomposes a residual into a sparse nonnegative combination of lens directions, of order ten to
twenty-five atoms. If the state a query computes on is ``h = sum_i a_i v_i`` over a support ``S`` with
``a_i >= 0``, then a set of directions whose removal stops that query yielding the object has to touch
``S`` -- and ``S`` is a must-hit set in exactly the store's sense.

WHAT IS ASSUMED AND WHAT IS TESTED, because the difference is the whole point. The argument above is
an argument, and this programme has closed five instruments that certified by not testing. So the
must-hit property is not assumed here: ``certify_must_hit`` ENUMERATES the subsets of the complement
and checks that none of them silences the query. Sparsity is what makes that affordable -- the
complement of a ten-atom support inside a twelve-atom pool has four elements and sixteen subsets,
where the complement inside the full residual dimension has 2**758 -- and affordability is the only
thing the J-lens buys the argument. It buys a great deal.

WHY THE BOUND IS NOT VACUOUS EVEN THOUGH EVERY SUPPORT CONTAINS THE OBJECT. A first reading says the
object's own lens direction is present in every phrasing's support, so no two supports are ever
disjoint and the bound is always one. That reading is right about the mechanism and wrong about the
conclusion: a fact whose phrasings all run through ONE shared direction has a deletion closure of one,
and a bound of one is the correct answer. The bound is large exactly when the phrasings run through
different directions -- when the representation stores the fact as copies rather than as a pod. So the
disjoint family is not a technicality; it is **the number of independent copies of the fact in the
representation**, which is the store's denormalisation degree read off a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = ["Support", "nonneg_pursuit", "MustHitCertificate", "certify_must_hit",
           "DisjointBound", "disjoint_lower_bound", "CertifiedClosure", "certified_closure"]


# ------------------------------------------------------------------------- the decomposition
@dataclass
class Support:
    """A state written as a nonnegative combination of dictionary atoms, and what is left over.

    ``residual_fraction`` is the part of the state the support does not explain. It is reported rather
    than hidden because the must-hit argument runs through it: an ablation outside the support can
    still change the readout by way of the residual, and a support that explains a third of the state
    is not a trace of anything. A caller that wants a certificate should read this number first.
    """

    directions: Tuple[int, ...] = ()
    coefficients: Tuple[float, ...] = ()
    residual_fraction: float = 1.0
    nonnegative: bool = True

    @property
    def size(self) -> int:
        return len(self.directions)

    def summary(self) -> str:
        return (f"{self.size} atom(s) {list(self.directions)}, "
                f"{100.0 * (1.0 - self.residual_fraction):.1f}% of the state explained"
                + ("" if self.nonnegative else ", SIGNS NOT CONSTRAINED"))


def _nnls(atoms: torch.Tensor, target: torch.Tensor, steps: int = 400) -> torch.Tensor:
    """``min_x ||atoms^T x - target||`` subject to ``x >= 0``, by projected gradient.

    Projected gradient with step ``1/L`` for ``L`` the largest eigenvalue of the Gram matrix, which is
    the standard guarantee. Nonnegativity is a constraint here rather than an observation, because the
    must-hit argument uses it: with signs free, two atoms can cancel and an ablation that removes one
    of them can RAISE the object's logit, which breaks the monotonicity the argument needs.
    """
    gram = atoms @ atoms.t()
    rhs = atoms @ target
    lip = torch.linalg.eigvalsh(gram).max().clamp(min=1e-6)
    x = torch.zeros(atoms.shape[0], dtype=atoms.dtype)
    for _ in range(steps):
        x = (x - (gram @ x - rhs) / lip).clamp(min=0.0)
    return x


def nonneg_pursuit(h: torch.Tensor, atoms: torch.Tensor, ids: Optional[Sequence[int]] = None,
                   n_atoms: int = 12, tol: float = 0.05, nonnegative: bool = True) -> Support:
    """Greedy nonnegative matching pursuit: the affordable version of the paper's gradient pursuit.

    ``atoms`` is ``(n, d)`` and need not be orthogonal -- a lens dictionary is overcomplete and
    correlated, which is why the coefficients are re-fitted over the whole chosen support at every
    step instead of being read off one inner product. ``ids`` names the atoms in the caller's own
    numbering so the support can be handed to an ablation that speaks in those terms.

    Stops at ``n_atoms`` or when the unexplained fraction falls below ``tol``. Both are recorded on
    the result: a support that stopped because it ran out of budget is a different object from one
    that stopped because it had explained the state, and only the second is a trace.
    """
    vec = h.reshape(-1).to(torch.float32)
    dic = atoms.to(torch.float32)
    dic = dic / dic.norm(dim=1, keepdim=True).clamp(min=1e-8)
    names = list(range(dic.shape[0])) if ids is None else [int(i) for i in ids]
    norm0 = vec.norm().clamp(min=1e-8)

    chosen: List[int] = []
    coef = torch.zeros(0)
    residual = vec.clone()
    for _ in range(min(n_atoms, dic.shape[0])):
        score = dic @ residual
        if chosen:
            score[torch.as_tensor(chosen, dtype=torch.long)] = float("-inf")
        if nonnegative:
            score = torch.where(score > 0, score, torch.full_like(score, float("-inf")))
        best = int(score.argmax())
        if score[best] == float("-inf"):
            break
        chosen.append(best)
        sub = dic[torch.as_tensor(chosen, dtype=torch.long)]
        coef = _nnls(sub, vec) if nonnegative else torch.linalg.lstsq(sub.t(), vec).solution
        residual = vec - sub.t() @ coef
        if float(residual.norm() / norm0) <= tol:
            break

    keep = [(names[c], float(a)) for c, a in zip(chosen, coef.tolist()) if a > 1e-8] if chosen else []
    return Support(tuple(k for k, _ in keep), tuple(a for _, a in keep),
                   float(residual.norm() / norm0), nonnegative)


# ------------------------------------------------------------------------- the certificate
@dataclass
class MustHitCertificate:
    """Whether a support really is a set every silencing ablation has to touch.

    ``holds`` is a measurement, not a property of the definition. ``counterexample`` is the subset of
    the complement that silenced the query when it should not have -- when one exists, the support is
    NOT a must-hit set for this query and any bound built on it is void.
    """

    support: Tuple[int, ...] = ()
    complement: Tuple[int, ...] = ()
    subsets_tested: int = 0
    max_size_exhausted: int = 0
    exhaustive: bool = False
    holds: bool = False
    counterexample: Optional[Tuple[int, ...]] = None
    vacuous: bool = False

    def summary(self) -> str:
        if self.vacuous:
            return ("VACUOUS: the support fills the pool, so there is no disjoint ablation to try and "
                    "the must-hit property holds by having nothing to test against")
        how = ("exhaustive over the complement" if self.exhaustive
               else f"exhaustive to size {self.max_size_exhausted} of {len(self.complement)}")
        if not self.holds:
            return (f"NOT a must-hit set: removing {list(self.counterexample or ())}, which is "
                    f"disjoint from the support, silenced the query ({self.subsets_tested} tested)")
        return (f"must-hit over this pool: {self.subsets_tested} disjoint ablation(s) tested, "
                f"{how}, none silenced the query")


def certify_must_hit(silences: Callable[[Sequence[int]], bool], support: Sequence[int],
                     pool: Sequence[int], budget: int = 4096) -> MustHitCertificate:
    """Test the must-hit property by removing everything the support does not contain.

    ``silences(dirs)`` removes those directions and returns whether the query stopped yielding the
    object. Subsets of the complement are enumerated smallest first, so a budget that runs out leaves
    a certificate that is exhaustive up to a stated size rather than one of unknown coverage.

    THE SCOPE, stated because the number invites over-reading: this certifies the must-hit property
    against ablations drawn from ``pool``. It says nothing about a direction outside the pool, exactly
    as ``certify_store_absence`` says nothing about a payload outside the domain it sweeps. What makes
    the scope worth having is that the closure search draws from the same pool, so the bound and the
    upper bound it is compared against are statements about the same universe.

    Raises when the query does not answer with nothing removed at all -- every subset would then
    "silence" it and the certificate would pass on a query carrying no fact.

    A support that fills the pool leaves NO disjoint ablation to try. The must-hit property is then
    true by vacuity -- any non-empty subset of the pool must intersect a support that is the pool --
    and the result is flagged ``vacuous`` so that a bound cannot be built on it. That is not a corner
    case: a pursuit run to a tight reconstruction tolerance on a small pool takes every atom, and a
    certificate that passed because it had nothing to test would be the sixth instrument in this
    programme that cannot fail.
    """
    if silences(()):
        raise ValueError(
            "certify_must_hit was asked about a query the model does not answer with nothing removed. "
            "Every subset would then count as silencing it and the certificate would pass on a query "
            "that carries no fact -- an instrument that cannot fail. Establish the answer first.")

    have = set(int(s) for s in support)
    comp = tuple(int(d) for d in pool if int(d) not in have)
    tested, largest, counter = 0, 0, None
    for size in range(1, len(comp) + 1):
        if tested + _n_choose_k(len(comp), size) > budget:
            break
        for sub in combinations(comp, size):
            tested += 1
            if silences(sub):
                counter = sub
                break
        if counter is not None:
            break
        largest = size
    return MustHitCertificate(tuple(sorted(have)), comp, tested, largest,
                              largest == len(comp) and counter is None, counter is None, counter,
                              vacuous=len(comp) == 0)


def _n_choose_k(n: int, k: int) -> int:
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out


# ------------------------------------------------------------------------- the bound
@dataclass
class DisjointBound:
    """The store's argument, licensed by tested must-hit sets instead of assumed ones.

    ``lower_bound`` is the size of a pairwise-disjoint family of certified must-hit supports. ANY such
    family is a sound bound -- a solution must hit each of them and cannot hit two with one direction
    -- so the greedy family below is valid even where it is not the largest, and the bound is
    therefore conservative in the direction that matters.

    ``certified`` is false the moment one support in the family failed its must-hit test, was tested
    only partially, or was VACUOUS -- a support filling the pool passes by having nothing to test
    against, and a bound resting on one of those is a bound resting on nothing. A bound reported with
    ``certified`` false is a conjecture and is labelled one.
    """

    family: Tuple[Tuple[int, ...], ...] = ()
    lower_bound: int = 0
    certified: bool = False
    n_supports: int = 0
    shared_atoms: Tuple[int, ...] = ()

    def summary(self) -> str:
        head = (f"lower bound {self.lower_bound} from {self.lower_bound} pairwise-disjoint support(s) "
                f"out of {self.n_supports}")
        if self.shared_atoms:
            head += f"; every access path runs through {list(self.shared_atoms)}"
        return head + ("" if self.certified else " -- NOT CERTIFIED, at least one support untested")


def disjoint_lower_bound(certs: Sequence[MustHitCertificate]) -> DisjointBound:
    """Greedily pick a pairwise-disjoint family, smallest supports first.

    Smallest first because a small support is the hardest to avoid and leaves the most room for the
    next one; the exact maximum independent set is NP-hard and unnecessary, since any disjoint family
    is already a sound bound.
    """
    if not certs:
        return DisjointBound()
    order = sorted(range(len(certs)), key=lambda i: len(certs[i].support))
    used: set = set()
    family: List[Tuple[int, ...]] = []
    ok = True
    for i in order:
        s = set(certs[i].support)
        if s and not (s & used):
            used |= s
            family.append(certs[i].support)
            ok = ok and certs[i].holds and certs[i].exhaustive and not certs[i].vacuous
    shared = set(certs[0].support)
    for c in certs[1:]:
        shared &= set(c.support)
    return DisjointBound(tuple(family), len(family), ok and bool(family), len(certs),
                         tuple(sorted(shared)))


@dataclass
class CertifiedClosure:
    """An interval for a representation's deletion closure, with a proof on the low side."""

    obj_id: int = -1
    upper: int = 0
    lower: int = 0
    certified: bool = False
    optimal: bool = False
    n_queries: int = 0
    workload: str = ""

    def summary(self) -> str:
        how = ("EXACT, greedy meets a certified bound" if self.optimal
               else ("certified interval" if self.certified else "interval, lower side not certified"))
        return (f"object {self.obj_id}: closure in [{self.lower}, {self.upper}] over {self.n_queries} "
                f"quer(ies) [{self.workload}] -- {how}")


def certified_closure(obj_id: int, upper: int, bound: DisjointBound, n_queries: int,
                      workload: str = "") -> CertifiedClosure:
    """Compose a greedy upper bound with a certified lower bound into one interval.

    ``optimal`` requires BOTH that the two meet and that the lower side is certified. Greedy meeting an
    uncertified bound proves nothing, and reporting it as exact would be the sixth instrument in this
    programme that cannot fail.
    """
    return CertifiedClosure(int(obj_id), int(upper), int(bound.lower_bound), bound.certified,
                            bound.certified and upper == bound.lower_bound, int(n_queries), workload)
