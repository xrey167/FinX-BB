"""Deletion closure: how many records must go before a fact is gone.

Today's certificates prove a model is independent of a deleted RECORD. That is not the same as the
FACT being unavailable, and the two come apart whenever the fact can be reached another way. This
repository already records the extreme case: `dependency/derivable_recovery_after_revoke_K3 = 1.0` in
every seed of E-000019 -- after revoking the target cell, every derivable fact is still recovered.

So a record-level guarantee needs a companion statistic that is a property of the STORE rather than of
a checkpoint: given a fact, how many records have to be removed before no query produces it. Call it
the deletion closure. It is computed here with the mechanical reference resolver and never touches the
model, so it says what the store makes possible rather than what one trained network happens to do.

Two honest limits, stated because a number without them invites over-reading:

  * The general problem is deletion propagation -- given a view tuple, find a minimal set of base
    tuples whose removal deletes it -- and it is NP-hard in general. What is computed here is a GREEDY
    upper bound: repeatedly remove the records the current derivation actually uses, until the answer
    changes. It is exact when the derivations are disjoint, which is the canonical and the duplicated
    case alike, and an upper bound when they share records.
  * It is a statement about the store's own semantics. A frozen model that already knew the fact from
    pretraining is outside it, which is exactly what E-000013 measured separately.

Why the number matters: a store where every fact has closure size one admits deletion as a single,
certifiable, constant-cost operation. A store where the distribution has a tail requires a search
before you can even begin, and every erasure claim about it is a claim about the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .mvcc import MVCCStore, Status
from .reference import ReferenceResolver
from .world import Query, UNKNOWN


@dataclass
class Closure:
    key: Tuple[int, int]
    answer: int
    records: Tuple[int, ...] = ()
    exhausted: bool = False

    @property
    def size(self) -> int:
        return len(self.records)

    def __str__(self) -> str:
        where = "exhausted the store" if self.exhausted else f"{self.size} record(s)"
        return f"{self.key} -> {self.answer}: {where} {list(self.records)}"


def _query(key: Tuple[int, int]) -> Query:
    return Query("fwd", int(key[0]), (int(key[1]),), (0,))


def deletion_closure(store: MVCCStore, key: Tuple[int, int], max_records: int = 64,
                     restore: bool = True) -> Closure:
    """The records that have to go before ``key`` stops resolving to its current answer.

    Greedy over the derivation the resolver reports: whatever the current answer is reached through is
    what gets removed next. ``restore`` puts the store back exactly as it was, so the function is a
    measurement rather than an edit; it uses EVICT, which leaves the payload in place.
    """
    ref = ReferenceResolver(store)
    q = _query(key)
    original = ref.resolve(q)
    if original.answer == UNKNOWN:
        return Closure(key, UNKNOWN, (), False)

    removed: List[int] = []
    exhausted = False
    try:
        while True:
            res = ReferenceResolver(store).resolve(q)
            if res.answer != original.answer:
                break
            trace = [int(k) for k in res.trace if int(k) not in removed]
            if not trace:
                exhausted = True
                break
            if len(removed) >= max_records:
                exhausted = True
                break
            store.evict(trace[0])
            removed.append(trace[0])
    finally:
        if restore:
            for kid in reversed(removed):
                store.restore(kid)
    return Closure(key, original.answer, tuple(removed), exhausted)


@dataclass
class ClosureProfile:
    """The distribution of closure sizes over a set of facts, which is the store-level statement."""

    sizes: Tuple[int, ...]
    exhausted: int
    keys: Tuple[Tuple[int, int], ...] = ()

    @property
    def n(self) -> int:
        return len(self.sizes)

    @property
    def mean(self) -> float:
        return float(np.mean(self.sizes)) if self.sizes else float("nan")

    @property
    def max(self) -> int:
        return int(max(self.sizes)) if self.sizes else 0

    def fraction_at(self, size: int) -> float:
        return float(np.mean([s == size for s in self.sizes])) if self.sizes else float("nan")

    def histogram(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for s in self.sizes:
            out[s] = out.get(s, 0) + 1
        return dict(sorted(out.items()))

    def summary(self) -> str:
        return (f"{self.n} facts: mean closure {self.mean:.3f}, max {self.max}, "
                f"{self.fraction_at(1):.1%} at size 1, {self.exhausted} exhausted")


def closure_profile(store: MVCCStore, keys: Sequence[Tuple[int, int]],
                    max_records: int = 64) -> ClosureProfile:
    sizes: List[int] = []
    used: List[Tuple[int, int]] = []
    exhausted = 0
    for key in keys:
        c = deletion_closure(store, key, max_records=max_records)
        if c.answer == UNKNOWN:
            continue
        sizes.append(c.size)
        used.append(key)
        exhausted += int(c.exhausted)
    return ClosureProfile(tuple(sizes), exhausted, tuple(used))



# --------------------------------------------------------------- the fact, not the key

@dataclass
class FactClosure:
    """How many records must go before an OBJECT stops being obtainable through any of its keys.

    Writing the key-level closure first made the distinction obvious and is worth stating, because it
    is the same confusion the record-level certificate makes one level up. Ask "how many records must
    I remove before THIS KEY stops answering" and a canonical pod and a duplicated store both say one:
    each key resolves through its own record either way. That number is not the privacy question.

    The privacy question is about the association, not the lookup: how many records before the object
    is unobtainable through EVERY key that currently yields it. A canonical pod answers one, because
    the k access paths share one object. A duplicated store answers k, because each copy is a separate
    place the fact lives. That is Codd's deletion anomaly, measured rather than described, and it is
    what the symlink buys.

    The search is greedy maximum coverage over the records the live derivations use. The general
    problem -- deletion propagation -- is NP-hard, so an exact answer is not on offer in general and is
    not claimed. What IS claimed, when ``optimal`` is set, is verified rather than assumed: see
    ``lower_bound``.
    """

    obj: int
    keys: Tuple[Tuple[int, int], ...]
    records: Tuple[int, ...] = ()
    exhausted: bool = False
    lower_bound: int = 0
    optimal: bool = False

    @property
    def size(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        how = (f"exact (matches a certified lower bound of {self.lower_bound})" if self.optimal
               else f"greedy upper bound, lower bound {self.lower_bound}")
        return (f"object {self.obj} reachable through {len(self.keys)} key(s): "
                f"closure {self.size} record(s), {how}"
                + (" -- SEARCH EXHAUSTED" if self.exhausted else ""))


def _disjoint_lower_bound(derivations: Iterable[frozenset]) -> int:
    """A certified lower bound on the minimum number of records that must go.

    Every listed derivation is a set of records the resolver actually used to produce the object for
    some key. A solution that leaves all of a derivation's records in place leaves that derivation
    working, so any solution must remove at least one record from EVERY derivation -- each derivation
    is a must-hit set. Pairwise disjoint must-hit sets therefore need pairwise distinct records, so the
    size of any pairwise-disjoint subfamily is a lower bound on the optimum.

    Finding the LARGEST such subfamily is set packing and NP-hard too, so a smallest-first greedy is
    used. That only ever makes the bound weaker, never unsound: any disjoint subfamily is a valid bound.
    """
    chosen: List[frozenset] = []
    used: set = set()
    for d in sorted((d for d in derivations if d), key=lambda s: (len(s), sorted(s))):
        if not (d & used):
            chosen.append(d)
            used |= set(d)
    return len(chosen)


def fact_closure(store: MVCCStore, keys: Sequence[Tuple[int, int]], obj: Optional[int] = None,
                 max_records: int = 256, restore: bool = True) -> FactClosure:
    """Remove records until no key in ``keys`` resolves to ``obj`` any more.

    ``restore`` puts the store back exactly as it was, so this is a measurement and not an edit.
    """
    ks = [(int(a), int(b)) for a, b in keys]
    live = {k: ReferenceResolver(store).resolve(_query(k)) for k in ks}
    if obj is None:
        answers = [r.answer for r in live.values() if r.answer != UNKNOWN]
        if not answers:
            return FactClosure(UNKNOWN, tuple(ks), (), False, 0, True)
        obj = max(set(answers), key=answers.count)
    obj = int(obj)
    targets = tuple(k for k in ks if live[k].answer == obj)

    # the must-hit sets as the store stands now; the bound is computed from these, before any edit
    initial = [frozenset(int(x) for x in live[k].trace) for k in targets]
    bound = _disjoint_lower_bound(initial)

    removed: List[int] = []
    exhausted = False
    try:
        while True:
            resolver = ReferenceResolver(store)
            traces: List[List[int]] = []
            for k in targets:
                res = resolver.resolve(_query(k))
                if res.answer == obj:
                    traces.append([int(x) for x in res.trace])
            if not traces:
                break
            if len(removed) >= max_records:
                exhausted = True
                break
            counts: Dict[int, int] = {}
            for t in traces:
                for kid in set(t):
                    if kid not in removed:
                        counts[kid] = counts.get(kid, 0) + 1
            if not counts:
                # every remaining derivation runs entirely through records already removed, which
                # cannot happen with a correct resolver; stop rather than loop
                exhausted = True
                break
            best = max(counts, key=lambda kid: (counts[kid], -kid))   # deterministic tie-break
            store.evict(best)
            removed.append(best)
    finally:
        if restore:
            for kid in reversed(removed):
                store.restore(kid)

    optimal = (not exhausted) and len(removed) == bound
    return FactClosure(obj, targets, tuple(removed), exhausted, bound, optimal)


def pod_keys(store: MVCCStore, target_kid: int) -> Tuple[Tuple[int, int], ...]:
    """Every key that reaches ``target_kid``: its own, plus every alias pointing at it.

    This is the pod -- one knowledge object and the access paths that share it -- and it is the set the
    fact-level closure has to be measured over. Only direct pointers are followed, which is the shape
    the experiments build; a chain of aliases would need the transitive closure of ``target``.
    """
    out: List[Tuple[int, int]] = []
    for kid, cell in store.cells.items():
        if cell.status in (Status.DELETED, Status.EVICTED) or not cell.versions:
            continue
        v = cell.version_obj(cell.active_version)
        if kid == target_kid or getattr(v, "target", None) == target_kid:
            out.append((int(v.subject), int(v.relation)))
    return tuple(out)


def duplicate_keys(store: MVCCStore, obj: int) -> Tuple[Tuple[int, int], ...]:
    """Every key that currently resolves to ``obj``, however it gets there.

    The counterpart to ``pod_keys``: the pod is defined by the object's identity, this by its value.
    Measuring the fact closure over this set is the store-wide question -- how many records hold the
    association at all -- and it is the set an erasure claim actually has to answer for.
    """
    view = store.resolved_view(respect_markers=True)
    return tuple(sorted(k for k, (o, _) in view.items() if int(o) == int(obj)))
