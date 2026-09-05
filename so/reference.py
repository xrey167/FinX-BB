"""Mechanical reference resolver (experiment E-000001-A).

Resolves queries symbolically against the *active view* of an ``MVCCStore`` and
returns the answer together with an exact provenance trace (the cells used).
It fixes the intended semantics of addressing, composition, provenance, update,
rollback, revocation, locality and alternative paths before any neural model is
involved (architecture document section 19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .mvcc import MVCCStore
from .world import Query, UNKNOWN, World


@dataclass(frozen=True)
class Resolution:
    answer: int
    trace: Tuple[int, ...]  # kids used, in order


class ReferenceResolver:
    def __init__(self, store: MVCCStore):
        self.store = store
        self._cache_revision = -1
        self._cache_view: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def view(self) -> Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]]:
        """``key -> (object, trace of cells read)``; alias chains are followed (E-000015).

        In a store without link cells every trace has exactly one element, so this is the same
        view as before and every earlier record stays reproducible.
        """
        if self._cache_revision != self.store.revision:
            self._cache_view = self.store.resolved_view(respect_markers=True)
            self._cache_revision = self.store.revision
        return self._cache_view

    def resolve(self, q: Query, view: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None) -> Resolution:
        v = self.view() if view is None else view
        if q.mode == "fwd":
            cur = q.start
            trace: List[int] = []
            for r in q.path:
                hit = v.get((cur, r))
                if hit is None:
                    return Resolution(UNKNOWN, tuple(trace))
                cur, tr = hit
                trace.extend(int(x) for x in tr)
            return Resolution(cur, tuple(trace))
        if q.mode == "rev":
            r, o = q.path[0], q.start
            # reverse addressing goes to the cell that HOLDS the object; an alias holds a pointer,
            # so only directly resolving keys (trace length 1) take part
            hits = [(s, tr[0]) for (s, rr), (oo, tr) in v.items() if rr == r and oo == o and len(tr) == 1]
            if len(hits) != 1:
                return Resolution(UNKNOWN, tuple())
            return Resolution(hits[0][0], (hits[0][1],))
        raise ValueError(q.mode)


def load_world(store: MVCCStore, world: World, provenance: str = "world") -> Dict[Tuple[int, int], int]:
    """Write every fact of ``world`` into ``store``; return ``(subject, relation) -> kid``."""
    kids: Dict[Tuple[int, int], int] = {}
    for f in world.facts:
        kids[f.key] = store.write(f.subject, f.relation, f.obj, provenance=provenance)
    return kids
