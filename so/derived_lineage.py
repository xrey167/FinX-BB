"""Object-scoped lineage for neural-derived runtime state.

This is a correctness prototype motivated by E-000079.  It is deliberately
small and conservative: a reusable runtime object records the exact CAVI
resolve witnesses whose neural memory reads contributed to its construction.
Before the object is reused, those witnesses are compared with the independent
live authority.

Nothing in this file is claimed as standalone novelty.  Version tags,
dependency sets, cache validation and lazy invalidation are established systems
techniques.  The research question is whether carrying *knowledge-object causal
ancestry* across neural runtime state (KV/hidden/activation/router/payload) can
provide the live-LLM lifecycle property while avoiding global cache flushes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Iterable, Tuple, TypeVar

from so.cavi import CAVIAuthority, ResolveWitness

T = TypeVar("T")


@dataclass(frozen=True)
class DerivedLineage:
    """Conservative dependency certificate for a reusable neural-derived object.

    A witness is alias-qualified as well as pod-qualified.  That matters because
    an alias RELINK can invalidate state even when the old pod remains live and
    unchanged.  Multiple witnesses are supported for inference state that
    depends on more than one knowledge object.
    """

    witnesses: Tuple[ResolveWitness, ...]

    @classmethod
    def of(cls, *witnesses: ResolveWitness) -> "DerivedLineage":
        # Stable de-duplication keeps serialization/validation deterministic.
        seen = set()
        out = []
        for w in witnesses:
            key = (w.alias_id, w.alias_incarnation, w.pod_id, w.pod_incarnation)
            if key not in seen:
                seen.add(key)
                out.append(w)
        return cls(tuple(out))

    @classmethod
    def union(cls, lineages: Iterable["DerivedLineage"]) -> "DerivedLineage":
        ws = []
        for lineage in lineages:
            ws.extend(lineage.witnesses)
        return cls.of(*ws)

    def is_current(self, authority: CAVIAuthority) -> bool:
        # validate_witness uses the authority RLock; grouping all checks under the
        # same RLock makes the multi-dependency decision one authority snapshot.
        with authority.lock:
            return all(authority.validate_witness(w) for w in self.witnesses)

    def stale_witnesses(self, authority: CAVIAuthority) -> Tuple[ResolveWitness, ...]:
        with authority.lock:
            return tuple(w for w in self.witnesses if not authority.validate_witness(w))

    @property
    def dependency_count(self) -> int:
        return len(self.witnesses)

    @property
    def packed_metadata_bytes(self) -> int:
        # Four unsigned 64-bit fields per ResolveWitness.  This is the compact
        # logical metadata size, not Python object overhead.
        return 32 * len(self.witnesses)


@dataclass
class LineagedState(Generic[T]):
    """Reusable neural-derived payload plus its authority dependency lineage."""

    payload: T
    lineage: DerivedLineage

    def reusable(self, authority: CAVIAuthority) -> bool:
        return self.lineage.is_current(authority)
