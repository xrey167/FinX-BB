"""The mutable knowledge layer: versioned knowledge cells with a lifecycle (Neural-MVCC).

Every knowledge unit is a *cell* with a stable id ``kid``.  A cell holds a list of
versions; exactly one version is *active* unless the cell is revoked or deleted.
Operations mirror the architecture document (section 16) and the ledger (section 18):

    WRITE  -> new cell, version 1, ACTIVE
    UPDATE -> new version appended, becomes active (old versions stay for rollback)
    REVOKE -> status REVOKED (the payload remains, routing is removed)
    RESTORE-> status ACTIVE again
    ROLLBACK -> active version pointer moved to an older version
    DELETE -> hard removal of the cell from the layer (component removal, F2)
    SHRED  -> the marker of the active version is destroyed; the payload remains
              in place but is no longer signed (crypto-shredding analogy, section 12)
    SWAP / REPLACE -> causal interventions (ledger section 25)

Every operation is appended to an operation log, and ``replay`` rebuilds a store
from a log so that replay determinism can be measured.

Markers: each version carries a marker vector.  Valid markers are drawn around a
per-store secret centre (the "key κ"); ``shred`` replaces the marker with noise
that is far from the centre.  ``marker_valid`` is the mechanical check; the neural
model must *learn* an equivalent check from data (see ``so.model``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Status(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    DELETED = "DELETED"
    EVICTED = "EVICTED"   # out of the addressable bank, still in the store (E-000030)


class CellKind(str, Enum):
    """FACT cells hold an object; LINK cells hold the address of another cell (E-000015)."""

    FACT = "FACT"
    LINK = "LINK"


LINK_OBJ = -1          # a link version has no object; -1 never denotes an entity


@dataclass
class Version:
    version: int
    subject: int
    relation: int
    obj: int
    marker: np.ndarray
    op_index: int
    kind: CellKind = CellKind.FACT
    target: Optional[int] = None      # LINK only: the kid this cell points at


@dataclass
class Cell:
    kid: int
    versions: List[Version] = field(default_factory=list)
    active_version: int = 1
    status: Status = Status.ACTIVE
    provenance: str = ""
    tombstone_key: Optional[Tuple[int, int]] = None    # the key this cell held when it was DELETED

    @property
    def active(self) -> Optional[Version]:
        if self.status != Status.ACTIVE:
            return None
        return self.versions[self.active_version - 1]

    def version_obj(self, v: int) -> Version:
        return self.versions[v - 1]


class MVCCStore:
    def __init__(self, marker_dim: int = 16, seed: int = 0, valid_radius: float = 0.35,
                 marker_centre: Optional[np.ndarray] = None):
        self.marker_dim = marker_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        drawn = self.rng.normal(size=marker_dim)          # always drawn: replay must reproduce the RNG sequence
        centre = drawn if marker_centre is None else np.asarray(marker_centre, dtype=float)
        self.marker_centre = centre / np.linalg.norm(centre)
        self.valid_radius = valid_radius
        self.cells: Dict[int, Cell] = {}
        self.log: List[Tuple[str, Dict[str, Any]]] = []
        self._next_kid = 1
        self.revision = 0

    # ------------------------------------------------------------------ markers
    def new_valid_marker(self) -> np.ndarray:
        m = self.marker_centre + self.rng.normal(scale=0.05, size=self.marker_dim)
        return m / np.linalg.norm(m)

    def new_invalid_marker(self) -> np.ndarray:
        m = self.rng.normal(size=self.marker_dim)
        m = m / np.linalg.norm(m)
        # make sure it is far from the centre (reject near-collisions)
        while np.linalg.norm(m - self.marker_centre) < 2 * self.valid_radius:
            m = self.rng.normal(size=self.marker_dim)
            m = m / np.linalg.norm(m)
        return m

    def marker_valid(self, marker: np.ndarray) -> bool:
        return bool(np.linalg.norm(marker - self.marker_centre) <= self.valid_radius)

    # ------------------------------------------------------------------ operations
    def _record(self, op: str, **args: Any) -> int:
        self.log.append((op, args))
        self.revision += 1
        return len(self.log) - 1

    def write(self, subject: int, relation: int, obj: int, provenance: str = "") -> int:
        kid = self._next_kid
        self._next_kid += 1
        op = self._record("write", subject=subject, relation=relation, obj=obj, provenance=provenance)
        cell = Cell(kid=kid, provenance=provenance)
        cell.versions.append(Version(1, subject, relation, obj, self.new_valid_marker(), op))
        self.cells[kid] = cell
        return kid

    def link(self, subject: int, relation: int, target: int, provenance: str = "") -> int:
        """Write a LINK cell: the key ``(subject, relation)`` resolves to whatever cell ``target`` holds.

        The link carries its OWN valid marker, so a resolution path through an alias has two
        independent signatures (the alias's and the payload's) and either can be shredded alone.
        """
        if target not in self.cells:
            raise KeyError(f"link target {target} does not exist")
        kid = self._next_kid
        self._next_kid += 1
        op = self._record("link", subject=subject, relation=relation, target=target, provenance=provenance)
        cell = Cell(kid=kid, provenance=provenance)
        cell.versions.append(Version(1, subject, relation, LINK_OBJ, self.new_valid_marker(), op,
                                     kind=CellKind.LINK, target=target))
        self.cells[kid] = cell
        return kid

    def relink(self, kid: int, target: int) -> int:
        """Point an existing alias at a different cell (a new version, like UPDATE on a fact)."""
        cell = self._alive(kid)
        if target not in self.cells:
            raise KeyError(f"link target {target} does not exist")
        op = self._record("relink", kid=kid, target=target)
        prev = cell.version_obj(cell.active_version)
        v = Version(len(cell.versions) + 1, prev.subject, prev.relation, LINK_OBJ, self.new_valid_marker(), op,
                    kind=CellKind.LINK, target=target)
        cell.versions.append(v)
        cell.active_version = v.version
        cell.status = Status.ACTIVE
        return v.version

    def refcount(self, kid: int) -> int:
        """How many non-deleted alias cells currently point at ``kid``."""
        n = 0
        for other in self.cells.values():
            if other.status in (Status.DELETED, Status.EVICTED) or not other.versions:
                continue
            v = other.version_obj(other.active_version)
            if v.kind == CellKind.LINK and v.target == kid:
                n += 1
        return n

    def read(self, kid: int) -> Optional[Version]:
        cell = self.cells.get(kid)
        if cell is None:
            return None
        v = cell.active
        if v is None or not self.marker_valid(v.marker):
            return None
        return v

    def update(self, kid: int, obj: int) -> int:
        cell = self._alive(kid)
        op = self._record("update", kid=kid, obj=obj)
        prev = cell.version_obj(cell.active_version)
        v = Version(len(cell.versions) + 1, prev.subject, prev.relation, obj, self.new_valid_marker(), op)
        cell.versions.append(v)
        cell.active_version = v.version
        cell.status = Status.ACTIVE
        return v.version

    def revoke(self, kid: int) -> None:
        cell = self._alive(kid)
        self._record("revoke", kid=kid)
        cell.status = Status.REVOKED

    def restore(self, kid: int) -> None:
        """Undo REVOKE or EVICT. A deleted cell has no versions left and cannot come back."""
        cell = self._alive(kid)
        self._record("restore", kid=kid)
        cell.status = Status.ACTIVE
        cell.tombstone_key = None

    def rollback(self, kid: int, version: int) -> None:
        cell = self._alive(kid)
        if not 1 <= version <= len(cell.versions):
            raise ValueError(f"cell {kid} has no version {version}")
        self._record("rollback", kid=kid, version=version)
        cell.active_version = version
        cell.status = Status.ACTIVE

    def delete(self, kid: int) -> None:
        cell = self._alive(kid)
        self._record("delete", kid=kid)
        v = cell.version_obj(cell.active_version)
        cell.tombstone_key = (v.subject, v.relation)   # aliases keep pointing at this key: the pointer is NOT erased
        cell.status = Status.DELETED
        cell.versions = []

    def evict(self, kid: int) -> None:
        """Take the row out of the addressable bank and KEEP the payload in the store.

        This exists because E-000030 measured a gap the lifecycle did not cover. SHRED keeps the row
        addressable and asks a learned gate to refuse it, which earns no certificate: the payload is
        still an input to the computation, and E-000028 recovered it at 1.0000 through a derived key
        the gate never touched. DELETE earns the certificate -- the row is not in the bank, so nothing
        the model computes can depend on it, for any payload over any domain -- but it also does
        ``cell.versions = []``, so the data is gone and there is nothing left to audit, roll back or
        hold for a legal case. That was what SHRED was FOR.

        EVICT is the operation the certificate prescribes and the store lacked: the row leaves
        ``bank()`` exactly as a deleted one does, so the model has no path to it, while every version
        stays in the store so RESTORE and ROLLBACK still work. Retention and unreachability were never
        actually in tension; they only looked that way because the payload was being kept in the same
        place the model reads.
        """
        cell = self._alive(kid)
        self._record("evict", kid=kid)
        v = cell.version_obj(cell.active_version)
        cell.tombstone_key = (v.subject, v.relation)
        cell.status = Status.EVICTED

    def shred(self, kid: int) -> None:
        """Destroy the marker of the active version; the payload stays in the layer.

        Kept as recorded. E-000030 finds it certified at neither level: see ``evict`` for the operation
        that earns the certificate without discarding the data.
        """
        cell = self._alive(kid)
        self._record("shred", kid=kid)
        cell.version_obj(cell.active_version).marker = self.new_invalid_marker()

    def resign(self, kid: int) -> None:
        """Give the active version a fresh valid marker (undo of shred, for restore tests)."""
        cell = self._alive(kid)
        self._record("resign", kid=kid)
        cell.version_obj(cell.active_version).marker = self.new_valid_marker()

    def swap(self, kid_a: int, kid_b: int) -> None:
        """Causal intervention: exchange the payload objects of two active cells."""
        a, b = self._alive(kid_a), self._alive(kid_b)
        self._record("swap", kid_a=kid_a, kid_b=kid_b)
        va, vb = a.version_obj(a.active_version), b.version_obj(b.active_version)
        va.obj, vb.obj = vb.obj, va.obj

    def replace(self, kid: int, obj: int) -> None:
        """Causal intervention: overwrite the active payload in place (no new version)."""
        cell = self._alive(kid)
        self._record("replace", kid=kid, obj=obj)
        cell.version_obj(cell.active_version).obj = obj

    def _alive(self, kid: int) -> Cell:
        cell = self.cells.get(kid)
        if cell is None or cell.status == Status.DELETED:
            # EVICTED is deliberately NOT rejected here. An evicted cell is out of the addressable
            # bank but still in the store -- that is the entire point of the operation -- so RESTORE
            # and ROLLBACK have to be able to reach it. Rejecting it here would make eviction
            # irreversible and turn it into a slower DELETE.
            raise KeyError(f"cell {kid} does not exist or was deleted")
        return cell

    # ------------------------------------------------------------------ views
    def active_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """``(subject, relation) -> (object, kid)`` over usable cells."""
        view: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for kid, cell in self.cells.items():
            v = cell.active
            if v is None:
                continue
            if respect_markers and not self.marker_valid(v.marker):
                continue
            view[(v.subject, v.relation)] = (v.obj, kid)
        return view

    def _key_index(self, respect_markers: bool) -> Dict[Tuple[int, int], Optional[Tuple[int, Version]]]:
        """``key -> (kid, version)`` for usable cells; ``key -> None`` when the holder is unsigned.

        Built once per resolution pass: resolving key by key would be quadratic in the bank size.
        """
        idx: Dict[Tuple[int, int], Optional[Tuple[int, Version]]] = {}
        for kid, cell in self.cells.items():
            v = cell.active
            if v is None:
                continue
            key = (v.subject, v.relation)
            if key in idx:
                continue                          # first holder wins, as before
            idx[key] = None if (respect_markers and not self.marker_valid(v.marker)) else (kid, v)
        return idx

    def _usable_cell_at(self, key: Tuple[int, int], respect_markers: bool,
                        index: Optional[Dict[Tuple[int, int], Optional[Tuple[int, Version]]]] = None):
        if index is not None:
            return index.get(key)
        for kid, cell in self.cells.items():
            v = cell.active
            if v is None or (v.subject, v.relation) != key:
                continue
            if respect_markers and not self.marker_valid(v.marker):
                return None                       # the cell holding the key is unsigned: the key is not resolvable
            return kid, v
        return None

    MAX_LINK_DEPTH = 4

    def resolve_key(self, key: Tuple[int, int], respect_markers: bool = True,
                    max_depth: Optional[int] = None,
                    index: Optional[Dict[Tuple[int, int], Optional[Tuple[int, Version]]]] = None
                    ) -> Tuple[Optional[int], Tuple[int, ...]]:
        """Follow ``key`` through any chain of aliases: ``(object or None, trace of kids)``.

        A miss (``None``) is returned when no usable cell holds the key, when the chain exceeds
        ``max_depth``, when it revisits a cell (cycle), or when it ends on a cell that no longer
        exists — a DANGLING alias.  The trace names every cell that was read, aliases included:
        revoking an alias changes the answer, so the alias is part of the dependency.
        """
        depth = self.MAX_LINK_DEPTH if max_depth is None else max_depth
        trace: List[int] = []
        seen: set = set()
        cur = key
        for _ in range(depth + 1):
            hit = self._usable_cell_at(cur, respect_markers, index)
            if hit is None:
                return None, tuple(trace)
            kid, v = hit
            if kid in seen:
                return None, tuple(trace)          # cycle
            seen.add(kid)
            trace.append(kid)
            if v.kind == CellKind.FACT:
                return v.obj, tuple(trace)
            target = self.cells.get(v.target)
            if target is None or target.status in (Status.DELETED, Status.EVICTED) or not target.versions:
                return None, tuple(trace)          # dangling: the referent is gone, the pointer remains
            tv = target.version_obj(target.active_version)
            cur = (tv.subject, tv.relation)
        return None, tuple(trace)                  # depth exceeded

    def resolved_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]]:
        """``(subject, relation) -> (object, trace)`` for every key that resolves, aliases followed."""
        idx = self._key_index(respect_markers)
        out: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {}
        for key in idx:
            obj, trace = self.resolve_key(key, respect_markers, index=idx)
            if obj is not None:
                out[key] = (obj, trace)
        return out

    def index_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], int]:
        """``(subject, relation) -> object``.  Aliases are followed, so a world containing links has
        the same interface as one without."""
        return {k: o for k, (o, _) in self.resolved_view(respect_markers).items()}

    def kid_of(self, key: Tuple[int, int]) -> Optional[int]:
        """kid of the cell currently holding ``key`` (any status except deleted)."""
        for kid, cell in self.cells.items():
            if cell.status in (Status.DELETED, Status.EVICTED):
                continue
            v = cell.version_obj(cell.active_version)
            if (v.subject, v.relation) == key:
                return kid
        return None

    def bank(self, respect_markers: bool = False) -> Dict[str, np.ndarray]:
        """Tensors describing the layer for a neural model.

        All non-deleted cells are returned (their payload physically remains in the
        layer); ``active`` flags routing availability.  Marker validity is NOT applied
        here unless ``respect_markers`` is set — the neural model has to learn to
        reject unsigned payloads itself.
        """
        kids, subj, rel, obj, markers, active = [], [], [], [], [], []
        is_link, l_subj, l_rel = [], [], []
        for kid, cell in self.cells.items():
            if cell.status in (Status.DELETED, Status.EVICTED):
                continue
            v = cell.version_obj(cell.active_version)
            usable = cell.status == Status.ACTIVE
            if respect_markers and not self.marker_valid(v.marker):
                usable = False
            kids.append(kid)
            subj.append(v.subject)
            rel.append(v.relation)
            markers.append(v.marker)
            active.append(usable)
            link = v.kind == CellKind.LINK
            is_link.append(link)
            # A link row carries the TARGET'S KEY, not its payload and not its state: whether that key
            # is held by a signed, active, existing cell is exactly what the model has to discover.
            # ``obj`` is a constant placeholder for link rows (never the target's object).
            obj.append(0 if link else v.obj)
            if link:
                t = self.cells.get(v.target)
                if t is not None and t.versions:
                    tv = t.version_obj(t.active_version)
                    l_subj.append(tv.subject); l_rel.append(tv.relation)
                elif t is not None and t.tombstone_key is not None:
                    l_subj.append(t.tombstone_key[0]); l_rel.append(t.tombstone_key[1])   # dangling: key kept
                else:
                    l_subj.append(v.subject); l_rel.append(v.relation)                    # self-reference = miss
            else:
                l_subj.append(0); l_rel.append(0)
        return {
            "kid": np.asarray(kids, dtype=np.int64),
            "subject": np.asarray(subj, dtype=np.int64),
            "relation": np.asarray(rel, dtype=np.int64),
            "obj": np.asarray(obj, dtype=np.int64),
            "marker": np.asarray(markers, dtype=np.float32).reshape(len(kids), self.marker_dim),
            "active": np.asarray(active, dtype=bool),
            "is_link": np.asarray(is_link, dtype=bool),
            "link_subject": np.asarray(l_subj, dtype=np.int64),
            "link_relation": np.asarray(l_rel, dtype=np.int64),
        }

    # ------------------------------------------------------------------ replay / hashing
    def state_hash(self) -> str:
        parts = []
        for kid in sorted(self.cells):
            c = self.cells[kid]
            vs = [(v.version, v.subject, v.relation, v.obj, self.marker_valid(v.marker), v.kind.value, v.target)
                  for v in c.versions]
            parts.append((kid, c.status.value, c.active_version, vs))
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    @classmethod
    def replay(cls, log: List[Tuple[str, Dict[str, Any]]], marker_dim: int, seed: int,
               valid_radius: float, marker_centre: Optional[np.ndarray] = None) -> "MVCCStore":
        store = cls(marker_dim=marker_dim, seed=seed, valid_radius=valid_radius, marker_centre=marker_centre)
        for op, args in log:
            getattr(store, op)(**args)
        return store

    def clone_by_replay(self) -> "MVCCStore":
        return MVCCStore.replay(list(self.log), self.marker_dim, self.seed, self.valid_radius, self.marker_centre)
