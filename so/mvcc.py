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


@dataclass
class Version:
    version: int
    subject: int
    relation: int
    obj: int
    marker: np.ndarray
    op_index: int


@dataclass
class Cell:
    kid: int
    versions: List[Version] = field(default_factory=list)
    active_version: int = 1
    status: Status = Status.ACTIVE
    provenance: str = ""

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
        cell = self._alive(kid)
        self._record("restore", kid=kid)
        cell.status = Status.ACTIVE

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
        cell.status = Status.DELETED
        cell.versions = []

    def shred(self, kid: int) -> None:
        """Destroy the marker of the active version; the payload stays in the layer."""
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

    def index_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], int]:
        return {k: o for k, (o, _) in self.active_view(respect_markers).items()}

    def kid_of(self, key: Tuple[int, int]) -> Optional[int]:
        """kid of the cell currently holding ``key`` (any status except deleted)."""
        for kid, cell in self.cells.items():
            if cell.status == Status.DELETED:
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
        for kid, cell in self.cells.items():
            if cell.status == Status.DELETED:
                continue
            v = cell.version_obj(cell.active_version)
            usable = cell.status == Status.ACTIVE
            if respect_markers and not self.marker_valid(v.marker):
                usable = False
            kids.append(kid)
            subj.append(v.subject)
            rel.append(v.relation)
            obj.append(v.obj)
            markers.append(v.marker)
            active.append(usable)
        return {
            "kid": np.asarray(kids, dtype=np.int64),
            "subject": np.asarray(subj, dtype=np.int64),
            "relation": np.asarray(rel, dtype=np.int64),
            "obj": np.asarray(obj, dtype=np.int64),
            "marker": np.asarray(markers, dtype=np.float32).reshape(len(kids), self.marker_dim),
            "active": np.asarray(active, dtype=bool),
        }

    # ------------------------------------------------------------------ replay / hashing
    def state_hash(self) -> str:
        parts = []
        for kid in sorted(self.cells):
            c = self.cells[kid]
            vs = [(v.version, v.subject, v.relation, v.obj, self.marker_valid(v.marker)) for v in c.versions]
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
