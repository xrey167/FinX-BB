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

``content_markers`` (default off; every recorded run used the default): E-000051 (ledger §31.41)
found that drawing markers from the store's seeded generator makes the POSITION in that stream a
history channel -- a store that wrote a pod and then evicted every row of it differs from one that
never wrote it in the markers of every row written after the pod, and a frozen reader separates the
two on bystander queries at AUC 0.948 at KL 0.  With the option on, a row's marker is derived from
its EXPORTED CONTENT (kind, key, object or pointed-at key) and a per-store secret ``marker_key``, an
HMAC mapped into the same distribution ``new_valid_marker`` draws from (a seeded normal draw of
scale 0.05 around the centre, normalised), so the marker of a row is a function of what the row
holds and not of when it was written.  Owned mechanism: content-derived / deterministic
signatures.  ``bank()``, ``state_hash`` and the gate semantics are unchanged; two rows with
identical exported content carry identical markers, which is the side effect E-000053 measures.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import numbers
import threading
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
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


def _integral(value: Any, name: str) -> int:
    """Validate and canonicalise an integer identifier without Python's bool/float aliasing."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not bool or float")
    return int(value)


def _key(value: Any) -> Tuple[int, int]:
    """Canonicalise a public ``(subject, relation)`` lookup key."""
    try:
        subject, relation = value
    except (TypeError, ValueError) as exc:
        raise TypeError("key must be a (subject, relation) pair of integers") from exc
    return _integral(subject, "subject"), _integral(relation, "relation")


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


@dataclass(frozen=True)
class VersionView:
    """Immutable, defensive value returned by :meth:`MVCCStore.read`.

    Internal ``Version`` objects remain mutable because interventions deliberately edit them in
    place.  Returning one of those objects leaked both the marker array and every scalar field to a
    caller, bypassing the operation log and revision counter.  The tuple marker makes this value
    recursively immutable rather than merely freezing the dataclass shell.
    """

    version: int
    subject: int
    relation: int
    obj: int
    marker: Tuple[float, ...]
    op_index: int
    kind: CellKind = CellKind.FACT
    target: Optional[int] = None


class RevisionConflict(RuntimeError):
    """A compare-and-swap mutation was attempted against a stale store revision."""


class StaleSnapshotError(RuntimeError):
    """A consumer attempted to use a bank captured from an older store revision."""


@dataclass(frozen=True)
class RevisionToken:
    """Process-local store identity plus the revision captured in one atomic snapshot."""

    revision: int
    state_fingerprint: str
    _store_identity: object = field(repr=False, compare=False)


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
        version = _integral(v, "version")
        if not 1 <= version <= len(self.versions):
            raise ValueError(f"cell {self.kid} has no version {version}")
        return self.versions[version - 1]


@dataclass(frozen=True)
class SnapshotRow:
    """One immutable exported row captured at an ``MVCCStore`` revision."""

    kid: int
    subject: int
    relation: int
    obj: int
    marker: Tuple[float, ...]
    status: Status
    is_holder: bool
    marker_valid: bool
    is_link: bool
    link_target_kid: int
    link_subject: int
    link_relation: int


@dataclass(frozen=True)
class MVCCSnapshot:
    """A single-revision, immutable materialisation of rows and resolver metadata."""

    revision: int
    marker_dim: int
    rows: Tuple[SnapshotRow, ...]
    holder_of_key: Tuple[Tuple[Tuple[int, int], int], ...]
    resolved_view: Tuple[Tuple[Tuple[int, int], Tuple[int, Tuple[int, ...]]], ...]
    index_view: Tuple[Tuple[Tuple[int, int], int], ...]
    revision_token: RevisionToken = field(repr=False, compare=False)


def _synchronized(method):
    """Serialize a store operation against mutations and atomic snapshot capture."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


class MVCCStore:
    MARKER_SCALE = 0.05          # the spread of a valid marker around the centre, both marker schemes

    def __init__(self, marker_dim: int = 16, seed: int = 0, valid_radius: float = 0.35,
                 marker_centre: Optional[np.ndarray] = None, content_markers: bool = False,
                 marker_key: Optional[bytes] = None):
        self.marker_dim = marker_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        drawn = self.rng.normal(size=marker_dim)          # always drawn: replay must reproduce the RNG sequence
        centre = drawn if marker_centre is None else np.asarray(marker_centre, dtype=float)
        self.marker_centre = centre / np.linalg.norm(centre)
        self.valid_radius = valid_radius
        # History-independent markers (module docstring). The per-store secret defaults to a digest of
        # the seed so that two stores built with the same seed -- E-000051's CASCADE(p) and NEVER(p) --
        # sign identical content identically; pass ``marker_key`` to give a store its own secret.
        self.content_markers = bool(content_markers)
        self.marker_key = (marker_key if marker_key is not None
                           else hashlib.sha256(b"so.mvcc.marker-key|" + str(seed).encode()).digest())
        self.cells: Dict[int, Cell] = {}
        self.log: List[Tuple[str, Dict[str, Any]]] = []
        self._next_kid = 1
        self.revision = 0
        self._lock = threading.RLock()
        self._identity = object()

    def _check_revision(self, expected_revision: Optional[int]) -> None:
        if expected_revision is not None:
            expected_revision = _integral(expected_revision, "expected_revision")
        if expected_revision is not None and expected_revision != self.revision:
            raise RevisionConflict(
                f"stale MVCC revision {expected_revision}; current revision is {self.revision}"
            )

    @contextmanager
    def _operation_attempt(self, expected_revision: Optional[int] = None,
                           touched_kids: Tuple[int, ...] = ()):
        """Rollback fallible pre-commit work, including RNG and a partially recorded log entry.

        Mutators stage validation and signing before touching a cell, then append the log, and only
        then perform non-fallible assignments.  This small checkpoint therefore need not deepcopy
        the potentially large cell store, but it still restores every piece of metadata a signing or
        injected ``_record`` failure can advance.
        """
        self._check_revision(expected_revision)
        log_len = len(self.log)
        revision = self.revision
        next_kid = self._next_kid
        rng_state = deepcopy(self.rng.bit_generator.state)
        missing = object()
        cell_state = {
            int(kid): deepcopy(self.cells[kid]) if kid in self.cells else missing
            for kid in touched_kids
        }
        try:
            yield
        except BaseException:
            del self.log[log_len:]
            self.revision = revision
            self._next_kid = next_kid
            self.rng.bit_generator.state = rng_state
            for kid, prior in cell_state.items():
                if prior is missing:
                    self.cells.pop(kid, None)
                else:
                    self.cells[kid] = prior
            raise

    @_synchronized
    def revision_token(self) -> RevisionToken:
        """Return a freshness token linearized with mutations."""
        return RevisionToken(int(self.revision), self._freshness_fingerprint(), self._identity)

    @_synchronized
    def is_current(self, token: RevisionToken) -> bool:
        """Whether ``token`` names this store at its current revision."""
        return (token._store_identity is self._identity and token.revision == self.revision
                and token.state_fingerprint == self._freshness_fingerprint())

    def _freshness_fingerprint(self) -> str:
        """Fast process-local digest that detects unsupported direct mutation of public legacy state."""
        h = hashlib.blake2b(digest_size=16)

        def put(value: Any) -> None:
            raw = value if isinstance(value, bytes) else repr(value).encode()
            h.update(len(raw).to_bytes(8, "big")); h.update(raw)

        def put_array(value: np.ndarray) -> None:
            array = np.ascontiguousarray(np.asarray(value))
            put((array.dtype.str, tuple(array.shape)))
            put(array.tobytes())

        put((self.revision, self._next_kid, self.marker_dim, self.seed, self.valid_radius,
             self.content_markers, self.MAX_LINK_DEPTH))
        put_array(self.marker_centre)
        put(self.marker_key)
        # Marker sampling is a deliberately low-level research helper rather than a logged
        # lifecycle operation, but advancing it changes later signatures and must stale any
        # already-exported consumer view.
        put(self.rng.bit_generator.state)
        for kid in sorted(self.cells):
            c = self.cells[kid]
            put((kid, c.kid, c.status.value, c.active_version, c.provenance, c.tombstone_key))
            for v in c.versions:
                put((v.version, v.subject, v.relation, v.obj, v.op_index, v.kind.value, v.target))
                put_array(v.marker)
        put(self.log)
        return h.hexdigest()

    # ------------------------------------------------------------------ markers
    @_synchronized
    def new_valid_marker(self) -> np.ndarray:
        """Sample a valid marker for research fixtures, advancing the persistent store RNG.

        This is intentionally not a lifecycle mutation: it creates no cell, log record, or
        revision.  Consequently ad-hoc calls are not reproducible from :meth:`clone_by_replay`;
        callers needing replayable state must use the logged mutation APIs.  Freshness tokens do
        include the RNG state, so a sample invalidates an already exported ``Bank``.
        """
        m = self.marker_centre + self.rng.normal(scale=0.05, size=self.marker_dim)
        return m / np.linalg.norm(m)

    @_synchronized
    def new_invalid_marker(self) -> np.ndarray:
        """Sample an invalid research marker under the same boundary as ``new_valid_marker``."""
        m = self.rng.normal(size=self.marker_dim)
        m = m / np.linalg.norm(m)
        # make sure it is far from the centre (reject near-collisions)
        while np.linalg.norm(m - self.marker_centre) < 2 * self.valid_radius:
            m = self.rng.normal(size=self.marker_dim)
            m = m / np.linalg.norm(m)
        return m

    def marker_valid(self, marker: np.ndarray) -> bool:
        return bool(np.linalg.norm(marker - self.marker_centre) <= self.valid_radius)

    # ---- content-derived markers (the ``content_markers`` option)
    def row_content(self, v: "Version") -> Tuple[Any, ...]:
        """What ``bank()`` exports for this version, as a tuple: the input to a content-derived marker.

        A FACT row is (kind, subject, relation, object). A LINK row is (kind, subject, relation,
        pointed-at subject, pointed-at relation) with the pointed-at key resolved exactly as ``bank()``
        resolves it -- the target's key, its tombstone key once it is gone, the row's own key once it
        is blanked -- so the marker follows the row's exported content through its lifecycle and never
        encodes a cell id (cell ids encode write order, which is the history the option removes).
        """
        if v.kind is CellKind.FACT:
            return ("FACT", int(v.subject), int(v.relation), int(v.obj))
        t = self.cells.get(v.target) if v.target is not None else None
        if t is not None and t.versions:
            tv = t.version_obj(t.active_version)
            ls, lr = int(tv.subject), int(tv.relation)
        elif t is not None and t.tombstone_key is not None:
            ls, lr = int(t.tombstone_key[0]), int(t.tombstone_key[1])
        else:
            ls, lr = int(v.subject), int(v.relation)
        return ("LINK", int(v.subject), int(v.relation), ls, lr)

    def _derived_rng(self, content: Tuple[Any, ...], valid: bool) -> np.random.Generator:
        tag = b"valid|" if valid else b"invalid|"
        digest = hmac.new(self.marker_key, tag + json.dumps(list(content)).encode(), hashlib.sha256).digest()
        return np.random.default_rng(int.from_bytes(digest[:16], "big"))

    def derived_marker(self, content: Tuple[Any, ...], valid: bool = True) -> np.ndarray:
        """The marker a row with this exported content carries under ``content_markers``.

        Same distribution as the generator scheme -- a normal draw of scale ``MARKER_SCALE`` around
        the centre, normalised (invalid: the same rejection loop as ``new_invalid_marker``) -- so a
        reader trained on generator-drawn markers sees the same family; only the SOURCE of the draw
        changes, from a position in a stream to a digest of the content.
        """
        rng = self._derived_rng(content, valid)
        if valid:
            m = self.marker_centre + rng.normal(scale=self.MARKER_SCALE, size=self.marker_dim)
            return m / np.linalg.norm(m)
        m = rng.normal(size=self.marker_dim)
        m = m / np.linalg.norm(m)
        while np.linalg.norm(m - self.marker_centre) < 2 * self.valid_radius:
            m = rng.normal(size=self.marker_dim)
            m = m / np.linalg.norm(m)
        return m

    def _sign(self, v: "Version") -> None:
        """Give ``v`` its marker under whichever scheme the store uses (called once per new version)."""
        v.marker = self.derived_marker(self.row_content(v)) if self.content_markers else self.new_valid_marker()

    def _unsign(self, v: "Version") -> None:
        v.marker = (self.derived_marker(self.row_content(v), valid=False) if self.content_markers
                    else self.new_invalid_marker())

    # ------------------------------------------------------------------ operations
    def _record(self, op: str, **args: Any) -> int:
        self.log.append((op, args))
        self.revision += 1
        return len(self.log) - 1

    @_synchronized
    def write(self, subject: int, relation: int, obj: int, provenance: str = "", *,
              expected_revision: Optional[int] = None) -> int:
        subject = _integral(subject, "subject")
        relation = _integral(relation, "relation")
        obj = _integral(obj, "obj")
        with self._operation_attempt(expected_revision, (self._next_kid,)):
            kid = self._next_kid
            v = Version(1, subject, relation, obj, None, len(self.log))
            self._sign(v)                         # stage all fallible work before publishing the cell
            op = self._record("write", subject=subject, relation=relation, obj=obj,
                              provenance=provenance)
            v.op_index = op
            self.cells[kid] = Cell(kid=kid, versions=[v], provenance=provenance)
            self._next_kid = kid + 1
            return kid

    @_synchronized
    def link(self, subject: int, relation: int, target: int, provenance: str = "", *,
             expected_revision: Optional[int] = None) -> int:
        """Write a LINK cell: the key ``(subject, relation)`` resolves to whatever cell ``target`` holds.

        The link carries its OWN valid marker, so a resolution path through an alias has two
        independent signatures (the alias's and the payload's) and either can be shredded alone.
        """
        subject = _integral(subject, "subject")
        relation = _integral(relation, "relation")
        target = _integral(target, "target")
        with self._operation_attempt(expected_revision, (self._next_kid,)):
            if target not in self.cells:
                raise KeyError(f"link target {target} does not exist")
            kid = self._next_kid
            v = Version(1, subject, relation, LINK_OBJ, None, len(self.log),
                        kind=CellKind.LINK, target=target)
            self._sign(v)
            op = self._record("link", subject=subject, relation=relation, target=target,
                              provenance=provenance)
            v.op_index = op
            self.cells[kid] = Cell(kid=kid, versions=[v], provenance=provenance)
            self._next_kid = kid + 1
            return kid

    @_synchronized
    def relink(self, kid: int, target: int, *, expected_revision: Optional[int] = None) -> int:
        """Point an existing alias at a different cell (a new version, like UPDATE on a fact)."""
        kid = _integral(kid, "kid")
        target = _integral(target, "target")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            if target not in self.cells:
                raise KeyError(f"link target {target} does not exist")
            prev = cell.version_obj(cell.active_version)
            if prev.kind is not CellKind.LINK:
                raise ValueError(f"relink is for LINK cells and cell {kid} is a {prev.kind.value}")
            v = Version(len(cell.versions) + 1, prev.subject, prev.relation, LINK_OBJ, None,
                        len(self.log), kind=CellKind.LINK, target=target)
            self._sign(v)
            op = self._record("relink", kid=kid, target=target)
            v.op_index = op
            cell.versions.append(v)
            cell.active_version = v.version
            # Updating retained state is not RESTORE: REVOKED/EVICTED remains non-addressable.
            return v.version

    @_synchronized
    def refcount(self, kid: int) -> int:
        """How many non-deleted alias cells currently point at ``kid``."""
        kid = _integral(kid, "kid")
        n = 0
        for other in self.cells.values():
            # EVICTED removes a row from routing/export, not its retained pointer.  Closure and
            # deletion guards therefore continue to count it until the alias is truly DELETED.
            if other.status is Status.DELETED or not other.versions:
                continue
            v = other.version_obj(other.active_version)
            if v.kind == CellKind.LINK and v.target == kid:
                n += 1
        return n

    @_synchronized
    def read(self, kid: int) -> Optional[VersionView]:
        kid = _integral(kid, "kid")
        cell = self.cells.get(kid)
        if cell is None:
            return None
        v = cell.active
        if v is None or not self.marker_valid(v.marker):
            return None
        return VersionView(
            version=int(v.version), subject=int(v.subject), relation=int(v.relation), obj=int(v.obj),
            marker=tuple(float(x) for x in v.marker), op_index=int(v.op_index), kind=v.kind,
            target=None if v.target is None else int(v.target),
        )

    @_synchronized
    def update(self, kid: int, obj: int, *, expected_revision: Optional[int] = None) -> int:
        kid = _integral(kid, "kid")
        obj = _integral(obj, "obj")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            prev = cell.version_obj(cell.active_version)
            if prev.kind is not CellKind.FACT:
                raise ValueError(f"update is for FACT cells and cell {kid} is a {prev.kind.value}")
            v = Version(len(cell.versions) + 1, prev.subject, prev.relation, obj, None, len(self.log))
            self._sign(v)
            op = self._record("update", kid=kid, obj=obj)
            v.op_index = op
            cell.versions.append(v)
            cell.active_version = v.version
            # UPDATE changes retained data but never changes its lifecycle state.
            return v.version

    @_synchronized
    def revoke(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            if cell.status is not Status.ACTIVE:
                raise ValueError(f"revoke requires ACTIVE, cell {kid} is {cell.status.value}")
            self._record("revoke", kid=kid)
            cell.status = Status.REVOKED

    @_synchronized
    def restore(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        """Undo REVOKE or EVICT. A deleted cell has no versions left and cannot come back."""
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            if cell.status not in (Status.REVOKED, Status.EVICTED):
                raise ValueError(
                    f"restore requires REVOKED or EVICTED, cell {kid} is {cell.status.value}"
                )
            self._record("restore", kid=kid)
            cell.status = Status.ACTIVE
            cell.tombstone_key = None

    @_synchronized
    def rollback(self, kid: int, version: int, *, expected_revision: Optional[int] = None) -> None:
        kid = _integral(kid, "kid")
        version = _integral(version, "version")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            if not 1 <= version <= len(cell.versions):
                raise ValueError(f"cell {kid} has no version {version}")
            self._record("rollback", kid=kid, version=version)
            cell.active_version = version
            # ROLLBACK selects retained data; only RESTORE makes it addressable again.

    @_synchronized
    def delete(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            v = cell.version_obj(cell.active_version)
            tombstone = (v.subject, v.relation)
            self._record("delete", kid=kid)
            cell.tombstone_key = tombstone   # pointer metadata remains, but the target kid never retargets
            cell.status = Status.DELETED
            cell.versions = []

    @_synchronized
    def evict(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
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
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            if cell.status not in (Status.ACTIVE, Status.REVOKED):
                raise ValueError(f"evict requires ACTIVE or REVOKED, cell {kid} is {cell.status.value}")
            v = cell.version_obj(cell.active_version)
            tombstone = (v.subject, v.relation)
            self._record("evict", kid=kid)
            cell.tombstone_key = tombstone
            cell.status = Status.EVICTED

    @_synchronized
    def shred(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        """Destroy the marker of the active version; the payload stays in the layer.

        Kept as recorded. E-000030 finds it certified at neither level: see ``evict`` for the operation
        that earns the certificate without discarding the data.
        """
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            v = cell.version_obj(cell.active_version)
            candidate = copy(v)
            candidate.marker = np.asarray(v.marker).copy()
            self._unsign(candidate)
            self._record("shred", kid=kid)
            v.marker = candidate.marker

    @_synchronized
    def blank(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        """Clear a LINK's target: the row stays live and addressable, and points at nothing.

        THE OPERATION THE LAW ASKS FOR. E-000041 measured that a fact on k access paths costs U =
        1 + copies to make unreachable and T = k to leave nothing pointing at what was removed;
        E-000046 then found that the T-side cost is the number of KEY-BEARING references, and that it
        can be paid in three currencies -- deletions, repairs, or an interface that declines to show
        the reference. The third is not payment: with the target hidden rather than cleared, the raw
        store still names the removed key in 0.8000 of cells.

        ``blank`` is the second currency, as a primitive. ``evict`` also closes the channel, but by
        removing the alias: every key that reached the fact stops being addressable at all, which is a
        visible change of a different kind and destroys rows a caller may still need. Blanking leaves
        the row live, so the key still resolves -- to UNKNOWN, which is what a caller asking for a
        deleted fact should get -- while ``bank()`` exports the row's own key in place of the target's
        and the dangling pointer is gone.

        E-000035 measured that closing this channel at the key is what removes the disclosure, at
        1.0000. This is that closure made into an operation rather than an analysis.
        """
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            v = cell.version_obj(cell.active_version)
            if v.kind is not CellKind.LINK:
                raise ValueError(
                    f"blank is for LINK cells and cell {kid} is a {v.kind.value}. Blanking a FACT would "
                    "clear a target it does not have, and would quietly do nothing -- which is how an "
                    "operation that cannot fail gets into a certificate.")
            marker = None
            if self.content_markers:
                candidate = copy(v)
                candidate.target = None
                candidate.marker = np.asarray(v.marker).copy()
                self._sign(candidate)
                marker = candidate.marker
            self._record("blank", kid=kid)
            v.target = None
            if marker is not None:
                v.marker = marker

    @_synchronized
    def resign(self, kid: int, *, expected_revision: Optional[int] = None) -> None:
        """Give the active version a fresh valid marker (undo of shred, for restore tests)."""
        kid = _integral(kid, "kid")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            v = cell.version_obj(cell.active_version)
            candidate = copy(v)
            candidate.marker = np.asarray(v.marker).copy()
            self._sign(candidate)
            self._record("resign", kid=kid)
            v.marker = candidate.marker

    @_synchronized
    def swap(self, kid_a: int, kid_b: int, *, expected_revision: Optional[int] = None) -> None:
        """Causal intervention: exchange the payload objects of two active cells."""
        kid_a = _integral(kid_a, "kid_a")
        kid_b = _integral(kid_b, "kid_b")
        with self._operation_attempt(expected_revision, (kid_a, kid_b)):
            a, b = self._alive(kid_a), self._alive(kid_b)
            va, vb = a.version_obj(a.active_version), b.version_obj(b.active_version)
            if va.kind is not CellKind.FACT or vb.kind is not CellKind.FACT:
                raise ValueError("swap requires two FACT cells")
            if a.status is not Status.ACTIVE or b.status is not Status.ACTIVE:
                raise ValueError("swap requires two ACTIVE cells")
            if not self.marker_valid(va.marker) or not self.marker_valid(vb.marker):
                raise ValueError("swap requires two marker-valid cells")
            self._record("swap", kid_a=kid_a, kid_b=kid_b)
            va.obj, vb.obj = vb.obj, va.obj

    @_synchronized
    def replace(self, kid: int, obj: int, *, expected_revision: Optional[int] = None) -> None:
        """Causal intervention: overwrite the active payload in place (no new version)."""
        kid = _integral(kid, "kid")
        obj = _integral(obj, "obj")
        with self._operation_attempt(expected_revision, (kid,)):
            cell = self._alive(kid)
            v = cell.version_obj(cell.active_version)
            if v.kind is not CellKind.FACT:
                raise ValueError(f"replace requires FACT, cell {kid} is {v.kind.value}")
            self._record("replace", kid=kid, obj=obj)
            v.obj = obj

    def _alive(self, kid: int) -> Cell:
        kid = _integral(kid, "kid")
        cell = self.cells.get(kid)
        if cell is None or cell.status == Status.DELETED:
            # EVICTED is deliberately NOT rejected here. An evicted cell is out of the addressable
            # bank but still in the store -- that is the entire point of the operation -- so RESTORE
            # and ROLLBACK have to be able to reach it. Rejecting it here would make eviction
            # irreversible and turn it into a slower DELETE.
            raise KeyError(f"cell {kid} does not exist or was deleted")
        return cell

    def _validate_cell_keys(self) -> None:
        """Fail closed when unsupported direct mutation breaks mapping-key/cell identity."""
        for mapping_kid, cell in self.cells.items():
            canonical_mapping = _integral(mapping_kid, "cell mapping kid")
            canonical_cell = _integral(cell.kid, "cell.kid")
            if canonical_mapping != canonical_cell:
                raise ValueError(
                    f"cell mapping key {canonical_mapping} disagrees with embedded kid {canonical_cell}"
                )

    # ------------------------------------------------------------------ views
    def _holder_index(self) -> Dict[Tuple[int, int], Tuple[int, Version]]:
        """Return the one status-active holder of each key, in deterministic write order.

        Duplicate rows are physical shadows, not simultaneous logical facts.  The first ACTIVE
        cell (the lowest ``kid``) owns the key; a
        REVOKE, DELETE or EVICT removes it from consideration and promotes the next ACTIVE shadow.
        Marker validity is deliberately *not* part of holder selection.  It is checked afterwards,
        so shredding the selected holder closes the key instead of resurrecting a shadow copy.
        """
        self._validate_cell_keys()
        idx: Dict[Tuple[int, int], Tuple[int, Version]] = {}
        for kid in sorted(self.cells):
            cell = self.cells[kid]
            v = cell.active
            if v is None:
                continue
            idx.setdefault((v.subject, v.relation), (kid, v))
        return idx

    @_synchronized
    def active_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], Tuple[int, int]]:
        """``(subject, relation) -> (object, kid)`` over the selected usable holders."""
        view: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for key, (kid, v) in self._holder_index().items():
            if respect_markers and not self.marker_valid(v.marker):
                continue
            view[key] = (v.obj, kid)
        return view

    def _key_index(self, respect_markers: bool) -> Dict[Tuple[int, int], Optional[Tuple[int, Version]]]:
        """``key -> (kid, version)`` for usable cells; ``key -> None`` when the holder is unsigned.

        Built once per resolution pass: resolving key by key would be quadratic in the bank size.
        """
        return {
            key: None if (respect_markers and not self.marker_valid(v.marker)) else (kid, v)
            for key, (kid, v) in self._holder_index().items()
        }

    def _usable_cell_at(self, key: Tuple[int, int], respect_markers: bool,
                        index: Optional[Dict[Tuple[int, int], Optional[Tuple[int, Version]]]] = None):
        if index is not None:
            return index.get(key)
        for kid in sorted(self.cells):
            cell = self.cells[kid]
            v = cell.active
            if v is None or (v.subject, v.relation) != key:
                continue
            if respect_markers and not self.marker_valid(v.marker):
                return None                       # the cell holding the key is unsigned: the key is not resolvable
            return kid, v
        return None

    MAX_LINK_DEPTH = 4

    @_synchronized
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
        key = _key(key)
        if index is None:
            self._validate_cell_keys()
        depth = self.MAX_LINK_DEPTH if max_depth is None else max_depth
        trace: List[int] = []
        seen: set = set()
        hit = self._usable_cell_at(key, respect_markers, index)
        for _ in range(depth + 1):
            if hit is None:
                return None, tuple(trace)
            kid, v = hit
            if kid in seen:
                return None, tuple(trace)          # cycle
            seen.add(kid)
            trace.append(kid)
            if v.kind == CellKind.FACT:
                return v.obj, tuple(trace)
            # A LINK is an identity reference to exactly one kid (E-000064), not a second lookup by
            # the target's (possibly duplicated) key.  Direct lookup may promote a shadow holder;
            # an alias must instead dangle until its registered target kid is explicitly restored.
            target = self.cells.get(v.target)
            if target is None or target.status is not Status.ACTIVE or not target.versions:
                return None, tuple(trace)          # dangling: the referent is gone, the pointer remains
            tv = target.version_obj(target.active_version)
            if respect_markers and not self.marker_valid(tv.marker):
                return None, tuple(trace)
            hit = (target.kid, tv)
        return None, tuple(trace)                  # depth exceeded

    @_synchronized
    def resolved_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]]:
        """``(subject, relation) -> (object, trace)`` for every key that resolves, aliases followed."""
        idx = self._key_index(respect_markers)
        out: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {}
        for key in idx:
            obj, trace = self.resolve_key(key, respect_markers, index=idx)
            if obj is not None:
                out[key] = (obj, trace)
        return out

    @_synchronized
    def index_view(self, respect_markers: bool = True) -> Dict[Tuple[int, int], int]:
        """``(subject, relation) -> object``.  Aliases are followed, so a world containing links has
        the same interface as one without."""
        return {k: o for k, (o, _) in self.resolved_view(respect_markers).items()}

    @_synchronized
    def kid_of(self, key: Tuple[int, int]) -> Optional[int]:
        """Kid of the selected status-active holder, whether its marker is valid or shredded."""
        hit = self._holder_index().get(_key(key))
        return None if hit is None else hit[0]

    @_synchronized
    def snapshot(self) -> MVCCSnapshot:
        """Capture all exported rows and resolution metadata at one linearizable revision.

        Marker arrays are converted to tuples and every collection is tuple-backed, so callers can
        safely finish materialising a neural bank after the store lock is released.  No caller needs
        to re-read ``cells`` (or invoke a second resolver pass) and accidentally combine revisions.
        """
        holders = self._holder_index()
        key_index = self._key_index(respect_markers=True)
        resolved: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {}
        for key in key_index:
            obj, trace = self.resolve_key(key, respect_markers=True, index=key_index)
            if obj is not None:
                resolved[key] = (int(obj), tuple(int(k) for k in trace))

        holder_kids = {kid for kid, _ in holders.values()}
        rows: List[SnapshotRow] = []
        for kid in sorted(self.cells):
            cell = self.cells[kid]
            if cell.status in (Status.DELETED, Status.EVICTED):
                continue
            v = cell.version_obj(cell.active_version)
            link = v.kind == CellKind.LINK
            link_target_kid = 0
            link_subject = 0
            link_relation = 0
            if link:
                link_target_kid = -1 if v.target is None else int(v.target)
                target = self.cells.get(v.target)
                if target is not None and target.versions:
                    tv = target.version_obj(target.active_version)
                    link_subject, link_relation = int(tv.subject), int(tv.relation)
                elif target is not None and target.tombstone_key is not None:
                    link_subject, link_relation = map(int, target.tombstone_key)
                else:
                    link_subject, link_relation = int(v.subject), int(v.relation)
            rows.append(SnapshotRow(
                kid=int(kid), subject=int(v.subject), relation=int(v.relation),
                obj=0 if link else int(v.obj), marker=tuple(float(x) for x in v.marker),
                status=cell.status, is_holder=kid in holder_kids,
                marker_valid=self.marker_valid(v.marker), is_link=link,
                link_target_kid=link_target_kid,
                link_subject=link_subject, link_relation=link_relation,
            ))

        return MVCCSnapshot(
            revision=int(self.revision), marker_dim=int(self.marker_dim), rows=tuple(rows),
            holder_of_key=tuple((key, int(kid)) for key, (kid, _) in holders.items()),
            resolved_view=tuple(resolved.items()),
            index_view=tuple((key, obj) for key, (obj, _) in resolved.items()),
            revision_token=RevisionToken(int(self.revision), self._freshness_fingerprint(), self._identity),
        )

    def bank(self, respect_markers: bool = False) -> Dict[str, np.ndarray]:
        """Tensors describing the layer for a neural model.

        All non-deleted, non-evicted cells are returned; ``active`` flags routing
        availability.  Marker validity is NOT applied
        here unless ``respect_markers`` is set — the neural model has to learn to
        reject unsigned payloads itself.
        """
        snapshot = self.snapshot()
        rows = snapshot.rows
        return {
            "kid": np.asarray([r.kid for r in rows], dtype=np.int64),
            "subject": np.asarray([r.subject for r in rows], dtype=np.int64),
            "relation": np.asarray([r.relation for r in rows], dtype=np.int64),
            "obj": np.asarray([r.obj for r in rows], dtype=np.int64),
            "marker": np.asarray([r.marker for r in rows], dtype=np.float32).reshape(len(rows), snapshot.marker_dim),
            "active": np.asarray([r.is_holder and (r.marker_valid or not respect_markers) for r in rows], dtype=bool),
            "is_link": np.asarray([r.is_link for r in rows], dtype=bool),
            "link_target_kid": np.asarray([r.link_target_kid for r in rows], dtype=np.int64),
            "link_subject": np.asarray([r.link_subject for r in rows], dtype=np.int64),
            "link_relation": np.asarray([r.link_relation for r in rows], dtype=np.int64),
        }

    # ------------------------------------------------------------------ replay / hashing
    @staticmethod
    def _json_value(value: Any) -> Any:
        """Convert state to a deterministic JSON value without losing ndarray bytes."""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            return {"dtype": a.dtype.str, "shape": list(a.shape), "bytes": a.tobytes().hex()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        if isinstance(value, dict):
            return {str(k): MVCCStore._json_value(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
        if isinstance(value, (list, tuple)):
            return [MVCCStore._json_value(v) for v in value]
        return value

    def _cell_state(self) -> List[Dict[str, Any]]:
        cells: List[Dict[str, Any]] = []
        for kid in sorted(self.cells):
            c = self.cells[kid]
            versions = []
            for v in c.versions:
                versions.append({
                    "version": self._json_value(v.version), "subject": self._json_value(v.subject),
                    "relation": self._json_value(v.relation), "obj": self._json_value(v.obj),
                    "marker": self._json_value(np.asarray(v.marker)),
                    "op_index": self._json_value(v.op_index), "kind": v.kind.value,
                    "target": self._json_value(v.target),
                })
            cells.append({
                "mapping_kid": self._json_value(kid), "kid": self._json_value(c.kid),
                "status": c.status.value,
                "active_version": self._json_value(c.active_version), "provenance": c.provenance,
                "tombstone_key": None if c.tombstone_key is None else list(c.tombstone_key),
                "versions": versions,
            })
        return cells

    def _config_state(self) -> Dict[str, Any]:
        return {
            "marker_dim": self._json_value(self.marker_dim), "seed": self._json_value(self.seed),
            "valid_radius": self._json_value(self.valid_radius),
            "marker_centre": self._json_value(np.asarray(self.marker_centre)),
            "content_markers": self._json_value(self.content_markers),
            "marker_key": self.marker_key.hex(), "max_link_depth": self._json_value(self.MAX_LINK_DEPTH),
        }

    @staticmethod
    def _digest_state(state: Dict[str, Any]) -> str:
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @_synchronized
    def logical_state_hash(self) -> str:
        """Hash current retained cells/config, deliberately excluding event history and RNG position.

        This is the compatibility oracle for counterfactual measurements that temporarily EVICT and
        RESTORE rows.  It answers "is the currently readable/retained state back?", not "did nothing
        happen?".  Use :meth:`state_hash` for replay/persistence identity.
        """
        return self._digest_state({
            "schema": "so.mvcc.logical.v2", "config": self._config_state(),
            "next_kid": self._json_value(self._next_kid), "cells": self._cell_state(),
        })

    @_synchronized
    def state_hash(self) -> str:
        """Complete canonical oracle for persistent state and deterministic future behaviour."""
        return self._digest_state({
            "schema": "so.mvcc.persistent.v2", "config": self._config_state(),
            "revision": self._json_value(self.revision), "next_kid": self._json_value(self._next_kid),
            "rng_state": self._json_value(self.rng.bit_generator.state),
            "log": self._json_value(self.log), "cells": self._cell_state(),
        })

    @classmethod
    def replay(cls, log: List[Tuple[str, Dict[str, Any]]], marker_dim: int, seed: int,
               valid_radius: float, marker_centre: Optional[np.ndarray] = None,
               content_markers: bool = False, marker_key: Optional[bytes] = None) -> "MVCCStore":
        store = cls(marker_dim=marker_dim, seed=seed, valid_radius=valid_radius, marker_centre=marker_centre,
                    content_markers=content_markers, marker_key=marker_key)
        # ``marker_centre`` supplied here is already the store's canonical, normalised persistent
        # value.  Normalising it a second time can change a final bit and therefore future markers.
        if marker_centre is not None:
            store.marker_centre = np.asarray(marker_centre, dtype=float).copy()
        for op, args in log:
            getattr(store, op)(**args)
        return store

    @_synchronized
    def clone_by_replay(self) -> "MVCCStore":
        return MVCCStore.replay(list(self.log), self.marker_dim, self.seed, self.valid_radius, self.marker_centre,
                                content_markers=self.content_markers, marker_key=self.marker_key)

    @_synchronized
    def marker_invariant_holds(self) -> bool:
        """Under ``content_markers``: every exported row's marker equals the marker derived from its
        exported content (a SHRED row: the derived invalid marker). Returns False for a store without
        the option, so a test cannot pass by running the wrong scheme. ``swap`` and ``replace`` are
        causal interventions that change a payload in place without re-signing; they break this on
        purpose and are outside the interface the option covers."""
        if not self.content_markers:
            return False
        for cell in self.cells.values():
            if cell.status in (Status.DELETED, Status.EVICTED) or not cell.versions:
                continue
            v = cell.version_obj(cell.active_version)
            want = self.derived_marker(self.row_content(v), valid=self.marker_valid(v.marker))
            if not np.array_equal(np.asarray(v.marker), want):
                return False
        return True
