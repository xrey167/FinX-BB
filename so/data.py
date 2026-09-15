"""Training-time knowledge banks and query batches.

During training the world is re-sampled for every batch.  The neural core can
therefore never memorise a fact in its weights: the only stable signal is
*how to read* the mutable knowledge layer.  Every bank also contains cells in
non-trivial lifecycle states — revoked cells (routing removed), shredded cells
(payload present, marker invalid) and stale historical versions (inactive
duplicates of a key) — so that the model must learn the lifecycle semantics,
not just look-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .world import Query, UNKNOWN, World

PAD_HOP = -1

_BANK_ARRAY_FIELDS = frozenset({
    "subject", "relation", "obj", "marker", "active", "usable", "kid", "marker_valid",
    "routable", "is_link", "link_subject", "link_relation", "link_target_kid",
    "link_target_pos", "link_target_exact", "deref_routable",
})


def _immutable_array(value: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array whose ultimate buffer is immutable ``bytes``.

    Merely clearing NumPy's WRITEABLE flag is reversible when an array owns writable storage.
    Rebuilding from ``bytes`` makes ``setflags(write=True)`` fail at the buffer boundary while
    preserving the public shape and dtype.
    """
    owned = np.ascontiguousarray(np.array(value, copy=True))
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def valid_markers(rng: np.random.Generator, centre: np.ndarray, n: int, scale: float = 0.05) -> np.ndarray:
    m = centre[None, :] + rng.normal(scale=scale, size=(n, centre.shape[0]))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def invalid_markers(rng: np.random.Generator, centre: np.ndarray, n: int, min_dist: float = 0.7) -> np.ndarray:
    out = np.empty((n, centre.shape[0]))
    for i in range(n):
        while True:
            m = rng.normal(size=centre.shape[0])
            m = m / np.linalg.norm(m)
            if np.linalg.norm(m - centre) >= min_dist:
                out[i] = m
                break
    return out


@dataclass(frozen=True)
class Bank:
    """One immutable-revision tensor/export view plus its symbolic ground truth.

    ``active`` means that the row is the store-selected logical holder and can be routed to; it does
    not mean merely that its physical cell has raw ``Status.ACTIVE``.  Duplicate shadows can retain
    that status while remaining non-routable.  ``revision`` and ``freshness_token`` let a lifecycle
    consumer reject use after the source store changes.
    """

    subject: np.ndarray
    relation: np.ndarray
    obj: np.ndarray
    marker: np.ndarray
    active: np.ndarray            # one logical/routable holder per key; marker checked by ``usable``
    usable: np.ndarray            # active AND marker valid -> the ground-truth view
    kid: np.ndarray
    index_view: Dict[Tuple[int, int], int]
    kid_of_key: Dict[Tuple[int, int], int]   # key -> position in the bank (usable cells only)
    active_pos: Dict[Tuple[int, int], int]   # key -> position of a ROUTABLE cell (active; marker may be invalid)
    marker_valid: Optional[np.ndarray] = None  # per cell: is the marker signed? (control-plane truth for gate supervision)
    routable: Optional[np.ndarray] = None      # status-gated designs: ACTIVE or REVOKED cells are routable, stale/deleted are not
    routable_pos: Optional[Dict[Tuple[int, int], int]] = None
    is_link: Optional[np.ndarray] = None       # E-000015: this row is an alias; its payload is the TARGET'S KEY
    link_subject: Optional[np.ndarray] = None
    link_relation: Optional[np.ndarray] = None
    trace_of_key: Optional[Dict[Tuple[int, int], Tuple[int, ...]]] = None   # resolution path as row positions
    link_target_kid: Optional[np.ndarray] = None  # exact LINK identity; -1 means BLANK, 0 means FACT
    link_target_pos: Optional[np.ndarray] = None  # exact target row, or -1 when absent/blank
    link_target_exact: Optional[np.ndarray] = None  # duplicate targets require constrained dereference
    deref_routable: Optional[np.ndarray] = None   # raw ACTIVE rows, including duplicate shadows
    revision: Optional[int] = None
    freshness_token: object = field(default=None, repr=False, compare=False)

    def __getattribute__(self, name):
        """Never expose the sealed array that backs a captured Bank.

        PyTorch deliberately permits zero-copy tensors over read-only NumPy arrays and can write
        through them despite NumPy's flag.  Returning another immutable bytes-backed snapshot makes
        that unsupported write affect only the caller's disposable copy, never Bank's sealed state.
        """
        value = object.__getattribute__(self, name)
        if (name in _BANK_ARRAY_FIELDS and value is not None
                and object.__getattribute__(self, "__dict__").get("_arrays_sealed", False)):
            return _immutable_array(value)
        return value

    @property
    def size(self) -> int:
        return int(self.subject.shape[0])

    def __post_init__(self) -> None:
        """Own and freeze every exported collection.

        ``torch.as_tensor`` can share NumPy storage, and the original maps were returned by reference.
        A consumer could therefore mutate a supposedly captured bank after freshness validation.  Bank
        keeps read-only private copies; :meth:`tensors` makes independent Torch copies as well.
        """
        for name in _BANK_ARRAY_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            owned = _immutable_array(value)
            object.__setattr__(self, name, owned)
        for name in ("index_view", "kid_of_key", "active_pos", "routable_pos", "trace_of_key"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "_arrays_sealed", True)

    def is_fresh(self, store) -> bool:
        """Whether this bank still names the current revision of its source store."""
        return self.freshness_token is not None and bool(store.is_current(self.freshness_token))

    def require_fresh(self, store) -> None:
        """Reject stale or cross-store consumption before a model uses this bank."""
        if not self.is_fresh(store):
            from .mvcc import StaleSnapshotError
            current = getattr(store, "revision", "unknown")
            raise StaleSnapshotError(
                f"bank revision {self.revision!r} is stale or belongs to another store; "
                f"current revision is {current!r}"
            )

    def tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        return {
            "subject": torch.tensor(np.array(self.subject), dtype=torch.long, device=device),
            "relation": torch.tensor(np.array(self.relation), dtype=torch.long, device=device),
            "obj": torch.tensor(np.array(self.obj), dtype=torch.long, device=device),
            "marker": torch.tensor(np.array(self.marker), dtype=torch.float32, device=device),
            "active": torch.tensor(np.array(self.active), dtype=torch.bool, device=device),
            "marker_valid": torch.tensor(np.array(
                self.marker_valid if self.marker_valid is not None else np.ones(self.size, dtype=bool)),
                dtype=torch.bool, device=device),
            "routable": torch.tensor(np.array(
                self.routable if self.routable is not None else self.active), dtype=torch.bool, device=device),
            "is_link": torch.tensor(np.array(
                self.is_link if self.is_link is not None else np.zeros(self.size, dtype=bool)),
                dtype=torch.bool, device=device),
            "link_target_kid": torch.tensor(np.array(
                self.link_target_kid if self.link_target_kid is not None else np.zeros(self.size, dtype=np.int64),
            ), dtype=torch.long, device=device),
            "link_target_pos": torch.tensor(np.array(
                self.link_target_pos if self.link_target_pos is not None else np.full(self.size, -1, dtype=np.int64),
            ), dtype=torch.long, device=device),
            "link_target_exact": torch.tensor(np.array(
                self.link_target_exact if self.link_target_exact is not None else np.zeros(self.size, dtype=bool),
            ), dtype=torch.bool, device=device),
            "deref_routable": torch.tensor(np.array(
                self.deref_routable if self.deref_routable is not None else self.active,
            ), dtype=torch.bool, device=device),
            "link_subject": torch.tensor(np.array(self.link_subject if self.link_subject is not None
                                          else np.zeros(self.size, dtype=np.int64)), dtype=torch.long, device=device),
            "link_relation": torch.tensor(np.array(self.link_relation if self.link_relation is not None
                                           else np.zeros(self.size, dtype=np.int64)), dtype=torch.long, device=device),
        }


def bank_from_world(rng: np.random.Generator, world: World, centre: np.ndarray, p_revoked: float = 0.10,
                    p_shred: float = 0.05, p_stale: float = 0.05) -> Bank:
    """A training bank: the world's facts with random lifecycle states."""
    n = len(world.facts)
    subject = np.array([f.subject for f in world.facts], dtype=np.int64)
    relation = np.array([f.relation for f in world.facts], dtype=np.int64)
    obj = np.array([f.obj for f in world.facts], dtype=np.int64)
    revoked = rng.random(n) < p_revoked
    shred = (~revoked) & (rng.random(n) < p_shred)
    marker = valid_markers(rng, centre, n)
    if shred.any():
        marker[shred] = invalid_markers(rng, centre, int(shred.sum()))
    active = ~revoked
    # stale historical versions: inactive duplicates of existing keys with a different object
    n_stale = int(rng.binomial(n, p_stale))
    if n_stale:
        pick = rng.choice(n, size=n_stale, replace=False)
        s_sub, s_rel = subject[pick], relation[pick]
        s_obj = (obj[pick] + 1 + rng.integers(0, world.n_entities - 1, size=n_stale)) % world.n_entities
        subject = np.concatenate([subject, s_sub]); relation = np.concatenate([relation, s_rel])
        obj = np.concatenate([obj, s_obj]); marker = np.concatenate([marker, valid_markers(rng, centre, n_stale)])
        active = np.concatenate([active, np.zeros(n_stale, dtype=bool)])
        shred = np.concatenate([shred, np.zeros(n_stale, dtype=bool)])
    usable = active & ~shred
    index_view = {(int(s), int(r)): int(o) for s, r, o, u in zip(subject, relation, obj, usable) if u}
    kid_of_key = {(int(s), int(r)): int(i) for i, (s, r, u) in enumerate(zip(subject, relation, usable)) if u}
    active_pos = {(int(s), int(r)): int(i) for i, (s, r, a) in enumerate(zip(subject, relation, active)) if a}
    routable = np.concatenate([np.ones(n, dtype=bool), np.zeros(subject.shape[0] - n, dtype=bool)])   # stale rows are not
    routable_pos = {(int(s), int(r)): int(i) for i, (s, r, rt) in enumerate(zip(subject, relation, routable)) if rt}
    return Bank(subject, relation, obj, marker.astype(np.float32), active, usable, np.arange(subject.shape[0]),
                index_view, kid_of_key, active_pos, marker_valid=~shred, routable=routable, routable_pos=routable_pos)


def bank_from_store(store, respect_markers: bool = False) -> Bank:
    """Bank view of a real ``MVCCStore`` (used for evaluation and lifecycle operations).

    Every non-evicted row remains available as physical/supervision data, but routing maps expose at
    most one row per key.  The store-selected ACTIVE holder wins; only when no ACTIVE holder exists
    does a status-gated reader route to the first REVOKED row to learn/refuse that status.  A shadow
    row therefore cannot disagree with the symbolic resolver or become live merely because the
    selected holder was shredded.
    """
    snapshot = store.snapshot()
    rows = snapshot.rows
    subject = np.asarray([row.subject for row in rows], dtype=np.int64)
    relation = np.asarray([row.relation for row in rows], dtype=np.int64)
    obj = np.asarray([row.obj for row in rows], dtype=np.int64)
    marker = np.asarray([row.marker for row in rows], dtype=np.float32).reshape(len(rows), snapshot.marker_dim)
    active = np.asarray([row.is_holder for row in rows], dtype=bool)
    valid = np.asarray([row.marker_valid for row in rows], dtype=bool)
    usable = valid & active
    kids = np.asarray([row.kid for row in rows], dtype=np.int64)
    # the view FOLLOWS aliases (a link row holds no object of its own) and the trace names every cell read
    row_of_kid = {row.kid: i for i, row in enumerate(rows)}
    resolved = dict(snapshot.resolved_view)
    index_view = {k: int(o) for k, (o, _) in resolved.items()}
    trace_of_key = {k: tuple(row_of_kid[int(x)] for x in tr if int(x) in row_of_kid) for k, (_, tr) in resolved.items()}
    kid_of_key = {(int(s), int(r)): int(i) for i, (s, r, u) in enumerate(zip(subject, relation, usable)) if u}
    active_pos = {(int(s), int(r)): int(i) for i, (s, r, a) in enumerate(zip(subject, relation, active)) if a}
    # Status-gated supervision keeps one REVOKED row routable only when the key has no ACTIVE holder.
    # Shadow rows remain in the tensors, but never compete in the routing map.
    routable = np.zeros(kids.shape[0], dtype=bool)
    chosen = set()
    for i, (s, r, is_active) in enumerate(zip(subject, relation, active)):
        key = (int(s), int(r))
        if is_active and key not in chosen:
            routable[i] = True
            chosen.add(key)
    for i, row in enumerate(rows):
        s, r = row.subject, row.relation
        key = (int(s), int(r))
        if key in chosen:
            continue
        if row.status.value == "REVOKED":
            routable[i] = True
            chosen.add(key)
    routable_pos = {(int(s), int(r)): int(i)
                    for i, (s, r, rt) in enumerate(zip(subject, relation, routable)) if rt}
    key_counts: Dict[Tuple[int, int], int] = {}
    for row in rows:
        key = (int(row.subject), int(row.relation))
        key_counts[key] = key_counts.get(key, 0) + 1
    link_target_pos = np.asarray([
        row_of_kid.get(row.link_target_kid, -1) if row.is_link else -1 for row in rows
    ], dtype=np.int64)
    link_target_exact = np.asarray([
        bool(row.is_link and row.link_target_kid > 0 and
             (row_of_kid.get(row.link_target_kid, -1) < 0 or
              key_counts.get((int(row.link_subject), int(row.link_relation)), 0) > 1))
        for row in rows
    ], dtype=bool)
    deref_routable = np.asarray([row.status.value == "ACTIVE" for row in rows], dtype=bool)
    return Bank(subject, relation, obj, marker, active, usable, kids, index_view, kid_of_key,
                active_pos, marker_valid=valid, routable=routable, routable_pos=routable_pos,
                is_link=np.asarray([row.is_link for row in rows], dtype=bool),
                link_subject=np.asarray([row.link_subject for row in rows], dtype=np.int64),
                link_relation=np.asarray([row.link_relation for row in rows], dtype=np.int64),
                trace_of_key=trace_of_key,
                link_target_kid=np.asarray([row.link_target_kid for row in rows], dtype=np.int64),
                link_target_pos=link_target_pos, link_target_exact=link_target_exact,
                deref_routable=deref_routable,
                revision=snapshot.revision, freshness_token=snapshot.revision_token)


def failing_hop_target(bank: Bank, q: Query, gt, status_gated: bool = False) -> int:
    """Routing target for the hop at which a forward path fails.

    If a *routable* cell holds that key (a shredded cell: active, marker invalid), the model
    must attend to it and discover the closed gate, so the target is that cell; otherwise
    (key absent or cell revoked) the target is the null cell (-1).  In a status-gated design
    (``status_gated``) revoked cells stay routable and read as unknown through the gate, so
    they are targets too.
    """
    cur = q.start
    for e in gt.edges:
        cur = bank.index_view[e]
    key = (cur, q.path[len(gt.edges)])
    if status_gated and bank.routable_pos is not None:
        return bank.routable_pos.get(key, -1)
    return bank.active_pos.get(key, -1)


def reverse_target(bank: Bank, q: Query, gt) -> int:
    """Routing target for a reverse query: the usable cell, else a single routable (shredded) cell, else null."""
    if gt.edges:
        return bank.kid_of_key[gt.edges[0]]
    r, o = q.path[0], q.start
    hits = [i for i, (rr, oo, a) in enumerate(zip(bank.relation, bank.obj, bank.active)) if a and rr == r and oo == o]
    return hits[0] if len(hits) == 1 else -1


@dataclass
class Batch:
    mode: torch.Tensor      # (B,)  0 = fwd, 1 = rev
    start: torch.Tensor     # (B,)
    rels: torch.Tensor      # (B, H) surface tokens, PAD = n_surface
    hop_valid: torch.Tensor  # (B, H) bool
    target: torch.Tensor    # (B,) entity id, UNKNOWN -> n_entities
    route: torch.Tensor     # (B, H) bank position of the cell to read at each hop, -1 = null cell, -2 = ignore
    queries: List[Query]


def encode_queries(queries: List[Query], bank: Bank, world: World, max_hops: int,
                   device: str = "cpu") -> Batch:
    B = len(queries)
    mode = np.zeros(B, dtype=np.int64)
    start = np.zeros(B, dtype=np.int64)
    rels = np.full((B, max_hops), world.n_surface, dtype=np.int64)
    hop_valid = np.zeros((B, max_hops), dtype=bool)
    target = np.zeros(B, dtype=np.int64)
    route = np.full((B, max_hops), -2, dtype=np.int64)
    for i, q in enumerate(queries):
        mode[i] = 0 if q.mode == "fwd" else 1
        start[i] = q.start
        rels[i, : q.hops] = q.surface
        hop_valid[i, : q.hops] = True
        gt = world.answer(q, bank.index_view)
        target[i] = world.n_entities if gt.answer == UNKNOWN else gt.answer
        if q.mode == "fwd":
            for t, e in enumerate(gt.edges):
                route[i, t] = bank.kid_of_key[e]
            if gt.answer == UNKNOWN and len(gt.edges) < q.hops:
                route[i, len(gt.edges)] = failing_hop_target(bank, q, gt)
        else:
            route[i, 0] = reverse_target(bank, q, gt)
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return Batch(t(mode, torch.long), t(start, torch.long), t(rels, torch.long), t(hop_valid, torch.bool),
                 t(target, torch.long), t(route, torch.long), queries)


def sample_training_queries(rng: np.random.Generator, world: World, bank: Bank, batch_size: int,
                            mix: Dict[str, float]) -> List[Query]:
    """``mix`` maps ``"fwd1" | "fwd2" | "fwd3" | "rev1"`` to fractions."""
    out: List[Query] = []
    kinds = list(mix.keys())
    counts = np.round(np.array([mix[k] for k in kinds]) * batch_size).astype(int)
    counts[0] += batch_size - counts.sum()
    for kind, n in zip(kinds, counts):
        if n <= 0:
            continue
        mode, hops = kind[:3], int(kind[3])
        # half answerable, half natural mix so that UNKNOWN is well represented but not dominant
        n_ans = n // 2
        out += world.sample_queries(rng, n_ans, hops, mode, require_answer=True, index=bank.index_view)
        out += world.sample_queries(rng, n - n_ans, hops, mode, require_answer=None, index=bank.index_view)
    rng.shuffle(out)
    return out[:batch_size]
