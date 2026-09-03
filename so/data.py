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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .world import Query, UNKNOWN, World

PAD_HOP = -1


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


@dataclass
class Bank:
    """Tensor view of a knowledge layer plus the symbolic ground-truth view that goes with it."""

    subject: np.ndarray
    relation: np.ndarray
    obj: np.ndarray
    marker: np.ndarray
    active: np.ndarray            # routing available (status ACTIVE)
    usable: np.ndarray            # active AND marker valid -> the ground-truth view
    kid: np.ndarray
    index_view: Dict[Tuple[int, int], int]
    kid_of_key: Dict[Tuple[int, int], int]   # key -> position in the bank (usable cells only)
    active_pos: Dict[Tuple[int, int], int]   # key -> position of a ROUTABLE cell (active; marker may be invalid)
    marker_valid: Optional[np.ndarray] = None  # per cell: is the marker signed? (control-plane truth for gate supervision)

    @property
    def size(self) -> int:
        return int(self.subject.shape[0])

    def tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        return {
            "subject": torch.as_tensor(self.subject, dtype=torch.long, device=device),
            "relation": torch.as_tensor(self.relation, dtype=torch.long, device=device),
            "obj": torch.as_tensor(self.obj, dtype=torch.long, device=device),
            "marker": torch.as_tensor(self.marker, dtype=torch.float32, device=device),
            "active": torch.as_tensor(self.active, dtype=torch.bool, device=device),
            "marker_valid": torch.as_tensor(self.marker_valid if self.marker_valid is not None else np.ones(self.size, dtype=bool),
                                            dtype=torch.bool, device=device),
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
    return Bank(subject, relation, obj, marker.astype(np.float32), active, usable, np.arange(subject.shape[0]),
                index_view, kid_of_key, active_pos, marker_valid=~shred)


def bank_from_store(store, respect_markers: bool = False) -> Bank:
    """Bank view of a real ``MVCCStore`` (used for evaluation and lifecycle operations)."""
    b = store.bank(respect_markers=False)
    valid = (np.linalg.norm(b["marker"].astype(float) - store.marker_centre[None, :], axis=1) <= store.valid_radius) \
        if b["marker"].shape[0] else np.zeros(0, dtype=bool)
    usable = valid & b["active"]
    index_view = {(int(s), int(r)): int(o) for s, r, o, u in zip(b["subject"], b["relation"], b["obj"], usable) if u}
    kid_of_key = {(int(s), int(r)): int(i) for i, (s, r, u) in enumerate(zip(b["subject"], b["relation"], usable)) if u}
    active_pos = {(int(s), int(r)): int(i) for i, (s, r, a) in enumerate(zip(b["subject"], b["relation"], b["active"])) if a}
    return Bank(b["subject"], b["relation"], b["obj"], b["marker"], b["active"], usable, b["kid"], index_view, kid_of_key,
                active_pos, marker_valid=valid)


def failing_hop_target(bank: Bank, q: Query, gt) -> int:
    """Routing target for the hop at which a forward path fails.

    If a *routable* cell holds that key (a shredded cell: active, marker invalid), the model
    must attend to it and discover the closed gate, so the target is that cell; otherwise
    (key absent or cell revoked) the target is the null cell (-1).
    """
    cur = q.start
    for e in gt.edges:
        cur = bank.index_view[e]
    key = (cur, q.path[len(gt.edges)])
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
