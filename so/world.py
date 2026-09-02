"""Synthetic knowledge worlds with exact ground truth.

A world is a set of functional facts ``(subject, relation) -> object`` over
``n_entities`` entities and ``n_relations`` relation types.  Every fact is one
knowledge unit ("cell").  Because the world is generated, provenance,
dependency paths, contradictions and updates are all exactly known, which is
what makes falsification possible (ledger sections 12 and 18).

Relations have several surface forms ("paraphrases"): surface token
``relation * n_synonyms + k`` denotes relation ``relation`` for every
``k < n_synonyms``.  The neural model only ever sees surface tokens, so a
deletion that generalises across paraphrases is a real test, not a tautology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

UNKNOWN = -1  # answer code for "no active knowledge path"


@dataclass(frozen=True)
class Fact:
    subject: int
    relation: int
    obj: int

    @property
    def key(self) -> Tuple[int, int]:
        return (self.subject, self.relation)


@dataclass(frozen=True)
class Query:
    """A question against a world.

    ``mode`` is ``"fwd"`` (follow ``path`` from ``start``) or ``"rev"`` (find the
    unique subject with ``(subject, path[0]) -> start``).  ``surface`` holds the
    surface (paraphrase) tokens actually shown to a neural model; ``path`` holds
    the underlying relation ids.
    """

    mode: str
    start: int
    path: Tuple[int, ...]
    surface: Tuple[int, ...]

    @property
    def hops(self) -> int:
        return len(self.path)


@dataclass(frozen=True)
class GroundTruth:
    answer: int                      # entity id or UNKNOWN
    edges: Tuple[Tuple[int, int], ...]  # (subject, relation) keys used, in order


class World:
    def __init__(self, n_entities: int, n_relations: int, n_synonyms: int, facts: Sequence[Fact]):
        self.n_entities = int(n_entities)
        self.n_relations = int(n_relations)
        self.n_synonyms = int(n_synonyms)
        self.facts: List[Fact] = list(facts)
        self.index: Dict[Tuple[int, int], int] = {}
        for f in self.facts:
            if f.key in self.index:
                raise ValueError(f"duplicate functional key {f.key}")
            self.index[f.key] = f.obj

    # ------------------------------------------------------------------ construction
    @staticmethod
    def sample(rng: np.random.Generator, n_entities: int, n_relations: int, n_facts: int,
               n_synonyms: int = 1) -> "World":
        """Sample ``n_facts`` distinct functional facts uniformly over all (s, r) pairs."""
        n_pairs = n_entities * n_relations
        if n_facts > n_pairs:
            raise ValueError(f"cannot place {n_facts} functional facts into {n_pairs} (subject, relation) pairs")
        chosen = rng.choice(n_pairs, size=n_facts, replace=False)
        objs = rng.integers(0, n_entities, size=n_facts)
        facts = [Fact(int(p // n_relations), int(p % n_relations), int(o)) for p, o in zip(chosen, objs)]
        return World(n_entities, n_relations, n_synonyms, facts)

    def surface_of(self, relation: int, synonym: int) -> int:
        return relation * self.n_synonyms + synonym

    def relation_of_surface(self, surface: int) -> int:
        return surface // self.n_synonyms

    @property
    def n_surface(self) -> int:
        return self.n_relations * self.n_synonyms

    # ------------------------------------------------------------------ ground truth
    def follow(self, start: int, path: Sequence[int],
               index: Optional[Dict[Tuple[int, int], int]] = None) -> GroundTruth:
        """Follow ``path`` from ``start`` through ``index`` (default: this world's facts)."""
        idx = self.index if index is None else index
        cur = start
        edges: List[Tuple[int, int]] = []
        for r in path:
            key = (cur, r)
            if key not in idx:
                return GroundTruth(UNKNOWN, tuple(edges))
            edges.append(key)
            cur = idx[key]
        return GroundTruth(cur, tuple(edges))

    def reverse(self, relation: int, obj: int,
                index: Optional[Dict[Tuple[int, int], int]] = None) -> GroundTruth:
        """Unique subject with (subject, relation) -> obj, else UNKNOWN."""
        idx = self.index if index is None else index
        subjects = [s for (s, r), o in idx.items() if r == relation and o == obj]
        if len(subjects) != 1:
            return GroundTruth(UNKNOWN, tuple())
        return GroundTruth(subjects[0], ((subjects[0], relation),))

    def answer(self, q: Query, index: Optional[Dict[Tuple[int, int], int]] = None) -> GroundTruth:
        if q.mode == "fwd":
            return self.follow(q.start, q.path, index)
        if q.mode == "rev":
            return self.reverse(q.path[0], q.start, index)
        raise ValueError(q.mode)

    # ------------------------------------------------------------------ query sampling
    def make_query(self, rng: np.random.Generator, mode: str, start: int, path: Sequence[int]) -> Query:
        surface = tuple(int(self.surface_of(r, int(rng.integers(0, self.n_synonyms)))) for r in path)
        return Query(mode, int(start), tuple(int(r) for r in path), surface)

    def sample_queries(self, rng: np.random.Generator, n: int, hops: int, mode: str = "fwd",
                       require_answer: Optional[bool] = None,
                       index: Optional[Dict[Tuple[int, int], int]] = None) -> List[Query]:
        """Sample ``n`` queries with ``hops`` relations.

        ``require_answer=True`` keeps only answerable queries, ``False`` only
        unanswerable ones, ``None`` keeps whatever comes out (natural mix).
        """
        out: List[Query] = []
        idx = self.index if index is None else index
        keys = list(idx.keys())
        attempts = 0
        while len(out) < n and attempts < 200 * n + 1000:
            attempts += 1
            if mode == "fwd":
                if require_answer:
                    # walk from an existing edge so that answerable paths are found efficiently
                    s, r = keys[int(rng.integers(0, len(keys)))]
                    path = [r]
                    cur = idx[(s, r)]
                    ok = True
                    for _ in range(hops - 1):
                        nxt = [rr for rr in range(self.n_relations) if (cur, rr) in idx]
                        if not nxt:
                            ok = False
                            break
                        rr = int(nxt[int(rng.integers(0, len(nxt)))])
                        path.append(rr)
                        cur = idx[(cur, rr)]
                    if not ok:
                        continue
                    q = self.make_query(rng, "fwd", s, path)
                else:
                    s = int(rng.integers(0, self.n_entities))
                    path = [int(x) for x in rng.integers(0, self.n_relations, size=hops)]
                    q = self.make_query(rng, "fwd", s, path)
            elif mode == "rev":
                if hops != 1:
                    raise ValueError("reverse queries are single-hop")
                if require_answer is False or (require_answer is None and rng.random() < 0.25):
                    r, o = int(rng.integers(0, self.n_relations)), int(rng.integers(0, self.n_entities))
                else:
                    s, r = keys[int(rng.integers(0, len(keys)))]
                    o = idx[(s, r)]
                n_hits = sum(1 for (ss, rr), oo in idx.items() if rr == r and oo == o)
                if n_hits > 1:
                    continue  # ambiguous reverse questions are excluded (not a lifecycle question)
                q = self.make_query(rng, "rev", o, [r])
            else:
                raise ValueError(mode)
            gt = self.answer(q, idx)
            if require_answer is True and gt.answer == UNKNOWN:
                continue
            if require_answer is False and gt.answer != UNKNOWN:
                continue
            out.append(q)
        return out

    # ------------------------------------------------------------------ structures
    def alternative_path_pairs(self, rng: np.random.Generator, n: int) -> List[Tuple[Query, Query, Tuple[int, int]]]:
        """Find ``n`` pairs of 2-hop queries from the same start to the same target.

        Returns ``(query_1, query_2, edge_to_revoke)`` where ``edge_to_revoke`` is the
        second edge of ``query_1``; revoking it must break ``query_1`` and must not
        affect ``query_2`` (architecture document section 9).
        """
        by_start: Dict[int, List[Tuple[Tuple[int, int], int]]] = {}
        for s in range(self.n_entities):
            routes = []
            for r1 in range(self.n_relations):
                if (s, r1) not in self.index:
                    continue
                mid = self.index[(s, r1)]
                for r2 in range(self.n_relations):
                    if (mid, r2) in self.index:
                        routes.append(((r1, r2), self.index[(mid, r2)]))
            by_start[s] = routes
        candidates = []
        for s, routes in by_start.items():
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j:
                        continue
                    (p1, t1), (p2, t2) = routes[i], routes[j]
                    if t1 != t2 or p1[0] == p2[0]:
                        continue  # need the same target through a different first edge
                    mid1 = self.index[(s, p1[0])]
                    mid2 = self.index[(s, p2[0])]
                    e1 = (mid1, p1[1])
                    if e1 == (mid2, p2[1]) or e1 == (s, p2[0]):
                        continue  # the revoked edge must not lie on the second path
                    candidates.append((s, p1, p2, e1))
        if not candidates:
            return []
        pick = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        out = []
        for k in pick:
            s, p1, p2, e1 = candidates[int(k)]
            out.append((self.make_query(rng, "fwd", s, p1), self.make_query(rng, "fwd", s, p2), e1))
        return out

    def derivable_shortcuts(self, rng: np.random.Generator, n: int) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """Triples ``(direct_edge, edge_1, edge_2)`` where the object of ``direct_edge`` equals
        the 2-hop result through ``edge_1`` then ``edge_2`` from the same subject.  This is the
        ledger's K3 = K1 + K2 dependency situation (section 23).  Degenerate triples (the
        intermediate entity equal to the subject, or an edge reused inside the triple) are
        excluded, and the returned triples share no edge with each other."""
        found = []
        for (s, r), o in self.index.items():
            for r1 in range(self.n_relations):
                if r1 == r or (s, r1) not in self.index:
                    continue
                mid = self.index[(s, r1)]
                if mid == s:
                    continue
                for r2 in range(self.n_relations):
                    e2 = (mid, r2)
                    if e2 in self.index and self.index[e2] == o and e2 != (s, r) and e2 != (s, r1):
                        found.append(((s, r), (s, r1), e2))
        if not found:
            return []
        order = rng.permutation(len(found))
        used: set = set()
        out = []
        for k in order:
            t = found[int(k)]
            if any(e in used for e in t):
                continue
            used.update(t)
            out.append(t)
            if len(out) >= n:
                break
        return out


def free_keys(world: World) -> List[Tuple[int, int]]:
    return [(s, r) for s in range(world.n_entities) for r in range(world.n_relations) if (s, r) not in world.index]


def inject_alternative_paths(rng: np.random.Generator, world: World, n_structures: int) -> World:
    """Return a new world in which ``n_structures`` explicit alternative-path structures
    ``A -r1-> B -r2-> C`` and ``A -r3-> X -r4-> C`` (r1 != r3) were added on free keys.

    Keys are sampled from the *free* key set so that dense worlds work; the caller
    should normally build these structures before filling the world randomly.
    """
    facts = list(world.facts)
    index = dict(world.index)
    ne, nr = world.n_entities, world.n_relations

    def free_relations(entity: int) -> List[int]:
        return [r for r in range(nr) if (entity, r) not in index]

    made = 0
    attempts = 0
    while made < n_structures and attempts < 200 * n_structures + 1000:
        attempts += 1
        a = int(rng.integers(0, ne))
        fa = free_relations(a)
        if len(fa) < 2:
            continue
        r1, r3 = [int(v) for v in rng.choice(fa, size=2, replace=False)]
        b, x, c = [int(v) for v in rng.choice([e for e in range(ne) if e != a], size=3, replace=False)]
        fb, fx = free_relations(b), free_relations(x)
        if not fb or not fx:
            continue
        r2, r4 = int(fb[int(rng.integers(0, len(fb)))]), int(fx[int(rng.integers(0, len(fx)))])
        new = {(a, r1): b, (b, r2): c, (a, r3): x, (x, r4): c}
        if len(new) < 4 or any(k in index for k in new):
            continue
        for (s, r), o in new.items():
            facts.append(Fact(s, r, o))
            index[(s, r)] = o
        made += 1
    if made < n_structures:
        raise RuntimeError(f"could only place {made} of {n_structures} alternative-path structures")
    return World(ne, nr, world.n_synonyms, facts)


def fill_random(rng: np.random.Generator, world: World, n_total: int) -> World:
    """Add uniformly random functional facts on free keys until the world has ``n_total`` facts."""
    need = n_total - len(world.facts)
    if need < 0:
        raise ValueError(f"world already has {len(world.facts)} > {n_total} facts")
    free = free_keys(world)
    if need > len(free):
        raise ValueError(f"need {need} free keys, only {len(free)} available")
    pick = rng.choice(len(free), size=need, replace=False)
    facts = list(world.facts) + [Fact(free[int(i)][0], free[int(i)][1], int(rng.integers(0, world.n_entities))) for i in pick]
    return World(world.n_entities, world.n_relations, world.n_synonyms, facts)
