import numpy as np

from so.world import UNKNOWN, Fact, World, fill_random, inject_alternative_paths


def test_sample_is_functional_and_sized():
    rng = np.random.default_rng(0)
    w = World.sample(rng, n_entities=64, n_relations=4, n_facts=200, n_synonyms=2)
    assert len(w.facts) == 200
    assert len(w.index) == 200
    assert all(0 <= f.obj < 64 for f in w.facts)


def test_follow_and_reverse():
    w = World(8, 2, 1, [Fact(0, 0, 1), Fact(1, 1, 2), Fact(2, 0, 3), Fact(5, 1, 2)])
    assert w.follow(0, [0]).answer == 1
    assert w.follow(0, [0, 1]).answer == 2
    assert w.follow(0, [0, 1]).edges == ((0, 0), (1, 1))
    assert w.follow(0, [0, 1, 0]).answer == 3
    assert w.follow(0, [1]).answer == UNKNOWN
    assert w.follow(0, [0, 0]).answer == UNKNOWN  # (1, 0) missing
    assert w.reverse(0, 1).answer == 0
    assert w.reverse(1, 2).answer == UNKNOWN  # two subjects (1 and 5) -> ambiguous


def test_surface_paraphrases_roundtrip():
    w = World(4, 3, 2, [])
    for r in range(3):
        for k in range(2):
            assert w.relation_of_surface(w.surface_of(r, k)) == r
    assert w.n_surface == 6


def test_sample_queries_respect_answerability():
    rng = np.random.default_rng(1)
    w = World.sample(rng, 64, 4, 240, 2)
    for hops in (1, 2, 3):
        qs = w.sample_queries(rng, 50, hops, "fwd", require_answer=True)
        assert len(qs) == 50 and all(w.answer(q).answer != UNKNOWN for q in qs)
        assert all(q.hops == hops for q in qs)
        assert all(w.relation_of_surface(s) == r for q in qs for s, r in zip(q.surface, q.path))
    broken = w.sample_queries(rng, 30, 2, "fwd", require_answer=False)
    assert len(broken) == 30 and all(w.answer(q).answer == UNKNOWN for q in broken)
    rev = w.sample_queries(rng, 20, 1, "rev")
    assert len(rev) == 20 and all(q.mode == "rev" for q in rev)


def test_inject_alternative_paths_and_pairs():
    rng = np.random.default_rng(2)
    base = World.sample(rng, 128, 4, 300, 1)
    w = inject_alternative_paths(rng, base, 20)
    assert len(w.facts) == 380
    pairs = w.alternative_path_pairs(rng, 15)
    assert len(pairs) == 15
    for q1, q2, edge in pairs:
        a1, a2 = w.answer(q1), w.answer(q2)
        assert a1.answer == a2.answer != UNKNOWN
        assert edge in a1.edges and edge not in a2.edges
        assert q1.start == q2.start


def test_derivable_shortcuts():
    w = World(6, 3, 1, [Fact(0, 0, 1), Fact(1, 1, 2), Fact(0, 2, 2)])
    triples = w.derivable_shortcuts(np.random.default_rng(0), 5)
    assert triples == [((0, 2), (0, 0), (1, 1))]


def test_dense_world_structures_first_then_fill():
    rng = np.random.default_rng(3)
    empty = World(256, 4, 2, [])
    structured = inject_alternative_paths(rng, empty, 25)
    assert len(structured.facts) == 100
    w = fill_random(rng, structured, 1000)
    assert len(w.facts) == 1000 and len(w.index) == 1000
    pairs = w.alternative_path_pairs(rng, 100)
    assert len(pairs) >= 25
