import numpy as np

from so.mvcc import MVCCStore
from so.reference import ReferenceResolver, load_world
from so.world import UNKNOWN, World


def test_resolver_matches_world_ground_truth():
    rng = np.random.default_rng(0)
    w = World.sample(rng, 64, 4, 240, 2)
    store = MVCCStore(seed=0)
    kids = load_world(store, w)
    res = ReferenceResolver(store)
    for hops in (1, 2, 3):
        for q in w.sample_queries(rng, 40, hops, "fwd"):
            gt = w.answer(q)
            r = res.resolve(q)
            assert r.answer == gt.answer
            assert r.trace == tuple(kids[e] for e in gt.edges)
    for q in w.sample_queries(rng, 20, 1, "rev"):
        assert res.resolve(q).answer == w.answer(q).answer


def test_revoke_breaks_only_the_targeted_path():
    rng = np.random.default_rng(1)
    w = World.sample(rng, 64, 4, 240, 1)
    store = MVCCStore(seed=1)
    kids = load_world(store, w)
    res = ReferenceResolver(store)
    q = w.sample_queries(rng, 1, 2, "fwd", require_answer=True)[0]
    gt = w.answer(q)
    target = kids[gt.edges[1]]
    others = [qq for qq in w.sample_queries(rng, 50, 2, "fwd") if gt.edges[1] not in w.answer(qq).edges]
    before = [res.resolve(qq).answer for qq in others]
    store.revoke(target)
    assert res.resolve(q).answer == UNKNOWN
    assert [res.resolve(qq).answer for qq in others] == before
    store.restore(target)
    assert res.resolve(q).answer == gt.answer
