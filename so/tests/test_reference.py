import numpy as np
import threading
from types import MappingProxyType

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


def test_cache_publishes_revision_and_view_from_one_snapshot_during_update():
    store = MVCCStore(seed=0)
    kid = store.write(3, 1, 7)
    resolver = ReferenceResolver(store)
    captured = threading.Event()
    release = threading.Event()
    original_snapshot = store.snapshot
    original_resolved_view = store.resolved_view

    # The old resolver called resolved_view and then separately read revision; the new resolver calls
    # snapshot.  Hook both seams so the same test deterministically exposes the former mixed-cache bug.
    def delayed_snapshot():
        snapshot = original_snapshot()
        captured.set()
        assert release.wait(timeout=5)
        return snapshot

    def delayed_resolved_view(*args, **kwargs):
        view = original_resolved_view(*args, **kwargs)
        captured.set()
        assert release.wait(timeout=5)
        return view

    store.snapshot = delayed_snapshot
    store.resolved_view = delayed_resolved_view
    outcome = {}

    def read_view():
        outcome["view"] = resolver.view()

    thread = threading.Thread(target=read_view)
    thread.start()
    assert captured.wait(timeout=5)
    store.update(kid, 9)
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert outcome["view"][(3, 1)][0] == 7
    assert resolver.view()[(3, 1)][0] == 9


def test_resolver_view_is_immutable_to_callers():
    store = MVCCStore(seed=0)
    store.write(3, 1, 7)
    view = ReferenceResolver(store).view()
    assert isinstance(view, MappingProxyType)
    try:
        view[(3, 1)] = (9, ())
    except TypeError:
        pass
    else:
        raise AssertionError("resolver cache escaped as a mutable mapping")


def test_resolver_invalidates_cache_after_direct_cell_mutation_without_revision_change():
    store = MVCCStore(seed=0)
    kid = store.write(3, 1, 7)
    resolver = ReferenceResolver(store)
    assert resolver.view()[(3, 1)][0] == 7
    revision = store.revision
    store.cells[kid].active.obj = 9
    assert store.revision == revision
    assert resolver.view()[(3, 1)][0] == 9
