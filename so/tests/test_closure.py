"""The deletion closure: what it must say about a canonical store and about a duplicated one."""
import numpy as np
import pytest

from so.closure import ClosureProfile, closure_profile, deletion_closure
from so.mvcc import MVCCStore, Status
from so.reference import ReferenceResolver
from so.world import Query, UNKNOWN


def _store():
    return MVCCStore(marker_dim=16, seed=0)


def test_a_single_record_has_closure_one():
    st = _store()
    st.write(3, 1, 7, provenance="w")
    c = deletion_closure(st, (3, 1))
    assert c.answer == 7 and c.size == 1


def test_the_measurement_leaves_the_store_as_it_found_it():
    """A statistic that edits the thing it measures is not a statistic."""
    st = _store()
    st.write(3, 1, 7, provenance="w")
    before = st.state_hash()
    deletion_closure(st, (3, 1))
    assert st.state_hash() == before
    assert ReferenceResolver(st).resolve(Query("fwd", 3, (1,), (0,))).answer == 7


def test_a_canonical_pod_has_closure_one_however_many_aliases_point_at_it():
    """The claim the symlink is for: k access paths, one object, one deletion."""
    st = _store()
    target = st.write(3, 1, 7, provenance="t")
    aliases = [st.link(10 + i, 1, target, provenance=f"a{i}") for i in range(5)]
    for i in range(5):
        assert ReferenceResolver(st).resolve(Query("fwd", 10 + i, (1,), (0,))).answer == 7
    # every alias's fact goes when the one object goes
    for i in range(5):
        assert deletion_closure(st, (10 + i, 1)).size == 1
    assert deletion_closure(st, (3, 1)).size == 1


def test_duplication_makes_the_closure_grow_with_the_number_of_copies():
    """The contrast: the same fact under k keys, stored k times, needs k deletions.

    Each key resolves through its OWN record, so removing one leaves the others readable -- which is
    a deletion anomaly in Codd's sense, measured here as a closure size rather than described.
    """
    st = _store()
    for i in range(5):
        st.write(10 + i, 1, 7, provenance=f"c{i}")
    for i in range(5):
        assert deletion_closure(st, (10 + i, 1)).size == 1     # per KEY it is still one record...
    prof = closure_profile(st, [(10 + i, 1) for i in range(5)])
    assert prof.n == 5 and prof.mean == 1.0
    # ...but the FACT "the object is 7" survives in four other records, which is what the profile of
    # the whole store has to show rather than the per-key number
    assert sum(1 for i in range(5)
               if ReferenceResolver(st).resolve(Query("fwd", 10 + i, (1,), (0,))).answer == 7) == 5


def test_a_derivable_fact_needs_more_than_one_record():
    """Two routes to the same answer means one deletion is not enough, which is the E-000019 case."""
    st = _store()
    st.write(3, 1, 7, provenance="direct")
    st.write(3, 2, 5, provenance="hop1")
    st.write(5, 3, 7, provenance="hop2")
    q = Query("fwd", 3, (1,), (0,))
    assert ReferenceResolver(st).resolve(q).answer == 7
    c = deletion_closure(st, (3, 1))
    assert c.size == 1                     # the direct key has one record...
    # ...and the 2-hop route still reaches 7, which the per-key closure does not see
    two_hop = Query("fwd", 3, (2, 3), (0, 0))
    st.evict(list(st.cells)[0])
    assert ReferenceResolver(st).resolve(two_hop).answer == 7


def test_the_profile_reports_a_distribution_not_an_average():
    st = _store()
    t = st.write(3, 1, 7, provenance="t")
    st.link(4, 1, t, provenance="a")
    st.write(5, 1, 9, provenance="w")
    prof = closure_profile(st, [(3, 1), (4, 1), (5, 1)])
    assert prof.n == 3
    assert prof.histogram() == {1: 3}
    assert "at size 1" in prof.summary()


def test_an_unresolvable_key_is_not_counted():
    st = _store()
    st.write(3, 1, 7, provenance="w")
    assert deletion_closure(st, (99, 3)).answer == UNKNOWN
    assert closure_profile(st, [(99, 3)]).n == 0


# ------------------------------------------------------- the fact-level closure (the symlink claim)

from so.closure import FactClosure, duplicate_keys, fact_closure, pod_keys   # noqa: E402


def _pod(st, n_aliases, subject=3, relation=1, obj=7):
    """One knowledge object with ``n_aliases`` access keys pointing at it."""
    target = st.write(subject, relation, obj, provenance="target")
    for i in range(n_aliases):
        st.link(100 + i, relation, target, provenance=f"alias{i}")
    return target


def _duplicates(st, n, relation=1, obj=7):
    """The same association written out ``n`` times under ``n`` keys."""
    return [st.write(100 + i, relation, obj, provenance=f"copy{i}") for i in range(n)]


def test_pod_keys_names_the_object_and_every_alias_that_points_at_it():
    st = _store()
    target = _pod(st, 4)
    keys = pod_keys(st, target)
    assert set(keys) == {(3, 1)} | {(100 + i, 1) for i in range(4)}


def test_duplicate_keys_names_every_key_that_currently_yields_the_object():
    st = _store()
    _duplicates(st, 4)
    st.write(9, 1, 55, provenance="unrelated")
    assert set(duplicate_keys(st, 7)) == {(100 + i, 1) for i in range(4)}
    assert duplicate_keys(st, 55) == ((9, 1),)


@pytest.mark.parametrize("n_aliases", [0, 1, 2, 5, 12])
def test_a_canonical_pod_has_fact_closure_one_however_many_keys_reach_it(n_aliases):
    """The central symlink claim: k access paths, one object, ONE deletion -- for every k."""
    st = _store()
    target = _pod(st, n_aliases)
    keys = pod_keys(st, target)
    assert len(keys) == n_aliases + 1
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 1
    assert fc.records == (target,)          # and it is the object itself that goes, not an alias
    assert fc.lower_bound == 1 and fc.optimal and not fc.exhausted


@pytest.mark.parametrize("n_copies", [1, 2, 5, 12])
def test_duplication_makes_the_fact_closure_grow_one_for_one_with_the_copies(n_copies):
    """The contrast, at the level where it matters: k copies of a fact cost k deletions."""
    st = _store()
    kids = _duplicates(st, n_copies)
    keys = duplicate_keys(st, 7)
    assert len(keys) == n_copies
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == n_copies
    assert set(fc.records) == set(kids)
    assert fc.lower_bound == n_copies and fc.optimal and not fc.exhausted


def test_the_two_stores_are_indistinguishable_per_key_and_differ_only_at_the_fact_level():
    """The headline, stated as one assertion pair.

    Same interface, same per-key closure, and an erasure cost that differs by a factor of k. A
    record-level guarantee cannot see this difference; the fact-level closure is what does.
    """
    k = 8
    pod, dup = _store(), _store()
    target = _pod(pod, k - 1)                      # k keys reaching one object
    _duplicates(dup, k)                            # k keys, k objects

    pod_ks, dup_ks = pod_keys(pod, target), duplicate_keys(dup, 7)
    assert len(pod_ks) == len(dup_ks) == k

    for st, keys in ((pod, pod_ks), (dup, dup_ks)):
        for key in keys:
            assert ReferenceResolver(st).resolve(Query("fwd", key[0], (key[1],), (0,))).answer == 7
            assert deletion_closure(st, key).size == 1      # per key: identical stores

    assert fact_closure(pod, pod_ks, obj=7).size == 1       # per fact: 1 ...
    assert fact_closure(dup, dup_ks, obj=7).size == k       # ... versus k


def test_the_fact_level_measurement_leaves_the_store_as_it_found_it():
    for build in (lambda st: _pod(st, 5), lambda st: _duplicates(st, 5)):
        st = _store()
        build(st)
        before = st.state_hash()
        keys = duplicate_keys(st, 7)
        fact_closure(st, keys, obj=7)
        assert st.state_hash() == before
        assert len(duplicate_keys(st, 7)) == len(keys)


def test_a_mixed_store_costs_the_pod_plus_one_per_stray_copy():
    """Partial normalization buys exactly the part it normalized, and no more."""
    st = _store()
    target = _pod(st, 4)                       # 5 keys sharing one object
    st.write(200, 1, 7, provenance="stray")    # one un-normalized copy of the same association
    keys = duplicate_keys(st, 7)
    assert len(keys) == 6
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 2 and fc.lower_bound == 2 and fc.optimal
    assert target in fc.records


def test_a_chain_of_aliases_still_costs_one_because_the_object_is_still_shared():
    """Dereference depth does not change the closure; sharing does."""
    st = _store()
    target = st.write(3, 1, 7, provenance="target")
    a = st.link(10, 1, target, provenance="a")
    b = st.link(11, 1, a, provenance="b")
    st.link(12, 1, b, provenance="c")
    keys = duplicate_keys(st, 7)
    assert len(keys) == 4
    fc = fact_closure(st, keys, obj=7)
    assert fc.size == 1 and fc.records == (target,) and fc.optimal


def test_removing_the_top_of_a_chain_is_not_enough_which_is_why_greedy_picks_the_object():
    """A check on the search itself: the alias is a valid cut for ONE key, never for the pod."""
    st = _store()
    target = st.write(3, 1, 7, provenance="target")
    a = st.link(10, 1, target, provenance="a")
    st.evict(a)
    assert ReferenceResolver(st).resolve(Query("fwd", 10, (1,), (0,))).answer == UNKNOWN
    assert ReferenceResolver(st).resolve(Query("fwd", 3, (1,), (0,))).answer == 7
    st.restore(a)
    assert fact_closure(st, pod_keys(st, target), obj=7).records == (target,)


def test_a_fact_nothing_reaches_has_an_empty_closure():
    st = _store()
    st.write(3, 1, 7, provenance="w")
    fc = fact_closure(st, [(99, 3)])
    assert fc.obj == UNKNOWN and fc.size == 0 and fc.records == ()


def test_the_object_is_inferred_from_the_keys_when_it_is_not_given():
    st = _store()
    target = _pod(st, 3)
    fc = fact_closure(st, pod_keys(st, target))
    assert fc.obj == 7 and fc.size == 1


def test_keys_that_do_not_yield_the_object_are_dropped_rather_than_counted():
    st = _store()
    target = _pod(st, 2)
    st.write(500, 1, 41, provenance="other")
    fc = fact_closure(st, list(pod_keys(st, target)) + [(500, 1), (999, 9)], obj=7)
    assert set(fc.keys) == set(pod_keys(st, target))
    assert fc.size == 1
    assert ReferenceResolver(st).resolve(Query("fwd", 500, (1,), (0,))).answer == 41


def test_the_lower_bound_is_a_certificate_and_not_a_restatement_of_the_answer():
    """It has to be computable before the search and it has to bind: k disjoint records, bound k."""
    st = _store()
    _duplicates(st, 6)
    from so.closure import _disjoint_lower_bound
    view = st.resolved_view(respect_markers=True)
    derivations = [frozenset(t) for (o, t) in view.values() if o == 7]
    assert _disjoint_lower_bound(derivations) == 6
    # a pod's derivations all share the object, so no two are disjoint and the bound is 1
    st2 = _store()
    t = _pod(st2, 5)
    view2 = st2.resolved_view(respect_markers=True)
    assert _disjoint_lower_bound([frozenset(tr) for (o, tr) in view2.values() if o == 7]) == 1
    assert t in fact_closure(st2, pod_keys(st2, t), obj=7).records


def test_the_search_reports_exhaustion_instead_of_lying_when_it_is_cut_short():
    st = _store()
    _duplicates(st, 5)
    fc = fact_closure(st, duplicate_keys(st, 7), obj=7, max_records=2)
    assert fc.exhausted and fc.size == 2 and not fc.optimal
    assert len(duplicate_keys(st, 7)) == 5      # and it still put the store back
