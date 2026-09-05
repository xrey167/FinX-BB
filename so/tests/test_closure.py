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

    Each key resolves through its OWN record, so removing one leaves the others readable. In Codd's
    vocabulary that is the MODIFICATION anomaly applied to a delete -- his DELETION anomaly is the
    opposite failure, unintended loss -- and it is measured here as a closure size.
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


def _survival_after_evicting_the_object(st, keys, obj, target):
    """Run the store's OWN resolver on the post-deletion store and count. No model anywhere."""
    from so.closure import _query
    st.evict(target)
    n = sum(1 for k in keys if ReferenceResolver(st).resolve(_query(k)).answer == obj)
    st.restore(target)
    return n / len(keys)


def test_closure_minus_one_over_keys_is_star_arithmetic_and_not_a_store_law():
    """E-000032 reported `(closure - 1) / keys_per_group` as a store-side forecast of what still reads
    after only the object is removed, and it matched at error 0.0000 in three arms. A review
    (ledger §31.33) showed why: on a STAR every non-target closure member backs exactly one key, so the
    formula is that invariant restated. On a CHAIN -- an alias pointing at a COPY rather than at the
    target -- the store's own resolver refutes it by a full grid step, with no model in the loop.
    The quantity that is a function of the store is the post-deletion resolver count, and against
    that the formula is redundant on stars and wrong off them."""
    obj, keys = 7, [(1, 0), (10, 0), (11, 0)]
    stars = {}
    for kind in ("link", "mixed", "copy"):
        st = _store()
        t = st.write(1, 0, obj, provenance="t")
        if kind == "link":
            st.link(10, 0, t, provenance="a"); st.link(11, 0, t, provenance="b")
        elif kind == "mixed":
            st.link(10, 0, t, provenance="a"); st.write(11, 0, obj, provenance="b")
        else:
            st.write(10, 0, obj, provenance="a"); st.write(11, 0, obj, provenance="b")
        fc = fact_closure(st, keys, obj=obj)
        stars[kind] = ((fc.size - 1) / len(keys), _survival_after_evicting_the_object(st, keys, obj, t))
    # the three published arms: the formula and the resolver agree exactly
    assert stars["link"] == (0.0, 0.0)
    assert stars["mixed"] == pytest.approx((1 / 3, 1 / 3))
    assert stars["copy"] == pytest.approx((2 / 3, 2 / 3))

    # the chain: closure 2 (object + copy), but evicting the object leaves BOTH the copy and the
    # alias that points at the copy answering
    st = _store()
    t = st.write(1, 0, obj, provenance="t")
    copy = st.write(10, 0, obj, provenance="copy")
    st.link(11, 0, copy, provenance="alias-of-copy")
    fc = fact_closure(st, keys, obj=obj)
    assert fc.size == 2 and fc.optimal
    predicted = (fc.size - 1) / len(keys)
    measured = _survival_after_evicting_the_object(st, keys, obj, t)
    assert predicted == pytest.approx(1 / 3)
    assert measured == pytest.approx(2 / 3)
    assert abs(predicted - measured) == pytest.approx(1 / 3)


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


# ------------------ the workload is part of the answer: canonicalisation does not touch derivations

def test_the_closure_is_one_over_single_hop_and_larger_over_a_workload_that_includes_the_derivation():
    """The limit the pod claim must state, and the one E-000019 already recorded at full strength.

    A canonical pod collapses the DUPLICATION term of the closure to one. It does nothing to the
    DERIVATION term: a two-hop path to the same object is a second way to answer, and removing the
    object's record leaves it working. Reporting the Q1 number as "the" closure would turn that into
    a false guarantee, so the workload travels with the number.
    """
    st = _store()
    target = st.write(3, 1, 7, provenance="target")
    st.link(10, 1, target, provenance="alias")
    st.write(3, 2, 5, provenance="hop1")          # 3 --2--> 5
    st.write(5, 3, 7, provenance="hop2")          # 5 --3--> 7, so 3 reaches 7 in two hops as well
    keys = pod_keys(st, target)

    narrow = fact_closure(st, keys, obj=7)
    assert narrow.size == 1 and narrow.records == (target,)
    assert "Q1" in narrow.workload and "single-hop" in narrow.summary()

    two_hop = Query("fwd", 3, (2, 3), (0, 0))
    wide = fact_closure(st, keys, obj=7,
                        queries=[Query("fwd", k[0], (k[1],), (0,)) for k in keys] + [two_hop],
                        workload="Q1 plus the two-hop path 3 -2-> 5 -3-> 7")
    assert wide.size == 2, wide.summary()
    assert target in wide.records
    assert "two-hop" in wide.summary()


def test_the_certificate_prints_the_workload_it_was_proved_over():
    """A fact-level claim that does not name its query set is not a claim."""
    from so.audit import AbsenceCheck, Certificate, certify_fact
    st = _store()
    target = st.write(3, 1, 7, provenance="target")
    st.link(10, 1, target, provenance="alias")
    keys = pod_keys(st, target)
    fc = fact_closure(st, keys, obj=7)
    st.evict(target)
    cert = certify_fact(Certificate(True, True, True, 0, 32, 1, 0, []), fc, [target],
                        store_after=st, keys=keys,
                        absence=AbsenceCheck(True, True, 1, 3, 2))
    assert cert.valid and cert.workload == fc.workload
    assert "Q1" in cert.summary()


def test_a_wider_workload_can_turn_a_valid_certificate_into_a_void_one():
    """The point of carrying the workload: the same removal, judged against more questions, fails."""
    from so.audit import AbsenceCheck, Certificate, certify_fact
    st = _store()
    target = st.write(3, 1, 7, provenance="target")
    st.write(3, 2, 5, provenance="hop1")
    st.write(5, 3, 7, provenance="hop2")
    keys = pod_keys(st, target)
    two_hop = Query("fwd", 3, (2, 3), (0, 0))
    qs = [Query("fwd", k[0], (k[1],), (0,)) for k in keys] + [two_hop]
    wide = fact_closure(st, keys, obj=7, queries=qs, workload="Q1 + one two-hop path")
    assert wide.size == 2

    st.evict(target)                                  # only the record the NARROW closure named
    cert = certify_fact(Certificate(True, True, True, 0, 32, 1, 0, []), wide, [target],
                        store_after=st, keys=keys, absence=AbsenceCheck(True, True, 1, 3, 2))
    assert not cert.valid
    assert "misses 1 record(s)" in cert.void_reason
    # and the store agrees: the two-hop question still answers 7
    assert ReferenceResolver(st).resolve(two_hop).answer == 7


# ---------------- the two ways the pod boundary was mis-drawn, and the tests that keep it drawn

def test_pod_keys_follows_a_chain_of_aliases_and_not_only_direct_pointers():
    """The earlier version missed a1 <- a2 and would have reported a short pod as covered."""
    st = _store()
    target = st.write(3, 1, 7, provenance="t")
    a1 = st.link(10, 1, target, provenance="a1")
    st.link(11, 1, a1, provenance="a2")            # points at the ALIAS, not at the target
    keys = pod_keys(st, target)
    assert set(keys) == {(3, 1), (10, 1), (11, 1)}
    for key in keys:                               # and every one of them really does reach the object
        assert ReferenceResolver(st).resolve(Query("fwd", key[0], (key[1],), (0,))).answer == 7


def test_pod_keys_terminates_on_a_cycle():
    st = _store()
    target = st.write(3, 1, 7, provenance="t")
    a1 = st.link(10, 1, target, provenance="a1")
    a2 = st.link(11, 1, a1, provenance="a2")
    st.relink(a1, a2)                              # a1 -> a2 -> a1: no answer, and no infinite walk
    keys = pod_keys(st, target)
    assert (3, 1) in keys and len(keys) <= 3


def test_selecting_on_the_value_would_delete_a_bystander_and_the_name_says_so():
    """value_keys selects on the object's VALUE, so two unrelated facts sharing it come back together.

    Measured rather than warned about: the closure over that set removes both records, which is
    over-deletion -- the dual of the failure the pod exists to prevent.
    """
    from so.closure import value_keys
    st = _store()
    mine = st.write(3, 1, 42, provenance="mine")
    bystander = st.write(9, 2, 42, provenance="bystander")
    keys = value_keys(st, 42)
    assert set(keys) == {(3, 1), (9, 2)}
    fc = fact_closure(st, keys, obj=42)
    assert set(fc.records) == {mine, bystander}    # BOTH go, and the bystander is not the fact
    # the right set for the fact "the object under (3,1) is 42" is that subject's pod alone
    assert fact_closure(st, pod_keys(st, mine), obj=42).records == (mine,)


def test_the_pod_of_a_removed_cell_is_empty_rather_than_wrong():
    st = _store()
    target = st.write(3, 1, 7, provenance="t")
    st.link(10, 1, target, provenance="a")
    st.evict(target)
    assert pod_keys(st, target) == ()
