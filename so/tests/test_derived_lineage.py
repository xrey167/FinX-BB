from so.cavi import CAVIAuthority
from so.derived_lineage import DerivedLineage, LineagedState


def _authority():
    a = CAVIAuthority()
    a.create_pod(1)
    a.create_pod(2)
    a.create_alias(101, 1)
    a.create_alias(102, 1)
    a.create_alias(201, 2)
    return a


def test_unrelated_pod_update_does_not_invalidate_other_lineage():
    a = _authority()
    one = LineagedState("one", DerivedLineage.of(a.witness(101)))
    two = LineagedState("two", DerivedLineage.of(a.witness(201)))
    a.update_pod(1)
    assert not one.reusable(a)
    assert two.reusable(a)


def test_one_pod_update_invalidates_all_alias_derived_state_without_alias_edits():
    a = _authority()
    s1 = LineagedState("a", DerivedLineage.of(a.witness(101)))
    s2 = LineagedState("b", DerivedLineage.of(a.witness(102)))
    before = (a.alias_incarnation(101), a.alias_incarnation(102))
    a.update_pod(1)
    assert (a.alias_incarnation(101), a.alias_incarnation(102)) == before
    assert not s1.reusable(a)
    assert not s2.reusable(a)


def test_alias_relink_invalidates_only_alias_qualified_state():
    a = _authority()
    x = LineagedState("x", DerivedLineage.of(a.witness(101)))
    sibling = LineagedState("sibling", DerivedLineage.of(a.witness(102)))
    a.relink_alias(101, 2)
    assert not x.reusable(a)
    assert sibling.reusable(a)


def test_aba_restore_cannot_revalidate_old_lineage():
    a = _authority()
    old = LineagedState("old", DerivedLineage.of(a.witness(101)))
    a.shred_pod(1)
    a.restore_pod(1)
    assert not old.reusable(a)
    assert a.witness(101).pod_incarnation > old.lineage.witnesses[0].pod_incarnation


def test_union_requires_all_dependencies_current():
    a = _authority()
    lineage = DerivedLineage.union([
        DerivedLineage.of(a.witness(101)),
        DerivedLineage.of(a.witness(201)),
    ])
    assert lineage.dependency_count == 2
    assert lineage.packed_metadata_bytes == 64
    assert lineage.is_current(a)
    a.update_pod(2)
    assert not lineage.is_current(a)
    stale = lineage.stale_witnesses(a)
    assert len(stale) == 1
    assert stale[0].pod_id == 2
