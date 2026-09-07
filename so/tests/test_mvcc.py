import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from so.mvcc import MVCCStore, RevisionConflict, Status


def test_write_read_update_rollback():
    s = MVCCStore(seed=0)
    k = s.write(1, 0, 7)
    assert s.read(k).obj == 7
    v2 = s.update(k, 9)
    assert v2 == 2 and s.read(k).obj == 9 and len(s.cells[k].versions) == 2
    s.rollback(k, 1)
    assert s.read(k).obj == 7
    s.rollback(k, 2)
    assert s.read(k).obj == 9


def test_revoke_restore_delete():
    s = MVCCStore(seed=0)
    k = s.write(1, 0, 7)
    s.revoke(k)
    assert s.read(k) is None and s.cells[k].status == Status.REVOKED
    assert (1, 0) not in s.index_view()
    assert s.bank()["active"].tolist() == [False]  # payload still physically present
    s.restore(k)
    assert s.read(k).obj == 7
    s.delete(k)
    assert s.read(k) is None and s.bank()["kid"].size == 0
    with pytest.raises(KeyError):
        s.update(k, 3)


def test_shred_and_resign():
    s = MVCCStore(seed=3)
    k = s.write(2, 1, 5)
    assert s.marker_valid(s.cells[k].active.marker)
    s.shred(k)
    assert s.read(k) is None                      # mechanical check refuses the unsigned payload
    assert s.cells[k].status == Status.ACTIVE     # routing untouched
    assert s.bank()["active"].tolist() == [True]  # neural bank still routes to it (model must reject)
    assert s.bank(respect_markers=True)["active"].tolist() == [False]
    s.resign(k)
    assert s.read(k).obj == 5


def test_swap_and_replace():
    s = MVCCStore(seed=0)
    a, b = s.write(1, 0, 7), s.write(2, 0, 8)
    s.swap(a, b)
    assert s.read(a).obj == 8 and s.read(b).obj == 7
    s.replace(a, 11)
    assert s.read(a).obj == 11 and len(s.cells[a].versions) == 1


def test_replay_is_deterministic():
    s = MVCCStore(seed=5)
    kids = [s.write(i, i % 3, (i * 7) % 11) for i in range(20)]
    s.update(kids[3], 4)
    s.revoke(kids[5])
    s.rollback(kids[3], 1)
    s.shred(kids[7])
    s.delete(kids[9])
    s.swap(kids[1], kids[2])
    clone = s.clone_by_replay()
    assert clone.state_hash() == s.state_hash()
    assert clone.index_view() == s.index_view()
    assert np.array_equal(clone.bank()["active"], s.bank()["active"])


def test_bank_shapes():
    s = MVCCStore(marker_dim=8, seed=1)
    for i in range(5):
        s.write(i, 0, i + 1)
    b = s.bank()
    assert b["marker"].shape == (5, 8) and b["kid"].shape == (5,)
    assert b["active"].all()


def test_duplicate_key_views_choose_one_holder_and_promote_only_on_status_change():
    """A shadow row is retained, but cannot disagree with the resolver about who owns the key."""
    st = MVCCStore(marker_dim=16, seed=0)
    first = st.write(3, 1, 7, provenance="first")
    shadow = st.write(3, 1, 9, provenance="shadow")

    assert st.active_view() == {(3, 1): (7, first)}
    assert st.index_view() == {(3, 1): 7}
    assert st.kid_of((3, 1)) == first
    assert st.bank()["active"].tolist() == [True, False]

    # Marker validity is checked after holder selection: SHRED closes this key and must not
    # accidentally promote a shadow copy containing the same key.
    st.shred(first)
    assert st.active_view() == {}
    assert st.index_view() == {}
    assert st.active_view(respect_markers=False) == {(3, 1): (7, first)}
    assert st.kid_of((3, 1)) == first
    assert st.bank()["active"].tolist() == [True, False]
    assert st.bank(respect_markers=True)["active"].tolist() == [False, False]

    # A status transition removes the first holder and deterministically promotes the next one.
    st.revoke(first)
    assert st.active_view() == {(3, 1): (9, shadow)}
    assert st.index_view() == {(3, 1): 9}
    assert st.kid_of((3, 1)) == shadow
    assert st.bank()["active"].tolist() == [False, True]


@pytest.mark.parametrize("retire", ["revoke", "evict", "delete"])
def test_duplicate_key_promotes_next_holder_for_every_routing_removal(retire):
    st = MVCCStore(marker_dim=16, seed=0)
    first = st.write(3, 1, 7)
    shadow = st.write(3, 1, 9)

    getattr(st, retire)(first)

    assert st.active_view() == {(3, 1): (9, shadow)}
    assert st.index_view() == {(3, 1): 9}
    assert st.kid_of((3, 1)) == shadow


def test_immutable_snapshot_survives_replay_with_duplicates_and_links():
    st = MVCCStore(marker_dim=16, seed=4)
    target = st.write(3, 1, 7, provenance="target")
    shadow = st.write(3, 1, 9, provenance="shadow")
    st.link(4, 1, target, provenance="alias")
    st.revoke(target)
    st.update(shadow, 11)

    snapshot = st.snapshot()
    replayed = st.clone_by_replay().snapshot()

    assert replayed == snapshot
    assert dict(snapshot.holder_of_key) == {(3, 1): shadow, (4, 1): 3}
    # The alias names its original target kid; duplicate-key promotion is only for direct lookup.
    assert dict(snapshot.index_view) == {(3, 1): 11}
    with pytest.raises((AttributeError, TypeError)):
        snapshot.revision = -1


# ------------------------------------------- EVICT: unreachable and retained, which SHRED and DELETE are not

def _one_cell():
    st = MVCCStore(marker_dim=16, seed=0)
    kid = st.write(3, 1, 7, provenance="t")
    return st, kid


def test_evict_takes_the_row_out_of_the_bank():
    st, kid = _one_cell()
    assert st.bank()["kid"].shape[0] == 1
    st.evict(kid)
    assert st.bank()["kid"].shape[0] == 0
    assert st.cells[kid].status is Status.EVICTED


def test_evict_keeps_the_payload_which_is_the_whole_point():
    """DELETE earns the same unreachability by discarding the data. EVICT does not have to."""
    st, kid = _one_cell()
    st.evict(kid)
    assert st.cells[kid].versions, "the versions must survive an eviction"
    assert st.cells[kid].version_obj(st.cells[kid].active_version).obj == 7


def test_delete_by_contrast_discards_it():
    st, kid = _one_cell()
    st.delete(kid)
    assert st.cells[kid].versions == []


def test_shred_by_contrast_leaves_the_row_addressable():
    st, kid = _one_cell()
    st.shred(kid)
    assert st.bank()["kid"].shape[0] == 1, "SHRED keeps the row in the bank; that is what E-000028 exploits"


def test_an_evicted_cell_comes_back_with_its_payload():
    st, kid = _one_cell()
    st.update(kid, 9)
    st.evict(kid)
    st.restore(kid)
    b = st.bank()
    assert b["kid"].shape[0] == 1 and int(b["obj"][0]) == 9
    st.rollback(kid, 1)
    assert int(st.bank()["obj"][0]) == 7


def test_eviction_is_in_the_operation_log_and_the_state_hash():
    st, kid = _one_cell()
    before = st.state_hash()
    st.evict(kid)
    assert st.state_hash() != before
    assert any(op == "evict" for op, _ in st.log)


def test_an_evicted_target_does_not_resolve_for_an_alias():
    """A link into an evicted cell must dangle exactly as it does into a deleted one."""
    st = MVCCStore(marker_dim=16, seed=0)
    target = st.write(3, 1, 7, provenance="t")
    alias = st.link(4, 1, target, provenance="a")
    assert st.resolve_key((4, 1))[0] == 7
    st.evict(target)
    assert st.resolve_key((4, 1))[0] is None
    st.restore(target)
    assert st.resolve_key((4, 1))[0] == 7


def test_aliases_keep_exact_first_and_shadow_target_identity_through_lifecycle():
    st = MVCCStore(marker_dim=16, seed=11)
    first = st.write(3, 1, 7, provenance="first")
    shadow = st.write(3, 1, 9, provenance="shadow")
    alias_first = st.link(4, 1, first)
    alias_shadow = st.link(5, 1, shadow)

    assert st.index_view() == {(3, 1): 7, (4, 1): 7, (5, 1): 9}

    st.revoke(first)
    assert st.index_view() == {(3, 1): 9, (5, 1): 9}
    st.restore(first)
    assert st.index_view() == {(3, 1): 7, (4, 1): 7, (5, 1): 9}

    st.shred(first)
    assert st.index_view() == {(5, 1): 9}  # SHRED never promotes the direct-key shadow
    st.resign(first)

    st.update(first, 8)
    st.evict(first)
    st.rollback(first, 1)                  # retained edit; not an implicit restore
    assert st.cells[first].status is Status.EVICTED
    assert st.index_view() == {(3, 1): 9, (5, 1): 9}
    st.restore(first)
    assert st.index_view() == {(3, 1): 7, (4, 1): 7, (5, 1): 9}

    st.revoke(shadow)
    assert st.index_view() == {(3, 1): 7, (4, 1): 7}
    st.restore(shadow)
    assert st.index_view()[(5, 1)] == 9

    st.delete(first)
    assert st.index_view() == {(3, 1): 9, (5, 1): 9}
    assert st.resolve_key((4, 1))[0] is None
    with pytest.raises(KeyError):
        st.restore(first)
    with pytest.raises(KeyError):
        st.rollback(first, 1)
    assert st.cells[alias_first].active.target == first
    assert st.cells[alias_shadow].active.target == shadow


def test_snapshot_exports_exact_link_target_kid_even_for_duplicate_keys_and_tombstones():
    st = MVCCStore(seed=2)
    first = st.write(3, 1, 7)
    shadow = st.write(3, 1, 9)
    alias = st.link(4, 1, shadow)
    row = next(r for r in st.snapshot().rows if r.kid == alias)
    assert row.link_target_kid == shadow
    assert (row.link_subject, row.link_relation) == (3, 1)
    st.delete(shadow)
    row = next(r for r in st.snapshot().rows if r.kid == alias)
    assert row.link_target_kid == shadow
    assert (row.link_subject, row.link_relation) == (3, 1)
    assert st.kid_of((3, 1)) == first


def test_read_returns_recursively_immutable_defensive_value():
    st = MVCCStore(seed=3)
    kid = st.write(2, 1, 5)
    before = st.state_hash()
    value = st.read(kid)
    assert value is not None and isinstance(value.marker, tuple)
    with pytest.raises(FrozenInstanceError):
        value.obj = 99
    with pytest.raises(TypeError):
        value.marker[0] = 0.0
    assert st.read(kid).obj == 5
    assert st.state_hash() == before


@pytest.mark.parametrize("operation", ["write", "update", "relink", "resign", "blank"])
def test_signing_failure_is_atomic(monkeypatch, operation):
    st = MVCCStore(seed=5, content_markers=True)
    fact = st.write(1, 0, 7)
    other = st.write(2, 0, 8)
    alias = st.link(3, 0, fact)
    before = st.state_hash()

    def fail(_version):
        raise RuntimeError("injected signing failure")

    monkeypatch.setattr(st, "_sign", fail)
    calls = {
        "write": lambda: st.write(9, 0, 9),
        "update": lambda: st.update(fact, 10),
        "relink": lambda: st.relink(alias, other),
        "resign": lambda: st.resign(fact),
        "blank": lambda: st.blank(alias),
    }
    with pytest.raises(RuntimeError, match="injected signing failure"):
        calls[operation]()
    assert st.state_hash() == before


def test_unsigning_failure_is_atomic(monkeypatch):
    st = MVCCStore(seed=7)
    kid = st.write(1, 0, 7)
    before = st.state_hash()
    monkeypatch.setattr(st, "_unsign", lambda _v: (_ for _ in ()).throw(RuntimeError("unsign")))
    with pytest.raises(RuntimeError, match="unsign"):
        st.shred(kid)
    assert st.state_hash() == before


@pytest.mark.parametrize("operation", [
    "write", "link", "update", "relink", "revoke", "restore", "rollback", "evict",
    "delete", "shred", "blank", "resign", "swap", "replace",
])
def test_record_failure_is_atomic_after_partial_log_append(monkeypatch, operation):
    st = MVCCStore(seed=13)
    kid = st.write(1, 0, 7)
    other = st.write(2, 0, 9)
    alias = st.link(3, 0, kid)
    st.update(kid, 8)
    if operation == "restore":
        st.revoke(kid)
    before = st.state_hash()
    original = st._record

    def record_then_fail(op, **args):
        original(op, **args)
        raise RuntimeError("record")

    monkeypatch.setattr(st, "_record", record_then_fail)
    calls = {
        "write": lambda: st.write(4, 0, 10),
        "link": lambda: st.link(4, 0, other),
        "update": lambda: st.update(kid, 9),
        "relink": lambda: st.relink(alias, other),
        "revoke": lambda: st.revoke(kid),
        "restore": lambda: st.restore(kid),
        "rollback": lambda: st.rollback(kid, 1),
        "evict": lambda: st.evict(kid),
        "delete": lambda: st.delete(kid),
        "shred": lambda: st.shred(kid),
        "blank": lambda: st.blank(alias),
        "resign": lambda: st.resign(kid),
        "swap": lambda: st.swap(kid, other),
        "replace": lambda: st.replace(kid, 11),
    }
    with pytest.raises(RuntimeError, match="record"):
        calls[operation]()
    assert st.state_hash() == before


def test_lifecycle_edits_do_not_implicitly_restore_or_reexpose():
    st = MVCCStore(seed=17)
    kid = st.write(1, 0, 7)
    st.update(kid, 8)
    st.revoke(kid)
    st.update(kid, 9)
    st.rollback(kid, 1)
    assert st.cells[kid].status is Status.REVOKED and st.index_view() == {}
    with pytest.raises(ValueError, match="requires ACTIVE"):
        st.revoke(kid)
    st.evict(kid)
    st.update(kid, 10)
    st.rollback(kid, 2)
    assert st.cells[kid].status is Status.EVICTED and st.bank()["kid"].size == 0
    with pytest.raises(ValueError, match="requires ACTIVE or REVOKED"):
        st.evict(kid)
    st.restore(kid)
    assert st.index_view() == {(1, 0): 8}
    with pytest.raises(ValueError, match="requires REVOKED or EVICTED"):
        st.restore(kid)


def test_expected_revision_is_backwards_compatible_cas_and_failure_is_atomic():
    st = MVCCStore(seed=19)
    kid = st.write(1, 0, 7, expected_revision=0)
    stale = st.revision - 1
    before = st.state_hash()
    with pytest.raises(RevisionConflict):
        st.update(kid, 8, expected_revision=stale)
    assert st.state_hash() == before
    st.update(kid, 8, expected_revision=st.revision)
    assert st.read(kid).obj == 8


@pytest.mark.parametrize("bad_revision", [True, False, 1.0, 1.5, "1"])
def test_expected_revision_rejects_non_integer_values_without_mutation(bad_revision):
    st = MVCCStore(seed=19)
    before = st.state_hash()
    with pytest.raises(TypeError, match="expected_revision"):
        st.write(1, 0, 7, expected_revision=bad_revision)
    assert st.state_hash() == before


@pytest.mark.parametrize("bad_version", [True, False, 1.0, 1.5, "1"])
def test_rollback_rejects_non_integer_versions_atomically(bad_version):
    st = MVCCStore(seed=19)
    kid = st.write(1, 0, 7)
    st.update(kid, 8)
    before = st.state_hash()
    with pytest.raises(TypeError, match="version"):
        st.rollback(kid, bad_version)
    assert st.state_hash() == before
    assert st.read(kid).obj == 8
    assert st.snapshot().revision == st.revision
    assert st.clone_by_replay().state_hash() == before


@pytest.mark.parametrize("bad_id", [True, 1.0, "1"])
@pytest.mark.parametrize("operation", [
    "read", "refcount", "update", "revoke", "restore", "rollback", "delete", "evict",
    "shred", "blank", "resign", "replace", "swap_left", "swap_right", "relink_kid",
    "link_target", "relink_target",
])
def test_public_kid_parameters_reject_bool_float_and_string_aliases_atomically(operation, bad_id):
    st = MVCCStore(seed=23)
    target = st.write(1, 0, 7)
    other = st.write(2, 0, 8)
    alias = st.link(3, 0, target)
    calls = {
        "read": lambda: st.read(bad_id),
        "refcount": lambda: st.refcount(bad_id),
        "update": lambda: st.update(bad_id, 9),
        "revoke": lambda: st.revoke(bad_id),
        "restore": lambda: st.restore(bad_id),
        "rollback": lambda: st.rollback(bad_id, 1),
        "delete": lambda: st.delete(bad_id),
        "evict": lambda: st.evict(bad_id),
        "shred": lambda: st.shred(bad_id),
        "blank": lambda: st.blank(bad_id),
        "resign": lambda: st.resign(bad_id),
        "replace": lambda: st.replace(bad_id, 9),
        "swap_left": lambda: st.swap(bad_id, other),
        "swap_right": lambda: st.swap(other, bad_id),
        "relink_kid": lambda: st.relink(bad_id, target),
        "link_target": lambda: st.link(4, 0, bad_id),
        "relink_target": lambda: st.relink(alias, bad_id),
    }
    before = st.state_hash()
    with pytest.raises(TypeError):
        calls[operation]()
    assert st.state_hash() == before


@pytest.mark.parametrize("bad_id", [True, 1.0, "1"])
@pytest.mark.parametrize("operation", [
    "write_subject", "write_relation", "write_obj", "link_subject", "link_relation",
    "update_obj", "replace_obj", "resolve_subject", "resolve_relation", "kid_of_subject",
    "kid_of_relation",
])
def test_public_entity_and_relation_ids_reject_non_integral_aliases(operation, bad_id):
    st = MVCCStore(seed=29)
    target = st.write(1, 0, 7)
    calls = {
        "write_subject": lambda: st.write(bad_id, 0, 8),
        "write_relation": lambda: st.write(2, bad_id, 8),
        "write_obj": lambda: st.write(2, 0, bad_id),
        "link_subject": lambda: st.link(bad_id, 0, target),
        "link_relation": lambda: st.link(2, bad_id, target),
        "update_obj": lambda: st.update(target, bad_id),
        "replace_obj": lambda: st.replace(target, bad_id),
        "resolve_subject": lambda: st.resolve_key((bad_id, 0)),
        "resolve_relation": lambda: st.resolve_key((1, bad_id)),
        "kid_of_subject": lambda: st.kid_of((bad_id, 0)),
        "kid_of_relation": lambda: st.kid_of((1, bad_id)),
    }
    before = st.state_hash()
    with pytest.raises(TypeError):
        calls[operation]()
    assert st.state_hash() == before


def test_numpy_integral_ids_are_normalised_before_storage_logging_and_replay():
    st = MVCCStore(seed=31)
    target = st.write(np.int64(1), np.int32(0), np.int64(7))
    st.update(np.int64(target), np.int32(8))
    st.rollback(np.int64(target), np.int32(1))
    alias = st.link(np.int64(2), np.int32(0), np.int64(target))
    st.relink(np.int64(alias), np.int64(target))
    assert st.read(np.int64(target)).obj == 7
    assert st.kid_of((np.int64(1), np.int32(0))) == target
    assert st.resolve_key((np.int64(2), np.int32(0)))[0] == 7
    assert all(type(value) is int
               for _, args in st.log for key, value in args.items()
               if key in {"kid", "target", "subject", "relation", "obj", "version"})
    assert st.clone_by_replay().state_hash() == st.state_hash()


@pytest.mark.parametrize("bad_version", [True, 1.0, "1"])
def test_cell_version_selector_rejects_non_integral_aliases(bad_version):
    st = MVCCStore(seed=37)
    kid = st.write(1, 0, 7)
    with pytest.raises(TypeError, match="version"):
        st.cells[kid].version_obj(bad_version)


@pytest.mark.parametrize("bad_version", [0, 2, -1])
def test_cell_version_selector_rejects_out_of_range_values(bad_version):
    st = MVCCStore(seed=37)
    kid = st.write(1, 0, 7)
    with pytest.raises(ValueError, match="no version"):
        st.cells[kid].version_obj(bad_version)


@pytest.mark.parametrize("field", ["marker", "marker_centre"])
@pytest.mark.parametrize("mutation", ["dtype", "shape"])
def test_freshness_fingerprint_binds_array_dtype_and_shape(field, mutation):
    st = MVCCStore(marker_dim=16, seed=41)
    kid = st.write(1, 0, 7)
    token = st.revision_token()
    before = st.state_hash()
    original = st.cells[kid].active.marker if field == "marker" else st.marker_centre
    changed = original.view(np.int64) if mutation == "dtype" else original.reshape(8, 2)
    if field == "marker":
        st.cells[kid].active.marker = changed
    else:
        st.marker_centre = changed
    assert changed.tobytes() == original.tobytes()
    assert not st.is_current(token)
    assert st.state_hash() != before


@pytest.mark.parametrize("invalid", [True, 1.0])
def test_state_hash_does_not_mask_invalid_direct_active_version_type(invalid):
    st = MVCCStore(seed=43)
    kid = st.write(1, 0, 7)
    before = st.state_hash()
    st.cells[kid].active_version = invalid
    assert st.state_hash() != before


def test_embedded_cell_kid_mutation_invalidates_freshness_and_snapshot_fails_closed():
    st = MVCCStore(seed=47)
    target = st.write(1, 0, 7)
    alias = st.link(2, 0, target)
    token = st.revision_token()
    before = st.state_hash()
    st.cells[target].kid = alias
    assert not st.is_current(token)
    assert st.state_hash() != before
    with pytest.raises(ValueError, match="mapping key .* disagrees"):
        st.snapshot()
    with pytest.raises(ValueError, match="mapping key .* disagrees"):
        st.resolve_key((2, 0))


def test_replay_rejects_a_non_integral_rollback_version_in_an_external_log():
    log = [
        ("write", {"subject": 1, "relation": 0, "obj": 7, "provenance": ""}),
        ("rollback", {"kid": 1, "version": 1.0}),
    ]
    with pytest.raises(TypeError, match="version"):
        MVCCStore.replay(log, marker_dim=16, seed=0, valid_radius=0.35)


@pytest.mark.parametrize("retire", ["revoke", "evict"])
def test_swap_rejects_non_active_cells_atomically(retire):
    st = MVCCStore(seed=29)
    a, b = st.write(1, 0, 7), st.write(2, 0, 8)
    getattr(st, retire)(a)
    before = st.state_hash()
    with pytest.raises(ValueError, match="two ACTIVE"):
        st.swap(a, b)
    assert st.state_hash() == before


def test_swap_rejects_shredded_cells_atomically():
    st = MVCCStore(seed=31)
    a, b = st.write(1, 0, 7), st.write(2, 0, 8)
    st.shred(a)
    before = st.state_hash()
    with pytest.raises(ValueError, match="marker-valid"):
        st.swap(a, b)
    assert st.state_hash() == before


def test_public_reinsertion_cannot_change_lowest_kid_holder_or_bank_order():
    st = MVCCStore(seed=37)
    first = st.write(1, 0, 7)
    shadow = st.write(1, 0, 9)
    before = st.snapshot()
    cell = st.cells.pop(first)
    st.cells[first] = cell
    assert st.index_view() == {(1, 0): 7}
    assert [row.kid for row in st.snapshot().rows] == [row.kid for row in before.rows] == [first, shadow]


def test_refcount_includes_retained_evicted_alias_until_delete():
    st = MVCCStore(seed=41)
    target = st.write(1, 0, 7)
    alias = st.link(2, 0, target)
    assert st.refcount(target) == 1
    st.evict(alias)
    assert st.refcount(target) == 1
    st.restore(alias)
    st.delete(alias)
    assert st.refcount(target) == 0


@pytest.mark.parametrize("sampler", ["new_valid_marker", "new_invalid_marker"])
def test_ad_hoc_marker_sampling_is_an_explicit_unlogged_boundary_but_stales_views(sampler):
    st = MVCCStore(seed=43)
    st.write(1, 0, 7)
    token = st.revision_token()
    revision, log = st.revision, list(st.log)
    getattr(st, sampler)()
    assert st.revision == revision
    assert st.log == log
    assert not st.is_current(token)
    # The helper is intentionally outside event replay: its extra RNG draw changes later markers.
    replay = st.clone_by_replay()
    assert replay.state_hash() != st.state_hash()


def test_persistent_state_hash_includes_history_while_logical_hash_tracks_restoration():
    st = MVCCStore(seed=23)
    kid = st.write(1, 0, 7, provenance="source-a")
    replay = st.clone_by_replay()
    assert replay.state_hash() == st.state_hash()
    persistent_before = st.state_hash()
    logical_before = st.logical_state_hash()
    st.evict(kid)
    st.restore(kid)
    assert st.logical_state_hash() == logical_before
    assert st.state_hash() != persistent_before

    same_view = MVCCStore(seed=23)
    same_view.write(1, 0, 7, provenance="source-b")
    assert same_view.index_view() == st.index_view()
    assert same_view.logical_state_hash() != st.logical_state_hash()
