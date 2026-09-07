import numpy as np
import threading
import pytest
import torch

from so.data import bank_from_store, bank_from_world, failing_hop_target
from so.mvcc import MVCCStore, StaleSnapshotError
from so.train import make_centre
from so.world import Query, World


def _bank(seed=0, p_revoked=0.5, p_shred=0.0, p_stale=0.2):
    rng = np.random.default_rng(seed)
    world = World.sample(rng, 64, 4, 200, 2)
    centre = make_centre(seed, 16)
    return world, bank_from_world(rng, world, centre, p_revoked, p_shred, p_stale)


def test_routable_marks_original_cells_and_not_stale_rows():
    world, bank = _bank()
    n = len(world.facts)
    assert bank.routable is not None and bank.routable_pos is not None
    assert bank.routable[:n].all() and not bank.routable[n:].any()
    assert (bank.tensors()["routable"].numpy() == bank.routable).all()
    # every revoked original cell is routable but not active
    revoked = bank.routable & ~bank.active
    assert revoked.any()
    for i in np.flatnonzero(revoked):
        assert bank.routable_pos[(int(bank.subject[i]), int(bank.relation[i]))] == i
        assert (int(bank.subject[i]), int(bank.relation[i])) not in bank.active_pos


def test_failing_hop_target_status_gated_points_at_revoked_cell():
    world, bank = _bank()
    n = len(world.facts)
    hit = 0
    for i in range(n):
        if bank.active[i]:
            continue
        q = Query("fwd", int(bank.subject[i]), (int(bank.relation[i]),), (world.surface_of(int(bank.relation[i]), 0),))
        gt = world.answer(q, bank.index_view)
        assert len(gt.edges) == 0                     # the fact is not in the view: the first hop fails
        assert failing_hop_target(bank, q, gt) == -1                       # mask design: null cell
        assert failing_hop_target(bank, q, gt, status_gated=True) == i     # status-gated design: the revoked cell itself
        hit += 1
    assert hit > 0


def test_failing_hop_target_absent_key_is_null_in_both_designs():
    world, bank = _bank()
    present = set(zip(bank.subject.tolist(), bank.relation.tolist()))
    for s in range(64):
        for r in range(4):
            if (s, r) not in present:
                q = Query("fwd", s, (r,), (world.surface_of(r, 0),))
                gt = world.answer(q, bank.index_view)
                assert failing_hop_target(bank, q, gt) == -1
                assert failing_hop_target(bank, q, gt, status_gated=True) == -1
                return
    raise AssertionError("no absent key found")


def test_bank_from_store_keeps_revoked_cells_routable_and_drops_deleted():
    centre = make_centre(0, 16)
    store = MVCCStore(marker_dim=16, seed=0, marker_centre=centre)
    k1 = store.write(1, 0, 5); k2 = store.write(2, 0, 6); k3 = store.write(3, 1, 7)
    store.revoke(k2)
    store.delete(k3)
    bank = bank_from_store(store)
    assert bank.size == 2
    assert bank.routable.all()
    pos2 = bank.routable_pos[(2, 0)]
    assert not bank.active[pos2] and (2, 0) not in bank.active_pos
    assert bank.active[bank.routable_pos[(1, 0)]]
    assert (3, 1) not in bank.routable_pos


def test_bank_from_store_duplicate_maps_follow_the_single_mvcc_holder():
    centre = make_centre(0, 16)
    store = MVCCStore(marker_dim=16, seed=0, marker_centre=centre)
    first = store.write(3, 1, 7, provenance="first")
    shadow = store.write(3, 1, 9, provenance="shadow")

    bank = bank_from_store(store)
    assert bank.kid.tolist() == [first, shadow]             # shadow stays physically present
    assert bank.active.tolist() == [True, False]
    assert bank.usable.tolist() == [True, False]
    assert bank.routable.tolist() == [True, False]
    assert bank.index_view == {(3, 1): 7}
    assert bank.kid_of_key == {(3, 1): 0}
    assert bank.active_pos == {(3, 1): 0}
    assert bank.routable_pos == {(3, 1): 0}

    # SHRED is a marker failure of the selected holder, not a routing-removal transition.
    store.shred(first)
    bank = bank_from_store(store)
    assert bank.active.tolist() == [True, False]
    assert bank.usable.tolist() == [False, False]
    assert bank.routable.tolist() == [True, False]
    assert bank.index_view == {}
    assert bank.kid_of_key == {}
    assert bank.active_pos == {(3, 1): 0}
    assert bank.routable_pos == {(3, 1): 0}

    store.revoke(first)
    bank = bank_from_store(store)
    assert bank.kid.tolist() == [first, shadow]
    assert bank.active.tolist() == [False, True]
    assert bank.usable.tolist() == [False, True]
    assert bank.routable.tolist() == [False, True]
    assert bank.index_view == {(3, 1): 9}
    assert bank.kid_of_key == {(3, 1): 1}
    assert bank.active_pos == {(3, 1): 1}
    assert bank.routable_pos == {(3, 1): 1}


def _bank_captured_before_mutation(store, mutate):
    """Pause bank construction after its immutable snapshot, then mutate the live store."""
    captured = threading.Event()
    release = threading.Event()
    original_snapshot = store.snapshot

    def snapshot_then_pause():
        snapshot = original_snapshot()
        captured.set()
        assert release.wait(timeout=5), "test did not release bank construction"
        return snapshot

    store.snapshot = snapshot_then_pause
    outcome = {}

    def build():
        try:
            outcome["bank"] = bank_from_store(store)
        except BaseException as exc:  # propagate worker failures in the test thread
            outcome["error"] = exc

    worker = threading.Thread(target=build)
    worker.start()
    assert captured.wait(timeout=5), "bank construction did not capture a snapshot"
    mutate()
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive(), "bank construction did not finish"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["bank"]


def test_bank_from_store_update_is_wholly_before_or_after_the_snapshot():
    store = MVCCStore(marker_dim=16, seed=0)
    target = store.write(3, 1, 7)
    store.link(4, 1, target)

    bank = _bank_captured_before_mutation(store, lambda: store.update(target, 9))

    assert bank.obj.tolist() == [7, 0]
    assert bank.index_view == {(3, 1): 7, (4, 1): 7}
    assert store.index_view() == {(3, 1): 9, (4, 1): 9}


def test_bank_from_store_revoke_does_not_mix_duplicate_promotion_into_old_rows():
    store = MVCCStore(marker_dim=16, seed=0)
    first = store.write(3, 1, 7)
    shadow = store.write(3, 1, 9)

    bank = _bank_captured_before_mutation(store, lambda: store.revoke(first))

    assert bank.active.tolist() == [True, False]
    assert bank.index_view == {(3, 1): 7}
    assert bank.kid_of_key == {(3, 1): 0}
    assert store.index_view() == {(3, 1): 9}
    assert bank_from_store(store).kid_of_key == {(3, 1): 1}


def test_bank_from_store_delete_does_not_mix_a_dangling_link_into_old_rows():
    store = MVCCStore(marker_dim=16, seed=0)
    target = store.write(3, 1, 7)
    store.link(4, 1, target)

    bank = _bank_captured_before_mutation(store, lambda: store.delete(target))

    assert bank.kid.tolist() == [target, 2]
    assert bank.index_view == {(3, 1): 7, (4, 1): 7}
    assert bank.trace_of_key == {(3, 1): (0,), (4, 1): (1, 0)}
    assert store.index_view() == {}


def test_bank_carries_snapshot_revision_and_rejects_stale_or_cross_store_use():
    store = MVCCStore(seed=0)
    kid = store.write(3, 1, 7)
    bank = bank_from_store(store)
    assert bank.revision == store.revision
    assert bank.is_fresh(store)
    bank.require_fresh(store)

    store.update(kid, 9)
    assert not bank.is_fresh(store)
    with pytest.raises(StaleSnapshotError, match="bank revision"):
        bank.require_fresh(store)

    other = MVCCStore(seed=0)
    other.write(3, 1, 7)
    assert other.revision == bank.revision
    assert not bank.is_fresh(other)


def test_bank_exports_exact_link_target_identity_when_keys_are_duplicated():
    store = MVCCStore(seed=0)
    first = store.write(3, 1, 7)
    shadow = store.write(3, 1, 9)
    alias = store.link(4, 1, shadow)
    bank = bank_from_store(store)
    pos = bank.kid.tolist().index(alias)
    assert int(bank.link_target_kid[pos]) == shadow
    assert int(bank.tensors()["link_target_kid"][pos]) == shadow
    assert bank.index_view == {(3, 1): 7, (4, 1): 9}

    store.revoke(first)
    promoted = bank_from_store(store)
    assert promoted.index_view == {(3, 1): 9, (4, 1): 9}
    assert int(promoted.link_target_kid[promoted.kid.tolist().index(alias)]) == shadow


def test_bank_collections_and_tensor_exports_are_defensive():
    store = MVCCStore(seed=0)
    store.write(3, 1, 7)
    bank = bank_from_store(store)
    with pytest.raises(ValueError):
        bank.obj[0] = 99
    with pytest.raises(TypeError):
        bank.index_view[(3, 1)] = 99
    with pytest.raises((AttributeError, TypeError)):
        bank.obj = np.asarray([99])
    tensors = bank.tensors()
    tensors["obj"][0] = 99
    assert int(bank.obj[0]) == 7
    assert int(bank.tensors()["obj"][0]) == 7


def test_every_bank_array_is_backed_by_immutable_storage_and_cannot_be_reenabled():
    store = MVCCStore(seed=0)
    target = store.write(3, 1, 7)
    store.link(4, 1, target)
    bank = bank_from_store(store)
    array_fields = (
        "subject", "relation", "obj", "marker", "active", "usable", "kid",
        "marker_valid", "routable", "is_link", "link_subject", "link_relation",
        "link_target_kid", "link_target_pos", "link_target_exact", "deref_routable",
    )
    expected = {name: np.array(getattr(bank, name), copy=True) for name in array_fields}
    metadata = {name: (getattr(bank, name).shape, getattr(bank, name).dtype) for name in array_fields}

    for name in array_fields:
        exported = getattr(bank, name)
        with pytest.raises(ValueError):
            exported.setflags(write=True)
        with pytest.raises(ValueError):
            exported.flat[0] = 99
        assert (exported.shape, exported.dtype) == metadata[name]
        assert np.array_equal(exported, expected[name])

    assert bank.is_fresh(store)
    tensors = bank.tensors()
    for name in array_fields:
        if name in {"usable", "kid"}:
            continue  # symbolic/export metadata not present in the model tensor map
        assert np.array_equal(tensors[name].cpu().numpy(), expected[name])


@pytest.mark.parametrize("converter", [torch.from_numpy, torch.as_tensor], ids=["from_numpy", "as_tensor"])
def test_pytorch_zero_copy_writes_cannot_mutate_encapsulated_bank_arrays(converter):
    store = MVCCStore(seed=0)
    target = store.write(3, 1, 7)
    store.link(4, 1, target)
    bank = bank_from_store(store)
    array_fields = (
        "subject", "relation", "obj", "marker", "active", "usable", "kid",
        "marker_valid", "routable", "is_link", "link_subject", "link_relation",
        "link_target_kid", "link_target_pos", "link_target_exact", "deref_routable",
    )
    expected = {name: np.array(getattr(bank, name), copy=True) for name in array_fields}

    for name in array_fields:
        exposed = getattr(bank, name)
        tensor = converter(exposed)
        current = tensor.reshape(-1)[0].item()
        replacement = (not current) if exposed.dtype == np.bool_ else current + 1
        tensor.reshape(-1)[0] = replacement
        assert np.array_equal(getattr(bank, name), expected[name]), name

    assert bank.is_fresh(store)
    tensors = bank.tensors()
    for name in array_fields:
        if name not in {"usable", "kid"}:
            assert np.array_equal(tensors[name].cpu().numpy(), expected[name]), name


def test_freshness_detects_direct_cell_and_log_mutation_without_revision_change():
    store = MVCCStore(seed=0)
    kid = store.write(3, 1, 7)
    bank = bank_from_store(store)
    revision = store.revision
    store.cells[kid].active.obj = 9
    assert store.revision == revision
    assert not bank.is_fresh(store)

    fresh = bank_from_store(store)
    store.log.append(("external", {}))
    assert store.revision == revision
    assert not fresh.is_fresh(store)


def test_evaluation_rejects_a_provided_stale_store_bank_before_model_use():
    from so.evaluation import predict

    rng = np.random.default_rng(0)
    world = World.sample(rng, 16, 2, 20, 1)
    store = MVCCStore(seed=0)
    fact = world.facts[0]
    kid = store.write(fact.subject, fact.relation, fact.obj)
    stale = bank_from_store(store)
    store.update(kid, (fact.obj + 1) % world.n_entities)
    with pytest.raises(StaleSnapshotError):
        predict(object(), store, world, [], bank=stale)


def test_evaluation_rejects_store_mutation_performed_inside_model_forward():
    from so.evaluation import predict

    rng = np.random.default_rng(0)
    world = World.sample(rng, 16, 2, 20, 1)
    store = MVCCStore(seed=0)
    fact = world.facts[0]
    kid = store.write(fact.subject, fact.relation, fact.obj)
    bank = bank_from_store(store)
    query = Query("fwd", fact.subject, (fact.relation,), (world.surface_of(fact.relation, 0),))

    class MutatingModel:
        cfg = type("Config", (), {"max_hops": 1})()

        def __call__(self, tensors, mode, start, rels, hop_valid, **kwargs):
            store.update(kid, (fact.obj + 1) % world.n_entities)
            batch = int(mode.shape[0])
            logits = torch.zeros(batch, world.n_entities + 1)
            routing = torch.zeros(batch, 1, bank.size + 1)
            return logits, routing, {}

    with pytest.raises(StaleSnapshotError):
        predict(MutatingModel(), store, world, [query], bank=bank)
