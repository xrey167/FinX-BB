import numpy as np

from so.data import bank_from_store, bank_from_world, failing_hop_target
from so.mvcc import MVCCStore
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
