import numpy as np
import pytest

from so.mvcc import MVCCStore, Status


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
