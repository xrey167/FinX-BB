"""The deletion-disclosure channel (E-000035): a pod's aliases point at what was removed.

The finding is mechanical and needs no model, so it can be pinned by tests rather than by a record
alone. What is asserted here is the asymmetry itself and the inversion that follows from it, because
both are properties of the store's own semantics and both are easy to lose in a refactor.
"""
import numpy as np

from so.closure import fact_closure, pod_keys
from so.experiments.e000035_deletion_disclosure import dangling_targets, live_keys
from so.mvcc import MVCCStore


def _pod(k_aliases=2, obj=7):
    st = MVCCStore(marker_dim=16, seed=0)
    target = st.write(3, 1, obj, provenance="t")
    aliases = [st.link(10 + i, 1, target, provenance=f"a{i}") for i in range(k_aliases)]
    return st, target, aliases


def _duplicates(k=3, obj=7):
    st = MVCCStore(marker_dim=16, seed=0)
    return st, [st.write(10 + i, 1, obj, provenance=f"c{i}") for i in range(k)]


def test_a_clean_pod_points_at_nothing_missing():
    st, _, _ = _pod()
    assert dangling_targets(st.bank()) == []


def test_evicting_a_pods_object_makes_every_alias_name_the_key_that_went():
    """The channel. The adversary reads the bank and nothing else."""
    st, target, aliases = _pod(k_aliases=3)
    st.evict(target)
    found = dangling_targets(st.bank())
    assert set(found) == {(3, 1)}                 # the removed key, named exactly
    assert len(found) == 3                        # once per surviving alias, all agreeing
    assert (3, 1) not in live_keys(st.bank())


def test_deleting_a_pods_object_names_it_too_through_the_tombstone():
    """DELETE drops the versions, and bank() falls back to tombstone_key, which is the same key."""
    st, target, _ = _pod(k_aliases=2)
    st.delete(target)
    assert set(dangling_targets(st.bank())) == {(3, 1)}


def test_a_duplicated_store_leaves_no_signpost_at_all():
    """The contrast: removing one copy leaves a store that looks like a store with one fewer copy."""
    st, kids = _duplicates(k=3)
    st.evict(kids[0])
    assert dangling_targets(st.bank()) == []


def test_the_closure_inverts_between_the_two_guarantees():
    """The result, as one assertion block.

    Unreachable to the reader: a pod costs one record, k duplicates cost k.
    No trace left in the bank: a pod costs k, k duplicates cost the one you were removing anyway.
    """
    k = 3
    pod, target, aliases = _pod(k_aliases=k - 1)
    dup, kids = _duplicates(k=k)

    # reachability
    assert fact_closure(pod, pod_keys(pod, target), obj=7).size == 1
    assert fact_closure(dup, [(10 + i, 1) for i in range(k)], obj=7).size == k

    # trace: the pod needs the object AND every alias; the duplicated store needs only the copy
    pod.evict(target)
    assert dangling_targets(pod.bank())
    for a in aliases:
        pod.evict(a)
    assert dangling_targets(pod.bank()) == []      # k records in total

    dup.evict(kids[0])
    assert dangling_targets(dup.bank()) == []      # 1 record, and it was the deletion itself


def test_blanking_the_pointer_closes_the_channel_and_makes_the_pointers_identical():
    """The mitigation and its price, as the experiment measures them."""
    st, target, _ = _pod(k_aliases=2)
    st.evict(target)
    b = st.bank()
    held = live_keys(b)
    blanked = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    rows = [i for i, l in enumerate(b["is_link"])
            if bool(l) and (int(b["link_subject"][i]), int(b["link_relation"][i])) not in held]
    assert rows
    for i in rows:
        blanked["link_subject"][i] = 0
        blanked["link_relation"][i] = 0
    assert (3, 1) not in dangling_targets(blanked)
    # and the price: every blanked pointer is now the same row, so which target went is unrecoverable
    assert len({(int(blanked["link_subject"][i]), int(blanked["link_relation"][i])) for i in rows}) == 1


def test_a_pointer_that_was_always_dangling_is_not_a_disclosure():
    """The control the experiment subtracts: E-000015 trains on pointers to nothing on purpose."""
    st = MVCCStore(marker_dim=16, seed=0)
    target = st.write(3, 1, 7, provenance="t")
    st.link(10, 1, target, provenance="a")
    ghost = st.write(90, 5, 1, provenance="ghost")
    st.link(11, 1, ghost, provenance="dangler")
    st.delete(ghost)
    base = set(dangling_targets(st.bank()))
    assert base == {(90, 5)}                       # dangling before any deletion of interest
    st.evict(target)
    new = [k for k in dangling_targets(st.bank()) if k not in base]
    assert set(new) == {(3, 1)}                    # only the new one counts
