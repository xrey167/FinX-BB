"""The tracelessness certificate, and the four ways it must refuse."""

from __future__ import annotations

import pytest

from so.audit import certify_traceless
from so.closure import fact_closure
from so.experiments.e000035_deletion_disclosure import dangling_targets
from so.experiments.e000041_traceless_cost import build

OBJ = 7


def _links(st):
    b = st.bank()
    return [int(b["kid"][i]) for i, l in enumerate(b["is_link"]) if bool(l)]


def _setup(k=4, n_links=3, seed=0):
    st, _kids, keys = build(k, n_links, seed)
    base = tuple(dangling_targets(st.bank()))
    n0 = int(st.bank()["kid"].shape[0])
    return st, keys, base, n0


def test_the_closure_alone_is_unreachable_but_not_traceless():
    """Exactly the gap E-000041 measured: U is paid, T is not."""
    st, keys, base, n0 = _setup()
    ops = 0
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid); ops += 1
    c = certify_traceless(st, keys, OBJ, baseline=base, ops=ops, n_live_before=n0)
    assert c.unreachable
    assert not c.certified
    assert not c.exported_clean
    assert "UNREACHABLE, NOT TRACELESS" in c.summary()


def test_blanking_the_aliases_earns_the_certificate():
    st, keys, base, n0 = _setup()
    ops = 0
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid); ops += 1
    for kid in _links(st):
        st.blank(kid); ops += 1
    c = certify_traceless(st, keys, OBJ, baseline=base, ops=ops, n_live_before=n0)
    assert c.certified
    assert c.raw_clean and c.exported_clean and c.store_retained
    assert "TRACELESS, CERTIFIED" in c.summary()


def test_evicting_the_aliases_is_clean_in_the_VIEW_and_not_in_the_STORE():
    """EVICT retains the row's data on purpose -- that is what it is for -- so the evicted alias goes
    on holding the removed key internally. The exported view is clean and the store is not, which is
    the same shape as the opaque case: the disclosure moved behind an interface."""
    st, keys, base, n0 = _setup()
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    for kid in _links(st):
        st.evict(kid)
    c = certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)
    assert c.unreachable and c.exported_clean
    assert not c.raw_clean and not c.certified
    assert "the interface hid the disclosure rather than removing it" in c.summary()


def test_blanking_is_strictly_stronger_than_evicting_and_keeps_more_rows():
    st, keys, base, n0 = _setup()
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    for kid in _links(st):
        st.evict(kid)
    ev = certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)

    st2, keys2, base2, n02 = _setup()
    for kid in fact_closure(st2, keys2, obj=OBJ).records:
        st2.evict(kid)
    for kid in _links(st2):
        st2.blank(kid)
    bl = certify_traceless(st2, keys2, OBJ, baseline=base2, n_live_before=n02)

    assert bl.certified and not ev.certified          # strictly stronger guarantee
    assert bl.n_live_after > ev.n_live_after          # AND strictly less destruction


def test_a_still_reachable_fact_is_refused():
    st, keys, base, n0 = _setup()
    c = certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)
    assert not c.certified and not c.unreachable
    assert "no deletion to be traceless about" in c.summary()


def test_emptying_the_store_is_refused_rather_than_certified():
    """Tracelessness by removing everything measures nothing."""
    st, keys, base, n0 = _setup()
    b = st.bank()
    for kid in [int(x) for x in b["kid"]][: int(0.8 * n0)]:
        st.evict(kid)
    c = certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)
    assert not c.store_retained and not c.certified
    assert "the store was emptied" in c.summary()


def test_an_empty_store_raises_instead_of_passing_vacuously():
    st, keys, base, n0 = _setup()
    for kid in [int(x) for x in st.bank()["kid"]]:
        st.evict(kid)
    with pytest.raises(ValueError, match="no live rows"):
        certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)


def test_a_preexisting_dangling_pointer_does_not_fail_this_deletion():
    st, keys, base, n0 = _setup()
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    for kid in _links(st):
        st.blank(kid)
    stale = certify_traceless(st, keys, OBJ, baseline=base, n_live_before=n0)
    assert stale.certified                      # the baseline is what makes this true
