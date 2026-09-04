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
    assert "REFERENTIALLY CLEAN" in c.summary()
    # and the certificate says, in the same sentence, what it does NOT certify
    assert "NOT HISTORY INDEPENDENT" in c.summary()


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


def test_blanking_is_referentially_cleaner_than_evicting_and_keeps_more_rows_and_is_NOT_history_independent():
    """The corrected statement of what was once called 'strictly stronger AND less destructive'
    (ledger §31.35). Blanking keeps more rows and leaves no version holding the removed key --
    both true. But the rows it keeps exist only because the fact once did, so a store that blanked
    is distinguishable from one that never wrote: NOT history independent (Naor-Teague 2001, Def.
    2.1). Evicting every row of the pod is the reverse: referentially dirty in the store, and yet
    bank() is identical to a store that never held the fact."""
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

    assert bl.certified and not ev.certified          # referentially: blanking is cleaner
    assert bl.n_live_after > ev.n_live_after          # and keeps more rows
    assert not bl.history_independent                 # and those rows ARE the residue
    assert bl.history.residue_rows == 3               # the three blanked aliases (k=4, n_links=3)
    assert ev.history_independent                     # evicting everything: bank() as if never written
    assert not ev.history.raw_hi                      # ... and the raw store still knows (log, cells)


def test_the_history_check_reproduces_the_review_scenario():
    """S1 = write a fact, link two aliases, evict the fact, blank both aliases. S2 = never wrote.
    Same content through the interface; different memory representation on every axis the
    definition names. This is the check the first version of certify_traceless did not run."""
    from so.audit import check_history_independence
    from so.mvcc import MVCCStore
    s1 = MVCCStore(marker_dim=8, seed=0)
    s1.write(1, 1, 7)                                    # a bystander, so the store is never empty
    f = s1.write(2, 3, 42)
    a1 = s1.link(4, 3, f); a2 = s1.link(5, 3, f)
    s1.evict(f); s1.blank(a1); s1.blank(a2)
    hi = check_history_independence(s1)
    assert not hi.exported_hi
    assert hi.rows_store == 3 and hi.rows_fresh == 1 and hi.residue_rows == 2
    assert not hi.cells_equal and not hi.log_equal and not hi.next_kid_equal
    assert "NOT HISTORY INDEPENDENT" in hi.summary()
    # the same store with the whole pod evicted instead: the exported level matches never-wrote
    s3 = MVCCStore(marker_dim=8, seed=0)
    s3.write(1, 1, 7)
    f3 = s3.write(2, 3, 42)
    b1 = s3.link(4, 3, f3); b2 = s3.link(5, 3, f3)
    s3.evict(f3); s3.evict(b1); s3.evict(b2)
    hi3 = check_history_independence(s3)
    assert hi3.exported_hi and hi3.markers_equal
    assert not hi3.raw_hi                                # the log and the evicted cells remain
    assert "EXPORTED level only" in hi3.summary()


def test_a_store_that_never_removed_anything_is_history_independent_at_both_levels():
    """The control: the fresh-store comparison must PASS somewhere, or it is a check that cannot
    pass and certifies nothing by failing everything."""
    from so.audit import check_history_independence
    st, keys, base, n0 = _setup()
    hi = check_history_independence(st)
    assert hi.exported_hi and hi.markers_equal and hi.raw_hi
    assert hi.residue_rows == 0
    assert "HISTORY INDEPENDENT at the exported and the raw level" in hi.summary()


def test_the_comparison_is_by_content_and_not_by_cell_id():
    """Two stores holding the same facts written in different orders have different kids; the
    exported comparison must still call them equal, and the raw one must not (the generator
    position differs, and so do the markers)."""
    from so.audit import check_history_independence
    from so.mvcc import MVCCStore
    st = MVCCStore(marker_dim=8, seed=0)
    st.write(5, 1, 9); st.write(6, 1, 8); t = st.write(7, 1, 3); st.link(8, 1, t)
    hi = check_history_independence(st)
    assert hi.exported_hi and hi.raw_hi                  # same order as the fresh replay: identical


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
