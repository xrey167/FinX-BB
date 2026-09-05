"""BLANK: clearing a link's target, and the two ways it must refuse."""

from __future__ import annotations

import pytest

from so.closure import fact_closure
from so.experiments.e000035_deletion_disclosure import dangling_targets
from so.experiments.e000041_traceless_cost import build
from so.reference import ReferenceResolver
from so.world import Query

OBJ = 7


def _unreachable(st, keys):
    return all(ReferenceResolver(st).resolve(Query("fwd", a, (r,), (0,))).answer != OBJ
               for a, r in keys)


def _links(st):
    b = st.bank()
    return [int(b["kid"][i]) for i, l in enumerate(b["is_link"]) if bool(l)]


def test_blank_closes_the_dangling_channel_the_closure_opens():
    st, _kids, keys = build(4, 3, 0)
    base = set(dangling_targets(st.bank()))
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    assert [k for k in dangling_targets(st.bank()) if k not in base]      # the channel is open
    for kid in _links(st):
        st.blank(kid)
    assert not [k for k in dangling_targets(st.bank()) if k not in base]  # and blanking closes it


def test_blank_keeps_the_alias_rows_live_where_evict_removes_them():
    """The whole point: the key still resolves, to UNKNOWN, instead of ceasing to exist."""
    st, _kids, keys = build(4, 3, 0)
    before = int(st.bank()["kid"].shape[0])
    n_links = len(_links(st))
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    after_closure = int(st.bank()["kid"].shape[0])
    for kid in _links(st):
        st.blank(kid)
    assert int(st.bank()["kid"].shape[0]) == after_closure          # blanking removes no row
    assert len(_links(st)) == n_links                                # and every alias survives
    assert after_closure < before                                    # only the closure removed rows


def test_the_fact_is_still_unreachable_after_blanking():
    st, _kids, keys = build(4, 3, 0)
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    for kid in _links(st):
        st.blank(kid)
    assert _unreachable(st, keys)


def test_blank_refuses_a_fact_cell():
    """An operation that quietly does nothing on the wrong input is how a certificate goes hollow."""
    st, _kids, keys = build(4, 1, 0)
    b = st.bank()
    facts = [int(b["kid"][i]) for i, l in enumerate(b["is_link"]) if not bool(l)]
    with pytest.raises(ValueError, match="blank is for LINK cells"):
        st.blank(facts[0])


def test_blank_on_an_evicted_cell_is_allowed_and_takes_effect_on_restore():
    """Not a refusal, and the reason is in ``_alive``: EVICTED is deliberately reachable so that
    RESTORE and ROLLBACK still work. Blanking an evicted row is therefore meaningful rather than a
    no-op -- it clears the target the row would come back with."""
    st, _kids, keys = build(4, 3, 0)
    base = set(dangling_targets(st.bank()))
    for kid in fact_closure(st, keys, obj=OBJ).records:
        st.evict(kid)
    link = _links(st)[0]
    st.evict(link)
    st.blank(link)                                   # allowed on an evicted row
    st.restore(link)
    b = st.bank()
    i = list(b["kid"]).index(link)
    assert bool(b["is_link"][i])
    assert (int(b["link_subject"][i]), int(b["link_relation"][i])) not in \
        {k for k in dangling_targets(b) if k not in base}


def test_blank_refuses_a_deleted_cell():
    st, _kids, keys = build(4, 3, 0)
    kid = _links(st)[0]
    st.delete(kid)
    with pytest.raises(KeyError, match="does not exist or was deleted"):
        st.blank(kid)


def test_blanking_a_duplicated_store_is_a_no_op_because_there_are_no_links():
    """T = k there is paid entirely in deletions; there is no reference to repair."""
    st, _kids, keys = build(4, 0, 0)
    assert _links(st) == []
