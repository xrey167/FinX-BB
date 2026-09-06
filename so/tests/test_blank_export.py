"""``MVCCStore(blank_export="foreign")``: how a BLANKed link is exported (ledger §31.48).

The recorded export of a blanked link is the row's own key ("self"). Measured on the retrained E-000020
seed-0 adapter with no weight changed, that export is read as a wrong entity -- the alias's own subject --
in 0.26-0.475 of reads, while an export pointing at an absent key under ANOTHER subject reads UNKNOWN at
0.93-0.98. These tests pin the store half: under the option the exported key has no live cell and a
different subject; the default is byte-identical to the recorded behaviour; the derived marker follows
the exported key; and the fallback when every subject is taken is the self key, never a live one.
"""

from __future__ import annotations

import numpy as np

from so.mvcc import MVCCStore


def _pod(blank_export: str, content_markers: bool = False) -> MVCCStore:
    st = MVCCStore(marker_dim=16, seed=3, content_markers=content_markers, blank_export=blank_export)
    f = st.write(5, 2, 40)
    st.link(7, 2, f)          # alias under subject 7, same relation as the target
    st.write(0, 2, 41)        # subject 0 already holds relation 2: the foreign key must skip it
    st.write(1, 3, 42)        # subject 1 holds a different relation: (1, 2) is free
    return st


def _blanked_row(st: MVCCStore):
    b = st.bank()
    i = int(np.where(b["is_link"])[0][0])
    return int(b["link_subject"][i]), int(b["link_relation"][i]), b


def test_default_exports_the_row_s_own_key():
    st = _pod("self")
    st.blank(2)
    ls, lr, _ = _blanked_row(st)
    assert (ls, lr) == (7, 2)


def test_foreign_export_is_an_absent_key_under_another_subject():
    st = _pod("foreign")
    st.blank(2)
    ls, lr, b = _blanked_row(st)
    assert lr == 2 and ls != 7
    assert ls not in (0, 5), "subjects 0 and 5 hold relation 2 and must be skipped"
    live = {(int(s), int(r)) for s, r, a in zip(b["subject"], b["relation"], b["active"]) if a}
    assert (ls, lr) not in live


def test_foreign_export_falls_back_to_the_own_key_when_every_seen_subject_is_taken():
    st = MVCCStore(marker_dim=16, seed=1, blank_export="foreign")
    f = st.write(0, 0, 9)
    st.link(1, 0, f)
    st.write(2, 0, 9); st.write(3, 0, 9)      # subjects 0..3 all hold relation 0; 1 is the row itself
    st.blank(2)
    ls, lr, _ = _blanked_row(st)
    assert (ls, lr) == (1, 0), "no seen subject is free, so the export is the row's own key (never an unseen index)"
    st.write(4, 1, 9)                          # subject 4 seen, relation 0 free there
    ls, lr, _ = _blanked_row(st)
    assert (ls, lr) == (4, 0)


def test_target_and_tombstone_exports_are_untouched_by_the_option():
    st = _pod("foreign")
    ls, lr, _ = _blanked_row(st)
    assert (ls, lr) == (5, 2)          # live target: its key
    st.evict(1)
    ls, lr, _ = _blanked_row(st)
    assert (ls, lr) == (5, 2)          # evicted: tombstone key kept (dangling)


def test_derived_marker_follows_the_foreign_export():
    a = _pod("self", content_markers=True)
    b = _pod("foreign", content_markers=True)
    a.blank(2); b.blank(2)
    ma = a.bank()["marker"][1]; mb = b.bank()["marker"][1]
    assert not np.allclose(ma, mb), "the exported content differs, so the derived marker must differ"
    assert b.marker_valid(mb)


def test_bad_option_is_refused():
    try:
        MVCCStore(blank_export="other")
    except ValueError:
        return
    raise AssertionError("an unknown blank_export value must be refused")
