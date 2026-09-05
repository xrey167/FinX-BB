"""History-independent markers (``MVCCStore(content_markers=True)``), the store half of E-000053.

E-000051 (ledger §31.41) read the store's seeded marker generator as a history channel: a store that
wrote a pod and evicted every row of it differs from one that never wrote it in the markers of every
row written after the pod, and a frozen reader separates the two on bystander queries at AUC 0.948.
These tests pin the store-level fact the reader-level run depends on: under the option
``check_history_independence`` reports ``markers_equal`` after CASCADE, and without it it does not.
Every test has a way to fail: the fresh store in the audit could be built without the option, a LINK
marker could be derived from the target's cell id (which encodes write order), a blanked row could
keep the marker of the pointer it no longer holds, the derived draw could leave the valid region.
"""

from __future__ import annotations

import numpy as np

from so.audit import check_history_independence
from so.mvcc import MVCCStore


def _pod_store(content_markers: bool, seed: int = 0) -> MVCCStore:
    """A bystander, a pod (fact + two aliases), and four rows written after the pod."""
    st = MVCCStore(marker_dim=16, seed=seed, content_markers=content_markers)
    st.write(1, 1, 7)
    f = st.write(2, 3, 42)
    a1 = st.link(4, 3, f); a2 = st.link(5, 3, f)
    for s in range(10, 14):
        st.write(s, 1, s + 100)
    t = st.write(20, 2, 9); st.link(21, 2, t)
    return st, (f, a1, a2)


def test_cascade_is_marker_equal_only_under_the_option():
    """The claim and its control in one place: same operations, same seed, the option is the only
    difference, and the column flips."""
    for flag, expect in ((False, False), (True, True)):
        st, (f, a1, a2) = _pod_store(flag)
        st.evict(f); st.evict(a1); st.evict(a2)
        hi = check_history_independence(st)
        assert hi.exported_hi and hi.residue_rows == 0          # E-000046: content is HI either way
        assert hi.markers_equal is expect, (flag, hi)


def test_a_never_wrote_store_and_a_cascaded_store_export_identical_banks():
    """Bit-identical ``bank()`` arrays, markers included, in the same row order: this is what makes
    E-000051's CASCADE-vs-NEVER cell a bank identity under the option, and the test that says so."""
    st, (f, a1, a2) = _pod_store(True)
    st.evict(f); st.evict(a1); st.evict(a2)
    never = MVCCStore(marker_dim=16, seed=0, content_markers=True)
    never.write(1, 1, 7)
    for s in range(10, 14):
        never.write(s, 1, s + 100)
    t = never.write(20, 2, 9); never.link(21, 2, t)
    b, n = st.bank(), never.bank()
    for col in ("subject", "relation", "obj", "is_link", "link_subject", "link_relation", "active", "marker"):
        assert np.array_equal(b[col], n[col]), col
    assert not np.array_equal(b["kid"], n["kid"])                # the cell ids still carry the history; the reader never sees them


def test_the_generator_scheme_is_untouched_by_default():
    """Default off: the recorded runs' markers reproduce bit for bit (replay determinism, E-000001a)."""
    a = MVCCStore(marker_dim=16, seed=3); b = MVCCStore(marker_dim=16, seed=3)
    assert not a.content_markers
    for st in (a, b):
        k = st.write(1, 1, 5); st.link(2, 1, k); st.update(k, 6); st.shred(k)
    assert np.array_equal(a.bank()["marker"], b.bank()["marker"])
    assert a.state_hash() == b.state_hash()


def test_derived_markers_are_valid_and_from_the_same_family():
    """Every derived valid marker is inside the gate's radius and its spread matches the generator's
    (scale 0.05 around the centre); a derived invalid marker is at least 2r away, like ``shred``."""
    st = MVCCStore(marker_dim=16, seed=1, content_markers=True)
    gen = MVCCStore(marker_dim=16, seed=1)
    kids = [st.write(s, r, o) for s in range(30) for r in range(3) for o in (s + r,)]
    m = st.bank()["marker"].astype(float)
    assert all(st.marker_valid(row) for row in m)
    d_derived = np.linalg.norm(m - st.marker_centre, axis=1)
    d_gen = np.array([np.linalg.norm(gen.new_valid_marker() - gen.marker_centre) for _ in range(len(kids))])
    assert abs(d_derived.mean() - d_gen.mean()) < 0.03 and abs(d_derived.std() - d_gen.std()) < 0.03
    st.shred(kids[0])
    assert np.linalg.norm(st.cells[kids[0]].versions[0].marker - st.marker_centre) >= 2 * st.valid_radius
    assert st.marker_invariant_holds()


def test_a_link_marker_follows_the_pointed_at_key_not_the_cell_id():
    """Two stores hold the same alias -> fact pointer with different cell ids for the fact (one store
    wrote and evicted a row first). A marker derived from the cell id would differ; from the key it
    must not."""
    a = MVCCStore(marker_dim=16, seed=0, content_markers=True)
    junk = a.write(99, 9, 99); a.evict(junk)
    fa = a.write(2, 3, 42); la = a.link(4, 3, fa)
    b = MVCCStore(marker_dim=16, seed=0, content_markers=True)
    fb = b.write(2, 3, 42); lb = b.link(4, 3, fb)
    assert fa != fb
    assert np.array_equal(a.cells[la].versions[0].marker, b.cells[lb].versions[0].marker)
    assert np.array_equal(a.cells[fa].versions[0].marker, b.cells[fb].versions[0].marker)


def test_blank_and_relink_re_derive_and_the_invariant_survives_the_lifecycle():
    """A blanked row exports its own key, so its marker must no longer be the one derived from the
    pointer it held (otherwise the marker still names the removed key). RELINK and UPDATE make new
    versions and are re-derived by construction; the invariant is checked after each step."""
    st = MVCCStore(marker_dim=16, seed=0, content_markers=True)
    f = st.write(2, 3, 42); g = st.write(6, 3, 43)
    a = st.link(4, 3, f)
    before = st.cells[a].versions[0].marker.copy()
    st.blank(a)
    assert not np.array_equal(before, st.cells[a].versions[0].marker)
    assert st.marker_invariant_holds()
    st.relink(a, g); assert st.marker_invariant_holds()
    st.update(g, 44); assert st.marker_invariant_holds()
    st.evict(g); assert st.marker_invariant_holds()                 # dangling: exports the tombstone key it was derived from
    st.restore(g); st.rollback(g, 1); assert st.marker_invariant_holds()
    st.shred(f); assert st.marker_invariant_holds()
    st.resign(f); assert st.marker_invariant_holds()
    assert np.array_equal(st.cells[f].versions[0].marker, MVCCStore(marker_dim=16, seed=0, content_markers=True)
                          .derived_marker(("FACT", 2, 3, 42)))


def test_replay_and_clone_carry_the_option_and_the_secret():
    st = MVCCStore(marker_dim=16, seed=0, content_markers=True, marker_key=b"per-store secret")
    f = st.write(2, 3, 42); st.link(4, 3, f); st.update(f, 43)
    clone = st.clone_by_replay()
    assert clone.content_markers and clone.marker_key == st.marker_key
    assert np.array_equal(clone.bank()["marker"], st.bank()["marker"])
    other = MVCCStore(marker_dim=16, seed=0, content_markers=True, marker_key=b"another secret")
    other.write(2, 3, 42)
    assert not np.array_equal(other.bank()["marker"][0], st.bank()["marker"][0])   # the secret is load-bearing


def test_the_side_effect_identical_content_identical_marker():
    """Registered, not hidden: a second write of the same fact carries the same marker under the
    option and a different one without. E-000053 measures whether the reader can see the difference."""
    for flag, same in ((True, True), (False, False)):
        st = MVCCStore(marker_dim=16, seed=0, content_markers=flag)
        k1 = st.write(2, 3, 42); k2 = st.write(2, 3, 42)
        d = float(np.linalg.norm(st.cells[k1].versions[0].marker - st.cells[k2].versions[0].marker))
        assert (d == 0.0) is same, (flag, d)
