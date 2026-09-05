"""What the E-000056 epoch arm actually measures, pinned so it cannot be over-read again.

WHERE THIS COMES FROM.  Two questions this repository registered against itself and never ran:

  * §31.12: the marker gate's operational radius is 0.90 against a declared 0.35, and "making it
    implement the predicate is a change to the data, not to the architecture ... That needs a
    training run and is not yet evaluated."
  * §31.47 (E-000056): "whether the learned acceptance function can host the freshness predicate at
    all ... a rotation of the marker centre large enough for the frozen gate to reject the previous
    epoch is, on the face of the geometry, also large enough for it to reject the current one."

`so/experiments/e000056_learned_custodian.py` runs both.  The first version of these tests asserted
that a frozen learned gate hosts no revocable epoch -- which is true, and vacuous: it is true of every
acceptance function that exists, learned or not, because a checker reading only the presented marker
returns the same number for the previous epoch whatever the rotation.  Asserting it proved nothing
about learning, and the sentence it was written to support ("revocation cannot be delegated to a
frozen LEARNED acceptance function") is withdrawn.

These tests pin what the arm does measure:

  A. THE LEARNED GATE AND THE STORE'S OWN HAND-WRITTEN PREDICATE BEHAVE IDENTICALLY under the epoch
     rotation.  Both host zero revocable epochs.  So the word carrying that result is "frozen", never
     "learned" and never "neural" -- and any claim resting on it is a claim about frozen references.

  B. LIVENESS IS THE VARIABLE THAT MOVES.  The same exact predicate reading the LIVE centre hosts
     many revocable epochs.  Frozen reference -> none; live reference -> many; and this is
     independent of whether the checker is learned.

  C. THE STORE PREDICATE IS EXACT ON THE WHOLE DOMAIN, where the learned gate trained on the
     programme's own separated classes accepts the entire annulus the store calls deleted.  That is
     §31.12's finding, and `so/experiments/e000056_learned_custodian.py` arm F shows it is closable.

None of this is a novelty claim.  It is the premise the E-000056 record is read under.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from so.data import invalid_markers, valid_markers  # noqa: E402
from so.experiments.e000056_learned_custodian import (  # noqa: E402
    BANDS,
    DECLARED_RADIUS,
    _accept,
    _epoch_scan,
    _fit,
    _labelled,
    deleted_region,
    gate_mlp,
    make_centre,
    shell,
)


def _margin_gate(seed: int, steps: int = 600):
    """The programme's own training distribution, at a budget a test can afford."""
    centre = make_centre(seed)

    def sampler(rng):
        return _labelled(valid_markers(rng, centre, 240), invalid_markers(rng, centre, 16))

    return _fit(gate_mlp(), sampler, seed, steps), centre


def _scans(seed: int):
    gate, centre = _margin_gate(seed)
    rng = np.random.default_rng(20 + seed)
    prev = valid_markers(rng, centre, 2000)

    # One rotation per grid point, shared by every checker — drawing a fresh one per call would
    # compare markers minted at one centre against a different centre and collapse `new`.
    rotations = {d: shell(rng, centre, np.full(1, d), 1)[0] for d in BANDS}
    new_markers = {d: valid_markers(rng, c, 2000) for d, c in rotations.items()}

    def store(m, c):
        return float((np.linalg.norm(m - c[None, :], axis=1) <= DECLARED_RADIUS).mean())

    learned = _epoch_scan(lambda d: (_accept(gate, new_markers[d]), _accept(gate, prev)))
    store_frozen = _epoch_scan(lambda d: (store(new_markers[d], centre), store(prev, centre)))
    store_live = _epoch_scan(lambda d: (store(new_markers[d], rotations[d]), store(prev, rotations[d])))
    return learned, store_frozen, store_live


def test_a_learned_gate_and_hand_written_predicate_agree_under_rotation():
    """The epoch arm does not separate learned from hand-coded: both host zero."""
    learned, store_frozen, _ = _scans(0)
    assert learned["hostable_revocable_epochs"] == 0
    assert store_frozen["hostable_revocable_epochs"] == 0


def test_a_the_previous_epoch_stays_accepted_at_every_rotation():
    """The concrete reason, and the reason it says nothing about learning."""
    learned, store_frozen, _ = _scans(1)
    for row in learned["grid"]:
        assert row["prev"] >= 0.99
    for row in store_frozen["grid"]:
        assert row["prev"] >= 0.99


def test_b_liveness_is_the_variable_that_moves():
    """The same exact predicate, reading the live centre, does host revocable epochs."""
    _, store_frozen, store_live = _scans(2)
    assert store_frozen["hostable_revocable_epochs"] == 0
    assert store_live["hostable_revocable_epochs"] > 0
    # and it needs a real rotation to do it: the old markers only leave the ball once the centre
    # has moved by at least about the validity radius (measured 0.40 against a declared 0.35).
    assert store_live["min_revoking_distance"] >= DECLARED_RADIUS


def test_c_the_store_predicate_is_exact_on_the_whole_domain():
    """`marker_valid` is a distance comparison: no sample of it can be wrong."""
    rng = np.random.default_rng(13)
    centre = make_centre(2)
    inside = shell(rng, centre, rng.uniform(0.0, DECLARED_RADIUS - 1e-6, size=20_000), 20_000)
    outside = deleted_region(rng, centre, 20_000)

    def store_valid(m):
        return np.linalg.norm(m - centre[None, :], axis=1) <= DECLARED_RADIUS

    assert store_valid(inside).all()
    assert not store_valid(outside).any()


def test_c_the_margin_trained_gate_accepts_the_annulus_the_store_deletes():
    """§31.12's finding: the gate certifies the margin it was shown, not the declared predicate."""
    gate, centre = _margin_gate(0)
    rng = np.random.default_rng(14)
    annulus = shell(rng, centre, rng.uniform(DECLARED_RADIUS + 0.01, 0.69, size=4000), 4000)
    assert _accept(gate, annulus) > 0.5


def test_bands_cover_the_declared_radius():
    """A grid that skipped the declared radius would make every row above unreadable."""
    assert DECLARED_RADIUS in BANDS
