"""Where the lifecycle-authority boundary can lie, pinned as behaviour.

WHERE THIS COMES FROM.  Two questions this repository registered against itself and never ran:

  * §31.12: the marker gate's operational radius is 0.90 against a declared 0.35, and "making it
    implement the predicate is a change to the data, not to the architecture ... That needs a
    training run and is not yet evaluated."
  * §31.47 (E-000056): "whether the learned acceptance function can host the freshness predicate at
    all ... a rotation of the marker centre large enough for the frozen gate to reject the previous
    epoch is, on the face of the geometry, also large enough for it to reject the current one."

`so/experiments/e000056_learned_custodian.py` measures both.  These tests pin the two parts of the
result that are structural rather than statistical, so they fail loudly if the substrate changes:

  A. A FROZEN ACCEPTANCE REGION CANNOT HOST A REVOCABLE EPOCH.  A credential rotation needs a centre
     displacement that is simultaneously usable (the frozen gate accepts the new epoch) and revocable
     (the frozen gate rejects an epoch at that displacement).  Those two sets are disjoint for any
     monotone-in-distance acceptance function, and measurably disjoint for the trained gate: every
     epoch that ever worked keeps working.  This is why revocation cannot be delegated to the gate.

  B. THE STORE-SIDE PREDICATE IS EXACT WHERE THE LEARNED ONE IS NOT.  `MVCCStore.marker_valid` is a
     distance comparison and is exact on the whole domain by construction; the learned gate is a
     sampled approximation of it and has a false-accept rate.  The asymmetry is the reason the
     authority boundary belongs before materialisation, not at the point of neural consumption.

Neither test claims novelty for anything.  They pin the premise that the E-000056 record reads.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from so.experiments.e000056_learned_custodian import (  # noqa: E402
    BANDS,
    DECLARED_RADIUS,
    _accept,
    _fit,
    _labelled,
    deleted_region,
    gate_mlp,
    make_centre,
    shell,
)
from so.data import invalid_markers, valid_markers  # noqa: E402


def _margin_gate(seed: int, steps: int = 600):
    """The programme's own training distribution, at a budget a test can afford."""
    centre = make_centre(seed)

    def sampler(rng):
        npos = 240
        return _labelled(valid_markers(rng, centre, npos), invalid_markers(rng, centre, 16))

    return _fit(gate_mlp(), sampler, seed, steps), centre


def test_a_frozen_gate_hosts_no_revocable_epoch():
    """Usable displacements and revocable displacements do not overlap."""
    gate, centre = _margin_gate(0)
    rng = np.random.default_rng(11)

    usable, revocable = [], []
    for d in BANDS:
        epoch_centre = shell(rng, centre, np.full(1, d), 1)[0]
        a = _accept(gate, valid_markers(rng, epoch_centre, 1000))
        if a >= 0.95:
            usable.append(d)
        if a <= 0.05:
            revocable.append(d)

    assert usable, "the gate accepts nothing: the fixture is broken, not the claim"
    assert revocable, "the gate accepts everything: the fixture is broken, not the claim"
    # The two conditions a rotation scheme needs are never satisfied by the same displacement.
    assert not (set(usable) & set(revocable))
    # And it misses by a margin rather than a hair.
    assert min(revocable) - max(usable) >= 0.05


def test_a_the_gate_accepts_its_own_training_centre_forever():
    """The concrete reason: the epoch the gate was frozen on is always accepted."""
    gate, centre = _margin_gate(1)
    rng = np.random.default_rng(12)
    assert _accept(gate, valid_markers(rng, centre, 2000)) >= 0.99


def test_b_the_store_predicate_is_exact_on_the_whole_domain():
    """`marker_valid` is a distance comparison: no sample of it can be wrong."""
    rng = np.random.default_rng(13)
    centre = make_centre(2)
    inside = shell(rng, centre, rng.uniform(0.0, DECLARED_RADIUS - 1e-6, size=20_000), 20_000)
    outside = deleted_region(rng, centre, 20_000)

    def store_valid(m):
        return np.linalg.norm(m - centre[None, :], axis=1) <= DECLARED_RADIUS

    assert store_valid(inside).all()
    assert not store_valid(outside).any()


def test_b_the_learned_gate_is_not_exact_where_the_store_is():
    """The same domain, asked of the gate: the annulus the store deletes is accepted."""
    gate, centre = _margin_gate(0)
    rng = np.random.default_rng(14)
    annulus = shell(rng, centre, rng.uniform(DECLARED_RADIUS + 0.01, 0.69, size=4000), 4000)
    assert _accept(gate, annulus) > 0.5
