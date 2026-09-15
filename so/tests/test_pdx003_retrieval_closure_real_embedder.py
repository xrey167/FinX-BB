"""Tests for PDX-003 — E-000033's protocol with one declared component swapped.

The danger this experiment carries is specific: swapping a component of a pre-registered experiment
until it passes is how a result gets manufactured. So the tests check the *discipline* as much as the
numbers — that E-000033 is untouched, that the swap is exactly one thing, and that the record says
what it does not show.

The heavy run is not repeated here. It is committed, and these read it.
"""

import json
from pathlib import Path

import pytest

RECORD = Path("so/results/pdx003/pdx003_retrieval_closure_real_embedder.json")

pytestmark = pytest.mark.skipif(not RECORD.exists(), reason="PDX-003 record not present")

_R = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}


def test_the_control_e000033_failed_is_now_met_on_every_seed():
    """E-000033 read 0.0250 on its worst seed against a required 0.80. This is the same criterion."""
    assert _R["control_met"] is True
    obs = _R["criteria"]["control/read_before_deletion"]
    assert obs["threshold"] == 0.80
    assert obs["observed"] >= 0.80
    assert _R["aggregate"]["control/read_before_deletion"]["min"] >= 0.80


def test_every_registered_criterion_passes():
    for name, c in _R["criteria"].items():
        assert c["pass"] is True, (name, c)
    assert _R["decision"] == "CLOSURE_REPRODUCED_IN_A_RETRIEVAL_STORE"


def test_the_closure_is_one_record_canonical_and_k_duplicated():
    """E-000032's result, in the substrate §13(b) asked for."""
    agg = _R["aggregate"]
    assert agg["canonical/fact_closure_mean"]["mean"] == 1.0
    assert agg["canonical/fact_closure_max"]["max"] == 1.0
    assert agg["duplicated/fact_closure_mean"]["mean"] == float(_R["k"])


def test_one_deletion_removes_the_fact_canonically_and_does_not_duplicated():
    """The half a practitioner meets: deleting the chunk the question retrieves is not enough."""
    agg = _R["aggregate"]
    canonical = agg["canonical/still_retrievable_after_one"]["mean"]
    duplicated = agg["duplicated/still_retrievable_after_one"]["mean"]
    assert canonical < 0.25
    assert duplicated > 0.90
    assert duplicated - canonical > 0.5


def test_exactly_one_thing_was_changed_and_it_is_named():
    change = _R["single_declared_change"]
    assert change["from"] == "mean-pooled gpt2"
    assert "MiniLM" in change["to"]
    assert _R["seeds"] == [0, 1, 2]
    assert _R["n_facts"] == 150


def test_e000033_is_not_amended_by_this_experiment():
    """The point of registering separately: the failing record must still stand as recorded."""
    assert "does_not_replace" in _R
    assert "E-000033" in _R["does_not_replace"]
    old = json.loads(Path("so/results/e000033_retrieval_closure.json").read_text(encoding="utf-8"))
    assert old["embedder"] == "gpt2"
    # E-000033's own control still fails, and that record is unchanged
    assert old["aggregate"]["control/read_before_deletion"]["min"] < 0.80


def test_e000033s_protocol_is_imported_rather_than_reimplemented():
    """`same protocol, one swap` has to be true of the code, not just of the prose."""
    src = Path("so/experiments/pdx003_retrieval_closure_real_embedder.py").read_text(encoding="utf-8")
    assert "from so.experiments import e000033_retrieval_closure as e33" in src
    for fn in ("e33.build_facts", "e33.question", "e33.build_index", "e33.read_rate",
               "e33.retrieve", "e33.resolve"):
        assert fn in src, fn


def test_the_record_states_what_it_does_not_show():
    nc = _R["not_claimed"].lower()
    assert "pre-registered" in nc
    assert "stands as recorded" in nc
    assert "does not show that e-000033's own configuration was sound" in nc
