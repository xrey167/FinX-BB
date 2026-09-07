"""Tests for PDX-002 — the audit run against a real vector index.

The floor matters more here than anywhere else in this branch, because the result is a *positive*
one about somebody else's software. A run that recovers a payload from a store it cannot read in the
first place is not a finding, it is a broken harness pointed at an innocent library.
"""

import pytest

hnswlib = pytest.importorskip("hnswlib", reason="PDX-002 tests the real index; without it there is "
                                               "nothing to test and skipping is honest")

from so.experiments.pdx002_real_index_deletion_audit import (  # noqa: E402
    DOMAIN,
    audit_policy,
    candidates_from_observation,
    embed,
    run,
)

_R = run()
_BY = {r["policy"]: r for r in _R["policies"]}


def test_the_attack_reads_a_live_payload_before_it_reports_on_a_deleted_one():
    """§31.15's floor. Without this the at-chance readings below mean nothing."""
    assert _R["attack_validity_floor_met"] is True
    assert _R["validity_control"]["top1_recovery"] == 1.0
    assert _R["validity_control"]["mean_candidates_remaining"] == 1.0


def test_rebuilding_without_the_row_leaves_the_whole_domain_open():
    """The registered clean control: a real erasure must reach chance, or the instrument is rigged."""
    assert _R["control_behaved_as_registered"] is True
    ctrl = _BY["rebuild_without_row"]
    assert ctrl["top1_recovery"] == 0.0
    assert ctrl["mean_candidates_remaining"] == DOMAIN
    assert ctrl["certified"] is True
    assert ctrl["mean_posterior_on_true_payload"] == pytest.approx(1.0 / DOMAIN)


def test_the_tombstone_is_reversible_through_the_public_api():
    """`unmark_deleted` is documented and public; recovery needs no exploit and no file parsing."""
    api = _BY["mark_deleted_api"]
    assert api["top1_recovery"] == 1.0
    assert api["mean_candidates_remaining"] == 1.0
    assert api["leaks_above_chance"] is True


def test_the_payload_bytes_survive_in_the_serialised_index():
    """The arrangement an adversary with disk access but no process meets."""
    f = _BY["mark_deleted_file"]
    assert f["top1_recovery"] == 1.0
    assert f["leaks_above_chance"] is True


def test_the_record_pins_the_version_actually_tested():
    """A system-under-test entry reading 'unknown' would not be a record of anything."""
    sut = _R["system_under_test"]
    assert sut["library"] == "hnswlib"
    assert sut["version"] not in ("", "unknown", None)
    assert sut["version"][0].isdigit()


def test_the_experiment_declines_to_call_this_a_vulnerability():
    """The finding is about deployments that serve a deletion request with a tombstone.

    hnswlib documents `mark_deleted` as reversible and ships `unmark_deleted` in the public API.
    Reporting that as a vulnerability would be false, and the record has to say so itself rather
    than relying on a reader's charity.
    """
    nc = _R["not_claimed"].lower()
    assert "not a vulnerability" in nc
    assert "undocumented" in nc
    assert "no hosted service" in nc


def test_the_embedding_separates_every_payload_in_the_domain():
    """If two payloads embedded alike the floor would fail for a reason that is not the store's."""
    seen = {tuple(embed(v)) for v in range(DOMAIN)}
    assert len(seen) == DOMAIN


def test_an_unobserved_payload_leaves_the_whole_domain():
    assert candidates_from_observation(None, range(DOMAIN)) == list(range(DOMAIN))


def test_an_observed_vector_names_exactly_one_payload():
    got = candidates_from_observation(embed(123), range(DOMAIN))
    assert got == [123]


def test_an_unknown_policy_raises_rather_than_returning_a_clean_result():
    """A typo must not silently produce a certified-looking null."""
    payloads = [(i * 37 + 11) % DOMAIN for i in range(8)]
    with pytest.raises(ValueError):
        audit_policy("no_such_policy", payloads, [0], "/tmp/pdx002-nonexistent.bin", hnswlib)
