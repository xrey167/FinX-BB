"""Floor tests for NOV-005.

The audit's whole point is that a search which has never been shown to find anything cannot support
a null. The same applies to the audit: a checker that reports AUDIT_CLEAN and has never been shown
to report anything else is not evidence either. So most of what follows constructs the failures and
requires them to be caught — per claim rather than on a sample, because a claim whose assertion
phrase silently stopped matching would pass a clean run too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so.experiments import nov005_novelty_claim_audit as nov005

ROOT = Path(__file__).resolve().parents[2]


def _mirror(tmp_path: Path, transform) -> Path:
    """Copy every file NOV-005 reads into tmp_path, passing each through `transform`."""
    for claim in nov005._CLAIMS:
        src = ROOT / claim.asserted_in
        dst = tmp_path / claim.asserted_in
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_text(transform(src.read_text(encoding="utf-8")), encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------- the audit on the real tree

def test_the_audit_is_clean_on_the_repository_as_committed():
    result = nov005.run(ROOT)
    assert result["contradictions"] == [], result["contradictions"]
    assert result["decision"] == "AUDIT_CLEAN"


def test_the_validity_floor_is_met():
    assert nov005.run(ROOT)["validity_floor_met"] is True


def test_every_claim_has_a_known_verdict():
    for claim in nov005._CLAIMS:
        assert claim.verdict in nov005._VERDICTS, claim.ident


def test_every_claim_names_a_record_that_exists():
    for claim in nov005._CLAIMS:
        if claim.record:
            assert (ROOT / claim.record).exists(), f"{claim.ident}: {claim.record}"


def test_every_assertion_phrase_is_actually_findable():
    """A registry whose phrases no longer match the documents would pass by not testing."""
    for claim in nov005._CLAIMS:
        text = (ROOT / claim.asserted_in).read_text(encoding="utf-8")
        assert claim.phrase in text, f"{claim.ident}: {claim.phrase!r} not in {claim.asserted_in}"


# --------------------------------------------------------------------- the enforcement must fire

@pytest.mark.parametrize(
    "claim", [c for c in nov005._CLAIMS if c.verdict in nov005._NEEDS_MARKER],
    ids=lambda c: c.ident,
)
def test_removing_the_correction_marker_is_caught_for_every_corrected_claim(claim, tmp_path):
    root = _mirror(tmp_path, lambda t: t.replace(nov005.MARKER, "XXX-000"))
    result = nov005.run(root)
    kinds = {(c["claim"], c["kind"]) for c in result["contradictions"]}
    assert (claim.ident, "UNCORRECTED_ASSERTION") in kinds, kinds
    assert result["decision"] == "CLAIMS_CONTRADICTED"


@pytest.mark.parametrize(
    "claim", [c for c in nov005._CLAIMS if c.verdict in nov005._NEEDS_MARKER],
    ids=lambda c: c.ident,
)
def test_deleting_a_withdrawn_claim_instead_of_marking_it_is_caught(claim, tmp_path):
    """A withdrawal that leaves no trace is not a withdrawal. §31.62's failure, one level up."""
    root = _mirror(tmp_path, lambda t: t.replace(claim.phrase, "(silently removed)"))
    kinds = {(c["claim"], c["kind"]) for c in nov005.run(root)["contradictions"]}
    assert (claim.ident, "ASSERTION_NOT_FOUND") in kinds, kinds


def test_a_narrowed_claim_without_a_remainder_is_caught(monkeypatch):
    narrowed = next(c for c in nov005._CLAIMS if c.verdict == "NARROWED")
    from dataclasses import replace
    monkeypatch.setattr(nov005, "_CLAIMS", (replace(narrowed, what_survives=""),))
    kinds = {c["kind"] for c in nov005.run(ROOT)["contradictions"]}
    assert "NARROWED_WITHOUT_REMAINDER" in kinds


def test_a_verdict_without_a_citation_is_caught(monkeypatch):
    dead = next(c for c in nov005._CLAIMS if c.verdict == "SUPERSEDED")
    from dataclasses import replace
    monkeypatch.setattr(nov005, "_CLAIMS", (replace(dead, evidence=()),))
    kinds = {c["kind"] for c in nov005.run(ROOT)["contradictions"]}
    assert "VERDICT_WITHOUT_EVIDENCE" in kinds


# --------------------------------------------------------------------- the calibration must fire

def test_a_positive_control_that_did_not_find_its_prior_art_invalidates_the_search(monkeypatch):
    """This is the whole argument. Without it every SURVIVES below is worthless."""
    broken = nov005.Control(
        ident="PC-BROKEN", proposition="…", query="…",
        must_find="Codd", found="nothing came back",
    )
    monkeypatch.setattr(nov005, "_CONTROLS", (broken,))
    result = nov005.run(ROOT)
    assert result["validity_floor_met"] is False
    assert result["decision"] == "SEARCH_INVALID_CONTROLS_MISSED"


def test_a_negative_control_that_matched_something_invalidates_the_search(monkeypatch):
    """A searcher returning a hit for a fabricated construct is as broken as one returning none."""
    monkeypatch.setattr(
        nov005, "_NEGATIVE_CONTROL",
        nov005.Control(ident="NC", proposition="…", query="…", must_find="",
                       found="a matching paper was returned"),
    )
    assert nov005.run(ROOT)["validity_floor_met"] is False


def test_every_positive_control_actually_found_its_target():
    for control in nov005._CONTROLS:
        assert control.must_find.lower() in control.found.lower(), control.ident


def test_the_negative_control_records_that_nothing_was_found():
    assert "no work" in nov005._NEGATIVE_CONTROL.found.lower()


# --------------------------------------------------------------------- evidence discipline

@pytest.mark.parametrize(
    "claim", [c for c in nov005._CLAIMS if c.verdict in nov005._NEEDS_MARKER],
    ids=lambda c: c.ident,
)
def test_each_corrected_claim_carries_an_identifier_and_a_quoted_sentence(claim):
    assert claim.evidence, claim.ident
    for cite in claim.evidence:
        assert cite.ident.strip(), f"{claim.ident}: citation without an identifier"
        assert len(cite.quote.strip()) > 20, f"{claim.ident}/{cite.ident}: quote too short to check"
        assert cite.read in {"FULL TEXT", "ABSTRACT", "SEARCH METADATA"}, cite.read


def test_every_claim_records_the_queries_that_produced_its_verdict():
    """A null nobody can attack is not a result. The queries are the attack surface."""
    for claim in nov005._CLAIMS:
        assert claim.queries, claim.ident
        for q in claim.queries:
            assert len(q.strip()) > 20, f"{claim.ident}: query too vague to reproduce"


def test_the_audit_declines_to_call_superseded_a_judgement_of_quality():
    assert "priority, not about quality" in nov005.run(ROOT)["not_claimed"]


def test_the_record_on_disk_matches_a_fresh_run():
    path = ROOT / "so/results/nov005/nov005_novelty_claim_audit.json"
    assert path.exists(), "run `make novelty` and commit the record"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    fresh = nov005.run(ROOT)
    assert on_disk["decision"] == fresh["decision"]
    assert on_disk["verdict_counts"] == fresh["verdict_counts"]
    assert len(on_disk["claims"]) == len(fresh["claims"])
