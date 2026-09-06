"""Tests for the paper-number registry, and the floor that makes its silence mean something.

A checker that prints ``0 failing`` and has never been shown to fail is not evidence. The first
four tests here are the validity floor: they mutate one thing at a time and require the checker
to notice, once for **every** registered figure and scope claim rather than for a sample. That is
what turns a clean run into a statement about the paper.
"""

import re

import pytest

from so.paper_numbers import (
    CLAIMS,
    PAPER,
    SCOPE_CLAIMS,
    Claim,
    ScopeClaim,
    _fmt,
    _resolve,
    check,
    coverage,
)

_PAPER_TEXT = PAPER.read_text(encoding="utf-8")
_REPORT = check(_PAPER_TEXT)


def _bump_last_digit(printed: str) -> str:
    """Change the printed figure by one in its last significant place."""
    idx = max(i for i, ch in enumerate(printed) if ch.isdigit())
    digit = printed[idx]
    replacement = "8" if digit == "9" else str(int(digit) + 1)
    return printed[:idx] + replacement + printed[idx + 1:]


# --------------------------------------------------------------------------- validity floor

@pytest.mark.parametrize("claim", CLAIMS, ids=[c.label for c in CLAIMS])
def test_every_figure_can_fail_on_a_wrong_number(claim):
    """Floor: perturb the printed figure and the checker must report MISMATCH.

    A claim whose path silently failed to resolve, or whose rule always rendered the expected
    string, would pass the clean run and pass here too if we only sampled. So this runs for all
    of them: each registered figure is proven to be reading a live value from a live path.
    """
    broken = Claim(claim.label, claim.record, claim.path,
                   _bump_last_digit(claim.printed), claim.rule, claim.reduce)
    assert broken.printed != claim.printed
    rows = check(_PAPER_TEXT, claims=(broken,), scope_claims=())["figures"]
    assert [r["status"] for r in rows] == ["MISMATCH"], rows


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.label for c in CLAIMS])
def test_every_figure_can_fail_on_a_paper_that_omits_it(claim):
    """Floor: the record still agrees, but the paper no longer prints the figure -> ABSENT."""
    text = _PAPER_TEXT.replace(claim.printed, "<removed>")
    rows = check(text, claims=(claim,), scope_claims=())["figures"]
    assert [r["status"] for r in rows] == ["ABSENT"], rows


@pytest.mark.parametrize("scope", SCOPE_CLAIMS, ids=[s.label for s in SCOPE_CLAIMS])
def test_every_scope_claim_can_fail_on_a_wrong_extent(scope):
    """Floor: an extent the record contradicts must be reported, not assumed."""
    broken = ScopeClaim(scope.label, scope.record, scope.path,
                        ["a value no record holds"], scope.must_appear)
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(broken,))["scope"]
    assert [r["status"] for r in rows] == ["MISMATCH"], rows


@pytest.mark.parametrize("scope", SCOPE_CLAIMS, ids=[s.label for s in SCOPE_CLAIMS])
def test_every_scope_claim_can_fail_on_a_paper_that_stays_silent(scope):
    """Floor: this is the failure mode that caught §7 — accurate numbers, undisclosed extent."""
    text = _PAPER_TEXT.replace(scope.must_appear, "<removed>")
    text = re.sub(re.escape(scope.must_appear), "<removed>", text, flags=re.IGNORECASE)
    rows = check(text, claims=(), scope_claims=(scope,))["scope"]
    assert [r["status"] for r in rows] == ["UNDISCLOSED"], rows


def test_an_unresolvable_path_is_reported_not_skipped():
    """A silent skip would let the registry grow while checking nothing."""
    bogus = Claim("bogus", CLAIMS[0].record, "aggregate/no_such_key/mean", "1.0000", "dp4")
    rows = check(_PAPER_TEXT, claims=(bogus,), scope_claims=())["figures"]
    assert rows[0]["status"] == "PATH"
    assert "no_such_key" in rows[0]["detail"]


def test_a_missing_record_is_reported_not_skipped():
    bogus = Claim("bogus", "e999999_not_a_record.json", "anything", "1.0000", "dp4")
    rows = check(_PAPER_TEXT, claims=(bogus,), scope_claims=())["figures"]
    assert rows[0]["status"] == "PATH"
    assert "e999999_not_a_record.json" in rows[0]["detail"]


# --------------------------------------------------------------------------- the live check

def test_the_paper_agrees_with_the_records():
    assert _REPORT["clean"] is True, _REPORT["failures"]
    assert _REPORT["registered_figures"] == len(CLAIMS)
    assert _REPORT["registered_scope_claims"] == len(SCOPE_CLAIMS)


def test_the_three_drifts_that_motivated_the_registry_are_bound():
    """Each of the three prose/record disagreements is now a registered figure."""
    labels = {c.label for c in CLAIMS}
    assert "E28 revoke mean rank" in labels
    assert "E29 band 0.70 mean" in labels
    assert "E24 is a single seed" in {s.label for s in SCOPE_CLAIMS}
    rank = next(c for c in CLAIMS if c.label == "E28 revoke mean rank")
    assert rank.printed == "128.02"          # not the 128.0 the prose had
    band = next(c for c in CLAIMS if c.label == "E29 band 0.70 mean")
    assert band.printed == "0.9999"          # not the 1.0000 the prose had


def test_coverage_is_declared_partial():
    cov = coverage()
    assert cov["figures"] == len(CLAIMS)
    assert cov["scope_claims"] == len(SCOPE_CLAIMS)
    assert len(cov["records_bound"]) >= 7
    assert "unchecked, not verified" in _REPORT["not_claimed"]


# --------------------------------------------------------------------------- resolver and rules

def test_resolver_walks_a_flat_key_that_contains_slashes():
    """These records hold `'active/margin_mean'` as one key, not as two levels."""
    doc = {"aggregate": {"active/margin_mean": {"mean": 0.6195}}}
    assert _resolve(doc, "aggregate/active/margin_mean/mean") == (0.6195, None)


def test_resolver_prefers_the_longest_key_but_falls_back_to_nesting():
    doc = {"aggregate": {"active": {"object_top1": {"mean": 1.0}}}}
    assert _resolve(doc, "aggregate/active/object_top1/mean") == (1.0, None)


def test_resolver_indexes_a_list_and_iterates_with_a_star():
    doc = {"per_checkpoint": [{"band": [1, 2]}, {"band": [3, 4]}]}
    assert _resolve(doc, "per_checkpoint/1/band/0") == (3, None)
    assert _resolve(doc, "per_checkpoint/*/band/1") == ([2, 4], None)


def test_resolver_reports_rather_than_raises():
    doc = {"a": [1, 2]}
    assert _resolve(doc, "a/x")[0] is None
    assert "index" in _resolve(doc, "a/x")[1]
    assert _resolve(doc, "a/9")[0] is None
    assert _resolve(doc, "b")[1] == "missing key 'b'"
    assert "cannot descend" in _resolve({"a": 1}, "a/b")[1]


def test_reducers_are_applied_to_the_swept_list():
    """The 0.80 band is one path read three ways; only the reducer separates them."""
    band = [c for c in CLAIMS if c.label.startswith("E29 band 0.80")]
    assert {c.reduce for c in band} == {"mean", "min", "max"}
    assert len({c.path for c in band}) == 1


def test_rounding_rules_render_as_the_paper_prints():
    assert _fmt(0.61954, "dp4") == "0.6195"
    assert _fmt(128.0234, "dp2") == "128.02"
    assert _fmt(2199996.0, "thousands") == "2,199,996"
    assert _fmt(11.0, "int") == "11"
    assert _fmt(0.000849, "exp2") == "8.49e-04"
    assert _fmt(0.01390, "exp3") == "1.390e-02"


def test_an_unknown_rounding_rule_raises_rather_than_passing():
    with pytest.raises(ValueError):
        _fmt(1.0, "nearest-convenient")
