"""The audit window, judged on numbers alone: the guard must be able to admit AND to refuse.

``so.siting`` decides whether an accessibility audit was in a position to say anything, from two
measurements per candidate site. The properties that make it worth having:

  1. IT CAN RETURN BOTH VERDICTS. A guard that refuses everything is as useless as one that admits
     everything; both are instruments that cannot fail. The window is non-empty on a site that has the
     content and a lens distinct from the unembedding, and empty when either half is missing.
  2. THE TWO HALVES ARE INDEPENDENT. Arrival and distinctness each refuse on their own, and the reason
     returned names which one fired, because "the audit is blind" and "the audit is the logit lens"
     call for different fixes -- the first is a siting error, the second a design one.
  3. IT REFUSES TO GUESS. A site with only one of the two measurements is not admitted by default.
  4. IT REPRODUCES THE RECORDED CASE. Fed the numbers this repository has already measured for
     ``read_layers=(8, 10)``, the window is empty and E-000063's headline verdict comes back VACUOUS.
"""

import math

import pytest

from so.siting import (ARRIVAL_FLOOR, DEGENERACY_CEILING, arrival_ratios, audit_window, certify,
                       lens_alignment, window_sites)

torch = pytest.importorskip("torch")

# Measured, not invented: WSC-001's mean max-abs state movement under SHRED on the recorded adapter
# (ledger 31.57) and the lens/unembedding cosine curve of ledger 31.56 at the hidden state each site
# is read at (site l is read at hidden_states[l + 1]).
RECORDED_MEDIATORS = {"site8": 2.81, "site9": 4.19, "site10": 83.55, "final": 27.06}
RECORDED_LENS_COS = {"site8": 0.783, "site9": 0.783, "site10": 1.000, "final": 1.000}


def test_arrival_ratios_are_relative_to_the_strongest_site():
    r = arrival_ratios({"a": 1.0, "b": 4.0})
    assert r["b"] == 1.0 and abs(r["a"] - 0.25) < 1e-12


def test_a_counterfactual_that_moves_nothing_admits_no_site():
    r = arrival_ratios({"a": 0.0, "b": 0.0})
    assert r == {"a": 0.0, "b": 0.0}
    w = audit_window({"a": 0.0, "b": 0.0}, {"a": 0.1, "b": 0.1})
    assert window_sites(w) == []


def test_a_negative_mediator_is_refused():
    with pytest.raises(ValueError, match="magnitude"):
        arrival_ratios({"a": -1.0})


def test_lens_alignment_is_one_on_the_unembedding_and_zero_on_an_orthogonal_family():
    w = torch.randn(6, 32, generator=torch.Generator().manual_seed(0))
    assert abs(lens_alignment(w * 3.0, w) - 1.0) < 1e-5          # scale-free
    assert abs(lens_alignment(-w, w) - 1.0) < 1e-5               # sign-free: a direction, not a vector
    q, _ = torch.linalg.qr(torch.randn(32, 12, generator=torch.Generator().manual_seed(1)))
    assert lens_alignment(q[:, :6].t(), q[:, 6:12].t()) < 1e-5


def test_lens_alignment_refuses_families_that_are_not_row_matched():
    with pytest.raises(ValueError, match="row for row"):
        lens_alignment(torch.randn(6, 32), torch.randn(5, 32))


def test_the_window_is_non_empty_when_a_site_has_the_content_and_a_distinct_lens():
    w = audit_window({"early": 1.0, "mid": 90.0, "late": 100.0},
                     {"early": 0.31, "mid": 0.70, "late": 1.00})
    assert window_sites(w) == ["mid"]
    mid = [r for r in w if r.site == "mid"][0]
    assert mid.in_window and mid.reasons == ()


def test_each_half_refuses_on_its_own_and_says_which():
    w = {r.site: r for r in audit_window({"blind": 1.0, "degenerate": 100.0},
                                         {"blind": 0.31, "degenerate": 1.00})}
    assert not w["blind"].in_window and len(w["blind"].reasons) == 1
    assert "arrival" in w["blind"].reasons[0]
    assert not w["degenerate"].in_window and len(w["degenerate"].reasons) == 1
    assert "logit lens" in w["degenerate"].reasons[0]


def test_both_halves_can_fire_at_once():
    w = audit_window({"bad": 1.0, "good": 100.0}, {"bad": 0.99, "good": 0.50})
    bad = [r for r in w if r.site == "bad"][0]
    assert len(bad.reasons) == 2


def test_a_site_measured_only_one_way_is_not_admitted_by_default():
    with pytest.raises(ValueError, match="both a mediator and a lens alignment"):
        audit_window({"a": 1.0, "b": 2.0}, {"a": 0.5})


def test_the_recorded_adapter_has_an_empty_window_and_its_verdict_is_vacuous():
    """The consolidation, pinned: on read_layers=(8, 10) no site passes both halves.

    Arrival and distinctness are anti-aligned here -- the two sites with a lens distinct from the
    unembedding are the two the pod has not reached, and the site the pod reaches is one block from
    the output, where the lens IS the unembedding. This is not a new measurement; both rows are
    already in the ledger (31.56, 31.57). It is pinned so that a change to the bars, to the recorded
    numbers, or to the window rule cannot silently turn E-000063's verdict admissible.
    """
    order = ["site8", "site9", "site10", "final"]
    w = audit_window(RECORDED_MEDIATORS, RECORDED_LENS_COS, order=order)
    assert window_sites(w) == []
    out = certify(w, "site8", {"jprobe_shred_minus_never": -0.0179, "claim": "no workspace trace after deletion"})
    assert out["status"] == "VACUOUS" and out["admissible"] is False
    assert out["window"] == [] and out["verdict"]["jprobe_shred_minus_never"] == -0.0179
    assert any("arrival" in r for r in out["reasons"])
    # site10 is where the content is, and it fails the other half instead
    assert certify(w, "site10", {})["status"] == "VACUOUS"
    assert any("logit lens" in r for r in certify(w, "site10", {})["reasons"])


def test_certify_passes_an_admissible_verdict_through_unchanged():
    w = audit_window({"early": 1.0, "mid": 90.0}, {"early": 0.31, "mid": 0.70})
    out = certify(w, "mid", {"jprobe_shred_minus_never": -0.01})
    assert out["status"] == "ADMITTED" and out["admissible"] is True and out["reasons"] == []
    assert out["verdict"] == {"jprobe_shred_minus_never": -0.01}


def test_certify_refuses_a_site_that_was_never_judged():
    w = audit_window({"mid": 1.0}, {"mid": 0.5})
    with pytest.raises(ValueError, match="not among the candidate sites"):
        certify(w, "site42", {})


def test_the_window_rule_is_not_sensitive_to_the_exact_bars_on_the_recorded_numbers():
    """How much of the recorded case turns on the bars, measured rather than asserted.

    Written first as "every floor in [0.05, 0.95] gives the same empty window", which is false and the
    test said so: site9's arrival ratio is 0.05015, so a floor of exactly 0.05 admits it. The true
    boundary is worth more than the round claim. The window is empty for EVERY floor at or above 0.06
    and every ceiling in [0.80, 0.99]; it stops being empty only below 0.0502, where the rule admits a
    site carrying five per cent of the movement -- a floor no one would defend, and the pre-registered
    one is ten times higher. The remaining sites are further out still: site8 sits at 0.0336, and
    site10 and the final state fail on distinctness at any ceiling below 1.0.
    """
    order = ["site8", "site9", "site10", "final"]
    for floor in (0.06, 0.25, 0.50, 0.75, 0.95):
        for ceiling in (0.80, 0.85, 0.90, 0.95, 0.99):
            assert window_sites(audit_window(RECORDED_MEDIATORS, RECORDED_LENS_COS, floor, ceiling,
                                             order=order)) == [], (floor, ceiling)
    ratios = arrival_ratios(RECORDED_MEDIATORS)
    assert abs(ratios["site9"] - 0.05015) < 1e-4 and abs(ratios["site8"] - 0.03363) < 1e-4
    # just below site9's ratio the rule admits it, and the reason it was excluded is arrival alone
    assert window_sites(audit_window(RECORDED_MEDIATORS, RECORDED_LENS_COS, 0.050, 0.90,
                                     order=order)) == ["site9"]


def test_the_defaults_are_the_pre_registered_ones():
    assert math.isclose(ARRIVAL_FLOOR, 0.50) and math.isclose(DEGENERACY_CEILING, 0.90)
