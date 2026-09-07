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
    FIGURE_CLAIMS,
    PAPER,
    SCOPE_CLAIMS,
    Claim,
    FigureClaim,
    ScopeClaim,
    VERDICT_CLAIMS,
    VerdictClaim,
    _CITED_BUT_UNRUN,
    _NOT_A_MEASUREMENT,
    _figure_text,
    check_record_freshness,
    check_references,
    _fmt,
    _resolve,
    check,
    check_self_description,
    coverage,
    occurrences,
    unregistered,
)

_PAPER_TEXT = PAPER.read_text(encoding="utf-8")
_REPORT = check(_PAPER_TEXT)
_FIG_TEXT = {name: _figure_text(name) for name in {c.figure for c in FIGURE_CLAIMS}}


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
                   _bump_last_digit(claim.printed), claim.rule, claim.reduce, claim.appears_as)
    assert broken.printed != claim.printed
    rows = check(_PAPER_TEXT, claims=(broken,), scope_claims=())["figures"]
    assert [r["status"] for r in rows] == ["MISMATCH"], rows


@pytest.mark.parametrize("claim", CLAIMS, ids=[c.label for c in CLAIMS])
def test_every_figure_can_fail_on_a_paper_that_omits_it(claim):
    """Floor: the record still agrees, but the paper no longer prints the figure -> ABSENT."""
    text = re.sub(re.escape(claim.sought), "<removed>", _PAPER_TEXT, flags=re.IGNORECASE)
    rows = check(text, claims=(claim,), scope_claims=())["figures"]
    assert [r["status"] for r in rows] == ["ABSENT"], rows


@pytest.mark.parametrize("fig", FIGURE_CLAIMS, ids=[c.label for c in FIGURE_CLAIMS])
def test_every_drawn_figure_can_fail_on_a_wrong_number(fig):
    """Floor, on the figures: a bar labelled with a number no record holds must be reported."""
    broken = FigureClaim(fig.figure, fig.label, fig.record, fig.path,
                         _bump_last_digit(fig.printed), fig.rule, fig.reduce)
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(broken,))["drawn"]
    assert [r["status"] for r in rows] == ["MISMATCH"], rows


@pytest.mark.parametrize("fig", FIGURE_CLAIMS, ids=[c.label for c in FIGURE_CLAIMS])
def test_every_drawn_figure_can_fail_when_the_svg_stops_printing_it(fig):
    """Floor: the record still agrees, but the number is no longer drawn -> ABSENT."""
    doctored = dict(_FIG_TEXT)
    doctored[fig.figure] = _FIG_TEXT[fig.figure].replace(fig.printed, "<removed>")
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(fig,),
                 figure_texts=doctored)["drawn"]
    assert [r["status"] for r in rows] == ["ABSENT"], rows


def test_a_missing_figure_file_is_reported_not_skipped():
    bogus = FigureClaim("figN-does-not-exist.svg", "bogus", CLAIMS[0].record,
                        "aggregate/active/object_top1/mean", "1.0000", "dp4")
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(bogus,))["drawn"]
    assert rows[0]["status"] == "PATH"
    assert "figN-does-not-exist.svg" in rows[0]["detail"]


def test_the_figure_text_reader_finds_the_drawn_numbers_and_not_the_geometry():
    """It reads what a reader sees. Path coordinates are geometry and are deliberately not checked."""
    drawn = _FIG_TEXT["fig2-swept-geometry.svg"]
    assert "0.2191" in drawn and "2,199,996" in drawn
    assert "M137.2,72.0" not in drawn          # the series path is geometry, not a claim
    assert "137.2" not in drawn


@pytest.mark.parametrize("v", VERDICT_CLAIMS, ids=[v.label for v in VERDICT_CLAIMS])
def test_every_verdict_can_fail_when_the_record_says_the_opposite(v):
    """Floor: a CERTIFIED the record does not support must be reported, not assumed."""
    flipped = VerdictClaim(v.label, v.record, v.path, not v.expect, v.row_prefix, v.says)
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(),
                 verdict_claims=(flipped,))["verdicts"]
    assert [r["status"] for r in rows] == ["MISMATCH"], rows


@pytest.mark.parametrize("v", VERDICT_CLAIMS, ids=[v.label for v in VERDICT_CLAIMS])
def test_every_verdict_can_fail_when_its_row_stops_saying_it(v):
    """Floor: the record still agrees, but the table cell no longer carries the verdict."""
    text = "\n".join(
        ln.replace(v.says, " — ") if ln.startswith(v.row_prefix) else ln
        for ln in _PAPER_TEXT.splitlines()
    )
    rows = check(text, claims=(), scope_claims=(), figure_claims=(), verdict_claims=(v,))["verdicts"]
    assert [r["status"] for r in rows] == ["CONTRADICTED"], rows


def test_a_verdict_whose_row_is_gone_is_reported_not_skipped():
    v = VERDICT_CLAIMS[0]
    gone = VerdictClaim(v.label, v.record, v.path, v.expect, "| no such row |", v.says)
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(),
                 verdict_claims=(gone,))["verdicts"]
    assert rows[0]["status"] == "ABSENT"
    assert "0 rows start" in rows[0]["detail"]


def test_a_verdict_is_checked_against_its_own_row_not_the_document():
    """`CERTIFIED` appears four times in §4; a claim must not pass on somebody else's row."""
    v = next(c for c in VERDICT_CLAIMS if c.label == "E30 soft-gate SHRED not certified")
    assert "**CERTIFIED**" in _PAPER_TEXT
    misdirected = VerdictClaim(v.label, v.record, v.path, v.expect, v.row_prefix, "**CERTIFIED**")
    rows = check(_PAPER_TEXT, claims=(), scope_claims=(), figure_claims=(),
                 verdict_claims=(misdirected,))["verdicts"]
    assert rows[0]["status"] == "CONTRADICTED", rows


def test_every_row_of_the_certificate_table_has_a_bound_verdict():
    """An unbound verdict cell is worse than an unbound number: it states whether the claim holds."""
    rows = [ln for ln in _PAPER_TEXT.splitlines()
            if ln.startswith("| synthetic |") or ln.startswith("| frozen GPT-2,")]
    assert len(rows) == 7, rows
    prefixes = {v.row_prefix for v in VERDICT_CLAIMS}
    for row in rows:
        assert any(row.startswith(p) for p in prefixes), f"unbound verdict row: {row}"


def test_every_yes_no_row_in_the_paper_is_bound_or_declared_an_argument():
    """Sweeping for verdict rows found one nobody had bound: §7's "a certificate is even available".

    It is not in any record and could not be -- it follows from a LoRA having no finite payload
    domain, which is a property of the representation, not an outcome. So it is the one exception,
    and the paper must say so rather than let it sit among the measured cells looking identical.
    """
    rows = [ln for ln in _PAPER_TEXT.splitlines()
            if ln.startswith("| ") and ln.rstrip().endswith(("| no |", "| **yes** |"))]
    prefixes = {v.row_prefix for v in VERDICT_CLAIMS}
    argued = "| **a certificate is even available** |"
    unbound = [r for r in rows if not any(r.startswith(p) for p in prefixes)]
    assert [r for r in unbound if not r.startswith(argued)] == [], unbound
    assert len(rows) - len(unbound) == 5, "the five PDX-001 policy rows must all be bound"
    assert "an argument, not a measurement" in _PAPER_TEXT


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
    assert cov["drawn_figures"] == len(FIGURE_CLAIMS)
    assert cov["scope_claims"] == len(SCOPE_CLAIMS)
    assert len(cov["records_bound"]) >= 7
    assert "unchecked, not verified" in _REPORT["not_claimed"]


def test_every_figure_the_paper_embeds_has_at_least_one_bound_number():
    """A figure nobody registered is a page of unchecked numbers in the reader's first glance."""
    embedded = set(re.findall(r"\(figures/([^)]+\.svg)\)", _PAPER_TEXT))
    assert embedded, "the paper embeds no figures; this test is checking nothing"
    bound = {c.figure for c in FIGURE_CLAIMS}
    assert embedded <= bound, f"unbound figures: {sorted(embedded - bound)}"


def test_no_registered_figure_has_gone_missing_from_the_paper():
    """The mirror: a registry entry for a figure the paper no longer shows is dead weight."""
    embedded = set(re.findall(r"\(figures/([^)]+\.svg)\)", _PAPER_TEXT))
    assert {c.figure for c in FIGURE_CLAIMS} <= embedded


def test_the_figures_are_numbered_in_the_order_they_appear():
    """Figure 1 was added to section 3 after Figures 1 and 2 already existed; both had to move."""
    order = re.findall(r"\(figures/(fig(\d+)-[^)]+\.svg)\)", _PAPER_TEXT)
    assert [int(n) for _, n in order] == list(range(1, len(order) + 1)), order
    captions = [int(n) for n in re.findall(r"^\*Figure (\d+) —", _PAPER_TEXT, flags=re.MULTILINE)]
    assert captions == list(range(1, len(order) + 1)), captions


# --------------------------------------------------------------------------- resolver and rules

def test_resolver_walks_a_flat_key_that_contains_slashes():
    """These records hold `'active/margin_mean'` as one key, not as two levels."""
    doc = {"aggregate": {"active/margin_mean": {"mean": 0.6195}}}
    assert _resolve(doc, "aggregate/active/margin_mean/mean") == (0.6195, None)


def test_resolver_prefers_the_longest_key_but_falls_back_to_nesting():
    doc = {"aggregate": {"active": {"object_top1": {"mean": 1.0}}}}
    assert _resolve(doc, "aggregate/active/object_top1/mean") == (1.0, None)


def test_resolver_selects_a_list_element_by_field_rather_than_by_position():
    """PDX-001's five policies are a list; a positional path would survive a reordering."""
    policies = [{"policy": "a", "top1": 1.0}, {"policy": "b", "top1": 0.0}]
    assert _resolve({"policies": policies}, "policies/policy=b/top1") == (0.0, None)
    # the same claim survives a reordering, which is the whole point
    assert _resolve({"policies": policies[::-1]}, "policies/policy=b/top1") == (0.0, None)
    # a positional path would not
    assert _resolve({"policies": policies}, "policies/1/top1") == (0.0, None)
    assert _resolve({"policies": policies[::-1]}, "policies/1/top1") == (1.0, None)


def test_a_selector_that_is_not_unique_is_an_error_not_a_first_match():
    doc = {"policies": [{"policy": "a"}, {"policy": "a"}]}
    assert _resolve(doc, "policies/policy=a")[0] is None
    assert "matched 2" in _resolve(doc, "policies/policy=a")[1]
    assert "matched 0" in _resolve(doc, "policies/policy=z")[1]


def test_every_pdx_claim_addresses_its_policy_by_name():
    """A positional index here would keep passing if the policy order ever changed."""
    pdx = [c for c in CLAIMS if c.record.startswith("pdx001/")]
    assert len(pdx) >= 10
    for c in pdx:
        assert "*" not in c.path
        assert not any(seg.isdigit() for seg in c.path.split("/")), c.path


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


# --------------------------------------------------------------------------- coverage

def test_no_number_in_the_paper_is_unbound():
    """The registry was green while `137`, `311` and `8.49e+06` were wrong, because none was bound.

    Coverage is the other half of the check: every numeric token in the paper must be bound to a
    record or excluded by a rule that states its reason. Zero unbound is the only reading that lets
    a clean `check()` mean anything about the paper as a whole.
    """
    cov = unregistered()
    assert cov["unbound_count"] == 0, cov["unbound"]
    assert cov["distinct_numeric_tokens"] > 40


def test_coverage_notices_a_number_nothing_binds():
    """Floor: introduce an unbound figure and it must be reported."""
    doctored = _PAPER_TEXT + "\n\nThe adapter recovered 0.4173 of the held-out facts.\n"
    cov = unregistered(doctored)
    assert "0.4173" in cov["unbound"]


def test_every_exclusion_states_why_it_is_not_a_measurement():
    """An exclusion list without reasons is a way to make coverage say whatever you want."""
    assert _NOT_A_MEASUREMENT
    for pattern, reason in _NOT_A_MEASUREMENT:
        assert pattern and reason
        assert len(reason) > 8, (pattern, reason)
        re.compile(pattern)


def test_the_three_wrong_numbers_in_section_7_are_bound_now():
    """`137` was a mean rank read as seconds; `311` was in no record; `8.49e+06` came from §5."""
    by_label = {c.label: c for c in CLAIMS}
    assert by_label["E24 ga delete seconds"].printed == "129"
    assert by_label["E24 relabel delete seconds"].printed == "335"
    assert by_label["E24 relabel perplexity"].printed == "6.39e+06"


def test_the_wrong_numbers_survive_only_where_the_paper_reports_them_as_wrong():
    """They are quoted in §9's methods narrative, and must not be back in §7's table."""
    table_row = next(line for line in _PAPER_TEXT.splitlines()
                     if line.startswith("| seconds to delete 50 facts"))
    assert "129" in table_row and "335" in table_row
    assert "137" not in table_row and "311" not in table_row
    ppl_row = next(line for line in _PAPER_TEXT.splitlines()
                   if line.startswith("| perplexity on ordinary prose"))
    assert "6.39e+06" in ppl_row and "8.49e+06" not in ppl_row
    # where they do survive, they are backticked -- the paper's mark for a numeral being discussed
    for quoted in ("`137`", "`311`", "`8.49e+06`"):
        assert quoted in _PAPER_TEXT


# --------------------------------------------------------------------------- references

def test_every_pointer_in_the_paper_points_at_something():
    """The fifth kind of claim: §0 promises reproducibility "by the `make` target named beside it".

    A target that does not exist makes that promise false in a way no amount of correct arithmetic
    would reveal, and nothing checked it until now.
    """
    rows = check_references()
    bad = [r for r in rows if r["status"] != "OK"]
    assert bad == [], bad
    kinds = {r["kind"] for r in rows}
    assert kinds == {"make target", "experiment", "section", "path", "figure"}, kinds
    assert len(rows) > 30


def test_the_sweeps_record_is_still_true_of_the_repository():
    """A registry that only compares two documents cannot notice the world moving underneath both."""
    rows = check_record_freshness()
    assert rows, "nothing is checked for freshness"
    assert [r["status"] for r in rows] == ["OK"] * len(rows), rows


def test_record_freshness_can_fail_when_the_repository_grows(tmp_path, monkeypatch):
    """Floor: add an experiment file and the sweep's record must be reported stale.

    This is the hole the sixth finding left. Binding the paper to NOV-004's record stopped the
    *paper* going stale silently; nothing stopped the *record* going stale, and then paper and
    record agree with each other while both disagree with the repository.
    """
    import so.paper_numbers as pn

    real = sorted(p.name for p in pn.EXPERIMENTS.glob("*.py")
                  if re.match(r"^e\d{6}[a-z]?_.*\.py$", p.name))
    fake = tmp_path / "experiments"
    fake.mkdir()
    for name in real:
        (fake / name).write_text("")
    (fake / "e999999_a_newly_landed_experiment.py").write_text("")

    monkeypatch.setattr(pn, "EXPERIMENTS", fake)
    rows = pn.check_record_freshness()
    assert rows[0]["status"] == "STALE", rows
    assert f"repository now holds {len(real) + 1}" in rows[0]["detail"]
    assert "re-run the sweep" in rows[0]["detail"]


def test_a_make_target_that_does_not_exist_is_caught():
    text = _PAPER_TEXT + "\n\nReproduce it with `make nosuchtarget`.\n"
    rows = check_references(text)
    assert any(r["name"] == "nosuchtarget" and r["status"] == "MISSING" for r in rows)


def test_an_experiment_cited_with_no_record_is_caught():
    text = _PAPER_TEXT + "\n\nSee also E-999999 for the follow-up.\n"
    rows = check_references(text)
    row = next(r for r in rows if r["name"] == "E-999999")
    assert row["status"] == "MISSING"
    assert "not declared unrun" in row["detail"]


def test_an_id_declared_unrun_that_acquires_a_record_is_a_contradiction():
    """This is the direction the first version missed, and it mattered within the hour.

    E-000033 sat in `_CITED_BUT_UNRUN` because §13(b) said it had never been run. When it *was*
    run, the reference pass took the "record present" branch and reported OK — while the paper
    still said "has never been run". A checker that only guards one direction of a pairing is
    the same defect as a comparison whose two sides share a source: it cannot fail the way it
    needs to.
    """
    import so.paper_numbers as pn
    declared = dict(pn._CITED_BUT_UNRUN)
    declared["E-000033"] = "never been run"          # E-000033 does have a record now
    monkey = pn._CITED_BUT_UNRUN
    try:
        pn._CITED_BUT_UNRUN = declared
        rows = pn.check_references(_PAPER_TEXT)
    finally:
        pn._CITED_BUT_UNRUN = monkey
    row = next(r for r in rows if r["name"] == "E-000033")
    assert row["status"] == "CONTRADICTED", row
    assert "a record now exists" in row["detail"]


def test_an_id_with_no_record_and_no_declaration_still_fails():
    """The other direction, kept alive now that the declaration list is empty."""
    import so.paper_numbers as pn
    rows = pn.check_references(_PAPER_TEXT + "\n\nSee E-999998 for the sequel.\n")
    row = next(r for r in rows if r["name"] == "E-999998")
    assert row["status"] == "MISSING"
    assert "not declared unrun" in row["detail"]


def test_section_13b_reports_the_failed_control_and_keeps_the_two_records_apart():
    """E-000033 failed its own control; PDX-003 answers the substance. Both must stay visible.

    The risk once a passing run exists is that the failing one quietly disappears and the paper reads
    as though §13(b) had worked first time. E-000033's numbers stay printed, its record stays
    unamended, and the replacement says it is a separate registration.
    """
    assert "fails its own pre-registered control" in _PAPER_TEXT
    assert "0.0467" in _PAPER_TEXT and "0.0250" in _PAPER_TEXT
    assert "must not be read" in _PAPER_TEXT
    # and the separately-registered answer, stated as separate
    assert "separately registered experiment (PDX-003)" in _PAPER_TEXT
    assert "is not amended" in _PAPER_TEXT
    assert "exactly one declared thing" in _PAPER_TEXT
    assert "does **not** show that E-000033's own configuration was sound" in _PAPER_TEXT


def test_a_dangling_section_reference_is_caught():
    text = _PAPER_TEXT + "\n\nAs shown in §99, this holds generally.\n"
    rows = check_references(text)
    assert any(r["name"] == "§99" and r["status"] == "MISSING" for r in rows)


def test_a_quoted_path_that_does_not_exist_is_caught():
    text = _PAPER_TEXT + "\n\nThe harness lives in `so/no_such_module.py`.\n"
    rows = check_references(text)
    assert any(r["name"] == "so/no_such_module.py" and r["status"] == "MISSING" for r in rows)


def test_an_embedded_figure_missing_from_disk_is_caught():
    text = _PAPER_TEXT + "\n\n![Nope](figures/fig9-not-real.svg)\n"
    rows = check_references(text)
    assert any(r["name"] == "fig9-not-real.svg" and r["status"] == "MISSING" for r in rows)


# --------------------------------------------------------------- the paper's self-description

def test_the_paper_states_this_registry_size_correctly():
    """These three numbers cannot be bound to a record: their source is the registry itself."""
    rows = check_self_description()
    assert rows, "the paper no longer describes the registry at all"
    assert [r["status"] for r in rows] == ["OK"] * len(rows), rows


def test_the_self_description_can_fail_on_a_stale_count():
    """Floor: it has drifted 33 -> 65 -> 70 -> 87, corrected by hand each time. Not any more."""
    rows = check_self_description(_PAPER_TEXT, claims=CLAIMS[:-1])
    assert [r["status"] for r in rows] == ["MISMATCH"] * len(rows), rows
    assert str(len(CLAIMS) - 1) in rows[0]["detail"]


def test_the_self_description_can_fail_when_the_paper_stops_making_it():
    text = _PAPER_TEXT.replace("figures printed in this text", "<removed>")
    text = text.replace("figures in the prose", "<removed>")
    rows = check_self_description(text)
    assert [r["status"] for r in rows] == ["ABSENT"] * len(rows), rows


# --------------------------------------------------------------------------- the presence test

def test_a_figure_is_not_found_inside_a_longer_number():
    """The bug this closes: '0.0' passed its presence test on '0.0040', and '12' on '128.02'."""
    assert occurrences("0.0", "the revoke arm reads 0.0040 here") == 0
    assert occurrences("12", "a mean rank of 128.02") == 0
    assert occurrences("1", "1,536 of 1,536 keys") == 0
    assert occurrences("0.0", "the mean rank is 0.0 on both arms") == 1


def test_a_trailing_full_stop_or_comma_is_punctuation_not_a_digit_group():
    assert occurrences("0.35", "a declared radius of 0.35.") == 1
    assert occurrences("0.0953", "(min 0.0953, max 0.4014)") == 1
    assert occurrences("0.4014", "(min 0.0953, max 0.4014)") == 1


def test_a_figure_the_paper_spells_in_words_is_sought_in_words():
    """'11' and '12' appear nowhere as numerals; the paper writes eleven and twelve."""
    spelled = {c.label: c for c in CLAIMS if c.appears_as}
    assert spelled["E29 checkpoints"].sought == "eleven"
    assert spelled["E25 templates"].sought == "twelve"
    assert occurrences("eleven", "over Eleven checkpoints") == 1
    assert occurrences("eleven", "elevenths") == 0


def test_a_pinned_claim_only_matches_in_its_own_context():
    """`100` is five different quantities here, and binding one let another go stale.

    E-000035's pod count and NOV-004's count of scanned experiments both rendered "100". The pod
    claim's presence test found *some* occurrence and passed, so when the repository first grew to
    106 experiments the paper's sweep count went stale behind a green check. `near` pins a claim to
    the paragraph that identifies which quantity it is; the bound value now tracks the consolidated
    repository.
    """
    pinned = [c for c in CLAIMS if c.near]
    assert pinned, "nothing is pinned; this test is checking nothing"
    for c in pinned:
        rows = check(_PAPER_TEXT, claims=(c,), scope_claims=(), figure_claims=(),
                     verdict_claims=())["figures"]
        assert rows[0]["status"] == "OK", rows

    # the pin has to bite: point one at a context it does not occur in
    c = next(c for c in CLAIMS if c.label == "NOV004 experiments scanned")
    misplaced = Claim(c.label, c.record, c.path, c.printed, c.rule, c.reduce, c.appears_as,
                      "pods each")
    rows = check(_PAPER_TEXT, claims=(misplaced,), scope_claims=(), figure_claims=(),
                 verdict_claims=())["figures"]
    assert rows[0]["status"] == "ABSENT", rows


def test_a_pin_naming_no_paragraph_is_reported_not_ignored():
    c = CLAIMS[0]
    nowhere = Claim(c.label, c.record, c.path, c.printed, c.rule, c.reduce, c.appears_as,
                    "a phrase this paper does not contain")
    rows = check(_PAPER_TEXT, claims=(nowhere,), scope_claims=(), figure_claims=(),
                 verdict_claims=())["figures"]
    assert rows[0]["status"] == "ABSENT"
    assert "no paragraph" in rows[0]["detail"]


def test_the_sweep_count_tracks_the_repository_rather_than_the_prose():
    """The generated audit, registry, and current prose move together as the repository grows."""
    import json
    from pathlib import Path
    rec = json.loads(Path("so/results/nov004/nov004_instrument_audit.json").read_text())
    assert rec["files_scanned"] == 118
    assert "sweep of all 118 recorded experiments" in _PAPER_TEXT
    # the old count survives only inside backticks, where §9 quotes it as the claim that went stale
    # -- the same convention that lets the coverage pass ignore it
    for m in re.finditer(r"100 recorded experiments", _PAPER_TEXT):
        line = _PAPER_TEXT[:m.start()].rsplit("\n", 1)[-1] + _PAPER_TEXT[m.start():].split("\n", 1)[0]
        assert "`" in line, f"un-quoted stale count: {line}"


def test_round_recurring_figures_are_reported_as_weak_not_counted_as_checks():
    """1.0000 appears all over the paper; its presence test proves nothing and says so."""
    weak = set(_REPORT["weak_presence_tests"])
    assert weak, "some figures are round enough to recur; none were flagged"
    assert "E28 active top-1" in weak
    assert all(r["occurrences"] > 0 for r in _REPORT["figures"])
    assert "no longer discriminates" in _REPORT["not_claimed"]
    strong = [r for r in _REPORT["figures"] if r.get("presence_discriminating")]
    assert len(strong) > len(weak)
