import ast

from so.experiments.nov004_instrument_audit import (
    _NONDETERMINISTIC,
    run,
    scan_class_a,
    scan_class_b,
)

_R = run()


def test_scanner_rediscovers_the_sites_found_by_hand():
    """Validity floor: if it misses the known positives its silence elsewhere means nothing."""
    assert _R["validity_floor_met"] is True
    assert _R["decision"] == "SCANNER_VALID"
    assert len(_R["known_sites_found"]) == 2


def test_it_scans_the_experiments_above_e000099():
    """The first version globbed e0000*.py, which silently excluded E-000100 onward."""
    assert _R["files_scanned"] >= 100
    scanned = {f["file"] if isinstance(f, dict) else f for f in _R["class_a_files"]}
    assert any("e000104" in s or "e000105" in s for s in scanned)


def test_exactly_three_confirmed_vacuous_comparisons():
    """Two known by hand, and E-000107 found by the sweep."""
    assert _R["class_a_confirmed_count"] == 3
    files = sorted(s["file"] for s in _R["class_a_confirmed"])
    assert any("e000104" in f for f in files)
    assert any("e000105" in f for f in files)
    assert any("e000107" in f for f in files)


def test_e000107_oracle_check_cannot_fail():
    """`oracle` and `direct` are the same pure call, and the difference feeds the kill predicate."""
    site = next(s for s in _R["class_a_confirmed"] if "e000107" in s["file"])
    assert site["callee"] == "fn"
    assert site["assigned_to"] == ["direct", "oracle"]
    assert site["results_are_compared"] is True


def test_duplicates_that_nothing_compares_are_kept_separate():
    """Allocating two zero vectors duplicates the expression, not the value."""
    assert _R["class_a_incidental_count"] > 0
    for s in _R["class_a_incidental_duplicates"]:
        assert s["results_are_compared"] is False


def test_nondeterministic_calls_are_not_flagged():
    """Two draws from a generator are a duplicated expression and two different values."""
    src = """
def f(rng):
    a = rng.randn(3)
    b = rng.randn(3)
    return a == b
"""
    assert scan_class_a(ast.parse(src)) == []
    assert "randn" in _NONDETERMINISTIC


def test_a_real_vacuous_pair_is_caught():
    src = """
def f(net):
    c = work(net)
    g = work(net)
    return c != g
"""
    found = scan_class_a(ast.parse(src))
    assert len(found) == 1
    assert found[0]["callee"] == "work"
    assert found[0]["results_are_compared"] is True


def test_an_intervening_mutation_ends_the_window():
    """bank_from_store(store) before and after a delete is the same call and a different value."""
    src = """
def f(store):
    before = bank_from_store(store)
    store = mutate(store)
    after = bank_from_store(store)
    return before != after
"""
    assert scan_class_a(ast.parse(src)) == []


def test_a_non_assignment_between_them_ends_the_window():
    src = """
def f(net):
    c = work(net)
    do_something_with_side_effects()
    g = work(net)
    return c != g
"""
    assert scan_class_a(ast.parse(src)) == []


def test_class_b_flags_a_loop_invariant_call():
    src = """
def f(rows, m):
    out = []
    for r in rows:
        setup = expensive(m)
        out.append(setup + r)
    return out
"""
    found = scan_class_b(ast.parse(src))
    assert any(s["callee"] == "expensive" for s in found)


def test_class_b_ignores_a_call_that_depends_on_the_loop():
    src = """
def f(rows, m):
    for r in rows:
        y = expensive(m, r)
    return y
"""
    assert scan_class_b(ast.parse(src)) == []


def test_class_b_is_reported_as_candidates_not_findings():
    """A loop-invariant call is only a defect when it sits on the baseline arm."""
    assert "class_b_candidates" in _R
    assert "candidate" in _R["how_to_read_this"].lower()
    assert _R["class_b_count"] > 0
