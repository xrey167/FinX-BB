from fractions import Fraction as F

from so.experiments.nov001_screen_calibration import (
    Arm,
    Cost,
    complete_screen,
    e000105_work_term_audit,
    m1_candidate,
    m1_generic,
    m2_candidate,
    m2_generic,
    m5_candidate,
    m5_generic,
    programme_screen,
    run,
)


def _by_id(result):
    return {m["id"]: m for m in result["mechanisms"]}


def test_screen_is_uncalibrated():
    r = run()
    assert r["screen_is_calibrated"] is False
    assert r["decision"] == "SCREEN_UNCALIBRATED"


def test_m1_is_a_false_kill():
    """Identical exact states, identical multiplies, strictly less memory traffic — refused."""
    m1 = _by_id(run())["M1"]
    assert m1["programme_screen"]["verdict"] == "KILL"
    assert m1["complete_screen"]["verdict"] == "PROMOTE"
    assert m1["candidate_cost"]["multiplies"] == m1["generic_cost"]["multiplies"]
    assert m1["candidate_cost"]["slow_memory_words"] < m1["generic_cost"]["slow_memory_words"]


def test_m5_is_a_false_promote():
    """Strictly more multiplies, better at nothing — passed, because the test is equality."""
    m5 = _by_id(run())["M5"]
    assert m5["programme_screen"]["verdict"] == "PROMOTE"
    assert m5["complete_screen"]["verdict"] == "KILL"
    assert m5["candidate_cost"]["multiplies"] > m5["generic_cost"]["multiplies"]
    assert m5["complete_screen"]["improved_coordinates"] == {}


def test_representation_subsidy_flips_the_verdict():
    """Same mechanism, same states, same queries — the accounting decides the verdict."""
    m4 = _by_id(run())["M4"]
    assert m4["subsidy_flips_verdict"] is True
    assert m4["complete_screen"]["verdict"] == "KILL"
    assert m4["complete_screen_vs_unsubsidised_baseline"]["verdict"] == "PROMOTE"
    assert m4["candidate_cost"]["representation_multiplies"] < m4["generic_unsubsidised_cost"]["multiplies"]


def test_screen_is_not_merely_unpassable():
    """The narrow finding needs this: the screen CAN promote, so it is blind, not vacuous."""
    r = run()
    assert "M3" in r["screens_agree_on"]
    assert _by_id(r)["M3"]["programme_screen"]["verdict"] == "PROMOTE"


def test_exactness_is_real_not_asserted():
    """Every mechanism reaches states identical to its baseline; that is what makes a kill possible."""
    r = run()
    for m in r["mechanisms"]:
        assert m["exhaustive_state_mismatches"] == 0, m["id"]
        assert m["programme_screen"]["same_exact_state"] is True, m["id"]


def test_m2_output_is_exactly_the_sequential_output_over_the_whole_domain():
    m2 = _by_id(run())["M2"]
    assert m2["domain_size"] == 189
    assert m2["exhaustive_state_mismatches"] == 0
    # and it really is cheaper in rounds while costing more arithmetic
    assert m2["candidate_cost"]["sequential_rounds"] < m2["generic_cost"]["sequential_rounds"]
    assert m2["candidate_cost"]["multiplies"] > m2["generic_cost"]["multiplies"]


def test_e000105_work_term_cannot_fail():
    """The work half of the E-000105 kill predicate is a tautology, not a measurement."""
    audit = e000105_work_term_audit()
    assert audit["cases"] == 512
    assert audit["observed_mismatches"] == 0
    assert audit["mismatch_is_reachable"] is False


def test_screens_are_not_the_same_function():
    """A guard against the two screens having been written to agree by construction."""
    r = run()
    assert r["false_kills"] == ["M1"]
    assert r["false_promotes"] == ["M5"]


def test_complete_screen_refuses_a_tie():
    """Negative control on the instrument: identical states and identical costs must not promote."""
    arm = Arm("s", Cost(multiplies=10, slow_memory_words=4, sequential_rounds=2))
    other = Arm("s", Cost(multiplies=10, slow_memory_words=4, sequential_rounds=2))
    assert complete_screen(arm, other)["verdict"] == "KILL"
    assert programme_screen(arm, other)["verdict"] == "KILL"


def test_underlying_mechanisms_agree_pointwise():
    """The aggregates above must not be hiding a per-case disagreement."""
    keys = tuple(tuple(F(1 + (i * j) % 5) for j in range(3)) for i in range(12))
    values = tuple(tuple(F((i + j) % 7) for j in range(3)) for i in range(12))
    q = (F(2), F(-1), F(3))
    assert m1_candidate(q, keys, values).state == m1_generic(q, keys, values).state

    for seed in [(0, 0, 0, 0), (3, 1, 2, 1), (6, 2, 2, 2)]:
        assert m2_candidate(seed, 6).state == m2_generic(seed, 6).state

    vals = tuple(F(i * i % 11) for i in range(16))
    assert m5_candidate(vals).state == m5_generic(vals).state
