from fractions import Fraction as F

from so.experiments.e000103_low_rank_resolvent_lifecycle_reduction import (
    inverse,
    matvec,
    rank1_inverse_update,
    rank1_solution_update,
    run,
    woodbury_rankk_inverse,
    woodbury_rankk_solution,
)


def test_rank1_inverse_and_solution_match_fresh():
    m = [[F(5), F(1)], [F(2), F(7)]]
    inv = inverse(m)
    u = [F(1), F(-2)]
    v = [F(2), F(1)]
    new_m = [[m[i][j] + u[i] * v[j] for j in range(2)] for i in range(2)]
    fresh_inv = inverse(new_m)
    got_inv, z, denom = rank1_inverse_update(inv, u, v)
    assert got_inv == fresh_inv
    b = [F(3), F(-1)]
    old_x = matvec(inv, b)
    got_x = rank1_solution_update(old_x, z, denom, v)
    assert got_x == matvec(fresh_inv, b)


def test_direct_rank2_woodbury_matches_fresh():
    m = [[F(11), F(1), F(0)], [F(2), F(13), F(1)], [F(0), F(-1), F(17)]]
    inv = inverse(m)
    u1, v1 = [F(1), F(0), F(-1)], [F(2), F(1), F(0)]
    u2, v2 = [F(0), F(2), F(1)], [F(-1), F(0), F(1)]
    new_m = [[m[i][j] + u1[i] * v1[j] + u2[i] * v2[j] for j in range(3)] for i in range(3)]
    fresh_inv = inverse(new_m)
    assert woodbury_rankk_inverse(inv, [u1, u2], [v1, v2]) == fresh_inv
    b = [F(2), F(3), F(-4)]
    old_x = matvec(inv, b)
    assert woodbury_rankk_solution(old_x, inv, [u1, u2], [v1, v2]) == matvec(fresh_inv, b)


def test_registered_logic_small_domain_passes():
    result = run(seed_count=2, dimensions=(4, 6))
    assert result["kill_screen_pass"] is True
    assert result["candidate_fresh_state_mismatches"] == 0
    assert result["generic_woodbury_fresh_state_mismatches"] == 0
    assert result["candidate_generic_inverse_mismatches"] == 0
    assert result["aba_mismatches"] == 0
    assert result["material_edit_rate"] >= 0.95
    assert result["interaction_rate"] >= 0.95


def test_registered_counts_small_domain():
    result = run(seed_count=1, dimensions=(4,))
    assert result["revision_cells"] == 4
    assert result["session_cases"] == 4 * 24
    assert result["interaction_cells"] == 4
