from fractions import Fraction as F

from so.experiments.e000104_tensor_train_lifecycle_quotient_reduction import (
    candidate_environments,
    candidate_local_update,
    full_contract,
    generate_cell,
    generic_cached_environment_update,
    materialize,
    run,
    third_order_interaction,
)


def test_candidate_and_generic_match_fresh_update():
    states, sessions = generate_cell(seed=3, pod_count=6, rank=3, session_count=2)
    target = 3
    old = materialize(states)
    for left, right in sessions:
        le, re = candidate_environments(old, target, left, right)
        fresh = full_contract(materialize(states, {target: "new"}), left, right)
        assert candidate_local_update(le, states[target]["new"], re) == fresh
        assert generic_cached_environment_update(le, states[target]["new"], re) == fresh


def test_delete_restore_and_aba_are_exact():
    states, sessions = generate_cell(seed=5, pod_count=8, rank=2, session_count=1)
    target = 4
    left, right = sessions[0]
    old = materialize(states)
    old_gold = full_contract(old, left, right)
    le, re = candidate_environments(old, target, left, right)
    for state_name in ("absent", "restore", "new", "old"):
        fresh = full_contract(materialize(states, {target: state_name}), left, right)
        assert candidate_local_update(le, states[target][state_name], re) == fresh
        assert generic_cached_environment_update(le, states[target][state_name], re) == fresh
    assert candidate_local_update(le, states[target]["new"], re) != old_gold
    assert candidate_local_update(le, states[target]["old"], re) == old_gold


def test_registered_interaction_is_genuinely_third_order():
    states, sessions = generate_cell(seed=7, pod_count=10, rank=4, session_count=1)
    left, right = sessions[0]
    target = 5
    interaction = third_order_interaction(states, (target - 1, target, target + 1), left, right)
    assert interaction != F(0)


def test_small_run_passes_reduction():
    result = run(seed_count=2, pod_counts=(4, 6), ranks=(2, 3), session_count=3)
    assert result["kill_screen_pass"] is True
    assert result["candidate_fresh_mismatch"] == 0
    assert result["generic_fresh_mismatch"] == 0
    assert result["candidate_generic_mismatch"] == 0
    assert result["work_mismatch"] == 0
    assert result["material_update_cases"] == result["session_cells"]
    assert result["interaction_material"] == result["interaction_checks"]
