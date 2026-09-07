from so.experiments.e000113_task_essential_incremental_accumulator_reduction import (
    DECISION,
    candidate_transport,
    generic_sparse_feature_diff,
    pod_feature_column,
    run_cell,
    run_registered,
)


def test_candidate_and_generic_single_update_are_identical() -> None:
    width = 8
    arm = "contextual"
    bucket = 3
    old_state = 0
    new_state = 1
    old_acc = tuple(range(width))

    candidate, candidate_work = candidate_transport(old_acc, arm, bucket, old_state, new_state)
    generic, generic_work = generic_sparse_feature_diff(
        old_acc,
        [pod_feature_column(arm, bucket, old_state, width)],
        [pod_feature_column(arm, bucket, new_state, width)],
    )

    assert candidate == generic
    assert candidate_work == generic_work == 2 * width


def test_global_arm_has_one_delta_and_exact_transport() -> None:
    result = run_cell(seed=0, width=8, arm="global", sessions=16)
    assert result["candidate_fresh_accumulator_mismatches"] == 0
    assert result["generic_fresh_accumulator_mismatches"] == 0
    assert result["candidate_fresh_final_mismatches"] == 0
    assert result["generic_fresh_final_mismatches"] == 0
    assert set(result["unique_exact_deltas_by_transition"].values()) == {1}


def test_contextual_arm_rejects_wrong_single_global_receipt() -> None:
    result = run_cell(seed=0, width=8, arm="contextual", sessions=16)
    assert result["candidate_fresh_accumulator_mismatches"] == 0
    assert result["generic_fresh_accumulator_mismatches"] == 0
    assert min(result["unique_exact_deltas_by_transition"].values()) == 8
    assert result["wrong_single_global_receipt_failures"] > 0


def test_registered_kill_screen() -> None:
    result = run_registered()
    assert result["passed"] is True
    assert result["decision"] == DECISION
    metrics = result["metrics"]
    assert metrics["lifecycle_cases"] == 24_576
    assert metrics["material_final_changes"] == 24_576
    assert metrics["candidate_mutation_adds"] == metrics["generic_mutation_adds"]
    assert metrics["candidate_ready_work"] == metrics["generic_ready_work"]
