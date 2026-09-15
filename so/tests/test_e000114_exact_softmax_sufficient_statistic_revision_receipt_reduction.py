from fractions import Fraction

from so.experiments.e000114_exact_softmax_sufficient_statistic_revision_receipt_reduction import (
    DECISION,
    candidate_revision_receipt,
    fresh_attention,
    generic_dynamic_normalized_aggregate,
    run_cell,
    run_registered,
)


def test_exact_single_record_softmax_patch_matches_fresh_and_generic() -> None:
    query = 3
    static = [
        (-1, (1, 2, 3, 4)),
        (0, (4, 3, 2, 1)),
        (2, (-1, 5, 0, 2)),
    ]
    old_record = (0, (10, 11, 12, 13))
    new_record = (2, (-7, -8, -9, -10))

    normalizer, output = fresh_attention(query, [*static, old_record])
    candidate_z, candidate_output, candidate_work = candidate_revision_receipt(
        query,
        output,
        normalizer,
        old_record,
        new_record,
    )
    generic_z, generic_output, generic_work = generic_dynamic_normalized_aggregate(
        query,
        output,
        normalizer,
        [old_record],
        [new_record],
    )
    fresh_z, fresh_output = fresh_attention(query, [*static, new_record])

    assert candidate_z == generic_z == fresh_z
    assert candidate_output == generic_output == fresh_output
    assert candidate_work == generic_work


def test_exact_attention_is_true_scaled_softmax_special_case() -> None:
    # With logits ln(2) * q*k, exp(logit) == 2**(q*k) exactly.
    normalizer, output = fresh_attention(
        1,
        [
            (0, (0,)),
            (1, (10,)),
        ],
    )
    assert normalizer == Fraction(3, 1)
    assert output == (Fraction(20, 3),)


def test_contexts_require_state_dependent_post_softmax_corrections() -> None:
    result = run_cell(seed=0, width=8, sessions=16)
    assert result["candidate_fresh_attention_mismatches"] == 0
    assert result["generic_fresh_attention_mismatches"] == 0
    assert result["candidate_fresh_final_mismatches"] == 0
    assert result["generic_fresh_final_mismatches"] == 0
    assert min(result["unique_exact_output_deltas_by_transition"].values()) == 16
    assert result["wrong_one_global_output_delta_failures"] > 0


def test_registered_kill_screen() -> None:
    result = run_registered()
    assert result["passed"] is True
    assert result["decision"] == DECISION
    metrics = result["metrics"]
    assert metrics["lifecycle_cases"] == 6_144
    assert metrics["material_final_changes"] == 6_144
    assert metrics["candidate_attention_update_work"] == metrics["generic_attention_update_work"]
    assert metrics["candidate_ready_work"] == metrics["generic_ready_work"]
    assert metrics["candidate_ready_work"] > metrics["oracle_exact_read_patch_plus_suffix_work"]
