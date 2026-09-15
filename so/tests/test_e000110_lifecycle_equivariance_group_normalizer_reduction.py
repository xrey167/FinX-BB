from so.experiments.e000110_lifecycle_equivariance_group_normalizer_reduction import (
    EditAction,
    generic_transport_action,
    independently_compile_transport_action,
    make_suffix,
    run_registered,
)


def test_nonlinear_normalizer_transport_matches_fresh_and_generic() -> None:
    suffix = make_suffix(seed=110, p=11, depth=8)
    action = EditAction(2, 0, 0)
    candidate, appeared = suffix.candidate_transport_action(action)
    generic = generic_transport_action(suffix, action)
    probe = independently_compile_transport_action(suffix, action)

    assert appeared
    assert candidate == generic == probe

    for a in range(suffix.p):
        for b in range(suffix.p):
            x = (a, b)
            old_final = suffix.forward(x)
            fresh_final = suffix.forward(action.apply(x, suffix.p))
            assert candidate.apply(old_final, suffix.p) == fresh_final
            assert generic.apply(old_final, suffix.p) == fresh_final


def test_layerwise_nonlinearity_turns_translation_into_state_dependent_action() -> None:
    suffix = make_suffix(seed=7, p=13, depth=2)
    action = EditAction(1, 0, 0)
    first = suffix.layers[0].candidate_push_action(action, suffix.p)
    assert first.v % suffix.p != 0


def test_registered_kill_screen() -> None:
    result = run_registered()
    totals = result["normalizer_transport"]["totals"]
    assert result["kill_screen_pass"]
    assert result["decision"] == (
        "KILL_LIFECYCLE_EQUIVARIANCE_AND_GROUP_NORMALIZER_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE"
    )
    assert totals["registered_cells"] == 192
    assert totals["edit_cells"] == 576
    assert totals["exact_state_transport_cases"] == 111168
    assert totals["candidate_fresh_mismatches"] == 0
    assert totals["generic_fresh_mismatches"] == 0
    assert totals["candidate_generic_action_mismatches"] == 0
    assert totals["independent_probe_action_mismatches"] == 0
    assert totals["aba_failures"] == 0
    assert totals["material_fraction"] == 1.0
    assert totals["state_dependent_action_fraction"] == 1.0
    assert totals["candidate_to_generic_normalized_work_ratio"] == 1.0
    assert totals["min_cell_replay_to_candidate_normalized_work_ratio"] > 1.9
