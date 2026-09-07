from so.experiments.e000109_reversible_suffix_conjugacy_exact_transport_reduction import (
    candidate_conjugate_table,
    candidate_state_only_transport,
    exhaustive_conjugacy_cell,
    independently_compile_generic_transport_table,
    make_suffix,
    pod_edit,
    run_registered,
)


def test_small_nonlinear_reversible_transport_is_exact_but_reduces_to_generic_map() -> None:
    suffix = make_suffix(seed=109, p=7, depth=4)
    delta = 1
    candidate_table = candidate_conjugate_table(suffix, delta)
    generic_table = independently_compile_generic_transport_table(suffix, delta)
    assert candidate_table == generic_table

    for a in range(suffix.p):
        for b in range(suffix.p):
            x = (a, b)
            old_y = suffix.forward(x)
            fresh_y = suffix.forward(pod_edit(x, delta, suffix.p))
            assert candidate_state_only_transport(suffix, old_y, delta) == fresh_y
            assert generic_table[old_y] == fresh_y
            assert candidate_state_only_transport(suffix, fresh_y, -delta) == old_y


def test_full_cycle_conjugacy_count_matches_expected_class() -> None:
    cell = exhaustive_conjugacy_cell(6)
    assert cell["class_match"]
    assert cell["suffix_bijections_enumerated"] == 720
    assert cell["distinct_exact_transport_maps"] == 120
    assert cell["expected_n_cycle_conjugacy_class"] == 120


def test_registered_kill_screen() -> None:
    result = run_registered()
    totals = result["reversible_suffix"]["totals"]
    assert result["kill_screen_pass"]
    assert result["decision"] == (
        "KILL_REVERSIBILITY_AND_CONJUGACY_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE"
    )
    assert totals["registered_cells"] == 192
    assert totals["candidate_fresh_mismatches"] == 0
    assert totals["generic_fresh_mismatches"] == 0
    assert totals["candidate_generic_table_mismatches"] == 0
    assert totals["aba_failures"] == 0
    assert totals["material_fraction"] == 1.0
    assert totals["candidate_to_baseline_layer_eval_ratio"] == 2.0
