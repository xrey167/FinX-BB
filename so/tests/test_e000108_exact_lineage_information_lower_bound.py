from so.experiments.e000108_exact_lineage_information_lower_bound import run, run_cell


def test_small_exact_lineage_lower_bound_matches_generic_ledger() -> None:
    cell = run_cell(q=2, n=5, seed=108)
    assert cell["unique_delete_profile_check"]
    assert cell["required_auxiliary_certificate_states"] == 16
    assert cell["generic_ledger_auxiliary_states"] == 16
    assert cell["generic_delete_failures"] == 0
    assert cell["generic_update_failures"] == 0
    assert cell["aba_failures"] == 0


def test_registered_kill_screen() -> None:
    result = run()
    assert result["kill_screen_pass"]
    assert result["decision"] == (
        "KILL_COMPACT_EXACT_CENTRAL_LINEAGE_CAPSULE_AS_GENERAL_ADVANTAGE"
    )
    assert result["registered_cells"] == 4
    assert result["generic_baseline_failures"] == 0
