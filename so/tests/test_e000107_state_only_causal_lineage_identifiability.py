from so.experiments.e000107_state_only_causal_lineage_identifiability import run


def test_all_registered_families_have_collision_witness():
    r = run()
    assert r["families_with_indistinguishability_witness"] == r["total_families"] == 4


def test_state_only_kill_screen_passes():
    r = run()
    assert r["kill_screen_pass"] is True
    assert r["decision"] == "KILL_STATE_ONLY_EXACT_CAUSAL_LINEAGE_AS_GENERAL_GUARANTEE"


def test_oracle_history_resolves_registered_ambiguity():
    r = run()
    assert r["oracle_history_failures"] == 0


def test_multiple_targets_are_nonidentifiable():
    r = run()
    assert r["target_witness_slots_across_families"] >= 4
