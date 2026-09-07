from so.experiments import e000094_noninterference_late_binding_reduction as E94


def test_firewall_and_sidecar_are_exactly_equal():
    rec = E94.run(range(3), n_x=5, n_q=4)
    assert rec["equality_cases"] > 0
    assert rec["equality_mismatches"] == 0
    assert rec["lifecycle_trace_mismatches"] == 0


def test_negative_control_detects_pod_contamination():
    rec = E94.run(range(2), n_x=4, n_q=3)
    assert rec["negative_control_pairs"] > 0
    assert rec["negative_control_detected"] == rec["negative_control_pairs"]


def test_mutable_cached_substate_is_not_reusable_across_lifecycle():
    rec = E94.run(range(2), n_x=4, n_q=3)
    assert rec["mutable_transition_cases"] > 0
    assert rec["mutable_substate_changed"] == rec["mutable_transition_cases"]


def test_registered_kill_screen_passes():
    rec = E94.run(range(2), n_x=4, n_q=3)
    assert rec["kill_screen_pass"] is True
    assert rec["decision"] == "KILL_NONINTERFERENCE_ALONE_AS_NOVELTY_SEAM"
