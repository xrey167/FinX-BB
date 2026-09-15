from so.experiments.pdx001_payload_derived_index_audit import (
    DOMAIN,
    Store,
    audit_policy,
    audit_row,
    run,
)


def _store():
    rows = [((i * 5 + 1) % 97, (i * 3 + 2) % 13, (i * 37 + 11) % DOMAIN) for i in range(24)]
    return Store(rows)


def _by_policy(result):
    return {p["policy"]: p for p in result["policies"]}


def test_attack_validity_floor_is_met():
    """E-000019's discipline: an at-chance reading means nothing unless the attack reads a live row."""
    r = run()
    assert r["attack_validity_floor_met"] is True
    assert r["validity_control"]["top1_recovery"] == 1.0


def test_decision_is_decisive():
    r = run()
    assert r["decision"] == "PAYLOAD_DERIVED_INDEX_CHANNEL_CONFIRMED_ACROSS_STORE_SHAPES"


def test_certificate_and_attack_agree_on_every_policy():
    """The two halves are the same statement; a disagreement is a broken instrument."""
    r = run()
    assert r["instrument_self_consistent"] is True
    for p in r["policies"]:
        assert p["certificate_matches_attack"] is True, p["policy"]


def test_three_store_shapes_leak_above_chance():
    r = run()
    assert set(r["policies_leaking_above_chance"]) == {
        "value_gated_shred",
        "hnsw_tombstone",
        "codebook_key",
    }


def test_value_gated_shred_names_the_payload():
    """E-000028's own defect, restated without torch."""
    p = _by_policy(run())["value_gated_shred"]
    assert p["top1_recovery"] == 1.0
    assert p["payload_derived_channels"] == ["key_rev"]
    assert p["certified"] is False


def test_codebook_key_names_the_payload():
    p = _by_policy(run())["codebook_key"]
    assert p["top1_recovery"] == 1.0
    assert p["payload_derived_channels"] == ["codebook_key"]


def test_tombstone_leaks_without_ever_naming_the_payload():
    """The case a top-1 headline would have called clean."""
    p = _by_policy(run())["hnsw_tombstone"]
    assert p["top1_recovery"] == 0.0
    assert p["leaks_above_chance"] is True
    assert p["mean_candidates_remaining"] < 5
    assert p["search_space_reduction_factor"] > 50
    assert p["mean_posterior_on_true_payload"] > 50 * p["chance_top1"]
    assert p["payload_derived_channels"] == ["adjacency"]


def test_both_controls_reach_chance_and_are_certified():
    r = run()
    assert set(r["policies_certified_independent"]) == {"revoke_unindex", "gate_all_derived"}
    for name in ("revoke_unindex", "gate_all_derived"):
        p = _by_policy(r)[name]
        assert p["certified"] is True
        assert p["payload_derived_channels"] == []
        assert p["mean_candidates_remaining"] == DOMAIN
        assert p["mean_posterior_on_true_payload"] == p["chance_top1"]


def test_a_live_row_is_never_certified():
    """Vacuity guard: an instrument that certifies a live payload is not testing anything."""
    store = _store()
    live = audit_policy(store, "live", list(range(12)))
    assert live["certified"] is False
    assert live["top1_recovery"] == 1.0


def test_sweep_is_exhaustive_over_the_payload_domain():
    """Certified means swept, not sampled: the clean policies leave the whole domain admissible."""
    store = _store()
    row = audit_row(store, "gate_all_derived", 0)
    assert row["candidates_remaining"] == DOMAIN
    assert row["certified_independent"] is True

    leaky = audit_row(store, "value_gated_shred", 0)
    assert leaky["candidates_remaining"] == 1
    assert leaky["recovered"] is True


def test_true_payload_is_always_inside_the_candidate_set():
    """Soundness of the attack: it never excludes the right answer, so 1/|set| is a real posterior."""
    store = _store()
    for policy in ("value_gated_shred", "hnsw_tombstone", "codebook_key", "gate_all_derived"):
        for target in range(6):
            true_obj = store.rows[target][2]
            observed = store.observe(policy, target, true_obj)
            assert store.observe(policy, target, true_obj) == observed
            consistent = [o for o in range(DOMAIN) if store.observe(policy, target, o) == observed]
            assert true_obj in consistent, (policy, target)
