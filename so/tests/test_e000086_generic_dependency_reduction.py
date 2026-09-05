from so.cavi import CAVIAuthority
from so.experiments.e000086_generic_dependency_reduction import (
    ARTIFACT_CLASSES,
    capture_generic,
    generic_dependency_valid,
    run,
    witness_projection,
)


def authority():
    a = CAVIAuthority()
    for p in (1, 2, 3):
        a.create_pod(p)
    a.create_alias(10, 1)
    a.create_alias(11, 3)
    return a


def assert_same(a, w, d):
    assert a.validate_witness(w) == generic_dependency_valid(a, d)


def test_witness_is_exact_dependency_projection():
    a = authority()
    w = a.witness(10)
    assert witness_projection(w) == capture_generic(a, 10)


def test_relink_invalidates_both_selective_methods():
    a = authority(); w = a.witness(10); d = capture_generic(a, 10)
    a.relink_alias(10, 2)
    assert not a.validate_witness(w)
    assert not generic_dependency_valid(a, d)


def test_pod_update_invalidates_both_selective_methods():
    a = authority(); w = a.witness(10); d = capture_generic(a, 10)
    a.update_pod(1)
    assert_same(a, w, d)
    assert not generic_dependency_valid(a, d)


def test_revoke_restore_aba_never_revives_old_snapshot():
    a = authority(); w = a.witness(10); d = capture_generic(a, 10)
    a.revoke_alias(10); assert_same(a, w, d)
    a.restore_alias(10); assert_same(a, w, d)
    assert not a.validate_witness(w)


def test_shred_restore_aba_never_revives_old_snapshot():
    a = authority(); w = a.witness(10); d = capture_generic(a, 10)
    a.shred_pod(1); assert_same(a, w, d)
    a.restore_pod(1); assert_same(a, w, d)
    assert not generic_dependency_valid(a, d)


def test_unrelated_pod_and_alias_changes_preserve_both():
    a = authority(); w = a.witness(10); d = capture_generic(a, 10)
    a.update_pod(3); assert_same(a, w, d)
    assert a.validate_witness(w)
    a.relink_alias(11, 2); assert_same(a, w, d)
    assert generic_dependency_valid(a, d)


def test_artifact_type_does_not_change_runtime_freshness_decision():
    result = run()
    assert len(ARTIFACT_CLASSES) == 5
    for row in result["trace_rows"]:
        for prefix in row["prefixes"]:
            values = prefix["decision"]["per_artifact"]
            assert set(values) == set(ARTIFACT_CLASSES)
            assert all(v["equal"] for v in values.values())


def test_registered_reduction_passes_and_kills_only_current_seam():
    result = run()
    assert all(result["checks"].values())
    assert result["reduction_pass"]
    assert result["decision"] == "KILL_CURRENT_COHERENCE_SEAM_AS_GENERIC_DEPENDENCY_SPECIALIZATION"
    assert "does not rule out" in result["scope"]
