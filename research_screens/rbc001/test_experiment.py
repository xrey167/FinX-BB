"""Correctness of countermodels, not satisfaction of the research utility gates."""
import pytest
import torch
from research_screens.rbc001 import experiment as R


def test_pinned_source_identity():
    assert R.verify_source()["git_blob"] == "c0cc8326660f480034547836c9f47f88dd16ccbb"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mutation", R.MUTATIONS)
@pytest.mark.parametrize("phase", R.PHASES)
@pytest.mark.parametrize("mode", R.MODES)
def test_hook_schedules(seed, mutation, phase, mode):
    row = R.schedule_case(seed, mutation, phase, mode)
    assert row["matches_registered_prediction"]
    assert row["trace"] == ["mutation_requested", "mutation_committed", "consume"]
    if mutation == "unrelated_pod_update":
        assert row["consumed"] and row["authoritative_valid_at_consume"]
    else:
        assert not row["authoritative_valid_at_consume"]
        if phase == "before_guard" or mode == "late_check_comparator":
            assert not row["consumed"]
            assert row["output_maxabs"] == 0
        else:
            assert row["consumed"] and row["stale_consumed"]
            assert row["output_maxabs"] > 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_other_thread_is_serialized(seed):
    row = R.other_thread_control(seed)
    assert row["competing_acquire_blocked"]
    assert not row["mutation_done_before_consume"]
    assert row["authoritative_valid_at_consume"]
    assert row["consumed"] and not row["stale_consumed"]
    assert row["old_witness_invalid_after_join"]
    assert row["trace"] == ["consume", "other_thread_mutation_committed"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binding_is_not_lineage_certification(seed):
    row = R.binding_case(seed)
    for key, val in row.items():
        if isinstance(val, bool):
            assert val, key


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_content_different_generation_is_observationally_ambiguous(seed):
    row = R.observation_collision_case(seed)
    assert not row["old_witness_valid"] and row["fresh_witness_valid"]
    assert row["all_observations_byte_equal"]
    assert all(x == 0 for x in row["maxabs"].values())


def test_exact_membership_counting_bound():
    row = R.small_membership_collision()
    assert row["possible_histories"] == "56"
    assert row["distinct_required_membership_vectors"] == 56
    assert row["all_outputs_byte_equal"]
    assert row["minimum_exact_auxiliary_bits"] == 6
    assert R.membership_bound(10, 0)["minimum_exact_auxiliary_bits"] == 0
    assert R.membership_bound(10, 10)["minimum_exact_auxiliary_bits"] == 0


def test_membership_domain_validation():
    with pytest.raises(ValueError):
        R.membership_bound(2, 3)
