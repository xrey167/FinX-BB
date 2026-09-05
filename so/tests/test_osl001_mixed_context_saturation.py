from so.experiments.osl001_mixed_context_saturation import (
    _implementation_cell, _segmentation_cell, _saturation_cell, _tiny_gpt2_control,
)


def test_actual_lineage_union_rejects_whole_state_on_one_stale_witness():
    row = _implementation_cell(8)
    assert row["stale_witnesses"] == 1
    assert row["other_witnesses_still_current"] == 7
    assert not row["whole_state_reusable_after_one_pod_update"]


def test_late_read_exposes_maximal_monolithic_collateral():
    row = _segmentation_cell(8, 4096)
    assert row["read_positions"][-1] == 4095
    assert row["ordinary_exact_suffix_recompute_tokens"][-1] == 1
    assert row["whole_cache_recompute_tokens_per_single_update"][-1] == 4096
    assert row["late_read_ratio"] == 4096.0
    assert row["late_read_exact_reusable_prefix_fraction"] > 0.999


def test_dependency_union_saturates_staleness_probability():
    one = _saturation_cell(1)["rates"]["0.01"]["whole_cache_stale_probability"]
    many = _saturation_cell(32)["rates"]["0.01"]["whole_cache_stale_probability"]
    assert abs(one - 0.01) < 1e-12
    assert many > 0.27


def test_actual_causal_transformer_prefix_is_exactly_reusable():
    row = _tiny_gpt2_control(0, seq_len=32, read_pos=24)
    assert row["repeat_forward_exact"]
    assert row["all_prefix_kv_before_read_byte_identical"]
    assert row["prefix_unequal_tensors"] == 0
    assert row["downstream_suffix_maxabs"] > 0
    assert row["unnecessary_prefix_rejection_tokens"] == 24
