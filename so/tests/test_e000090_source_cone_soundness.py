"""Independent regressions for E90; random operators, not language backbones."""
import numpy as np
import pytest
from so.experiments.e000090_source_cone_soundness import (
    EFFECT_FLOOR, averaged_lens_screen, block_case, denominator_screen,
    difference, full_blocks, identical, normalization_screen, normalize,
    replay_local, routing_screen,
)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("mode", ["identity", "local_layer", "local_rms", "global_layer", "global_rms"])
def test_local_support_is_not_preserved_by_global_normalization(seed, mode):
    c = block_case(seed)
    old = full_blocks(c, c["source_old"], mode)
    new = full_blocks(c, c["source_new"], mode)
    assert identical(old, full_blocks(c, c["source_old"], mode))
    d = np.abs(new-old)
    if mode.startswith("global_"):
        assert np.count_nonzero(d[0] > EFFECT_FLOOR) == 128
        assert np.any(d[:,16:] > EFFECT_FLOOR)
        with pytest.raises(ValueError):
            replay_local(c, c["source_new"], old, mode)
    else:
        assert identical(old[:,16:], new[:,16:])
        assert identical(replay_local(c, c["source_new"], old, mode), new)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("mode", ["identity", "local_layer", "local_rms"])
def test_update_and_delete_equal_fresh_at_every_write(seed, mode):
    c = block_case(seed)
    old = full_blocks(c, c["source_old"], mode)
    for src in [c["source_new"], np.zeros(16)]:
        assert identical(replay_local(c, src, old, mode), full_blocks(c, src, mode))


@pytest.mark.parametrize("seed", range(5))
def test_dormant_routing_source_has_no_old_payload_descendants(seed):
    row = routing_screen(seed)
    assert row["source_selected_before"] == 0
    assert row["source_selected_after"] == 64
    assert row["changed_persistent_vectors"] == 192
    assert row["vectors_missed_by_old_payload_lineage"] == 192
    assert not row["old_payload_cone_patch_vs_fresh"]["byte_identical"]
    assert row["decision_aware_vs_fresh"]["byte_identical"]


@pytest.mark.parametrize("seed", range(5))
def test_unchanged_top1_does_not_imply_unchanged_denominator(seed):
    row = denominator_screen(seed)
    assert row["winner_before"] == row["winner_after"] == 1
    assert row["global_denominator_effect"]["maxabs"] > EFFECT_FLOOR
    assert row["postselection_renormalization_control"]["byte_identical"]


@pytest.mark.parametrize("seed", range(5))
def test_average_jacobian_cancellation_is_not_no_causal_effect(seed):
    row = averaged_lens_screen(seed)
    assert row["averaged_lens_stale_vs_never"]["byte_identical"]
    assert row["averaged_jacobian_source_column_maxabs"] == 0
    assert row["conditional_jacobian_source_column_maxabs"] > 0
    assert row["positive_context_logit_effect_maxabs"] > EFFECT_FLOOR


def test_byte_equality_is_stricter_than_numerical_equality():
    a, b = np.array([0.]), np.array([-0.])
    assert np.array_equal(a, b)
    assert not identical(a, b)


@pytest.mark.parametrize("seed", range(5))
def test_averaged_jacobian_with_independent_torch_autograd(seed):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3000+seed)
    a, b = rng.normal(size=(2,8))
    p = float(rng.uniform(.5,1.5))
    a, b = torch.tensor(a), torch.tensor(b)
    h = torch.tensor([.25,p], dtype=torch.float64, requires_grad=True)
    def model(x, context):
        return a*x[0] + context*b*torch.tanh(x[1])
    jp = torch.autograd.functional.jacobian(lambda x: model(x,1.), h)
    jn = torch.autograd.functional.jacobian(lambda x: model(x,-1.), h)
    avg = (jp+jn)/2
    assert torch.count_nonzero(avg[:,1]) == 0
    never = torch.tensor([.25,0.], dtype=torch.float64)
    assert torch.equal(avg @ h, avg @ never)
    assert torch.max(torch.abs(model(h,1.)-model(never,1.))) > EFFECT_FLOOR


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", ["layer", "rms"])
def test_global_normalization_density_with_independent_torch(seed, kind):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    c = block_case(seed)
    x = c["initial"].copy()
    y = x.copy()
    x[:16] += c["source_old"]
    y[:16] += c["source_new"]
    x, y = torch.tensor(x), torch.tensor(y)
    if kind == "layer":
        u, v = F.layer_norm(x, (128,), eps=1e-5), F.layer_norm(y, (128,), eps=1e-5)
    else:
        u, v = F.rms_norm(x, (128,), eps=1e-5), F.rms_norm(y, (128,), eps=1e-5)
    assert int(torch.count_nonzero(torch.abs(u-v) > EFFECT_FLOOR)) == 128
