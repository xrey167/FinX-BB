"""Analytic and finite-difference checks; no pretrained checkpoints or network required."""
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from so.jlens_frozen import estimate_frozen_jlens


class CausalBlock(nn.Module):
    def __init__(self, d, nonlinear=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d, d, dtype=torch.float64) / d ** .5)
        self.nonlinear = nonlinear

    def forward(self, hidden_states, attention_mask=None):
        # Deliberate cross-position dependence, unlike a pointwise-only test.
        h = hidden_states * attention_mask[..., None]
        pooled = h.cumsum(1)
        update = pooled @ self.weight
        if self.nonlinear:
            update = torch.tanh(update)
        return (hidden_states + update,)


class TinyCausalLM(nn.Module):
    def __init__(self, seed=0, nonlinear=False):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(11, 5, dtype=torch.float64)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([CausalBlock(5, nonlinear) for _ in range(3)])
        self.output = nn.Parameter(torch.randn(11, 5, dtype=torch.float64))
        self.final_norm = nn.LayerNorm(5, dtype=torch.float64)
        self.config = SimpleNamespace(n_embd=5)
        self.fail = False
        self.keyword_blocks = False
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, input_ids, attention_mask, output_hidden_states=True,
                use_cache=False, return_dict=True):
        h = self.embedding(input_ids)
        states = []
        for i, block in enumerate(self.transformer.h):
            states.append(h)
            h, = (block(hidden_states=h, attention_mask=attention_mask) if self.keyword_blocks
                  else block(h, attention_mask=attention_mask))
            if self.fail and i == 1:
                raise RuntimeError('deliberate test failure')
        states.append(self.final_norm(h))
        return SimpleNamespace(hidden_states=tuple(states))


IDS = torch.tensor([[1, 2, 3], [4, 5, 0]])
MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])


def lens(model, **kwargs):
    return estimate_frozen_jlens(model, 1, [2, 4], IDS, MASK, model.output, **kwargs)


def test_historical_vjp_has_no_gradient_on_fully_frozen_backbone():
    m = TinyCausalLM()
    out = m(IDS, MASK)
    h, z = out.hidden_states[1], out.hidden_states[-2]
    assert not h.requires_grad and not z.requires_grad
    # This is the computation in the historical jlens_vectors, without a source leaf.
    with pytest.raises(RuntimeError, match='does not require grad'):
        torch.autograd.grad((z @ m.output[2]).sum(), h)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_analytic_cross_position_vjp(seed):
    m = TinyCausalLM(seed)
    jl = lens(m, batch_size=1)
    # 5 diagonal identity contributions and 9 causal pairs through W.
    expected = (5 * m.output[[2, 4]] + 9 * (m.output[[2, 4]] @ m.transformer.h[1].weight.T)) / 9
    torch.testing.assert_close(jl.mean_vectors, expected, rtol=1e-12, atol=1e-12)
    assert (jl.n_prompts, jl.n_source_positions, jl.n_position_pairs) == (2, 5, 9)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_nonlinear_vjp_matches_finite_difference(seed):
    m = TinyCausalLM(seed, nonlinear=True)
    jl = lens(m)
    d = torch.randn(5, dtype=torch.float64)
    epsilon = 1e-5
    def measure(sign):
        def shift(_module, args):
            return (args[0] + sign * epsilon * MASK.to(dtype=d.dtype)[..., None] * d,)
        handle = m.transformer.h[1].register_forward_pre_hook(shift)
        try:
            with torch.no_grad():
                z = m(IDS, MASK).hidden_states[2]
                return ((z * MASK[..., None]) @ m.output[[2, 4]].T).sum((0, 1))
        finally:
            handle.remove()
    finite_difference = (measure(1) - measure(-1)) / (2 * epsilon * 9)
    torch.testing.assert_close(jl.mean_vectors @ d, finite_difference, rtol=2e-8, atol=2e-9)


def test_batching_and_duplicate_corpus_do_not_change_raw_scale():
    m = TinyCausalLM(nonlinear=True)
    a, b = lens(m, batch_size=1), lens(m, batch_size=2)
    c = estimate_frozen_jlens(m, 1, [2, 4], IDS.repeat(2, 1), MASK.repeat(2, 1), m.output)
    torch.testing.assert_close(a.mean_vectors, b.mean_vectors)
    torch.testing.assert_close(a.raw_norms, c.raw_norms)
    assert c.n_position_pairs == 2 * a.n_position_pairs


def test_weights_flags_grads_and_hooks_are_unchanged():
    m = TinyCausalLM()
    before = [p.detach().clone() for p in m.parameters()]
    hooks = len(m.transformer.h[1]._forward_pre_hooks)
    lens(m)
    assert len(m.transformer.h[1]._forward_pre_hooks) == hooks
    for old, parameter in zip(before, m.parameters()):
        assert torch.equal(old, parameter)
        assert not parameter.requires_grad and parameter.grad is None
    assert not m.training


def test_failure_removes_hook_and_keeps_model_frozen():
    m = TinyCausalLM(); m.fail = True
    hooks = len(m.transformer.h[1]._forward_pre_hooks)
    with pytest.raises(RuntimeError, match='deliberate test failure'):
        lens(m)
    assert len(m.transformer.h[1]._forward_pre_hooks) == hooks
    assert all(not p.requires_grad and p.grad is None for p in m.parameters())


def test_no_grad_is_supported_but_inference_mode_is_rejected():
    m = TinyCausalLM()
    with torch.no_grad():
        assert torch.isfinite(lens(m).mean_vectors).all()
    with torch.inference_mode(), pytest.raises(ValueError, match='inference_mode'):
        lens(m)


@pytest.mark.parametrize('target', [1, 3, -1, 7])
def test_rejects_same_upstream_or_final_normalized_target(target):
    with pytest.raises(ValueError, match='require 0 <= source'):
        lens(TinyCausalLM(), target=target)


def test_training_mode_is_rejected_without_changing_it():
    m = TinyCausalLM(); m.train()
    with pytest.raises(ValueError, match='lm.eval'):
        lens(m)
    assert m.training


def test_invalid_masks_and_tokens_are_rejected():
    m = TinyCausalLM()
    for mask in (torch.zeros_like(MASK), MASK.float() + .5):
        with pytest.raises(ValueError):
            estimate_frozen_jlens(m, 1, [2, 4], IDS, mask, m.output)
    with pytest.raises(ValueError):
        estimate_frozen_jlens(m, 1, [2, 2], IDS, MASK, m.output)


def test_raw_projection_preserves_vjp_scale():
    m = TinyCausalLM(); jl = lens(m)
    x = torch.randn(4, 5, dtype=torch.float64)
    torch.testing.assert_close(jl.project(x), x @ jl.mean_vectors.T)
    torch.testing.assert_close(jl.project(x, unit_directions=True), x @ jl.directions.T)


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.TransformerEncoderLayer(5, 1, dim_feedforward=12, dropout=0.0,
                                               batch_first=True, norm_first=True, dtype=torch.float64)

    def forward(self, hidden_states, attention_mask=None):
        t = hidden_states.shape[1]
        causal = torch.ones(t, t, device=hidden_states.device, dtype=torch.bool).triu(1)
        return (self.core(hidden_states, src_mask=causal,
                          src_key_padding_mask=~attention_mask.bool()),)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_real_torch_attention_stack_matches_finite_difference(seed):
    m = TinyCausalLM(seed)
    m.transformer.h = nn.ModuleList([AttentionBlock() for _ in range(3)])
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    jl = lens(m)
    delta = torch.randn(5, dtype=torch.float64)
    epsilon = 1e-5
    scores = []
    for sign in (1, -1):
        def shift(_module, args):
            return (args[0] + sign * epsilon * MASK.to(delta.dtype)[..., None] * delta,)
        handle = m.transformer.h[1].register_forward_pre_hook(shift)
        try:
            with torch.no_grad():
                z = m(IDS, MASK).hidden_states[2]
                scores.append(((z * MASK[..., None]) @ m.output[[2, 4]].T).sum((0, 1)))
        finally:
            handle.remove()
    numerical = (scores[0] - scores[1]) / (2 * epsilon * jl.n_position_pairs)
    torch.testing.assert_close(jl.mean_vectors @ delta, numerical, rtol=1e-7, atol=1e-8)
    assert all(not p.requires_grad and p.grad is None for p in m.parameters())


def test_keyword_only_block_input_supported():
    m = TinyCausalLM(nonlinear=True)
    expected = lens(m).mean_vectors
    m.keyword_blocks = True
    torch.testing.assert_close(lens(m).mean_vectors, expected)


def test_padding_is_excluded_from_positions_and_raw_scale():
    m = TinyCausalLM(nonlinear=True)
    expected = lens(m)
    ids = torch.cat((IDS, torch.zeros(2, 2, dtype=IDS.dtype)), dim=1)
    mask = torch.cat((MASK, torch.zeros(2, 2, dtype=MASK.dtype)), dim=1)
    actual = estimate_frozen_jlens(m, 1, [2, 4], ids, mask, m.output)
    torch.testing.assert_close(actual.mean_vectors, expected.mean_vectors)
    assert actual.n_position_pairs == expected.n_position_pairs
