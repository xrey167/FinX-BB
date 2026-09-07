"""Numerical and gradient equivalence on randomly initialized GPT-2, not capability evidence."""
import copy
import pytest
import torch
import torch.nn.functional as F
from so.last_token_adapter import LastTokenGPT2Adapter
from so.tests.test_llm_adapter import _adapter, _bank, _prompt


@pytest.mark.parametrize("memory", [False, True])
@pytest.mark.parametrize("train", [False, True])
def test_last_token_head_matches_original_outputs_and_gradients(memory, train):
    original = _adapter(status_gated=True)
    fast = LastTokenGPT2Adapter(copy.deepcopy(original.lm), original.cfg,
                               original.entity_token_ids.tolist(), int(original.candidate_ids[-1]))
    # deepcopy also copies decoder hooks closing over the original object: remove
    # those before re-registering in independent tests, otherwise test is invalid.
    for block in fast.lm.transformer.h:
        block._forward_hooks.clear()
    fast._hooks = [fast.lm.transformer.h[l].register_forward_hook(fast._make_hook(i, l))
                   for i, l in enumerate(fast.cfg.read_layers)]
    fast.load_state_dict(original.state_dict())
    original.train(train); fast.train(train)
    ids, am, last = _prompt()
    am[0, -2:] = 0; last[0] -= 2
    bank = _bank() if memory else None
    torch.manual_seed(44)
    a = original(bank, ids, am, last)
    torch.manual_seed(44)
    b = fast(bank, ids, am, last)
    for x, y in zip(a, b):
        if x is None:
            assert y is None
        else:
            torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
    if memory:
        F.cross_entropy(a[0], torch.tensor([1, 2, 3, 4])).backward()
        F.cross_entropy(b[0], torch.tensor([1, 2, 3, 4])).backward()
        ga = dict(original.named_parameters()); gb = dict(fast.named_parameters())
        for name, p in ga.items():
            if p.grad is not None:
                assert gb[name].grad is not None
                torch.testing.assert_close(p.grad, gb[name].grad, atol=2e-6, rtol=5e-4)
