"""GEN-001 diagnostic/reference contracts; no trained capability claims."""
import pytest
import torch

from so.experiments.gen001_token_feedback_boundary import (
    Stamp, ManagedToken, append_managed, dependencies_valid, first_invalid,
    max_difference, run_case,
)


def test_indirect_descendant_inherits_without_new_read():
    p = [ManagedToken(1)]
    p.append(append_managed(p, 2, [Stamp('p', 1)]))
    p.append(append_managed(p, 3))
    assert p[-1].dependencies == frozenset({Stamp('p', 1)})
    assert not dependencies_valid(p[-1].dependencies, {})
    assert dependencies_valid([], {})  # naive direct-read-only receipt misses ancestry


def test_multiple_pods_union_and_single_revoke():
    p = [ManagedToken(2, frozenset({Stamp('p', 1)}))]
    t = append_managed(p, 3, [Stamp('q', 2)])
    assert t.dependencies == frozenset({Stamp('p', 1), Stamp('q', 2)})
    assert dependencies_valid(t.dependencies, {'p': 1, 'q': 2})
    assert not dependencies_valid(t.dependencies, {'q': 2})


def test_update_and_aba_never_revalidate_old_generation():
    deps = [Stamp('p', 1)]
    assert dependencies_valid(deps, {'p': 1})
    assert not dependencies_valid(deps, {'p': 2})
    assert not dependencies_valid(deps, {'p': 3})


def test_missing_pod_fails_closed():
    assert not dependencies_valid([Stamp('missing', 1)], {'p': 1})


def test_first_invalid_preserves_only_clean_prefix():
    p = [ManagedToken(1), ManagedToken(2)]
    p.append(append_managed(p, 3, [Stamp('p', 1)]))
    p.append(append_managed(p, 4))
    assert first_invalid(p, {'p': 1}) == 4
    assert first_invalid(p, {}) == 2
    assert first_invalid([], {}) == 0


def test_independent_session_not_invalidated():
    t = append_managed([], 3, [Stamp('q', 1)])
    assert dependencies_valid(t.dependencies, {'q': 1})


def test_same_token_id_does_not_erase_conservative_ancestry():
    p = [ManagedToken(7, frozenset({Stamp('p', 1)}))]
    t = append_managed(p, 7)
    assert t.token_id == 7
    assert not dependencies_valid(t.dependencies, {})


def test_cache_shape_mismatch_is_not_silently_truncated():
    with pytest.raises(ValueError):
        max_difference([torch.zeros(1)], [])
    with pytest.raises(ValueError):
        max_difference([torch.zeros(1)], [torch.zeros(2)])


@pytest.mark.parametrize('transition', ['revoke', 'update'])
def test_actual_adapter_generated_token_feedback(transition):
    pytest.importorskip('transformers')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = run_case(101, transition)
    assert result['pass'], [k for k, v in result['checks'].items() if not v]
