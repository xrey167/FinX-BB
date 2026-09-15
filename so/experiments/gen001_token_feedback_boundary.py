"""GEN-001: fixed-token cache purity is not autoregressive lifecycle closure.

Diagnostic only. Uses the actual KnowledgeAdapterLM final-block write on a tiny,
randomly initialized and frozen GPT-2. One pod and a deliberately constant router
make the first read material. These are NOT trained reader/capability results.
The reference is a fresh continuation with the lifecycle transition applied
BEFORE the first managed token is sampled. Already published external text is
outside this experiment; the generated tokens are an uncommitted internal buffer.

The lineage code is a conservative, ordinary reference monitor, not production
CAVI, not an authentication scheme and not a concurrent publication protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Iterable, Mapping, Sequence

import torch


@dataclass(frozen=True, order=True)
class Stamp:
    pod: str
    generation: int


@dataclass(frozen=True)
class ManagedToken:
    token_id: int
    dependencies: frozenset[Stamp] = frozenset()


def dependencies_valid(dependencies: Iterable[Stamp], live: Mapping[str, int]) -> bool:
    """Fail closed for absent identities and stale incarnations (including ABA)."""
    return all(live.get(d.pod) == d.generation for d in dependencies)


def append_managed(prefix: Sequence[ManagedToken], token_id: int,
                   reads: Iterable[Stamp] = ()) -> ManagedToken:
    """Conservative ancestry: all visible prefix inputs plus this step's reads.

    Scans the prefix for clarity. This is NOT a minimal-dependency or fast-path claim.
    Dense computation can depend on an input even when its current attention is tiny.
    """
    inherited = frozenset(d for t in prefix for d in t.dependencies)
    return ManagedToken(int(token_id), inherited | frozenset(reads))


def first_invalid(prefix: Sequence[ManagedToken], live: Mapping[str, int]) -> int:
    """A valid continuation can retain only the prefix before this position."""
    return next((i for i, t in enumerate(prefix)
                 if not dependencies_valid(t.dependencies, live)), len(prefix))


def cache_tensors(cache) -> list[torch.Tensor]:
    if hasattr(cache, 'layers'):
        return [t for layer in cache.layers for t in (layer.keys, layer.values)]
    if hasattr(cache, 'key_cache'):
        return [t for kv in zip(cache.key_cache, cache.value_cache) for t in kv]
    return [t for kv in cache for t in kv]


def cache_length(cache) -> int:
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):
        return int(cache.get_seq_length())
    return int(cache[0][0].shape[-2])


def max_difference(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> float:
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError('Cannot compare caches with different shapes')
    return max((float((x-y).abs().max()) for x, y in zip(a, b)), default=0.0)


def make_model(seed: int):
    from transformers import GPT2Config, GPT2LMHeadModel
    from so.llm_adapter import AdapterConfig, KnowledgeAdapterLM
    torch.manual_seed(seed)
    lm_cfg = GPT2Config(vocab_size=64, n_positions=64, n_embd=64, n_layer=4,
                        n_head=4, resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
                        bos_token_id=0, eos_token_id=63)
    lm_cfg._attn_implementation = 'eager'
    lm = GPT2LMHeadModel(lm_cfg).eval()
    cfg = AdapterConfig(read_layers=(1, 2), write_layer=3, d_key=16,
                        marker_dim=16, use_marker_gate=False, status_gated=True)
    m = KnowledgeAdapterLM(lm, cfg, list(range(10, 30)), 5).eval()
    with torch.no_grad():
        # Controlled material read, NOT a learned routing result.
        m.inject_gain.fill_(8.0)
        m.null_value.zero_()
        for q in m.q_proj.values():
            q.weight.zero_()
            q.bias.zero_()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def one_pod(obj: int, active: bool = True) -> dict[str, torch.Tensor]:
    return {'subject': torch.tensor([0]), 'relation': torch.tensor([0]),
            'obj': torch.tensor([obj]), 'marker': torch.zeros(1, 16),
            'active': torch.tensor([active]), 'routable': torch.tensor([True])}


@torch.no_grad()
def forward_cached(m, ids: torch.Tensor, bank=None, past=None):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError('GEN-001 expects one nonempty token sequence')
    if m._ctx is not None:
        raise RuntimeError('Adapter context leaked between calls')
    try:
        if bank is not None:
            enc = m.encode_bank(bank)
            m._ctx = {'keys': enc['keys'], 'values': enc['values'],
                      'allowed': enc['active'],
                      'last_idx': torch.tensor([ids.shape[1]-1]), 'routing': []}
        mask = torch.ones(1, cache_length(past)+ids.shape[1], dtype=torch.long)
        out = m.lm(input_ids=ids, attention_mask=mask, past_key_values=past,
                   use_cache=True)
        if m._ctx is not None and m._ctx.get('deferred'):
            raise AssertionError('Late write was not consumed')
        return out
    finally:
        m._ctx = None


@torch.no_grad()
def continuation(m, prompt: list[int], bank, count: int) -> dict:
    out = forward_cached(m, torch.tensor([prompt]), bank=bank)
    prefill = [t.clone() for t in cache_tensors(out.past_key_values)]
    first_logits = out.logits[0, -1].clone()
    chosen = int(first_logits.argmax())
    generated = []
    for _ in range(count):
        generated.append(chosen)
        # No pod read on any of these downstream decoding steps.
        out = forward_cached(m, torch.tensor([[chosen]]), past=out.past_key_values)
        chosen = int(out.logits[0, -1].argmax())
    return {'prefill': prefill, 'first_logits': first_logits, 'tokens': generated,
            'logits': out.logits[0, -1].clone(),
            'cache': [t.clone() for t in cache_tensors(out.past_key_values)]}


def managed_trace(prompt: list[int], generated: list[int], stamp: Stamp) -> list[ManagedToken]:
    tokens = [ManagedToken(t) for t in prompt]
    for i, t in enumerate(generated):
        tokens.append(append_managed(tokens, t, (stamp,) if i == 0 else ()))
    return tokens


@torch.no_grad()
def run_case(seed: int, transition: str, count: int = 6) -> dict:
    if transition not in ('revoke', 'update') or not 2 <= count <= 32:
        raise ValueError('Use revoke/update and 2..32 managed tokens')
    m = make_model(seed)
    prompt = [1, 2, 3, 4, 6, 7, 8]
    obj = seed % 20
    old_bank = one_pod(obj)
    current_bank = one_pod(obj, False) if transition == 'revoke' else one_pod((obj+1) % 20)
    current_live = {'q': 1} if transition == 'revoke' else {'p': 2, 'q': 1}
    old = continuation(m, prompt, old_bank, count)
    fresh = continuation(m, prompt, current_bank, count)
    expected_current_token = 5 if transition == 'revoke' else 10 + (obj+1) % 20
    pure = forward_cached(m, torch.tensor([prompt]))
    pure_cache = [t.clone() for t in cache_tensors(pure.past_key_values)]

    # Deleting K/V alone and rebuilding from the old generated text still retains it.
    stale_text_rebuild = forward_cached(m, torch.tensor([prompt + old['tokens']]))
    stale_text_logits = stale_text_rebuild.logits[0, -1]
    stale_text_cache = [t.clone() for t in cache_tensors(stale_text_rebuild.past_key_values)]

    tokens = managed_trace(prompt, old['tokens'], Stamp('p', 1))
    cut = first_invalid(tokens, current_live)
    kept = [t.token_id for t in tokens[:cut]]
    # Independent recomputation; the repair does not copy the reference's outputs.
    # Only the exogenous prompt remains. Full prompt prefill is the conservative baseline.
    repaired = continuation(m, kept, current_bank, count)

    bystander_bank = one_pod((obj+7) % 20)
    bystander_before = continuation(m, prompt, bystander_bank, count)
    bystander_tokens = managed_trace(prompt, bystander_before['tokens'], Stamp('q', 1))
    bystander_after = continuation(m, prompt, bystander_bank, count)

    prefill_delta = max_difference(old['prefill'], pure_cache)
    after_token_delta = max_difference(old['cache'], fresh['cache'])
    final_logit_delta = float((old['logits']-fresh['logits']).abs().max())
    retained_text_delta = float((stale_text_logits-fresh['logits']).abs().max())
    replay_old_delta = float((stale_text_logits-old['logits']).abs().max())
    checks = {
        'controlled_first_read_selects_old_payload': old['tokens'][0] == 10+obj,
        'controlled_current_read_selects_updated_or_unknown': fresh['tokens'][0] == expected_current_token,
        'prefill_kv_exactly_pure': prefill_delta == 0.0,
        'current_prefill_kv_exactly_pure': max_difference(fresh['prefill'], pure_cache) == 0.0,
        'generated_tokens_differ': old['tokens'] != fresh['tokens'],
        'token_feedback_changes_persisted_kv': after_token_delta > 1e-6,
        'continuation_without_new_pod_read_still_differs': final_logit_delta > 1e-6,
        'rebuild_from_stale_text_still_differs': retained_text_delta > 1e-6,
        'stale_text_rebuild_matches_old_path_numerically': torch.allclose(stale_text_logits, old['logits'], atol=5e-6, rtol=1e-5),
        'direct_read_only_empty_receipt_would_accept': dependencies_valid((), current_live),
        'transitive_receipt_rejects_last_descendant': not dependencies_valid(tokens[-1].dependencies, current_live),
        'cut_is_before_first_generated_token': cut == len(prompt),
        'regenerate_from_clean_prefix_matches_fresh_tokens': repaired['tokens'] == fresh['tokens'],
        'regenerate_from_clean_prefix_matches_fresh_logits_exactly': torch.equal(repaired['logits'], fresh['logits']),
        'regenerate_from_clean_prefix_matches_fresh_kv_exactly': max_difference(repaired['cache'], fresh['cache']) == 0.0,
        'aba_does_not_revalidate_old_descendant': not dependencies_valid(tokens[-1].dependencies, {'p': 3, 'q': 1}),
        'independent_bystander_receipt_stays_valid': dependencies_valid(bystander_tokens[-1].dependencies, current_live),
        'independent_bystander_tokens_unchanged': bystander_before['tokens'] == bystander_after['tokens'],
        'independent_bystander_logits_exactly_unchanged': torch.equal(bystander_before['logits'], bystander_after['logits']),
        'independent_bystander_kv_exactly_unchanged': max_difference(bystander_before['cache'], bystander_after['cache']) == 0.0,
        'adapter_context_cleared': m._ctx is None,
    }
    return {'seed': seed, 'transition': transition, 'initialization_only_not_training_seed': True,
            'old_tokens': old['tokens'], 'fresh_tokens': fresh['tokens'], 'cut_position': cut,
            'managed_tokens': count, 'old_payload_token': 10+obj,
            'downstream_old_payload_repetitions': sum(t == 10+obj for t in old['tokens'][1:]),
            'prefill_kv_maxabs': prefill_delta,
            'after_feedback_kv_maxabs_to_fresh': after_token_delta,
            'after_feedback_logits_maxabs_to_fresh': final_logit_delta,
            'stale_text_rebuild_logits_maxabs_to_fresh': retained_text_delta,
            'stale_text_rebuild_logits_maxabs_to_old_path': replay_old_delta,
            'stale_text_rebuild_kv_maxabs_to_old_path': max_difference(stale_text_cache, old['cache']),
            'repair_logits_maxabs_to_fresh': float((repaired['logits']-fresh['logits']).abs().max()),
            'checks': {k: bool(v) for k, v in checks.items()}, 'pass': all(checks.values())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', nargs='+', type=int, default=[101, 102, 103, 104, 105])
    p.add_argument('--tokens', type=int, default=6)
    p.add_argument('--results-dir', type=Path, default=Path('so/results'))
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds):
        p.error('Seeds must be distinct')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import transformers
    from so import llm_adapter
    rows = [run_case(s, tr, a.tokens) for s in a.seeds for tr in ('revoke', 'update')]
    result = {
        'experiment': 'GEN-001', 'kind': 'structural_counterexample_and_conservative_reference',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'adapter_sha256': hashlib.sha256(Path(llm_adapter.__file__).read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'transformers': transformers.__version__, 'device': 'cpu', 'threads': 1},
        'rows': rows, 'all_pass': all(r['pass'] for r in rows),
        'breakthrough': False, 'novelty_claim': False,
        'reference': 'Transition applied before the first internally buffered generated token; same prompt, model and greedy schedule',
        'scope': 'One controlled pod read, actual repository late-write adapter, tiny frozen random GPT-2; no training or NLP capability evidence',
        'limitations': ['Managed buffer only; already externalized output cannot be retracted',
                        'Conservative lineage is by construction, not learned or minimal',
                        'No cryptographic binding or concurrent publish/race claim',
                        'No J-space certificate, same-session mixed-pod locality, latency or multi-backbone claim',
                        'Repair is ordinary prefix cut and deterministic regeneration, not a new algorithm'],
    }
    a.results_dir.mkdir(parents=True, exist_ok=True)
    (a.results_dir/'gen001_token_feedback_boundary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['all_pass']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
