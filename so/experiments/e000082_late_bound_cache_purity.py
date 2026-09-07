"""E-000082 -- late-bound memory and autoregressive cache-purity boundary.

This is a falsification/architecture experiment, not a novelty claim.

E-000080 showed that object-scoped dependency/version tags can selectively
invalidate stale KV, but that is ordinary systems engineering. E-000082 asks a
more useful question: can a memory read be placed after the last cache-writing
transformer block so the persistent self-attention KV never contains the
memory contribution at all?

The experiment uses controlled residual payloads on public causal LMs. It
compares a penultimate-block memory injection (contaminating) with a final-block
late-bound injection (cache-pure). After an old->new memory lifecycle change it
requires:

* final-block memory changes logits but leaves every KV tensor exactly equal to
  the no-memory cache;
* an old final-block cache can be reused with the new payload and match a fresh
  current recomputation under a teacher-forced continuation;
* an old penultimate-block cache remains stale and materially differs from the
  current result unless the prompt is recomputed;
* reuse provides a measured lifecycle repair speedup over full prompt
  recomputation as prompt length grows;
* the experiment separately attacks the discrete-token boundary: if old and
  current memory produce different committed/generated tokens, replaying the
  old token after revocation is expected to remain different from the current
  counterfactual. That is an explicit rollback boundary, not something cache
  purity can erase.

Therefore a PASS is only evidence that architecture can avoid internal-KV
contamination for late-bound memory. It does not establish a real symlink
reader, deletion safety, or novelty, and it does not solve already committed
memory-dependent token history.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    if torch.is_tensor(output):
        return hidden
    raise TypeError(f"unsupported block output type: {type(output).__name__}")


def _payload(model, target_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings()
    if emb is None:
        emb = model.get_input_embeddings()
    v = emb.weight[target_id].detach().float().clone()
    return v * (float(rms) / float(v.pow(2).mean().sqrt().clamp_min(1e-8)))


def _inject_hook(payload: torch.Tensor):
    def hook(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(device=h.device, dtype=h.dtype)
        return _replace_hidden(output, h2)
    return hook


def _prefill(model, blocks, ids: torch.Tensor, layer: int | None,
             payload: torch.Tensor | None) -> Tuple[Any, torch.Tensor, float]:
    handle = None
    if layer is not None:
        assert payload is not None
        handle = blocks[layer].register_forward_hook(_inject_hook(payload))
    t0 = time.perf_counter_ns()
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
    finally:
        if handle is not None:
            handle.remove()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    if out.past_key_values is None:
        raise RuntimeError("backbone did not return past_key_values")
    return out.past_key_values, out.logits[:, -1, :].detach().float(), elapsed_ms


def _decode(model, blocks, past, prompt_len: int, token_id: int, layer: int | None,
            payload: torch.Tensor | None) -> Tuple[torch.Tensor, float]:
    handle = None
    if layer is not None:
        assert payload is not None
        handle = blocks[layer].register_forward_hook(_inject_hook(payload))
    tok = torch.tensor([[int(token_id)]], dtype=torch.long)
    mask = torch.ones((1, prompt_len + 1), dtype=torch.long)
    t0 = time.perf_counter_ns()
    try:
        with torch.no_grad():
            out = model(input_ids=tok, attention_mask=mask, past_key_values=past, use_cache=True)
    finally:
        if handle is not None:
            handle.remove()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    return out.logits[:, -1, :].detach().float(), elapsed_ms


def _cache_tensors(cache: Any) -> List[torch.Tensor]:
    """Return K/V tensors from both legacy tuple caches and HF DynamicCache."""
    out: List[torch.Tensor] = []
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if torch.is_tensor(t):
                    out.append(t.detach().float())
        if out:
            return out
    if hasattr(cache, "to_legacy_cache"):
        try:
            cache = cache.to_legacy_cache()
        except Exception:
            pass
    if isinstance(cache, (tuple, list)):
        for layer in cache:
            if isinstance(layer, (tuple, list)):
                out.extend(t.detach().float() for t in layer if torch.is_tensor(t))
            elif torch.is_tensor(layer):
                out.append(layer.detach().float())
    if not out:
        raise TypeError(f"cannot enumerate cache tensors from {type(cache).__name__}")
    return out


def _cache_maxabs(a: Any, b: Any) -> float:
    aa, bb = _cache_tensors(a), _cache_tensors(b)
    if len(aa) != len(bb):
        return float("inf")
    if any(x.shape != y.shape for x, y in zip(aa, bb)):
        return float("inf")
    return max((float((x - y).abs().max()) for x, y in zip(aa, bb)), default=0.0)


def _cache_bytes(cache: Any) -> int:
    return sum(t.numel() * t.element_size() for t in _cache_tensors(cache))


def _maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _kl(reference: torch.Tensor, observed: torch.Tensor) -> float:
    p = torch.softmax(reference.float(), -1)
    return float((p * (torch.log_softmax(reference.float(), -1) -
                       torch.log_softmax(observed.float(), -1))).sum(-1).mean())


def _make_prompt(vocab: int, length: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, length), generator=g, dtype=torch.long)


def run(model_name: str, seed: int, payload_rms: float,
        lengths: Iterable[int]) -> Dict[str, object]:
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = AutoModelForCausalLM.from_pretrained(model_name).float()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = transformer_blocks(model)
    if len(blocks) < 2:
        raise ValueError("need at least two transformer blocks")
    final_layer = len(blocks) - 1
    penultimate_layer = len(blocks) - 2
    vocab = int(model.get_input_embeddings().weight.shape[0])
    old_target = (42 + 17 * seed) % vocab
    new_target = (314 + 29 * seed) % vocab
    if new_target == old_target:
        new_target = (new_target + 1) % vocab
    old_payload = _payload(model, old_target, payload_rms)
    new_payload = _payload(model, new_target, payload_rms)
    rows: List[Dict[str, object]] = []

    for length in lengths:
        ids = _make_prompt(vocab, int(length), 91000 + seed * 1000 + int(length))
        fixed_cont = (521 + seed) % vocab

        base_cache, base_logits, base_ms = _prefill(model, blocks, ids, None, None)
        late_old_cache, late_old_logits, late_old_ms = _prefill(
            model, blocks, ids, final_layer, old_payload
        )
        late_new_cache, late_new_logits, late_new_ms = _prefill(
            model, blocks, ids, final_layer, new_payload
        )
        contam_old_cache, _, contam_old_ms = _prefill(
            model, blocks, ids, penultimate_layer, old_payload
        )
        contam_new_cache, _, contam_new_ms = _prefill(
            model, blocks, ids, penultimate_layer, new_payload
        )

        late_old_cache_vs_base = _cache_maxabs(late_old_cache, base_cache)
        late_new_cache_vs_base = _cache_maxabs(late_new_cache, base_cache)
        contam_old_cache_vs_base = _cache_maxabs(contam_old_cache, base_cache)
        contam_old_vs_new_cache = _cache_maxabs(contam_old_cache, contam_new_cache)

        late_reuse_logits, late_reuse_ms = _decode(
            model, blocks, late_old_cache, int(length), fixed_cont, final_layer, new_payload
        )
        late_fresh_logits, late_fresh_decode_ms = _decode(
            model, blocks, late_new_cache, int(length), fixed_cont, final_layer, new_payload
        )
        contam_stale_logits, contam_stale_decode_ms = _decode(
            model, blocks, contam_old_cache, int(length), fixed_cont, final_layer, new_payload
        )
        contam_current_logits, contam_current_decode_ms = _decode(
            model, blocks, contam_new_cache, int(length), fixed_cont, final_layer, new_payload
        )

        late_reuse_vs_fresh = _maxabs(late_reuse_logits, late_fresh_logits)
        contam_stale_vs_current = _maxabs(contam_stale_logits, contam_current_logits)

        old_token = int(late_old_logits.argmax(-1)[0])
        new_token = int(late_new_logits.argmax(-1)[0])
        stale_token_cache = _prefill(model, blocks, ids, final_layer, old_payload)[0]
        current_token_cache = _prefill(model, blocks, ids, final_layer, new_payload)[0]
        stale_token_logits, _ = _decode(
            model, blocks, stale_token_cache, int(length), old_token, final_layer, new_payload
        )
        current_token_logits, _ = _decode(
            model, blocks, current_token_cache, int(length), new_token, final_layer, new_payload
        )
        token_history_maxabs = _maxabs(stale_token_logits, current_token_logits)

        repair_full_ms = contam_new_ms + contam_current_decode_ms
        repair_late_ms = late_reuse_ms
        checks = {
            "late_bound_changes_current_logits": _maxabs(late_old_logits, base_logits) > 1e-4,
            "late_bound_old_new_effect_differs": _maxabs(late_old_logits, late_new_logits) > 1e-4,
            "late_bound_old_cache_exact_base": late_old_cache_vs_base == 0.0,
            "late_bound_new_cache_exact_base": late_new_cache_vs_base == 0.0,
            "penultimate_injection_contaminates_cache": contam_old_cache_vs_base > 1e-6,
            "penultimate_old_new_cache_differs": contam_old_vs_new_cache > 1e-6,
            "late_bound_old_cache_reuse_matches_fresh_current": late_reuse_vs_fresh < 5e-5,
            "contaminated_old_cache_is_stale_after_update": contam_stale_vs_current > 1e-4,
        }
        rows.append({
            "prompt_tokens": int(length),
            "cache_bytes": _cache_bytes(base_cache),
            "old_target_id": old_target,
            "new_target_id": new_target,
            "old_top1": old_token,
            "new_top1": new_token,
            "old_new_top1_changed": old_token != new_token,
            "late_old_cache_vs_base_maxabs": late_old_cache_vs_base,
            "late_new_cache_vs_base_maxabs": late_new_cache_vs_base,
            "contam_old_cache_vs_base_maxabs": contam_old_cache_vs_base,
            "contam_old_vs_new_cache_maxabs": contam_old_vs_new_cache,
            "late_old_vs_new_prompt_logits_maxabs": _maxabs(late_old_logits, late_new_logits),
            "late_reuse_vs_fresh_current_maxabs": late_reuse_vs_fresh,
            "contam_stale_vs_current_maxabs": contam_stale_vs_current,
            "contam_stale_vs_current_kl_nats": _kl(contam_current_logits, contam_stale_logits),
            "old_token_after_update_vs_current_token_path_maxabs": token_history_maxabs,
            "token_history_requires_rollback_when_token_changed": bool(
                old_token != new_token and token_history_maxabs > 1e-4
            ),
            "timing_ms": {
                "base_prefill": base_ms,
                "late_old_prefill": late_old_ms,
                "late_new_prefill": late_new_ms,
                "contam_old_prefill": contam_old_ms,
                "contam_current_prefill": contam_new_ms,
                "late_update_reuse_decode": repair_late_ms,
                "late_fresh_decode": late_fresh_decode_ms,
                "contam_stale_decode": contam_stale_decode_ms,
                "contam_current_decode": contam_current_decode_ms,
                "contaminated_full_repair_prefill_plus_decode": repair_full_ms,
            },
            "full_repair_over_late_reuse_ratio": repair_full_ms / max(repair_late_ms, 1e-9),
            "checks": checks,
            "pass": all(checks.values()),
        })

    return {
        "model": model_name,
        "seed": seed,
        "n_blocks": len(blocks),
        "final_layer": final_layer,
        "penultimate_layer": penultimate_layer,
        "payload_rms": payload_rms,
        "rows": rows,
        "all_pass": all(bool(r["pass"]) for r in rows),
        "token_boundary_observed": any(
            bool(r["token_history_requires_rollback_when_token_changed"]) for r in rows
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--payload-rms", type=float, default=8.0)
    ap.add_argument("--lengths", type=int, nargs="*", default=[16, 64, 256])
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()
    rows = [run(a.model, s, a.payload_rms, a.lengths) for s in a.seeds]
    rec = {
        "experiment": "E-000082",
        "all_pass": all(bool(r["all_pass"]) for r in rows),
        "model_rows": rows,
        "interpretation": (
            "A final-block late-bound memory contribution can alter neural output while remaining absent from "
            "persistent self-attention KV; therefore an old cache can survive a memory lifecycle update and be "
            "reused with current memory under teacher-forced continuation. Earlier-layer injection cannot. "
            "Committed memory-dependent token history remains a separate rollback/effect boundary."
        ),
        "not_claimed": (
            "No novelty claim for late fusion, external memory, sidecars, cache purity, selective recomputation, "
            "rollback, effect boundaries, versioning or cache reuse. Controlled payloads mean this is not a "
            ">=0.95 real-symlink capability result."
        ),
    }
    out_dir = Path(a.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("e000082_late_bound_cache_purity_" + a.model.replace("/", "_") + ".json")
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
