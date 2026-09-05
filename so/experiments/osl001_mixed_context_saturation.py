"""OSL001 — mixed-context saturation boundary for E80 whole-cache lineage.

This is a correctness/falsification screen, not a novelty claim.  It tests the
actual DerivedLineage semantics, causal prefix/suffix arithmetic, and a tiny
random GPT-2 control showing that a late neural memory write cannot affect
prefix KV created before the write even though one whole-cache LineagedState
would be rejected when any witness becomes stale.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from so.cavi import CAVIAuthority
from so.derived_lineage import DerivedLineage, LineagedState


def _implementation_cell(m: int) -> Dict[str, Any]:
    auth = CAVIAuthority()
    witnesses = []
    for i in range(m):
        pod = i + 1
        alias = 10_000 + i
        auth.create_pod(pod)
        auth.create_alias(alias, pod)
        witnesses.append(auth.witness(alias))
    lineage = DerivedLineage.of(*witnesses)
    state = LineagedState(payload=np.zeros(1, dtype=np.uint8), lineage=lineage)
    assert state.reusable(auth)
    target = m // 2 + 1
    auth.update_pod(target)
    stale = lineage.stale_witnesses(auth)
    current_others = sum(auth.validate_witness(w) for w in witnesses if w.pod_id != target)
    return {
        "dependencies": m,
        "target_pod": target,
        "whole_state_reusable_after_one_pod_update": state.reusable(auth),
        "stale_witnesses": len(stale),
        "other_witnesses_still_current": current_others,
        "expected_other_current": m - 1,
        "packed_lineage_bytes": lineage.packed_metadata_bytes,
    }


def _segmentation_cell(m: int, length: int = 4096) -> Dict[str, Any]:
    # The preregistration requires every cell to contain the adversarial final-token
    # read. np.linspace(start, stop, 1) returns start, so handle m=1 explicitly.
    # For m>1, unique positions are distributed from token 0 through the final token.
    positions = (
        np.asarray([length - 1], dtype=int)
        if m == 1
        else np.rint(np.linspace(0, length - 1, m)).astype(int)
    )
    suffix = np.asarray([length - int(pos) for pos in positions], dtype=np.int64)
    coarse = np.full(m, length, dtype=np.int64)
    ratio = coarse / suffix
    collateral = coarse - suffix
    return {
        "dependencies": m,
        "sequence_length": length,
        "read_positions": positions.tolist(),
        "whole_cache_recompute_tokens_per_single_update": coarse.tolist(),
        "ordinary_exact_suffix_recompute_tokens": suffix.tolist(),
        "whole_over_suffix_ratio": ratio.tolist(),
        "mean_whole_over_suffix_ratio": float(np.mean(ratio)),
        "max_whole_over_suffix_ratio": float(np.max(ratio)),
        "mean_unnecessarily_discarded_prefix_tokens": float(np.mean(collateral)),
        "late_read_ratio": float(ratio[-1]),
        "late_read_exact_reusable_prefix_fraction": float(positions[-1] / length),
        "whole_cache_invalidated_fraction": 1.0,
        "mean_exact_suffix_recompute_fraction": float(np.mean(suffix / length)),
        "global_epoch_equivalent_within_this_cache": True,
    }


def _saturation_cell(m: int) -> Dict[str, Any]:
    rates = {}
    for q in (0.001, 0.01, 0.05):
        # Independent per-object chance of changing before next reuse.
        rates[str(q)] = {
            "per_object_change_probability": q,
            "whole_cache_stale_probability": float(1.0 - (1.0 - q) ** m),
        }
    return {"dependencies": m, "rates": rates}


def _legacy_cache(cache: Any) -> Tuple[Tuple[Any, Any], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple((pair[0], pair[1]) for pair in cache)


def _tiny_gpt2_control(seed: int, seq_len: int = 32, read_pos: int = 24) -> Dict[str, Any]:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=4,
        n_head=4,
        use_cache=True,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = (torch.arange(seq_len, dtype=torch.long)[None] * 7 + 3) % cfg.vocab_size
    direction = torch.randn(cfg.n_embd)
    direction = direction / direction.pow(2).mean().sqrt()

    def run(amplitude: float):
        handles = []

        def inject(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            h2 = h.clone()
            h2[:, read_pos, :] += float(amplitude) * direction.to(h2)
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

        handles.append(model.transformer.h[1].register_forward_hook(inject))
        try:
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        return _legacy_cache(out.past_key_values)

    old = run(2.0)
    fresh = run(-3.0)
    repeat = run(2.0)
    assert len(old) == len(fresh) == 4
    repeat_exact = True
    prefix_exact = True
    prefix_unequal_tensors = 0
    suffix_maxabs = 0.0
    suffix_unequal_tensors = 0
    per_layer = []
    for layer, ((ok, ov), (fk, fv), (rk, rv)) in enumerate(zip(old, fresh, repeat)):
        layer_repeat = torch.equal(ok, rk) and torch.equal(ov, rv)
        repeat_exact &= layer_repeat
        # HF causal-LM K/V shape is [batch, heads, sequence, head_dim].
        pk_equal = torch.equal(ok[:, :, :read_pos, :], fk[:, :, :read_pos, :])
        pv_equal = torch.equal(ov[:, :, :read_pos, :], fv[:, :, :read_pos, :])
        prefix_exact &= pk_equal and pv_equal
        if not pk_equal:
            prefix_unequal_tensors += 1
        if not pv_equal:
            prefix_unequal_tensors += 1
        kd = (ok[:, :, read_pos:, :] - fk[:, :, read_pos:, :]).abs()
        vd = (ov[:, :, read_pos:, :] - fv[:, :, read_pos:, :]).abs()
        kmax = float(kd.max()) if kd.numel() else 0.0
        vmax = float(vd.max()) if vd.numel() else 0.0
        suffix_maxabs = max(suffix_maxabs, kmax, vmax)
        suffix_unequal_tensors += int(kmax > 0) + int(vmax > 0)
        per_layer.append({
            "layer": layer,
            "repeat_exact": layer_repeat,
            "prefix_key_exact": pk_equal,
            "prefix_value_exact": pv_equal,
            "suffix_key_maxabs": kmax,
            "suffix_value_maxabs": vmax,
        })
    assert repeat_exact
    assert prefix_exact
    assert suffix_maxabs > 0.0
    # The memory write is after block 1, so layers 0 and 1 K/V must be unchanged.
    assert per_layer[0]["suffix_key_maxabs"] == 0.0 and per_layer[1]["suffix_key_maxabs"] == 0.0
    assert any(row["suffix_key_maxabs"] > 0 or row["suffix_value_maxabs"] > 0 for row in per_layer[2:])
    return {
        "seed": seed,
        "architecture": "random_tiny_gpt2",
        "layers": 4,
        "sequence_length": seq_len,
        "memory_write_after_block": 1,
        "memory_read_token_position": read_pos,
        "repeat_forward_exact": repeat_exact,
        "all_prefix_kv_before_read_byte_identical": prefix_exact,
        "prefix_unequal_tensors": prefix_unequal_tensors,
        "downstream_suffix_maxabs": suffix_maxabs,
        "downstream_suffix_unequal_tensors": suffix_unequal_tensors,
        "exact_reusable_prefix_tokens": read_pos,
        "whole_cache_tokens_rejected_by_monolithic_lineage": seq_len,
        "unnecessary_prefix_rejection_tokens": read_pos,
        "per_layer": per_layer,
        "not_pretrained_or_reader_capability": True,
    }


def run() -> Dict[str, Any]:
    ms = (1, 2, 4, 8, 16, 32)
    implementation = [_implementation_cell(m) for m in ms]
    segmentation = [_segmentation_cell(m) for m in ms]
    saturation = [_saturation_cell(m) for m in ms]
    transformer = [_tiny_gpt2_control(seed) for seed in (0, 1, 2)]
    for row in implementation:
        assert not row["whole_state_reusable_after_one_pod_update"]
        assert row["stale_witnesses"] == 1
        assert row["other_witnesses_still_current"] == row["expected_other_current"]
    for row in segmentation:
        assert row["late_read_ratio"] == row["sequence_length"]
        assert row["mean_exact_suffix_recompute_fraction"] <= 1.0
    assert all(row["all_prefix_kv_before_read_byte_identical"] for row in transformer)
    return {
        "experiment": "OSL-001",
        "title": "mixed-context saturation boundary for object-scoped neural lineage",
        "implementation": implementation,
        "segmentation": segmentation,
        "saturation": saturation,
        "transformer_controls": transformer,
        "decision": (
            "E80 whole-cache LineagedState is sound but not all-and-only at sub-cache granularity: "
            "a single stale witness rejects causally independent prefix KV. Exact token/layer segmentation "
            "or another finer representation is required; ordinary dependency/prefix invalidation is the mandatory baseline."
        ),
        "major_invention": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="so/results/osl001_mixed_context_saturation.json")
    args = ap.parse_args()
    rec = run()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": rec["experiment"],
        "decision": rec["decision"],
        "late_read_whole_over_exact_suffix": rec["segmentation"][-1]["late_read_ratio"],
        "transformer_prefix_exact": all(r["all_prefix_kv_before_read_byte_identical"] for r in rec["transformer_controls"]),
    }, indent=2))


if __name__ == "__main__":
    main()
