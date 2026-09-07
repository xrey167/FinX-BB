"""E-000099 -- dense nonlinear influence / sparse exact-repair kill screen.

One controlled old->new Pod payload edit is injected three blocks from the end of a
frozen causal LM.  We inspect ordinary downstream residual/QKV/MLP coordinates to
ask whether an exact selective-coordinate repair could leave most suffix arithmetic
untouched.  This is a scoped systems falsifier, not an impossibility theorem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from so.llm_adapter import transformer_blocks


THRESHOLDS: Tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4)


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, h: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def _payload(model, token_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings() or model.get_input_embeddings()
    v = emb.weight[int(token_id)].detach().float().clone()
    denom = float(v.pow(2).mean().sqrt().clamp_min(1e-8))
    return v * (float(rms) / denom)


def _projection_modules(block: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # GPT-2 / DistilGPT-2
    if hasattr(block, "attn") and hasattr(block.attn, "c_attn") and hasattr(block, "mlp") and hasattr(block.mlp, "c_fc"):
        return block.attn.c_attn, block.mlp.c_fc
    # GPT-NeoX / Pythia
    if (
        hasattr(block, "attention")
        and hasattr(block.attention, "query_key_value")
        and hasattr(block, "mlp")
        and hasattr(block.mlp, "dense_h_to_4h")
    ):
        return block.attention.query_key_value, block.mlp.dense_h_to_4h
    raise TypeError(f"Unsupported transformer block type for registered E-000099 capture: {type(block)!r}")


def _tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Expected tensor-like module output, got {type(output)!r}")


def _forward_capture(
    model: torch.nn.Module,
    blocks: List[torch.nn.Module],
    read_layer: int,
    ids: torch.Tensor,
    payload: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    captures: Dict[str, torch.Tensor] = {}
    handles = []

    def inject(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(h.device, h.dtype)
        captures["post_read"] = h2.detach().float().cpu()
        return _replace_hidden(output, h2)

    handles.append(blocks[read_layer].register_forward_hook(inject))

    for i in range(read_layer + 1, len(blocks)):
        qkv, mlp = _projection_modules(blocks[i])

        def make_tensor_hook(name: str):
            def hook(module, inputs, output):
                t = _tensor_output(output)
                captures[name] = t[:, -1, :].detach().float().cpu()
            return hook

        def make_block_hook(name: str):
            def hook(module, inputs, output):
                h = _hidden(output)
                captures[name] = h.detach().float().cpu()
            return hook

        handles.append(qkv.register_forward_hook(make_tensor_hook(f"block{i}_qkv")))
        handles.append(mlp.register_forward_hook(make_tensor_hook(f"block{i}_mlp")))
        handles.append(blocks[i].register_forward_hook(make_block_hook(f"block{i}_residual")))

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    captures["logits"] = out.logits[:, -1, :].detach().float().cpu()
    return captures


def _quantile05(x: torch.Tensor) -> float:
    x = x.float().flatten()
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, 0.05))


def _change_stats(old: torch.Tensor, new: torch.Tensor, target_only: bool = True) -> Dict[str, object]:
    if target_only and old.ndim == 3:
        old = old[:, -1, :]
        new = new[:, -1, :]
    delta = (new.float() - old.float()).abs()
    rows: Dict[str, Dict[str, float]] = {}
    for eps in THRESHOLDS:
        changed = (delta != 0) if eps == 0.0 else (delta > eps)
        per_context = changed.float().reshape(changed.shape[0], -1).mean(-1)
        rows[f"{eps:.0e}" if eps else "exact_nonzero"] = {
            "mean_fraction": float(per_context.mean()),
            "min_fraction": float(per_context.min()),
            "p05_fraction": _quantile05(per_context),
        }
    return {
        "shape": list(old.shape),
        "maxabs_delta": float(delta.max()),
        "meanabs_delta": float(delta.mean()),
        "changed_fraction": rows,
    }


def _exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a, b))


def run(model_name: str, seed: int, n_contexts: int, seq_len: int, payload_rms: float) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    blocks = list(transformer_blocks(model))
    read_layer = max(0, len(blocks) - 3)
    suffix_blocks = len(blocks) - read_layer - 1
    vocab = int(model.get_input_embeddings().weight.shape[0])
    old_p = _payload(model, (41 + 7 * seed) % vocab, payload_rms)
    new_p = _payload(model, (313 + 11 * seed) % vocab, payload_rms)

    g = torch.Generator().manual_seed(99000 + seed)
    ids = torch.randint(0, vocab, (n_contexts, seq_len), generator=g, dtype=torch.long)

    old = _forward_capture(model, blocks, read_layer, ids, old_p)
    old_repeat = _forward_capture(model, blocks, read_layer, ids, old_p)
    new = _forward_capture(model, blocks, read_layer, ids, new_p)

    component_keys = sorted(
        k for k in old
        if k.startswith("block") and (k.endswith("_qkv") or k.endswith("_mlp") or k.endswith("_residual"))
    )
    stats: Dict[str, object] = {}
    dense_flags: Dict[str, bool] = {}
    deterministic_flags: Dict[str, bool] = {}

    for key in component_keys:
        stats[key] = _change_stats(old[key], new[key], target_only=True)
        c = stats[key]["changed_fraction"]
        exact_p05 = float(c["exact_nonzero"]["p05_fraction"])
        material_p05 = float(c["1e-06"]["p05_fraction"])
        dense_flags[key] = exact_p05 >= 0.99 and material_p05 >= 0.95
        deterministic_flags[key] = _exact_equal(old[key], old_repeat[key])

    logits_delta = (new["logits"] - old["logits"]).abs().max(-1).values
    material_rate = float((logits_delta > 1e-4).float().mean())
    deterministic_logits = _exact_equal(old["logits"], old_repeat["logits"])

    first_suffix = read_layer + 1
    first_res_key = f"block{first_suffix}_residual"
    prefix_old = old[first_res_key][:, :-1, :]
    prefix_new = new[first_res_key][:, :-1, :]
    causal_prefix_exact = _exact_equal(prefix_old, prefix_new)

    controls = {
        "V1_material_edit_rate_ge_095": material_rate >= 0.95,
        "V2_suffix_blocks_ge_2": suffix_blocks >= 2,
        "V3_all_registered_old_repeat_exact": bool(all(deterministic_flags.values()) and deterministic_logits),
        "V4_first_suffix_prefix_positions_exactly_unchanged": causal_prefix_exact,
    }
    valid = all(bool(v) for v in controls.values())
    dense_all = bool(component_keys) and all(dense_flags.values())
    decision = "DENSE_MATERIAL_INFLUENCE" if valid and dense_all else "DO_NOT_KILL_SPARSE_COORDINATE_REPAIR"

    return {
        "model": model_name,
        "seed": seed,
        "n_contexts": n_contexts,
        "seq_len": seq_len,
        "read_layer": read_layer,
        "n_blocks": len(blocks),
        "suffix_blocks": suffix_blocks,
        "payload_rms": payload_rms,
        "material_final_logit_edit_rate": material_rate,
        "component_stats": stats,
        "dense_component_flags": dense_flags,
        "deterministic_component_flags": deterministic_flags,
        "deterministic_logits": deterministic_logits,
        "controls": controls,
        "decision": decision,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-contexts", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--payload-rms", type=float, default=2.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [run(a.model, s, a.n_contexts, a.seq_len, a.payload_rms) for s in a.seeds]
    record = {
        "experiment": "E-000099",
        "title": "Dense nonlinear influence / sparse exact-repair kill screen",
        "registered_thresholds": {
            "exact_nonzero_p05_ge": 0.99,
            "abs_delta_gt_1e6_p05_ge": 0.95,
            "material_final_logit_edit_rate_ge": 0.95,
        },
        "rows": rows,
        "all_cells_dense_material_influence": all(r["decision"] == "DENSE_MATERIAL_INFLUENCE" for r in rows),
        "not_claimed": "No universal impossibility theorem; no claim against compact nonlinear coordinate transforms, architectural sparsification, or exact causal quotients outside ordinary coordinates.",
    }

    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("e000099_" + a.model.replace("/", "_") + ".json")
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
