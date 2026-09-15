"""E-000100 -- oracle sparse-coordinate FLOP ceiling.

Fresh intervention seeds give an ordinary output-row selective repair engine perfect,
free knowledge of exactly which dense-projection outputs changed.  We compute the
maximum projection arithmetic it could possibly skip while retaining exact state.
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


def _hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, h: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


def _tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(type(output))


def _payload(model, token_id: int, rms: float) -> torch.Tensor:
    emb = model.get_output_embeddings() or model.get_input_embeddings()
    v = emb.weight[int(token_id)].detach().float().clone()
    denom = float(v.pow(2).mean().sqrt().clamp_min(1e-8))
    return v * (float(rms) / denom)


def _projection_modules(block: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    if hasattr(block, "attn") and hasattr(block.attn, "c_attn") and hasattr(block.attn, "c_proj"):
        if not (hasattr(block, "mlp") and hasattr(block.mlp, "c_fc") and hasattr(block.mlp, "c_proj")):
            raise TypeError("Incomplete GPT2 projection family")
        return [
            ("attn_qkv", block.attn.c_attn),
            ("attn_out", block.attn.c_proj),
            ("mlp_up", block.mlp.c_fc),
            ("mlp_down", block.mlp.c_proj),
        ]
    if hasattr(block, "attention") and hasattr(block.attention, "query_key_value") and hasattr(block.attention, "dense"):
        if not (hasattr(block, "mlp") and hasattr(block.mlp, "dense_h_to_4h") and hasattr(block.mlp, "dense_4h_to_h")):
            raise TypeError("Incomplete GPTNeoX projection family")
        return [
            ("attn_qkv", block.attention.query_key_value),
            ("attn_out", block.attention.dense),
            ("mlp_up", block.mlp.dense_h_to_4h),
            ("mlp_down", block.mlp.dense_4h_to_h),
        ]
    raise TypeError(f"Unsupported block: {type(block)!r}")


def _forward_capture(model, blocks, read_layer: int, ids: torch.Tensor, payload: torch.Tensor) -> Dict[str, torch.Tensor]:
    cap: Dict[str, torch.Tensor] = {}
    handles = []

    def inject(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(h.device, h.dtype)
        return _replace_hidden(output, h2)

    handles.append(blocks[read_layer].register_forward_hook(inject))

    for i in range(read_layer + 1, len(blocks)):
        for short, mod in _projection_modules(blocks[i]):
            key = f"block{i}_{short}"

            def make_proj_hook(name: str):
                def hook(module, inputs, output):
                    t = _tensor_output(output)
                    cap[name] = t[:, -1, :].detach().float().cpu()
                return hook

            handles.append(mod.register_forward_hook(make_proj_hook(key)))

        def make_block_hook(name: str):
            def hook(module, inputs, output):
                cap[name] = _hidden(output).detach().float().cpu()
            return hook

        handles.append(blocks[i].register_forward_hook(make_block_hook(f"block{i}_residual")))

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    cap["logits"] = out.logits[:, -1, :].detach().float().cpu()
    return cap


def _row_cost(module: torch.nn.Module, output_width: int) -> float:
    w = getattr(module, "weight", None)
    if w is None:
        raise TypeError(f"Projection without weight: {type(module)!r}")
    return float(w.numel()) / float(output_width)


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

    # Fresh directions, deliberately distinct from E95-E99 token formulas.
    old_p = _payload(model, (907 + 29 * seed) % vocab, payload_rms)
    new_p = _payload(model, (2029 + 31 * seed) % vocab, payload_rms)
    g = torch.Generator().manual_seed(1000000 + 101 * seed)
    ids = torch.randint(0, vocab, (n_contexts, seq_len), generator=g, dtype=torch.long)

    old = _forward_capture(model, blocks, read_layer, ids, old_p)
    old2 = _forward_capture(model, blocks, read_layer, ids, old_p)
    new = _forward_capture(model, blocks, read_layer, ids, new_p)

    projection_modules: Dict[str, torch.nn.Module] = {}
    for i in range(read_layer + 1, len(blocks)):
        for short, mod in _projection_modules(blocks[i]):
            projection_modules[f"block{i}_{short}"] = mod

    exact_changed_weight = torch.zeros(n_contexts, dtype=torch.float64)
    material_changed_weight = torch.zeros(n_contexts, dtype=torch.float64)
    total_weight = 0.0
    projection_rows: Dict[str, object] = {}
    deterministic = True

    for key, mod in projection_modules.items():
        a = old[key]
        b = new[key]
        deterministic = deterministic and torch.equal(a, old2[key])
        delta = (b - a).abs()
        out_width = int(delta.shape[-1])
        rcost = _row_cost(mod, out_width)
        total_weight += rcost * out_width
        exact_counts = (delta != 0).sum(-1).double()
        material_counts = (delta > 1e-6).sum(-1).double()
        exact_changed_weight += exact_counts * rcost
        material_changed_weight += material_counts * rcost
        exact_fraction = exact_counts / float(out_width)
        material_fraction = material_counts / float(out_width)
        projection_rows[key] = {
            "output_width": out_width,
            "weight_elements": int(getattr(mod, "weight").numel()),
            "row_cost_coefficients": rcost,
            "exact_changed_fraction_mean": float(exact_fraction.mean()),
            "exact_changed_fraction_min": float(exact_fraction.min()),
            "material_gt_1e6_fraction_mean": float(material_fraction.mean()),
            "material_gt_1e6_fraction_min": float(material_fraction.min()),
        }

    exact_skip = 1.0 - exact_changed_weight / total_weight
    material_skip = 1.0 - material_changed_weight / total_weight

    logits_delta = (new["logits"] - old["logits"]).abs().max(-1).values
    material_logit_rate = float((logits_delta > 1e-4).float().mean())
    deterministic = deterministic and torch.equal(old["logits"], old2["logits"])

    first_suffix = read_layer + 1
    first_res = f"block{first_suffix}_residual"
    prefix_exact = torch.equal(old[first_res][:, :-1, :], new[first_res][:, :-1, :])

    controls = {
        "V1_material_logit_rate_ge_095": material_logit_rate >= 0.95,
        "V2_suffix_blocks_ge_2": suffix_blocks >= 2,
        "V3_repeat_old_exact": bool(deterministic),
        "V4_causal_prefix_exact": bool(prefix_exact),
    }
    valid = all(controls.values())
    no_headroom = float(exact_skip.max()) <= 0.05 and float(material_skip.max()) <= 0.05

    return {
        "model": model_name,
        "seed": seed,
        "n_contexts": n_contexts,
        "seq_len": seq_len,
        "read_layer": read_layer,
        "n_blocks": len(blocks),
        "suffix_blocks": suffix_blocks,
        "payload_rms": payload_rms,
        "material_final_logit_edit_rate": material_logit_rate,
        "registered_dense_projection_weight_elements": int(total_weight),
        "projection_stats": projection_rows,
        "oracle_exact_skip_fraction": {
            "mean": float(exact_skip.mean()),
            "max": float(exact_skip.max()),
            "p95": float(torch.quantile(exact_skip.float(), 0.95)),
        },
        "oracle_material_gt_1e6_skip_fraction": {
            "mean": float(material_skip.mean()),
            "max": float(material_skip.max()),
            "p95": float(torch.quantile(material_skip.float(), 0.95)),
        },
        "controls": controls,
        "decision": "NO_MATERIAL_SPARSE_FLOP_HEADROOM" if valid and no_headroom else "DO_NOT_KILL_SPARSE_FLOP_ROUTE",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[10, 11, 12])
    ap.add_argument("--n-contexts", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--payload-rms", type=float, default=2.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [run(a.model, s, a.n_contexts, a.seq_len, a.payload_rms) for s in a.seeds]
    record = {
        "experiment": "E-000100",
        "title": "Oracle sparse-coordinate FLOP ceiling",
        "oracle_advantage": "Changed projection output rows are supplied for free; detection and all non-projection repair costs are ignored.",
        "fresh_intervention_seeds": list(a.seeds),
        "rows": rows,
        "all_cells_no_material_sparse_flop_headroom": all(r["decision"] == "NO_MATERIAL_SPARSE_FLOP_HEADROOM" for r in rows),
        "not_claimed": "No lower bound against algebraic dense transforms, new coordinate systems, or retrained sparse architectures.",
    }
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("e000100_" + a.model.replace("/", "_") + ".json")
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
