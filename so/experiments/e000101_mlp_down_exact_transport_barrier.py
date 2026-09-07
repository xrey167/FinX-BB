"""E-000101 -- first-MLP exact-transport barrier.

Preregistered in docs/novelty/e000101-mlp-down-exact-transport-barrier-preregister.md.
The screen gives ordinary-coordinate delta propagation unusually favorable oracle
advantages, then measures whether the first downstream nonlinear MLP-down input
still exposes sparse or fleet-low-rank exact structure.
"""
from __future__ import annotations

import argparse
import json
import math
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


def _mlp_down(block: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    if hasattr(block, "mlp") and hasattr(block.mlp, "c_proj") and hasattr(block.mlp, "c_fc"):
        return "gpt2_c_proj", block.mlp.c_proj
    if hasattr(block, "mlp") and hasattr(block.mlp, "dense_4h_to_h") and hasattr(block.mlp, "dense_h_to_4h"):
        return "neox_dense_4h_to_h", block.mlp.dense_4h_to_h
    raise TypeError(f"Unsupported MLP family: {type(block)!r}")


def _forward_capture(
    model,
    blocks,
    read_layer: int,
    ids: torch.Tensor,
    payload: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    first_suffix = read_layer + 1
    _, down = _mlp_down(blocks[first_suffix])
    cap: Dict[str, torch.Tensor] = {}
    handles = []

    def inject(module, inputs, output):
        h = _hidden(output)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + payload.to(h.device, h.dtype)
        return _replace_hidden(output, h2)

    def down_pre(module, inputs):
        if not inputs:
            raise RuntimeError("MLP-down received no inputs")
        t = inputs[0]
        if not torch.is_tensor(t):
            raise TypeError(type(t))
        cap["mlp_down_input"] = t[:, -1, :].detach().float().cpu()

    def down_post(module, inputs, output):
        t = _tensor_output(output)
        cap["mlp_down_output"] = t[:, -1, :].detach().float().cpu()

    def block_post(module, inputs, output):
        cap["first_suffix_residual"] = _hidden(output).detach().float().cpu()

    handles.append(blocks[read_layer].register_forward_hook(inject))
    handles.append(down.register_forward_pre_hook(down_pre))
    handles.append(down.register_forward_hook(down_post))
    handles.append(blocks[first_suffix].register_forward_hook(block_post))

    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    cap["logits"] = out.logits[:, -1, :].detach().float().cpu()
    return cap


def _rank_screen(delta: torch.Tensor, entry_tol: float) -> Dict[str, object]:
    # SVD is deliberately evaluated in float64.  The Frobenius bound is a
    # necessary condition for every-entry <= entry_tol, not a sufficient one.
    d = delta.detach().double().cpu()
    s = torch.linalg.svdvals(d)
    sq = s.square()
    # residual_after[r] = optimal Frobenius residual after rank-r approximation.
    residual_after = torch.zeros(len(s) + 1, dtype=torch.float64)
    if len(s):
        tail = torch.flip(torch.cumsum(torch.flip(sq, dims=[0]), dim=0), dims=[0])
        residual_after[:-1] = tail.sqrt()
    bound = float(entry_tol) * math.sqrt(float(d.numel()))
    min_rank = len(s)
    for r in range(len(s) + 1):
        if float(residual_after[r]) <= bound:
            min_rank = r
            break

    ranks = sorted(set([0, 1, 2, 4, 8, 16, 32, 64, 96, 115, 128, len(s)]))
    samples = {}
    for r in ranks:
        if 0 <= r <= len(s):
            samples[str(r)] = float(residual_after[r])

    return {
        "matrix_shape": [int(d.shape[0]), int(d.shape[1])],
        "entry_tolerance": float(entry_tol),
        "necessary_frobenius_bound": bound,
        "min_rank_not_excluded_by_bound": int(min_rank),
        "rank_fraction_of_contexts": float(min_rank / max(1, d.shape[0])),
        "singular_value_max": float(s[0]) if len(s) else 0.0,
        "singular_value_min": float(s[-1]) if len(s) else 0.0,
        "residual_samples": samples,
    }


def _fraction_stats(mask: torch.Tensor) -> Dict[str, float]:
    frac = mask.float().mean(-1)
    return {
        "mean": float(frac.mean()),
        "median": float(frac.median()),
        "min": float(frac.min()),
        "max": float(frac.max()),
        "p05": float(torch.quantile(frac, 0.05)),
    }


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
    first_suffix = read_layer + 1
    suffix_blocks = len(blocks) - read_layer - 1
    hidden_width = int(model.get_input_embeddings().weight.shape[1])
    vocab = int(model.get_input_embeddings().weight.shape[0])
    down_name, down = _mlp_down(blocks[first_suffix])

    # Fresh formulas, disjoint from E95-E100 intervention seeds and token formulas.
    old_p = _payload(model, (3137 + 37 * seed) % vocab, payload_rms)
    new_p = _payload(model, (4219 + 41 * seed) % vocab, payload_rms)
    g = torch.Generator().manual_seed(2000000 + 131 * seed)
    ids = torch.randint(0, vocab, (n_contexts, seq_len), generator=g, dtype=torch.long)

    old = _forward_capture(model, blocks, read_layer, ids, old_p)
    old2 = _forward_capture(model, blocks, read_layer, ids, old_p)
    new = _forward_capture(model, blocks, read_layer, ids, new_p)

    deterministic = all(torch.equal(old[k], old2[k]) for k in old)
    prefix_exact = torch.equal(
        old["first_suffix_residual"][:, :-1, :],
        new["first_suffix_residual"][:, :-1, :],
    )

    din = new["mlp_down_input"] - old["mlp_down_input"]
    dout = new["mlp_down_output"] - old["mlp_down_output"]
    in_width = int(din.shape[-1])
    out_width = int(dout.shape[-1])

    in_exact_mask = din != 0
    in_material_mask = din.abs() > 1e-6
    out_exact_mask = dout != 0
    out_material_mask = dout.abs() > 1e-6

    in_exact_frac = in_exact_mask.float().mean(-1)
    out_exact_frac = out_exact_mask.float().mean(-1)
    in_material_frac = in_material_mask.float().mean(-1)
    out_material_frac = out_material_mask.float().mean(-1)

    # Unrealistically favorable oracle: if either an input column or output row is
    # unchanged, its corresponding coefficient rectangle may be skipped for free.
    exact_skip = 1.0 - in_exact_frac * out_exact_frac
    material_skip = 1.0 - in_material_frac * out_material_frac

    logits_delta = (new["logits"] - old["logits"]).abs().max(-1).values
    material_logit_rate = float((logits_delta > 1e-4).float().mean())

    rank = _rank_screen(din, entry_tol=1e-6)

    controls = {
        "V1_material_logit_rate_ge_095": material_logit_rate >= 0.95,
        "V2_suffix_blocks_ge_2": suffix_blocks >= 2,
        "V3_repeat_old_exact": bool(deterministic),
        "V4_causal_prefix_exact": bool(prefix_exact),
        "V5_expanded_mlp_input": in_width > hidden_width,
    }
    valid = all(controls.values())

    kill_cell = (
        valid
        and float(in_exact_frac.median()) >= 0.99
        and float(out_exact_frac.median()) >= 0.99
        and float(exact_skip.max()) <= 0.05
        and int(rank["min_rank_not_excluded_by_bound"]) >= math.ceil(0.90 * n_contexts)
    )

    w = getattr(down, "weight", None)
    weight_elements = int(w.numel()) if w is not None else None

    return {
        "model": model_name,
        "seed": seed,
        "n_contexts": n_contexts,
        "seq_len": seq_len,
        "read_layer": read_layer,
        "first_suffix_block": first_suffix,
        "suffix_blocks": suffix_blocks,
        "payload_rms": payload_rms,
        "mlp_down_family": down_name,
        "hidden_width": hidden_width,
        "mlp_down_input_width": in_width,
        "mlp_down_output_width": out_width,
        "mlp_down_weight_elements": weight_elements,
        "material_final_logit_edit_rate": material_logit_rate,
        "input_exact_changed_fraction": _fraction_stats(in_exact_mask),
        "input_material_gt_1e6_changed_fraction": _fraction_stats(in_material_mask),
        "output_exact_changed_fraction": _fraction_stats(out_exact_mask),
        "output_material_gt_1e6_changed_fraction": _fraction_stats(out_material_mask),
        "oracle_cartesian_exact_skip_fraction": {
            "mean": float(exact_skip.mean()),
            "max": float(exact_skip.max()),
            "p95": float(torch.quantile(exact_skip, 0.95)),
        },
        "oracle_cartesian_material_gt_1e6_skip_fraction": {
            "mean": float(material_skip.mean()),
            "max": float(material_skip.max()),
            "p95": float(torch.quantile(material_skip, 0.95)),
        },
        "fleet_activation_delta_rank_screen": rank,
        "controls": controls,
        "decision": "FIRST_MLP_ORDINARY_DELTA_BARRIER" if kill_cell else "DO_NOT_KILL_OPERATOR_LOCAL_ROUTE",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[20, 21, 22])
    ap.add_argument("--n-contexts", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--payload-rms", type=float, default=2.0)
    ap.add_argument("--results-dir", default="so/results")
    a = ap.parse_args()

    rows = [run(a.model, s, a.n_contexts, a.seq_len, a.payload_rms) for s in a.seeds]
    record = {
        "experiment": "E-000101",
        "title": "First-MLP exact-transport barrier",
        "oracle_advantage": "All update algebra before the first MLP-down is free; unchanged MLP input columns/output rows are supplied for free.",
        "fresh_intervention_seeds": list(a.seeds),
        "rows": rows,
        "all_cells_first_mlp_ordinary_delta_barrier": all(
            r["decision"] == "FIRST_MLP_ORDINARY_DELTA_BARRIER" for r in rows
        ),
        "not_claimed": "No information-theoretic MVM lower bound and no claim against new exact matrix representations, symbolic nonlinear quotients, or retrained lifecycle-native architectures.",
    }
    out = Path(a.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / ("e000101_" + a.model.replace("/", "_") + ".json")
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
