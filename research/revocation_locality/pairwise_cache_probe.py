"""RL-MIX-002: sparse pairwise source-interaction KV cache probe.

Purpose
-------
RL-MIX-001 showed that a source-independent/additive (order-1) reconstruction
can miss real cross-source cache interactions, while a full Boolean-lattice
Möbius reconstruction is exact but exponentially expensive.  RL-MIX-002 tests
whether the *ordinary* order-2 truncation is a useful strong baseline:
materialize only empty, singleton and pair coalitions, then reconstruct larger
source sets from those terms.

This is NOT a novelty claim. Functional-ANOVA/Möbius interaction decompositions,
data provenance, incremental view maintenance and sparse cache factorization are
prior art.  The experiment is allowed to motivate a later architecture only if
it demonstrates a practical gap between source-isolated/order-1 reuse and dense
recomputation that pairwise interaction terms close at polynomial cost.

Validity rules
--------------
* Frozen public causal LM; no training.
* Controlled residual payloads, not semantic symlink reads.
* Prompts and continuation tokens are exogenous.
* Gold cache for every evaluated coalition is independently recomputed.
* Cached decode is independently checked against a no-cache full forward.
* Pair terms are derived ONLY from empty/singleton/pair prefills.  No higher
  coalition is used to fit the approximation.
* No numerical threshold is used to omit a dependency.  Revoking source i
  removes every materialized term whose exact source-set contains i.
* Results are structural/engineering evidence only and cannot satisfy the
  >=0.95 real-symlink capability gate.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

PROTOCOL = "RL-MIX-002-v1.0-pairwise-source-set"


def digest_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def legacy_cache(past: Any) -> tuple:
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return tuple(tuple(t.detach().clone() for t in layer) for layer in past)


def pack(past: tuple) -> tuple[torch.Tensor, list[list[tuple]]]:
    layout = [[(tuple(t.shape), t.dtype) for t in layer] for layer in past]
    flat = torch.cat([t.flatten().double() for layer in past for t in layer])
    return flat, layout


def unpack(flat: torch.Tensor, layout: list[list[tuple]]) -> tuple:
    offset, layers = 0, []
    for layer in layout:
        out = []
        for shape, dtype in layer:
            count = math.prod(shape)
            out.append(flat[offset:offset+count].to(dtype).reshape(shape).clone())
            offset += count
        layers.append(tuple(out))
    if offset != flat.numel():
        raise ValueError("cache layout does not match tensor size")
    return tuple(layers)


def blocks_of(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    raise ValueError("unsupported model block layout")


@torch.inference_mode()
def prefill(model, prompt, payloads, mask, read_layers, positions, return_logits=False):
    handles, counts = [], []
    for index in read_layers:
        counts.append(0)
        slot = len(counts) - 1

        def hook(module, inputs, output, slot=slot):
            counts[slot] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            new = hidden.clone()
            for source, pos in enumerate(positions):
                if mask & (1 << source):
                    new[0, pos] += payloads[source].to(new.dtype)
            return (new,) + output[1:] if isinstance(output, tuple) else new

        handles.append(blocks_of(model)[index].register_forward_hook(hook))
    try:
        out = model(input_ids=prompt, use_cache=not return_logits)
    finally:
        for handle in handles:
            handle.remove()
    if counts != [1] * len(read_layers):
        raise RuntimeError(f"read sites not executed exactly once: {counts}")
    if return_logits:
        return out.logits[0, -1].double()
    return pack(legacy_cache(out.past_key_values))


@torch.inference_mode()
def decode(model, flat, layout, continuation, length):
    past = unpack(flat, layout)
    if hasattr(model, "gpt_neox"):
        from transformers.cache_utils import DynamicCache
        past = DynamicCache.from_legacy_cache(past)
    out = model(
        input_ids=continuation,
        past_key_values=past,
        attention_mask=torch.ones((1, length + 1), dtype=torch.long),
        use_cache=False,
    )
    return out.logits[0, -1].double()


def kl(reference, observed):
    lp = reference.log_softmax(-1)
    lq = observed.log_softmax(-1)
    return max(0.0, float((lp.exp() * (lp - lq)).sum()))


def summarize(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "max": float(x.max()),
    }


def masks_to_test(n_sources: int, seed: int) -> list[int]:
    full = (1 << n_sources) - 1
    masks = [full]
    masks.extend(full ^ (1 << i) for i in range(n_sources))
    # Deterministic half-density coalitions exercise composition without
    # exploding to all 2^n masks.
    g = torch.Generator().manual_seed(771000 + seed + n_sources * 100)
    for _ in range(4):
        bits = torch.randperm(n_sources, generator=g)[: max(2, n_sources // 2)].tolist()
        mask = 0
        for i in bits:
            mask |= 1 << int(i)
        masks.append(mask)
    # unique, stable order
    out = []
    for m in masks:
        if m not in out:
            out.append(m)
    return out


def source_positions(n_sources: int) -> tuple[int, list[int]]:
    # Three-token spacing avoids all interventions sharing one token while
    # keeping the prompt compact enough for CPU CI.
    positions = [2 + 3 * i for i in range(n_sources)]
    return positions[-1] + 4, positions


def derive_terms(model, prompt, payloads, read_layers, positions):
    n = len(positions)
    t0 = time.perf_counter()
    base, layout = prefill(model, prompt, payloads, 0, read_layers, positions)
    unary = []
    for i in range(n):
        cache, layout_i = prefill(model, prompt, payloads, 1 << i, read_layers, positions)
        if layout_i != layout:
            raise RuntimeError("cache layout changed across coalitions")
        unary.append(cache - base)
    pair = {}
    for i, j in itertools.combinations(range(n), 2):
        cache, layout_ij = prefill(model, prompt, payloads, (1 << i) | (1 << j), read_layers, positions)
        if layout_ij != layout:
            raise RuntimeError("cache layout changed across coalitions")
        pair[(i, j)] = cache - base - unary[i] - unary[j]
    return base, unary, pair, layout, time.perf_counter() - t0


def reconstruct(base, unary, pair, mask: int, order: int):
    out = base.clone()
    active = [i for i in range(len(unary)) if mask & (1 << i)]
    for i in active:
        out += unary[i]
    if order >= 2:
        for i, j in itertools.combinations(active, 2):
            out += pair[(i, j)]
    return out


def run_case(model, model_name, revision, seed, rms, n_sources):
    generator = torch.Generator().manual_seed(seed + 944200 + n_sources * 100)
    vocab = model.get_input_embeddings().weight.shape[0]
    prompt_len, positions = source_positions(n_sources)
    prompt = torch.randint(0, vocab, (1, prompt_len), generator=generator)
    continuation = torch.randint(0, vocab, (1, 1), generator=generator)
    target_ids = torch.randperm(vocab, generator=generator)[:n_sources]
    values = model.get_output_embeddings().weight[target_ids].detach().double()
    values = values * (rms / values.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12))
    n_blocks = len(blocks_of(model))
    if n_blocks < 4:
        raise RuntimeError("need at least four blocks")
    read_layers = [1, min(3, n_blocks - 2)]

    base, unary, pair, layout, materialize_seconds = derive_terms(
        model, prompt, values, read_layers, positions
    )
    masks = masks_to_test(n_sources, seed)
    extended = torch.cat([prompt, continuation], dim=1)

    order_rows = {1: [], 2: []}
    native_errors = []
    dense_prefill_seconds = 0.0
    gold_cache_bytes = None
    for mask in masks:
        t0 = time.perf_counter()
        gold_cache, gold_layout = prefill(model, prompt, values, mask, read_layers, positions)
        dense_prefill_seconds += time.perf_counter() - t0
        if gold_layout != layout:
            raise RuntimeError("gold cache layout mismatch")
        gold_logits = decode(model, gold_cache, layout, continuation, prompt_len)
        full_logits = prefill(
            model, extended, values, mask, read_layers, positions, return_logits=True
        )
        native_errors.append(float((gold_logits - full_logits).abs().max()))
        if gold_cache_bytes is None:
            gold_cache_bytes = int(gold_cache.numel() * gold_cache.element_size())
        for order in (1, 2):
            approx = reconstruct(base, unary, pair, mask, order)
            pred = decode(model, approx, layout, continuation, prompt_len)
            order_rows[order].append({
                "mask": mask,
                "active_sources": mask.bit_count(),
                "cache_maxabs": float((approx - gold_cache).abs().max()),
                "logit_maxabs": float((pred - gold_logits).abs().max()),
                "kl_to_dense_current": kl(gold_logits, pred),
                "top1_agreement": bool(pred.argmax() == gold_logits.argmax()),
            })

    full = (1 << n_sources) - 1
    revoke_masks = [full ^ (1 << i) for i in range(n_sources)]
    results = []
    for order in (1, 2):
        rows = order_rows[order]
        rev = [r for r in rows if r["mask"] in revoke_masks]
        results.append({
            "order": order,
            "materialized_terms": 1 + n_sources + (math.comb(n_sources, 2) if order == 2 else 0),
            "terms_invalidated_by_one_revoke": 1 if order == 1 else n_sources,
            "fraction_terms_invalidated_by_one_revoke": (
                (1 if order == 1 else n_sources)
                / (1 + n_sources + (math.comb(n_sources, 2) if order == 2 else 0))
            ),
            "kl_to_dense_current": summarize([r["kl_to_dense_current"] for r in rows]),
            "logit_maxabs": summarize([r["logit_maxabs"] for r in rows]),
            "cache_maxabs": summarize([r["cache_maxabs"] for r in rows]),
            "top1_agreement": sum(r["top1_agreement"] for r in rows) / len(rows),
            "single_revoke_kl": summarize([r["kl_to_dense_current"] for r in rev]),
            "single_revoke_top1_agreement": sum(r["top1_agreement"] for r in rev) / len(rev),
            "rows": rows,
        })

    pair_bytes = sum(x.numel() * x.element_size() for x in pair.values())
    unary_bytes = sum(x.numel() * x.element_size() for x in unary)
    base_bytes = base.numel() * base.element_size()
    return {
        "model": model_name,
        "model_revision": revision,
        "seed": seed,
        "payload_rms": rms,
        "source_count": n_sources,
        "read_layers": read_layers,
        "positions": positions,
        "prompt_tokens": prompt_len,
        "continuation_id": continuation.tolist(),
        "target_ids": target_ids.tolist(),
        "payload_sha256": digest_tensor(values),
        "evaluated_masks": masks,
        "materialization_prefills": 1 + n_sources + math.comb(n_sources, 2),
        "materialization_seconds": materialize_seconds,
        "independent_gold_prefill_seconds": dense_prefill_seconds,
        "single_dense_cache_bytes_fp64": gold_cache_bytes,
        "pairwise_term_store_bytes_fp64": int(base_bytes + unary_bytes + pair_bytes),
        "order_results": results,
        "native_cached_vs_full_forward_maxabs": summarize(native_errors),
        "native_cached_vs_full_forward_control_pass": max(native_errors) < 5e-7,
        "pairwise_beats_unary_on_max_kl": results[1]["kl_to_dense_current"]["max"] < results[0]["kl_to_dense_current"]["max"],
        "note": (
            "Controlled payloads only. Order-1 is an additive source decomposition, not an implementation of ReCache. "
            "Order-2 is standard pairwise interaction truncation, not claimed novel."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rms", type=float, nargs="+", default=[1.0, 4.0])
    ap.add_argument("--sources", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--output", default="results/pairwise_cache_probe.json")
    a = ap.parse_args()

    torch.set_num_threads(2)
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM
    import transformers

    revision = a.revision or model_info(a.model).sha
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        revision=revision,
        torch_dtype=torch.float64,
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rec = {
        "protocol": PROTOCOL,
        "source_commit": os.getenv("GITHUB_SHA"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_revision": revision,
        "training_steps": 0,
        "symlink_validity": "NOT_MEASURED",
        "breakthrough": False,
        "rows": [],
    }
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for n_sources in a.sources:
        for seed in a.seeds:
            for rms in a.rms:
                row = run_case(model, a.model, revision, seed, rms, n_sources)
                rec["rows"].append(row)
                dest.write_text(json.dumps(rec, indent=2))
                print(json.dumps({
                    "sources": n_sources,
                    "seed": seed,
                    "rms": rms,
                    "native_control": row["native_cached_vs_full_forward_control_pass"],
                    "unary_maxkl": row["order_results"][0]["kl_to_dense_current"]["max"],
                    "pair_maxkl": row["order_results"][1]["kl_to_dense_current"]["max"],
                    "pair_revoke_maxkl": row["order_results"][1]["single_revoke_kl"]["max"],
                }), flush=True)

    valid = all(x["native_cached_vs_full_forward_control_pass"] for x in rec["rows"])
    improved = all(x["pairwise_beats_unary_on_max_kl"] for x in rec["rows"])
    rec["all_controls_pass"] = valid
    rec["pairwise_improves_all_cells"] = improved
    rec["breakthrough"] = False
    dest.write_text(json.dumps(rec, indent=2))
    if not valid:
        raise SystemExit("invalid cached-vs-full-forward control; no inference may be drawn")


if __name__ == "__main__":
    main()
