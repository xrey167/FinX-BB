"""RL-MIX-001: controlled cache interaction screen, NOT a symlink qualification.

Known Boolean-lattice/Moebius decomposition is used as a diagnostic and an
expensive ordinary baseline. No novelty is claimed for the transform.
Every cache is computed on an independently authorized coalition of four
controlled payload sources. Prompts and continuation tokens are exogenous.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any
import torch

PROTOCOL = "RL-MIX-001-v1.1-cache-compat"


def mobius(table: torch.Tensor) -> torch.Tensor:
    n = len(table)
    if n == 0 or n & (n - 1):
        raise ValueError("number of coalitions must be a positive power of two")
    result = table.to(torch.float64).clone()
    for bit in range(n.bit_length() - 1):
        for mask in range(n):
            if mask & (1 << bit):
                result[mask] -= result[mask ^ (1 << bit)]
    return result


def reconstruct(terms: torch.Tensor, mask: int, order: int) -> torch.Tensor:
    if not 0 <= mask < len(terms) or order < 0:
        raise ValueError("invalid coalition or order")
    result = torch.zeros_like(terms[0])
    sub = mask
    while True:
        if sub.bit_count() <= order:
            result += terms[sub]
        if not sub:
            return result
        sub = (sub - 1) & mask


def digest_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(x.mean()), "max": float(x.max()),
            "median": float(x.median())}


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
    # GPT-NeoX 4.45.2 requires Cache with use_cache=False. Rebuild on each
    # decode because DynamicCache.update mutates its tensors during execution.
    if hasattr(model, "gpt_neox"):
        from transformers.cache_utils import DynamicCache
        past = DynamicCache.from_legacy_cache(past)
    out = model(input_ids=continuation, past_key_values=past,
                attention_mask=torch.ones((1, length+1), dtype=torch.long),
                use_cache=False)
    return out.logits[0, -1].double()


def kl(reference, observed):
    lp = reference.log_softmax(-1)
    lq = observed.log_softmax(-1)
    return max(0.0, float((lp.exp() * (lp - lq)).sum()))


def run_case(model, model_name, revision, seed, rms, n_sources=4):
    generator = torch.Generator().manual_seed(seed + 932700)
    vocab = model.get_input_embeddings().weight.shape[0]
    prompt = torch.randint(0, vocab, (1, 14), generator=generator)
    continuation = torch.randint(0, vocab, (1, 1), generator=generator)
    target_ids = torch.randperm(vocab, generator=generator)[:n_sources]
    values = model.get_output_embeddings().weight[target_ids].detach().float()
    values = values * (rms / values.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-9))
    positions = [1, 4, 7, 10]
    read_layers = [1, 3]
    t0 = time.perf_counter()
    snapshots, layout = [], None
    for mask in range(1 << n_sources):
        flat, layout = prefill(model, prompt, values, mask, read_layers, positions)
        snapshots.append(flat)
    table = torch.stack(snapshots)
    prefill_seconds = time.perf_counter() - t0
    terms = mobius(table)
    full = (1 << n_sources) - 1
    targets = [full ^ (1 << source) for source in range(n_sources)]
    gold = [decode(model, row, layout, continuation, prompt.shape[1]) for row in table]
    # Independent control: replay must match a no-cache full forward, not
    # merely another call through the same reconstruction/decode function.
    extended = torch.cat([prompt, continuation], dim=1)
    native_errors = []
    for mask in range(1 << n_sources):
        full_logits = prefill(model, extended, values, mask, read_layers, positions, return_logits=True)
        native_errors.append(float((gold[mask] - full_logits).abs().max()))
    rows = []
    for order in range(1, n_sources+1):
        cache_errors, logit_errors, kls, top1 = [], [], [], []
        for mask in range(1 << n_sources):
            approximation = reconstruct(terms, mask, order)
            predicted = decode(model, approximation, layout, continuation, prompt.shape[1])
            cache_errors.append(float((approximation - table[mask]).abs().max()))
            logit_errors.append(float((predicted - gold[mask]).abs().max()))
            kls.append(kl(gold[mask], predicted))
            top1.append(bool(predicted.argmax() == gold[mask].argmax()))
        rows.append({"order": order,
                     "materialized_terms": sum(math.comb(n_sources, j) for j in range(order+1)),
                     "cache_maxabs": summarize(cache_errors), "logit_maxabs": summarize(logit_errors),
                     "kl_to_dense_current": summarize(kls), "top1_agreement_to_dense": sum(top1)/len(top1),
                     "single_revoke_kl": [kls[m] for m in targets],
                     "single_revoke_logit_maxabs": [logit_errors[m] for m in targets],
                     "per_coalition_kl": kls, "per_coalition_cache_maxabs": cache_errors})
    direct_subtraction = []
    for source, mask in enumerate(targets):
        wrong = table[full] - terms[1 << source]
        prediction = decode(model, wrong, layout, continuation, prompt.shape[1])
        direct_subtraction.append({"source": source, "cache_maxabs": float((wrong-table[mask]).abs().max()),
                                   "logit_maxabs": float((prediction-gold[mask]).abs().max()),
                                   "kl": kl(gold[mask], prediction)})
    coefficient_rms = {str(order): summarize([float(terms[m].square().mean().sqrt())
                       for m in range(1 << n_sources) if m.bit_count() == order])
                       for order in range(1, n_sources+1)}
    # No numerical or geometrical cutoff is allowed to remove a dependency.
    full_error = rows[-1]["logit_maxabs"]["max"]
    exact_small = all(row["per_coalition_cache_maxabs"][m] < 1e-8
                      for row in rows for m in range(1 << n_sources) if m.bit_count() <= row["order"])
    record = {"model": model_name, "model_revision": revision, "intervention_seed": seed,
              "payload_rms": rms, "prompt_ids": prompt.tolist(), "continuation_id": continuation.tolist(),
              "target_ids": target_ids.tolist(), "payload_sha256": digest_tensor(values),
              "read_layers": read_layers, "positions": positions, "source_count": n_sources,
              "prefill_count": 1 << n_sources, "snapshot_table_sha256": digest_tensor(table),
              "single_cache_bytes_fp32": int(table.shape[1]*4),
              "measured_table_bytes_fp64": int(table.numel()*8),
              "prefill_table_seconds": prefill_seconds,
              "coefficient_rms_by_order": coefficient_rms, "orders": rows,
              "direct_unary_subtraction": direct_subtraction,
              "native_cached_vs_full_forward_maxabs": summarize(native_errors),
              "native_cached_vs_full_forward_control_pass": max(native_errors) < 5e-4,
              "full_lattice_reconstruction_control_pass": full_error < 5e-4 and exact_small,
              "note": "Controlled injected payloads, not learned alias reads. Seeds vary inputs/payloads, not training. KL here is dense-cache approximation KL, NOT generic-language KL or deleted-fact leakage."}
    return record


def selftest():
    torch.manual_seed(23)
    for n in range(1, 6):
        data = torch.randn(1 << n, 17, dtype=torch.float64)
        terms = mobius(data)
        for mask in range(1 << n):
            assert torch.allclose(reconstruct(terms, mask, n), data[mask], atol=1e-12, rtol=0)
    # A cross-source product is invisible to singleton subtraction.
    table = torch.tensor([[0.], [0.], [0.], [1.]])
    terms = mobius(table)
    assert float(reconstruct(terms, 3, 1)) == 0
    assert float(table[3] - terms[1]) == 1
    assert float(reconstruct(terms, 2, 2)) == 0
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--revision")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rms", type=float, nargs="+", default=[0.1, 1., 4.])
    ap.add_argument("--output", default="results/coalition_probe.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.selftest:
        selftest()
        if not a.model:
            return
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM
    import transformers
    revision = a.revision or model_info(a.model).sha
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=revision, torch_dtype=torch.float32,
                                                attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rec = {"protocol": PROTOCOL, "source_commit": os.getenv("GITHUB_SHA"),
           "python": platform.python_version(), "torch": torch.__version__,
           "transformers": transformers.__version__, "model_revision": revision,
           "training_steps": 0, "symlink_validity": "NOT_MEASURED", "breakthrough": False,
           "rows": []}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for rms in a.rms:
            row = run_case(model, a.model, revision, seed, rms)
            rec["rows"].append(row)
            dest.write_text(json.dumps(rec, indent=2))
            print(json.dumps({"seed": seed, "rms": rms, "control": row["full_lattice_reconstruction_control_pass"],
                              "unary_maxkl": row["orders"][0]["kl_to_dense_current"]["max"]}), flush=True)
    if not all(x["full_lattice_reconstruction_control_pass"] and
               x["native_cached_vs_full_forward_control_pass"] for x in rec["rows"]):
        raise SystemExit("invalid reconstruction control; no inference may be drawn")

if __name__ == "__main__":
    main()
