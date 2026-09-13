from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    cache_bytes,
    concat_cache,
    legacy_cache,
    query_logits,
    run_segment,
    sha256_file,
    slice_cache,
)
from research.qwen25_05b_r288_shared_latent_port_codec import (
    MODEL_ID,
    REVISION,
    EXPECTED_WEIGHTS,
    VALUE_POOL,
    render,
    top_word,
)

OUT = Path(os.environ.get("SO_R290_REPORT", "ci-qwen-r290/report.json"))


def mean_cache(caches):
    cs = [legacy_cache(c) for c in caches]
    out = []
    for layer in range(len(cs[0])):
        ks = torch.stack([c[layer][0].float() for c in cs], dim=0).mean(dim=0)
        vs = torch.stack([c[layer][1].float() for c in cs], dim=0).mean(dim=0)
        out.append((ks.to(cs[0][layer][0].dtype), vs.to(cs[0][layer][1].dtype)))
    return tuple(out)


def _quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    group_size: int,
    anchor: torch.Tensor | None = None,
    correction_topk: int = 0,
):
    x = tensor.float().reshape(-1)
    base = torch.zeros_like(x) if anchor is None else anchor.float().reshape(-1)
    delta = x - base
    qmax = (1 << (bits - 1)) - 1
    if qmax < 1:
        raise ValueError(bits)

    recon_delta = torch.empty_like(delta)
    code_bytes = 0
    scale_bytes = 0
    for start in range(0, delta.numel(), group_size):
        end = min(start + group_size, delta.numel())
        chunk = delta[start:end]
        max_abs = float(chunk.abs().max().item())
        scale = max_abs / qmax if max_abs > 0 else 1.0
        q = torch.round(chunk / scale).clamp(-qmax, qmax)
        recon_delta[start:end] = q * scale
        code_bytes += math.ceil((end - start) * bits / 8)
        scale_bytes += 2  # fp16 scale

    correction_bytes = 0
    if correction_topk > 0:
        err = delta - recon_delta
        k = min(correction_topk, err.numel())
        if k:
            idx = torch.topk(err.abs(), k=k, largest=True).indices
            # Store an fp16 residual correction plus a uint16 local index.
            corr = err[idx].to(torch.float16).float()
            recon_delta[idx] += corr
            correction_bytes = k * 4

    recon = (base + recon_delta).reshape_as(tensor).to(tensor.dtype).contiguous()
    rel = float(
        torch.linalg.vector_norm(recon.float().reshape(-1) - x)
        / max(float(torch.linalg.vector_norm(x).item()), 1e-12)
    )
    return recon, code_bytes + scale_bytes + correction_bytes, rel


def quantize_cache(
    cache,
    *,
    k_bits: int,
    v_bits: int,
    group_size: int,
    anchor=None,
    correction_topk: int = 0,
):
    c = legacy_cache(cache)
    a = legacy_cache(anchor) if anchor is not None else None
    out = []
    storage = 0
    rel_errors = []
    for i, (k, v) in enumerate(c):
        ak = None if a is None else a[i][0]
        av = None if a is None else a[i][1]
        rk, kb, kre = _quantize_tensor(
            k, k_bits, group_size, anchor=ak, correction_topk=correction_topk
        )
        rv, vb, vre = _quantize_tensor(
            v, v_bits, group_size, anchor=av, correction_topk=correction_topk
        )
        out.append((rk, rv))
        storage += kb + vb
        rel_errors.extend([kre, vre])
    return tuple(out), storage, sum(rel_errors) / len(rel_errors)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
        ],
    ))
    hashes = {name: sha256_file(model_dir / name) for name in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS

    tok = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, padding_side="left", trust_remote_code=False
    )
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    layouts = {}
    groups = {}
    for word in VALUE_POOL:
        layout = render(tok, word)
        layouts[word] = layout
        key = (tuple(layout["static_ids"]), len(layout["dynamic_ids"]))
        groups.setdefault(key, []).append(word)
    group_key, eligible = max(groups.items(), key=lambda kv: len(kv[1]))
    eligible = list(eligible)
    static_ids = list(group_key[0])
    slot_width = int(group_key[1])
    if len(eligible) < 16:
        raise RuntimeError(f"stable group too small: {eligible}")

    S = len(static_ids)
    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        static_out = model(
            input_ids=st,
            attention_mask=torch.ones_like(st),
            use_cache=True,
            return_dict=True,
        )
    shared = legacy_cache(static_out.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    capsules = {}
    prefixes = {}
    for word in eligible:
        layout = layouts[word]
        out = run_segment(model, layout["dynamic_ids"], shared, shared_mask)
        prefix = legacy_cache(out.past_key_values)
        prefixes[word] = prefix
        capsules[word] = slice_cache(prefix, S, S + slot_width)

    split = max(8, int(len(eligible) * 0.70))
    split = min(split, len(eligible) - 4)
    train_words = eligible[:split]
    holdout_words = eligible[split:]
    anchor = mean_cache([capsules[w] for w in train_words])
    shared_anchor_bytes = cache_bytes(anchor)
    full_capsule_bytes = cache_bytes(capsules[eligible[0]])
    prefix_mask = torch.ones((1, S + slot_width), dtype=torch.long)

    exact = {}
    for word in holdout_words:
        logits = query_logits(
            model, layouts[word]["query_ids"], prefixes[word], prefix_mask
        )
        top, _ = top_word(tok, logits, eligible)
        exact[word] = {"logits": logits, "top": top, "correct": top == word}

    variants = [
        {"name": "direct-k8-v8-g64", "k": 8, "v": 8, "g": 64, "anchor": False, "corr": 0},
        {"name": "direct-k6-v4-g64", "k": 6, "v": 4, "g": 64, "anchor": False, "corr": 0},
        {"name": "direct-k4-v4-g64", "k": 4, "v": 4, "g": 64, "anchor": False, "corr": 0},
        {"name": "direct-k4-v2-g64", "k": 4, "v": 2, "g": 64, "anchor": False, "corr": 0},
        {"name": "direct-k3-v3-g32", "k": 3, "v": 3, "g": 32, "anchor": False, "corr": 0},
        {"name": "direct-k3-v2-g32", "k": 3, "v": 2, "g": 32, "anchor": False, "corr": 0},
        {"name": "direct-k2-v2-g32", "k": 2, "v": 2, "g": 32, "anchor": False, "corr": 0},
        {"name": "residual-k4-v4-g64", "k": 4, "v": 4, "g": 64, "anchor": True, "corr": 0},
        {"name": "residual-k4-v2-g64", "k": 4, "v": 2, "g": 64, "anchor": True, "corr": 0},
        {"name": "residual-k3-v3-g32", "k": 3, "v": 3, "g": 32, "anchor": True, "corr": 0},
        {"name": "residual-k3-v2-g32", "k": 3, "v": 2, "g": 32, "anchor": True, "corr": 0},
        {"name": "residual-k2-v2-g32", "k": 2, "v": 2, "g": 32, "anchor": True, "corr": 0},
        {"name": "residual-k3-v2-g32-corr4", "k": 3, "v": 2, "g": 32, "anchor": True, "corr": 4},
        {"name": "residual-k2-v2-g32-corr8", "k": 2, "v": 2, "g": 32, "anchor": True, "corr": 8},
    ]

    rows = []
    summary = {}
    for spec in variants:
        matches = []
        correct = []
        logit_deltas = []
        rel_errors = []
        storage_values = []
        for word in holdout_words:
            qc, stored_bytes, rel = quantize_cache(
                capsules[word],
                k_bits=spec["k"],
                v_bits=spec["v"],
                group_size=spec["g"],
                anchor=anchor if spec["anchor"] else None,
                correction_topk=spec["corr"],
            )
            cache = concat_cache(shared, qc)
            logits = query_logits(model, layouts[word]["query_ids"], cache, prefix_mask)
            top, _ = top_word(tok, logits, eligible)
            exact_top = exact[word]["top"]
            delta = float(torch.max(torch.abs(logits - exact[word]["logits"])).item())
            matches.append(top == exact_top)
            correct.append(top == word)
            logit_deltas.append(delta)
            rel_errors.append(rel)
            storage_values.append(stored_bytes)
            rows.append({
                "variant": spec["name"],
                "word": word,
                "exact_top": exact_top,
                "compressed_top": top,
                "exact_correct": exact[word]["correct"],
                "compressed_correct": top == word,
                "top_match": top == exact_top,
                "max_vocab_logit_delta": delta,
                "mean_tensor_relative_l2_error": rel,
                "per_fact_storage_bytes": stored_bytes,
            })

        per_fact_bytes = max(storage_values)
        shared_bytes = shared_anchor_bytes if spec["anchor"] else 0
        summary[spec["name"]] = {
            "k_bits": spec["k"],
            "v_bits": spec["v"],
            "group_size": spec["g"],
            "uses_shared_anchor": spec["anchor"],
            "correction_topk_per_tensor": spec["corr"],
            "top_match_rate_vs_exact": sum(matches) / len(matches),
            "compressed_accuracy": sum(correct) / len(correct),
            "mean_tensor_relative_l2_error": sum(rel_errors) / len(rel_errors),
            "max_vocab_logit_delta": max(logit_deltas),
            "per_fact_storage_bytes": per_fact_bytes,
            "per_fact_storage_ratio_vs_bf16_capsule": per_fact_bytes / full_capsule_bytes,
            "shared_anchor_bytes": shared_bytes,
            "amortized_ratio_at_1k_facts": (
                shared_bytes + 1000 * per_fact_bytes
            ) / (1000 * full_capsule_bytes),
            "amortized_ratio_at_1m_facts": (
                shared_bytes + 1_000_000 * per_fact_bytes
            ) / (1_000_000 * full_capsule_bytes),
        }

    passing = [
        (name, m) for name, m in summary.items()
        if m["top_match_rate_vs_exact"] >= 0.95
        and m["compressed_accuracy"] >= 0.95
        and m["per_fact_storage_ratio_vs_bf16_capsule"] <= 0.50
    ]
    best = None
    if passing:
        best = min(
            passing,
            key=lambda x: (
                x[1]["per_fact_storage_ratio_vs_bf16_capsule"],
                -x[1]["top_match_rate_vs_exact"],
            ),
        )[0]

    report = {
        "stage": "R290-QUANTIZED-RESIDUAL-PORTS",
        "architecture_candidate": "Quantized Generation Port Capsules",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "eligible_words": eligible,
        "train_words_for_shared_anchor": train_words,
        "future_holdout_words": holdout_words,
        "exact_holdout_accuracy": sum(exact[w]["correct"] for w in holdout_words) / len(holdout_words),
        "full_bf16_capsule_bytes": full_capsule_bytes,
        "shared_anchor_bytes": shared_anchor_bytes,
        "variant_summary": summary,
        "best_passing_variant": best,
        "compactness_gate_pass": best is not None,
        "gate_definition": "holdout top-match >=95%, holdout accuracy >=95%, <=50% bf16 per-fact storage",
        "rows": rows,
        "interpretation": (
            "R288 showed that low-rank projection can destroy attention behavior despite modest L2 error. "
            "R290 preserves every KV dimension and reduces precision instead, including mixed K/V bit widths, "
            "a train-only shared anchor, and sparse residual error compensation. The holdout values are never used "
            "to fit the anchor and receive no gradient optimization."
        ),
        "dod_status": "NOT_DOD; compactness mechanism gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ranked = sorted(
        summary.items(),
        key=lambda x: (
            -x[1]["top_match_rate_vs_exact"],
            -x[1]["compressed_accuracy"],
            x[1]["per_fact_storage_ratio_vs_bf16_capsule"],
        ),
    )
    print(json.dumps({
        "exact_holdout_accuracy": report["exact_holdout_accuracy"],
        "full_bf16_capsule_bytes": full_capsule_bytes,
        "best_passing_variant": best,
        "top_variants": dict(ranked[:8]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
