from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    MODEL_ID,
    REVISION,
    EXPECTED_WEIGHTS,
    COLORS,
    SPECS,
    sha256_file,
    render,
    expected,
    top_from_logits,
    legacy_cache,
    slice_cache,
    concat_cache,
    cache_bytes,
    query_logits,
    run_segment,
)

OUT = Path(os.environ.get("SO_R285_REPORT", "ci-qwen-r285/report.json"))


def sparse_capsule(capsule, kept_layers: set[int]):
    out = []
    for i, (k, v) in enumerate(legacy_cache(capsule)):
        if i in kept_layers:
            out.append((k.clone(), v.clone()))
        else:
            out.append((torch.zeros_like(k), torch.zeros_like(v)))
    return tuple(out)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
        ],
    ))
    verified = {name: sha256_file(md / name) for name in EXPECTED_WEIGHTS}
    assert verified == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(
        md, local_files_only=True, padding_side="left", trust_remote_code=False
    )
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    layouts = {(v, spec): render(tok, v, spec) for v in range(4) for spec in SPECS}
    static_ids = layouts[(0, "read")]["static_ids"]
    assert all(x["static_ids"] == static_ids for x in layouts.values())
    S = len(static_ids)

    dyn_by_value = {}
    for v in range(4):
        ids = layouts[(v, "read")]["dynamic_ids"]
        assert all(layouts[(v, spec)]["dynamic_ids"] == ids for spec in SPECS)
        dyn_by_value[v] = ids
    widths = {len(x) for x in dyn_by_value.values()}
    assert len(widths) == 1
    D = next(iter(widths))

    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        so = model(
            input_ids=st,
            attention_mask=torch.ones_like(st),
            use_cache=True,
            return_dict=True,
        )
    shared = legacy_cache(so.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    capsules = {}
    full_prefix = {}
    for v in range(4):
        o = run_segment(model, dyn_by_value[v], shared, shared_mask)
        c = legacy_cache(o.past_key_values)
        full_prefix[v] = c
        capsules[v] = slice_cache(c, S, S + D)

    L = len(capsules[0])
    variants: dict[str, set[int]] = {
        "all": set(range(L)),
        "early8": set(range(min(8, L))),
        "middle8": set(range(max(0, L // 2 - 4), min(L, L // 2 + 4))),
        "late1": set(range(max(0, L - 1), L)),
        "late2": set(range(max(0, L - 2), L)),
        "late4": set(range(max(0, L - 4), L)),
        "late8": set(range(max(0, L - 8), L)),
        "every2": set(range(0, L, 2)),
        "every3": set(range(0, L, 3)),
        "every4": set(range(0, L, 4)),
        "every6": set(range(0, L, 6)),
        "every9": set(range(0, L, 9)),
    }
    for i in range(L):
        variants[f"layer_{i:02d}"] = {i}

    reference = {}
    for v in range(4):
        pmask = torch.ones((1, S + D), dtype=torch.long)
        for spec in SPECS:
            logits = query_logits(model, layouts[(v, spec)]["query_ids"], full_prefix[v], pmask)
            top, scores = top_from_logits(tok, logits, spec)
            reference[(v, spec)] = (logits, top, scores)

    rows = []
    summary = {}
    for name, kept in variants.items():
        matches = []
        correct = []
        deltas = []
        for v in range(4):
            sparse = sparse_capsule(capsules[v], kept)
            cache = concat_cache(shared, sparse)
            pmask = torch.ones((1, S + D), dtype=torch.long)
            for spec in SPECS:
                logits = query_logits(model, layouts[(v, spec)]["query_ids"], cache, pmask)
                ref_logits, ref_top, _ = reference[(v, spec)]
                top, _ = top_from_logits(tok, logits, spec)
                exp = expected(spec, v)
                delta = float(torch.max(torch.abs(logits - ref_logits)).item())
                matches.append(top == ref_top)
                correct.append(top == exp)
                deltas.append(delta)
                rows.append({
                    "variant": name,
                    "kept_layers": sorted(kept),
                    "value": v,
                    "color": COLORS[v],
                    "spec": spec,
                    "expected": exp,
                    "reference_top": ref_top,
                    "sparse_top": top,
                    "top_match": top == ref_top,
                    "correct": top == exp,
                    "max_vocab_logit_delta": delta,
                })
        retained_fraction = len(kept) / L
        summary[name] = {
            "kept_layer_count": len(kept),
            "kept_layers": sorted(kept),
            "retained_dynamic_kv_fraction": retained_fraction,
            "top_match_rate": sum(matches) / len(matches),
            "accuracy": sum(correct) / len(correct),
            "median_logit_delta": float(torch.tensor(deltas).median().item()),
            "max_logit_delta": max(deltas),
        }

    full_acc = summary["all"]["accuracy"]
    full_match = summary["all"]["top_match_rate"]
    candidates = [
        (name, m) for name, m in summary.items()
        if name != "all"
        and m["kept_layer_count"] <= 8
        and m["top_match_rate"] >= 0.90
        and m["accuracy"] >= 0.90
    ]
    best = None
    if candidates:
        best = min(
            candidates,
            key=lambda x: (x[1]["kept_layer_count"], -x[1]["top_match_rate"], -x[1]["accuracy"]),
        )[0]

    capsule_total_bytes = cache_bytes(capsules[0])
    bytes_per_layer = capsule_total_bytes // L
    report = {
        "stage": "R285-LAYER-SPARSE-GENERATION-PORTS",
        "architecture_candidate": "Layer-Sparse Generation-Gated Latent Ports",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": verified,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "dynamic_slots": D,
        "model_layers": L,
        "full_capsule_bytes": capsule_total_bytes,
        "approx_dynamic_bytes_per_kept_layer": bytes_per_layer,
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "reference_full_port_accuracy": full_acc,
        "reference_self_match": full_match,
        "best_leq8_layer_variant": best,
        "variant_summary": summary,
        "rows": rows,
        "scientific_scope": (
            "Real frozen Qwen2.5-3B-Instruct. Starting from the exact factorized R284 "
            "port capsule, zeroes dynamic K/V at omitted transformer layers while keeping "
            "the shared static prelude unchanged. Tests whether per-fact knowledge can be "
            "stored only at a sparse layer subset without per-fact training. Zeroed layers "
            "are treated as implicit and need not be persisted."
        ),
        "dod_status": "NOT_DOD; sparse-storage mechanism search",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    top_small = sorted(
        [(n, m) for n, m in summary.items() if n != "all"],
        key=lambda x: (-x[1]["top_match_rate"], -x[1]["accuracy"], x[1]["kept_layer_count"]),
    )[:12]
    print(json.dumps({
        "layers": L,
        "dynamic_slots": D,
        "reference_accuracy": full_acc,
        "best_leq8_layer_variant": best,
        "top_sparse_variants": {n: m for n, m in top_small},
    }, indent=2))

    if full_match != 1.0:
        return 2
    if full_acc < 0.90:
        return 3
    if best is None:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
