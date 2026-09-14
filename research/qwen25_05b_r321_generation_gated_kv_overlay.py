from __future__ import annotations

import copy
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS, VALUE_STRINGS

OUT = Path(os.environ.get("SO_R321_REPORT", "ci-qwen-r321/report.json"))


def run_from_prefix(model, suffix_ids, prefix_cache, prefix_len):
    ids = torch.tensor([suffix_ids], dtype=torch.long)
    mask = torch.ones((1, prefix_len + len(suffix_ids)), dtype=torch.long)
    pos = torch.arange(prefix_len, prefix_len + len(suffix_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=copy.deepcopy(prefix_cache),
            use_cache=True,
            return_dict=True,
        )


def mutate_stale_slots(cache, prefix_len: int, stale_len: int, seed: int):
    """Keep the live immutable prefix intact while corrupting physically resident stale slots."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for k, v in legacy_cache(cache):
        k2 = k.clone()
        v2 = v.clone()
        if stale_len:
            ks = torch.randn(k2[:, :, prefix_len:prefix_len + stale_len, :].shape, generator=g, dtype=torch.float32).to(k2.dtype)
            vs = torch.randn(v2[:, :, prefix_len:prefix_len + stale_len, :].shape, generator=g, dtype=torch.float32).to(v2.dtype)
            k2[:, :, prefix_len:prefix_len + stale_len, :] = ks
            v2[:, :, prefix_len:prefix_len + stale_len, :] = vs
        out.append((k2, v2))
    return tuple(out)


def run_overlay(model, current_suffix_ids, physical_cache, prefix_len: int, stale_len: int):
    """Append a new generation after stale physical slots while reusing its logical positions.

    Physical cache positions and logical/RoPE positions are intentionally decoupled:
      [live prefix][stale physical generation][new current generation]
    Attention authority mask:
      [1 ... 1][0 ... 0][1 ... 1]
    Logical positions of the current generation are reset to immediately follow the
    live prefix. The stale generation can remain resident until asynchronous compaction.
    """
    ids = torch.tensor([current_suffix_ids], dtype=torch.long)
    total = prefix_len + stale_len + len(current_suffix_ids)
    mask = torch.ones((1, total), dtype=torch.long)
    if stale_len:
        mask[:, prefix_len:prefix_len + stale_len] = 0
    logical_pos = torch.arange(prefix_len, prefix_len + len(current_suffix_ids), dtype=torch.long).unsqueeze(0)
    # cache_position remains the append-only physical location; position_ids is the
    # logical semantic location used by RoPE. Eager attention receives the explicit
    # authority mask for the stale physical slots.
    with torch.inference_mode():
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=logical_pos,
            past_key_values=copy.deepcopy(physical_cache),
            use_cache=True,
            return_dict=True,
        )


def cache_bytes(cache) -> int:
    return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in legacy_cache(cache)))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS

    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    messages = [
        {"role": "system", "content": (
            "The trusted CKCA runtime supplies the authoritative current governed value. "
            "Continue from that value; never recover a superseded value from cache state."
        )},
        {"role": "user", "content": "Report the current governed value for canonical POD-KV-321."},
    ]
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    static_text = base + "Current value:\n"
    static_ids = tok(static_text, add_special_tokens=False).input_ids
    S = len(static_ids)
    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        prefix_out = model(input_ids=st, attention_mask=torch.ones_like(st), use_cache=True, return_dict=True)
    prefix_cache = legacy_cache(prefix_out.past_key_values)

    stable = []
    for value in VALUE_STRINGS:
        full_text = static_text + value + "\nStatus:"
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if full_ids[:S] == static_ids:
            stable.append((value, full_ids[S:], full_ids))
    if len(stable) < 8:
        raise RuntimeError(f"too few prefix-stable values: {len(stable)}")
    probes = stable[:8]

    rows = []
    direct_vs_overlay = []
    direct_vs_corrupt_overlay = []
    full_vs_overlay = []
    top_match = []
    overlay_times = []
    direct_times = []
    physical_overhead = []

    # Every transition leaves the prior generation physically resident and overlays
    # the next generation in the same logical token positions.
    for i in range(1, len(probes)):
        stale_value, stale_suffix, _ = probes[i - 1]
        current_value, current_suffix, current_full = probes[i]

        stale_out = run_from_prefix(model, stale_suffix, prefix_cache, S)
        stale_physical_cache = legacy_cache(stale_out.past_key_values)
        stale_len = len(stale_suffix)

        t0 = time.perf_counter()
        direct = run_from_prefix(model, current_suffix, prefix_cache, S)
        direct_times.append(time.perf_counter() - t0)
        direct_logits = direct.logits[0, -1].float()

        t0 = time.perf_counter()
        overlay = run_overlay(model, current_suffix, stale_physical_cache, S, stale_len)
        overlay_times.append(time.perf_counter() - t0)
        overlay_logits = overlay.logits[0, -1].float()

        corrupted = mutate_stale_slots(stale_physical_cache, S, stale_len, 321000 + i)
        corrupt_overlay = run_overlay(model, current_suffix, corrupted, S, stale_len)
        corrupt_logits = corrupt_overlay.logits[0, -1].float()

        ft = torch.tensor([current_full], dtype=torch.long)
        with torch.inference_mode():
            full = model(input_ids=ft, attention_mask=torch.ones_like(ft), use_cache=True, return_dict=True)
        full_logits = full.logits[0, -1].float()

        d1 = float((direct_logits - overlay_logits).abs().max())
        d2 = float((direct_logits - corrupt_logits).abs().max())
        d3 = float((full_logits - overlay_logits).abs().max())
        direct_vs_overlay.append(d1)
        direct_vs_corrupt_overlay.append(d2)
        full_vs_overlay.append(d3)
        top_match.append(int(direct_logits.argmax()) == int(overlay_logits.argmax()) == int(corrupt_logits.argmax()) == int(full_logits.argmax()))
        physical_overhead.append(cache_bytes(stale_physical_cache) - cache_bytes(prefix_cache))
        rows.append({
            "transition": i,
            "stale_value": stale_value,
            "current_value": current_value,
            "stale_physical_tokens": stale_len,
            "current_logical_tokens": len(current_suffix),
            "direct_vs_overlay_max_logit_delta": d1,
            "direct_vs_corrupted_stale_overlay_max_logit_delta": d2,
            "full_vs_overlay_max_logit_delta": d3,
            "top_token_all_paths_match": top_match[-1],
            "stale_slots_physically_present": True,
            "stale_slots_authority_masked": True,
        })

    report = {
        "stage": "R321-GENERATION-GATED-KV-OVERLAY",
        "architecture_candidate": "Generation-Gated Logical KV Overlay (GG-LKVO)",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "transitions": len(rows),
        "immutable_prefix_tokens": S,
        "max_direct_vs_overlay_logit_delta": max(direct_vs_overlay),
        "max_direct_vs_corrupted_stale_overlay_logit_delta": max(direct_vs_corrupt_overlay),
        "max_full_recompute_vs_overlay_logit_delta": max(full_vs_overlay),
        "top_token_match_rate_all_paths": sum(top_match) / len(top_match),
        "median_direct_current_splice_seconds": statistics.median(direct_times),
        "median_append_only_overlay_seconds": statistics.median(overlay_times),
        "median_stale_physical_cache_bytes_retained": statistics.median(physical_overhead),
        "mechanism": (
            "KV physical residency is decoupled from semantic authority. Superseded generation slots can remain in an append-only "
            "cache, but their lifetime bit becomes an attention authority mask of zero. A repaired current suffix is appended using "
            "the same logical/RoPE positions as the expired suffix. This permits asynchronous compaction while stale KV bytes remain "
            "provably outside the current attention path in the tested eager-attention configuration."
        ),
        "architectural_hypothesis": (
            "CKCA caches should be log-structured and generation-aware: physical_slot != logical_position != authority. "
            "Revocation first changes authority in O(1)/factor time; physical memory reclamation can happen later without making "
            "stale bytes semantically live."
        ),
        "dod_status": "NOT_DOD; real-model KV authority-overlay mechanism gate",
        "claim_boundary": (
            "KV caching, attention masks, logical positions, append-only storage and MVCC are established individually. R321 tests "
            "their use as a generation-lifetime authority overlay inside CKCA neural execution; standalone component novelty is not claimed."
        ),
        "rows": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "transitions",
        "max_direct_vs_overlay_logit_delta",
        "max_direct_vs_corrupted_stale_overlay_logit_delta",
        "max_full_recompute_vs_overlay_logit_delta",
        "top_token_match_rate_all_paths",
        "median_direct_current_splice_seconds",
        "median_append_only_overlay_seconds",
        "median_stale_physical_cache_bytes_retained",
        "report_sha256",
    ]}, indent=2))

    # Strict gate. If the underlying transformer implementation does not support
    # physical/logical position decoupling exactly, preserve the negative result.
    if report["top_token_match_rate_all_paths"] != 1.0:
        return 2
    if report["max_direct_vs_overlay_logit_delta"] > 0.02:
        return 3
    if report["max_direct_vs_corrupted_stale_overlay_logit_delta"] > 0.02:
        return 4
    if report["max_full_recompute_vs_overlay_logit_delta"] > 0.02:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
