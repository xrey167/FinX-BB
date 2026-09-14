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

OUT = Path(os.environ.get("SO_R321B_REPORT", "ci-qwen-r321b/report.json"))


def run_suffix(model, suffix_ids, prefix_cache, prefix_len):
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
    g = torch.Generator().manual_seed(seed)
    out = []
    for k, v in legacy_cache(cache):
        k2 = k.clone()
        v2 = v.clone()
        if stale_len:
            shape = k2[:, :, prefix_len:prefix_len + stale_len, :].shape
            k2[:, :, prefix_len:prefix_len + stale_len, :] = torch.randn(shape, generator=g, dtype=torch.float32).to(k2.dtype)
            shape = v2[:, :, prefix_len:prefix_len + stale_len, :].shape
            v2[:, :, prefix_len:prefix_len + stale_len, :] = torch.randn(shape, generator=g, dtype=torch.float32).to(v2.dtype)
        out.append((k2, v2))
    return tuple(out)


def admit_pages(physical_cache, page_table):
    """Materialize the logical KV working set from authority-admitted physical ranges.

    A production paged-attention kernel would consume the block/page table directly.
    The HF reference path concatenates admitted ranges so the transformer sees the
    exact clean logical geometry. Stale physical pages remain in the backing cache.
    """
    logical = []
    for k, v in legacy_cache(physical_cache):
        ks = [k[:, :, start:end, :] for start, end in page_table]
        vs = [v[:, :, start:end, :] for start, end in page_table]
        logical.append((torch.cat(ks, dim=2), torch.cat(vs, dim=2)))
    return tuple(logical)


def cache_bytes(cache) -> int:
    return int(sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in legacy_cache(cache)))


def fingerprint(cache) -> str:
    h = hashlib.sha256()
    for k, v in legacy_cache(cache):
        h.update(k.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


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
            "The CKCA runtime supplies an authoritative current value. Superseded KV pages may remain physically resident, "
            "but only authority-admitted pages belong to the logical neural working set."
        )},
        {"role": "user", "content": "Report the current governed value for canonical POD-KV-321B."},
    ]
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    static_text = base + "Current value:\n"
    prefix_ids = tok(static_text, add_special_tokens=False).input_ids
    S = len(prefix_ids)
    pt = torch.tensor([prefix_ids], dtype=torch.long)
    with torch.inference_mode():
        prefix_out = model(input_ids=pt, attention_mask=torch.ones_like(pt), use_cache=True, return_dict=True)
    clean_prefix_cache = legacy_cache(prefix_out.past_key_values)
    clean_prefix_fingerprint = fingerprint(clean_prefix_cache)

    stable = []
    for value in VALUE_STRINGS:
        full_text = static_text + value + "\nStatus:"
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if full_ids[:S] == prefix_ids:
            stable.append((value, full_ids[S:], full_ids))
    if len(stable) < 8:
        raise RuntimeError(f"too few prefix-stable values: {len(stable)}")
    probes = stable[:8]

    rows = []
    deltas = []
    corrupt_deltas = []
    full_deltas = []
    page_table_ns = []
    stale_bytes = []
    stale_fingerprint_changed = []

    for i in range(1, len(probes)):
        stale_value, stale_suffix, _ = probes[i - 1]
        current_value, current_suffix, current_full = probes[i]

        # Append the now-superseded generation to the backing cache.
        stale_out = run_suffix(model, stale_suffix, clean_prefix_cache, S)
        physical = legacy_cache(stale_out.past_key_values)
        stale_len = len(stale_suffix)
        before_corruption = fingerprint(physical)

        # Corrupt only superseded physical pages to make non-admission observable.
        corrupted_physical = mutate_stale_slots(physical, S, stale_len, 321100 + i)
        after_corruption = fingerprint(corrupted_physical)
        stale_fingerprint_changed.append(before_corruption != after_corruption)

        # Authority admits only the immutable live prefix. The expired suffix is
        # still physically present in `corrupted_physical` but absent from the page table.
        page_table = ((0, S),)
        t0 = time.perf_counter_ns()
        admitted = admit_pages(corrupted_physical, page_table)
        page_table_ns.append(time.perf_counter_ns() - t0)
        assert fingerprint(admitted) == clean_prefix_fingerprint

        direct = run_suffix(model, current_suffix, clean_prefix_cache, S)
        paged = run_suffix(model, current_suffix, admitted, S)
        direct_logits = direct.logits[0, -1].float()
        paged_logits = paged.logits[0, -1].float()

        # Also admit from the uncorrupted physical backing cache. Both page-table
        # views must be identical because the stale range is not admitted.
        admitted_uncorrupted = admit_pages(physical, page_table)
        paged_uncorrupted = run_suffix(model, current_suffix, admitted_uncorrupted, S)
        paged_uncorrupted_logits = paged_uncorrupted.logits[0, -1].float()

        ft = torch.tensor([current_full], dtype=torch.long)
        with torch.inference_mode():
            full = model(input_ids=ft, attention_mask=torch.ones_like(ft), use_cache=True, return_dict=True)
        full_logits = full.logits[0, -1].float()

        d = float((direct_logits - paged_logits).abs().max())
        dc = float((paged_uncorrupted_logits - paged_logits).abs().max())
        df = float((full_logits - paged_logits).abs().max())
        deltas.append(d)
        corrupt_deltas.append(dc)
        full_deltas.append(df)
        stale_bytes.append(cache_bytes(physical) - cache_bytes(admitted))
        rows.append({
            "transition": i,
            "stale_value": stale_value,
            "current_value": current_value,
            "stale_physical_tokens": stale_len,
            "page_table": page_table,
            "stale_backing_fingerprint_changed_by_corruption": stale_fingerprint_changed[-1],
            "admitted_prefix_matches_clean_fingerprint": fingerprint(admitted) == clean_prefix_fingerprint,
            "direct_vs_pagetable_max_logit_delta": d,
            "uncorrupted_vs_corrupted_backing_max_logit_delta": dc,
            "full_recompute_vs_pagetable_max_logit_delta": df,
        })

    report = {
        "stage": "R321B-GENERATION-AWARE-KV-PAGE-TABLE",
        "architecture_candidate": "Generation-Aware Neural Page Table (GANPT) for KV authority",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "transitions": len(rows),
        "immutable_prefix_tokens": S,
        "max_direct_vs_pagetable_logit_delta": max(deltas),
        "max_uncorrupted_vs_corrupted_backing_logit_delta": max(corrupt_deltas),
        "max_full_recompute_vs_pagetable_logit_delta": max(full_deltas),
        "all_stale_backing_corruptions_changed_physical_bytes": all(stale_fingerprint_changed),
        "median_page_table_materialization_ns_reference_python": statistics.median(page_table_ns),
        "median_unadmitted_stale_physical_bytes_retained": statistics.median(stale_bytes),
        "mechanism": (
            "The physical KV store is append-only/log-structured, while a generation-aware neural page table defines the logical "
            "attention working set. Expired generation pages may remain physically resident or even be corrupted, but they are not "
            "admitted into the logical KV sequence. Current repair runs on the exact clean logical geometry."
        ),
        "architectural_hypothesis": (
            "For neural caches, physical possession must not imply semantic authority. CKCA can make revocation fast by flipping "
            "generation/page admission first and reclaiming physical KV memory asynchronously. A production paged-attention kernel "
            "would consume page references directly rather than concatenate admitted ranges as this HF reference does."
        ),
        "dod_status": "NOT_DOD; real-model generation-aware KV page-table gate",
        "claim_boundary": (
            "Paged attention, block tables, cache indirection and MVCC are established. R321b tests generation-lifetime authority as "
            "the admission predicate for a neural KV page table inside CKCA; component-level novelty is not claimed."
        ),
        "rows": rows,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in [
        "stage", "transitions", "max_direct_vs_pagetable_logit_delta",
        "max_uncorrupted_vs_corrupted_backing_logit_delta",
        "max_full_recompute_vs_pagetable_logit_delta",
        "all_stale_backing_corruptions_changed_physical_bytes",
        "median_page_table_materialization_ns_reference_python",
        "median_unadmitted_stale_physical_bytes_retained",
        "report_sha256",
    ]}, indent=2))

    if max(deltas) != 0.0:
        return 2
    if max(corrupt_deltas) != 0.0:
        return 3
    if max(full_deltas) != 0.0:
        return 4
    if not all(stale_fingerprint_changed):
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
