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

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    COLORS,
    SPECS,
    candidates,
    candidate_token,
    concat_cache,
    legacy_cache,
    render,
    run_segment,
    sha256_file,
    slice_cache,
    zero_lane_like,
    mutated_lane,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R289_REPORT", "ci-qwen-r289/report.json"))


def cache_len(cache) -> int:
    return int(legacy_cache(cache)[0][0].shape[2])


def query_logits_at(model, query_ids, cache, past_mask, logical_start: int):
    q = torch.tensor([query_ids], dtype=torch.long)
    qmask = torch.ones((1, len(query_ids)), dtype=torch.long)
    mask = torch.cat([past_mask, qmask], dim=1)
    pos = torch.arange(logical_start, logical_start + len(query_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        out = model(
            input_ids=q,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=False,
            return_dict=True,
        )
    return out.logits[0, -1, :].float()


def top_candidate(tok, logits, spec: str):
    scores = {}
    for c in candidates(spec):
        scores[c.casefold()] = float(logits[candidate_token(tok, c)])
    return max(scores, key=scores.get)


def timed(fn, warmup: int = 4, repeats: int = 16):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return {
        "median_ms": statistics.median(xs) * 1000.0,
        "mean_ms": statistics.mean(xs) * 1000.0,
        "min_ms": min(xs) * 1000.0,
        "max_ms": max(xs) * 1000.0,
    }


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

    layouts = {(v, s): render(tok, v, s) for v in range(4) for s in SPECS}
    static_variants = {tuple(x["static_ids"]) for x in layouts.values()}
    if len(static_variants) != 1:
        raise RuntimeError("static prefix varied")
    static_ids = list(next(iter(static_variants)))
    S = len(static_ids)

    dynamic = {}
    for v in range(4):
        variants = {tuple(layouts[(v, s)]["dynamic_ids"]) for s in SPECS}
        if len(variants) != 1:
            raise RuntimeError(f"value {v} dynamic ids varied by operation")
        dynamic[v] = list(next(iter(variants)))
    widths = {len(x) for x in dynamic.values()}
    if len(widths) != 1:
        raise RuntimeError(f"slot width varied: {widths}")
    D = next(iter(widths))

    static_tensor = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        static_out = model(
            input_ids=static_tensor,
            attention_mask=torch.ones_like(static_tensor),
            use_cache=True,
            return_dict=True,
        )
    shared = legacy_cache(static_out.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    # Compile every generation at the SAME logical position interval [S, S+D).
    # Physical storage lanes are introduced only after RoPE has been applied.
    caps = {}
    for v in range(4):
        out = run_segment(model, dynamic[v], shared, shared_mask)
        prefix = legacy_cache(out.past_key_values)
        caps[v] = slice_cache(prefix, S, S + D)

    rows = []
    overlay_deltas = []
    stale_mutation_deltas = []
    reverse_overlay_deltas = []
    revoked_mutation_deltas = []
    revoked_vs_null_deltas = []
    top_matches = []

    transitions = [(0, 1), (1, 2), (2, 3), (3, 0)]
    logical_query_start = S + D

    for old_v, new_v in transitions:
        single = concat_cache(shared, caps[new_v])
        single_mask = torch.ones((1, S + D), dtype=torch.long)

        # Lane 0 stale, lane 1 authoritative. Both K/V capsules carry the SAME
        # logical RoPE positions; the physical extra lane does not advance time.
        overlay = concat_cache(shared, caps[old_v], caps[new_v])
        overlay_mask = torch.cat([
            shared_mask,
            torch.zeros((1, D), dtype=torch.long),
            torch.ones((1, D), dtype=torch.long),
        ], dim=1)
        mutated_overlay = concat_cache(
            shared, mutated_lane(caps[old_v], 9000 + old_v), caps[new_v]
        )

        # Reverse physical lane ordering to prove authority is not tied to slot index.
        reverse = concat_cache(shared, caps[new_v], caps[old_v])
        reverse_mask = torch.cat([
            shared_mask,
            torch.ones((1, D), dtype=torch.long),
            torch.zeros((1, D), dtype=torch.long),
        ], dim=1)

        # Revoked: both materialized generations hidden. Compare against a logical
        # empty-port execution with the same query position, and randomize stale bytes.
        revoked = concat_cache(shared, caps[old_v], caps[new_v])
        revoked_mut = concat_cache(
            shared,
            mutated_lane(caps[old_v], 10000 + old_v),
            mutated_lane(caps[new_v], 11000 + new_v),
        )
        revoked_mask = torch.cat([
            shared_mask,
            torch.zeros((1, D), dtype=torch.long),
            torch.zeros((1, D), dtype=torch.long),
        ], dim=1)
        null_mask = shared_mask

        for spec in SPECS:
            qids = layouts[(new_v, spec)]["query_ids"]
            exact = query_logits_at(model, qids, single, single_mask, logical_query_start)
            over = query_logits_at(model, qids, overlay, overlay_mask, logical_query_start)
            over_mut = query_logits_at(model, qids, mutated_overlay, overlay_mask, logical_query_start)
            rev_order = query_logits_at(model, qids, reverse, reverse_mask, logical_query_start)

            d = float(torch.max(torch.abs(exact - over)).item())
            sd = float(torch.max(torch.abs(over - over_mut)).item())
            rd = float(torch.max(torch.abs(exact - rev_order)).item())
            overlay_deltas.append(d)
            stale_mutation_deltas.append(sd)
            reverse_overlay_deltas.append(rd)
            exact_top = top_candidate(tok, exact, spec)
            overlay_top = top_candidate(tok, over, spec)
            top_matches.append(exact_top == overlay_top)

            r0 = query_logits_at(model, qids, revoked, revoked_mask, logical_query_start)
            r1 = query_logits_at(model, qids, revoked_mut, revoked_mask, logical_query_start)
            n0 = query_logits_at(model, qids, shared, null_mask, logical_query_start)
            rmd = float(torch.max(torch.abs(r0 - r1)).item())
            rnd = float(torch.max(torch.abs(r0 - n0)).item())
            revoked_mutation_deltas.append(rmd)
            revoked_vs_null_deltas.append(rnd)

            rows.append({
                "old_value": old_v,
                "old_color": COLORS[old_v],
                "new_value": new_v,
                "new_color": COLORS[new_v],
                "spec": spec,
                "exact_top": exact_top,
                "overlay_top": overlay_top,
                "top_match": exact_top == overlay_top,
                "max_vocab_delta_overlay_vs_single": d,
                "max_vocab_delta_after_randomizing_masked_stale_lane": sd,
                "max_vocab_delta_reverse_physical_lane_order": rd,
                "max_vocab_delta_revoked_stale_randomization": rmd,
                "max_vocab_delta_revoked_vs_logical_empty_port": rnd,
            })

    # Mechanism overhead: same logical query and answer, single vs dual physical lanes.
    bench_spec = "read"
    bench_new = 1
    bench_old = 0
    qids = layouts[(bench_new, bench_spec)]["query_ids"]
    single = concat_cache(shared, caps[bench_new])
    single_mask = torch.ones((1, S + D), dtype=torch.long)
    overlay = concat_cache(shared, caps[bench_old], caps[bench_new])
    overlay_mask = torch.cat([
        shared_mask,
        torch.zeros((1, D), dtype=torch.long),
        torch.ones((1, D), dtype=torch.long),
    ], dim=1)
    single_t = timed(lambda: query_logits_at(model, qids, single, single_mask, logical_query_start))
    overlay_t = timed(lambda: query_logits_at(model, qids, overlay, overlay_mask, logical_query_start))
    overhead = overlay_t["median_ms"] / single_t["median_ms"] - 1.0

    report = {
        "stage": "R289-POSITION-OVERLAY-GENERATION-PORTS",
        "architecture_candidate": "Position-Overlay Generation Ports (POGP)",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "static_tokens": S,
        "logical_port_width": D,
        "physical_generation_lanes": 2,
        "logical_generation_lanes": 1,
        "logical_query_start": logical_query_start,
        "physical_dual_lane_cache_length": S + 2 * D,
        "single_lane_cache_length": S + D,
        "top_match_rate_overlay_vs_single": sum(top_matches) / len(top_matches),
        "max_vocab_delta_overlay_vs_single": max(overlay_deltas),
        "max_vocab_delta_after_randomizing_masked_stale_lane": max(stale_mutation_deltas),
        "max_vocab_delta_reverse_physical_lane_order": max(reverse_overlay_deltas),
        "max_vocab_delta_revoked_stale_randomization": max(revoked_mutation_deltas),
        "max_vocab_delta_revoked_vs_logical_empty_port": max(revoked_vs_null_deltas),
        "timing_single_lane": single_t,
        "timing_dual_physical_lane": overlay_t,
        "median_dual_lane_mechanism_overhead_fraction": overhead,
        "mechanism": (
            "Compile every generation at the same logical RoPE coordinates, then store revisions "
            "in separate physical KV lanes. Authority is an attention mask applied before softmax. "
            "The query advances past one logical port width, not past the count of materialized revisions. "
            "Thus an update can keep stale K/V bytes resident without positional drift."
        ),
        "scientific_scope": (
            "Real frozen Qwen2.5-0.5B, four value revisions, six operations, full-vocabulary logit "
            "comparisons. This tests a lifecycle/position mechanism, not broad language quality or novelty."
        ),
        "rows": rows,
        "dod_status": "NOT_DOD; real-model position-overlay lifecycle gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "logical_port_width": D,
        "top_match_rate": report["top_match_rate_overlay_vs_single"],
        "max_overlay_delta": report["max_vocab_delta_overlay_vs_single"],
        "max_stale_mutation_delta": report["max_vocab_delta_after_randomizing_masked_stale_lane"],
        "max_reverse_order_delta": report["max_vocab_delta_reverse_physical_lane_order"],
        "max_revoked_mutation_delta": report["max_vocab_delta_revoked_stale_randomization"],
        "max_revoked_vs_null_delta": report["max_vocab_delta_revoked_vs_logical_empty_port"],
        "median_mechanism_overhead_fraction": overhead,
    }, indent=2))

    if report["top_match_rate_overlay_vs_single"] != 1.0:
        return 2
    if max(overlay_deltas) > 0.02:
        return 3
    if max(stale_mutation_deltas) > 0.02:
        return 4
    if max(reverse_overlay_deltas) > 0.02:
        return 5
    if max(revoked_mutation_deltas) > 0.02:
        return 6
    if max(revoked_vs_null_deltas) > 0.02:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
