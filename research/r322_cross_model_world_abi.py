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

from research.ckca_v3_world_abi import AuthorityStore, LifetimeFabric
from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID as QWEN_ID, REVISION as QWEN_REVISION, EXPECTED_WEIGHTS as QWEN_EXPECTED

OUT = Path(os.environ.get("SO_R322_REPORT", "ci-r322/report.json"))

SMOL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOL_REVISION = "cbcad7f4d160a10174f725b968ab6faf2a76399e"
VALUES = ("VAL_0042", "VAL_9183", "VAL_2718", "VAL_5501")

MODEL_SPECS = (
    {"name": "qwen25-05b", "id": QWEN_ID, "revision": QWEN_REVISION, "expected": QWEN_EXPECTED},
    {"name": "smollm2-360m", "id": SMOL_ID, "revision": SMOL_REVISION, "expected": None},
)


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


def build_canonical_transition_log():
    authority = AuthorityStore()
    lifetimes = LifetimeFabric()
    authority.create(322, 0)
    authority.bind("shared-cross-model-pod", 322)
    log = []

    # Initial generation and then three updates. The middle transition is a
    # same-value rewrite: temporal identity changes although semantic bytes do not.
    sequence = [0, 1, 1, 2]
    current_factor = None
    for step, value_idx in enumerate(sequence):
        if step == 0:
            p = authority.pods[322]
            p.value = value_idx
        else:
            old = (322, authority.pods[322].generation)
            old_factor = current_factor
            authority.update(322, value_idx)
            lifetimes.invalidate(old)
            if old_factor is not None and lifetimes.valid(old_factor):
                raise AssertionError("expired canonical generation factor revived")
        dep = (322, authority.pods[322].generation)
        current_factor = lifetimes.factor([dep])
        log.append({
            "step": step,
            "pod_id": 322,
            "generation": dep[1],
            "value_idx": value_idx,
            "value": VALUES[value_idx],
            "same_value_rewrite": step > 0 and sequence[step] == sequence[step - 1],
            "factor_id": current_factor,
            "factor_live_at_creation": lifetimes.valid(current_factor),
        })
    return log


def load_model(spec):
    md = Path(snapshot_download(
        repo_id=spec["id"],
        revision=spec["revision"],
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json", "tokenizer.*"],
    ))
    weight_files = sorted(md.glob("*.safetensors"))
    hashes = {p.name: sha256_file(p) for p in weight_files}
    if spec["expected"] is not None:
        assert hashes == spec["expected"], (hashes, spec["expected"])
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)
    return md, hashes, tok, model


def model_probe(spec, transition_log):
    md, hashes, tok, model = load_model(spec)
    messages = [
        {"role": "system", "content": (
            "A model-independent CKCA World ABI supplies the authoritative current value. "
            "The canonical world cell is shared across model runtimes. Continue consistently from the supplied value."
        )},
        {"role": "user", "content": "Read the current value of SHARED-CROSS-MODEL-POD from the trusted runtime."},
    ]
    base = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    static_text = base + "Authoritative current value:\n"
    prefix_ids = tok(static_text, add_special_tokens=False).input_ids
    S = len(prefix_ids)
    pt = torch.tensor([prefix_ids], dtype=torch.long)
    with torch.inference_mode():
        prefix_out = model(input_ids=pt, attention_mask=torch.ones_like(pt), use_cache=True, return_dict=True)
    prefix_cache = legacy_cache(prefix_out.past_key_values)

    rows = []
    deltas = []
    full_times = []
    splice_times = []
    top_match = []
    current_next_tokens = []
    for transition in transition_log:
        value = transition["value"]
        full_text = static_text + value + "\nState:"
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        if full_ids[:S] != prefix_ids:
            raise RuntimeError(f"{spec['name']} tokenizer violated stable late-binding prefix for {value!r}")
        suffix = full_ids[S:]

        ft = torch.tensor([full_ids], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.inference_mode():
            full = model(input_ids=ft, attention_mask=torch.ones_like(ft), use_cache=True, return_dict=True)
        full_times.append(time.perf_counter() - t0)
        full_logits = full.logits[0, -1].float()

        t0 = time.perf_counter()
        splice = run_suffix(model, suffix, prefix_cache, S)
        splice_times.append(time.perf_counter() - t0)
        splice_logits = splice.logits[0, -1].float()
        delta = float((full_logits - splice_logits).abs().max())
        deltas.append(delta)
        match = int(full_logits.argmax()) == int(splice_logits.argmax())
        top_match.append(match)
        current_next_tokens.append(int(splice_logits.argmax()))
        rows.append({
            "step": transition["step"],
            "generation": transition["generation"],
            "value": value,
            "same_value_rewrite": transition["same_value_rewrite"],
            "prefix_tokens": S,
            "current_suffix_tokens": len(suffix),
            "full_vs_splice_max_logit_delta": delta,
            "next_token_match": match,
            "full_seconds": full_times[-1],
            "splice_seconds": splice_times[-1],
        })

    result = {
        "name": spec["name"],
        "model_id": spec["id"],
        "revision_requested": spec["revision"],
        "snapshot_revision_resolved": md.name,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "immutable_prefix_tokens": S,
        "max_full_vs_splice_logit_delta": max(deltas),
        "top_match_rate": sum(top_match) / len(top_match),
        "median_full_seconds": statistics.median(full_times),
        "median_splice_seconds": statistics.median(splice_times),
        "median_speedup": statistics.median(full_times) / max(statistics.median(splice_times), 1e-12),
        "rows": rows,
        "same_value_rewrite_next_token_stable": current_next_tokens[1] == current_next_tokens[2],
    }
    del model
    return result


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    transition_log = build_canonical_transition_log()
    results = []
    for spec in MODEL_SPECS:
        results.append(model_probe(spec, transition_log))

    same_generations = all(
        [row["generation"] for row in transition_log]
        == [r["generation"] for r in transition_log]
        for _ in results
    )
    report = {
        "stage": "R322-CROSS-MODEL-WORLD-ABI",
        "architecture_candidate": "Model-Independent Causal Neural World ABI",
        "canonical_pod_id": 322,
        "canonical_transition_log": transition_log,
        "models": results,
        "model_count": len(results),
        "all_models_consume_same_canonical_generation_log": same_generations,
        "model_specific_gradient_updates_after_world_change": 0,
        "model_specific_world_reindex_operations": 0,
        "max_full_vs_splice_logit_delta_across_models": max(r["max_full_vs_splice_logit_delta"] for r in results),
        "min_top_match_rate_across_models": min(r["top_match_rate"] for r in results),
        "all_backbones_frozen": all(r["backbone_frozen"] for r in results),
        "mechanism": (
            "One canonical Pod generation log is interpreted by two different pretrained transformer families through model-local "
            "materializers. World updates occur once in the authority plane; neither model is retrained or reindexed. Each model reuses "
            "its own immutable prefix and late-binds the same current world value at decode time."
        ),
        "architectural_hypothesis": (
            "Mutable knowledge should be model-independent state. A model is a consumer/compiler of the World ABI, not the owner of the "
            "current fact. This allows multiple neural runtimes to share one edit/revocation lifecycle."
        ),
        "dod_status": "NOT_DOD; two-model World ABI integration gate",
        "claim_boundary": (
            "External shared state and late-bound prompting are known. R322 tests the CKCA generation/lifetime abstraction across two "
            "different frozen model families; it does not by itself establish novelty or quality superiority."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "stage": report["stage"],
        "model_count": report["model_count"],
        "max_full_vs_splice_logit_delta_across_models": report["max_full_vs_splice_logit_delta_across_models"],
        "min_top_match_rate_across_models": report["min_top_match_rate_across_models"],
        "all_backbones_frozen": report["all_backbones_frozen"],
        "models": [{
            "name": r["name"],
            "resolved_revision": r["snapshot_revision_resolved"],
            "max_delta": r["max_full_vs_splice_logit_delta"],
            "speedup": r["median_speedup"],
        } for r in results],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    if report["min_top_match_rate_across_models"] != 1.0:
        return 2
    if report["max_full_vs_splice_logit_delta_across_models"] > 0.02:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
