from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    cache_bytes,
    candidate_token,
    concat_cache,
    legacy_cache,
    query_logits,
    run_segment,
    sha256_file,
    slice_cache,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R288_REPORT", "ci-qwen-r288/report.json"))

PORT_STEM = "Memory-port value:"
QUERY = "What is the memory-port value? Return only the value word."
SYSTEM = (
    "The user provides an authoritative memory-port value followed by a question. "
    "Use the memory-port value as current world state. For the requested format, "
    "answer with only the single value word and no punctuation."
)

VALUE_POOL = (
    "red", "blue", "green", "yellow", "black", "white", "orange", "purple",
    "brown", "pink", "gray", "gold", "silver", "violet", "beige", "cyan",
    "circle", "square", "north", "south", "east", "west", "summer", "winter",
    "spring", "autumn", "alpha", "beta", "gamma", "delta", "small", "large",
)


def render(tok, word: str):
    user = f"{PORT_STEM} {word}. {QUERY}"
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    stem_idx = full_text.find(PORT_STEM)
    q_idx = full_text.find(QUERY)
    if stem_idx < 0 or q_idx < 0 or stem_idx >= q_idx:
        raise RuntimeError("prompt boundaries missing")
    static_end = stem_idx + len(PORT_STEM)
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)

    def token_boundary(char_pos: int) -> int:
        for i, (start, end) in enumerate(offsets):
            if start >= char_pos:
                return i
            if start < char_pos < end:
                return i
        return len(ids)

    s = token_boundary(static_end)
    q = token_boundary(q_idx)
    if not (0 < s < q < len(ids)):
        raise RuntimeError(f"bad boundaries s={s} q={q} total={len(ids)}")
    return {
        "full_ids": ids,
        "static_ids": ids[:s],
        "dynamic_ids": ids[s:q],
        "query_ids": ids[q:],
        "dynamic_text": full_text[static_end:q_idx],
    }


def flatten_cache(cache) -> torch.Tensor:
    pieces = []
    for k, v in legacy_cache(cache):
        pieces.append(k.float().reshape(-1))
        pieces.append(v.float().reshape(-1))
    return torch.cat(pieces, dim=0)


def unflatten_like(vector: torch.Tensor, template):
    out = []
    offset = 0
    for k, v in legacy_cache(template):
        nk = k.numel()
        nv = v.numel()
        kk = vector[offset: offset + nk].reshape_as(k).to(k.dtype).contiguous()
        offset += nk
        vv = vector[offset: offset + nv].reshape_as(v).to(v.dtype).contiguous()
        offset += nv
        out.append((kk, vv))
    if offset != vector.numel():
        raise RuntimeError("unflatten length mismatch")
    return tuple(out)


def candidate_scores(tok, logits, words):
    return {w: float(logits[candidate_token(tok, w)]) for w in words}


def top_word(tok, logits, words):
    scores = candidate_scores(tok, logits, words)
    return max(scores, key=scores.get), scores


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

    # Keep only words that have a single-token candidate representation; then choose
    # the largest prompt-tokenization group with identical static prefix and slot width.
    eligible = []
    rejected = {}
    groups = defaultdict(list)
    layouts = {}
    for word in VALUE_POOL:
        try:
            candidate_token(tok, word)
            layout = render(tok, word)
        except Exception as exc:
            rejected[word] = repr(exc)
            continue
        layouts[word] = layout
        key = (tuple(layout["static_ids"]), len(layout["dynamic_ids"]))
        groups[key].append(word)
    if not groups:
        raise RuntimeError("no tokenization group")
    group_key, eligible = max(groups.items(), key=lambda kv: len(kv[1]))
    eligible = list(eligible)
    if len(eligible) < 8:
        raise RuntimeError(f"largest stable slot group too small: {eligible}")

    static_ids = list(group_key[0])
    slot_width = int(group_key[1])
    S = len(static_ids)
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

    capsules = {}
    exact_prefix = {}
    vectors = {}
    for word in eligible:
        layout = layouts[word]
        if layout["static_ids"] != static_ids or len(layout["dynamic_ids"]) != slot_width:
            raise RuntimeError("stable group violated")
        out = run_segment(model, layout["dynamic_ids"], shared, shared_mask)
        prefix = legacy_cache(out.past_key_values)
        exact_prefix[word] = prefix
        capsule = slice_cache(prefix, S, S + slot_width)
        capsules[word] = capsule
        vectors[word] = flatten_cache(capsule)

    # Exact-cache/read baseline first; low baseline items stay visible in the report.
    exact_rows = []
    exact_correct = {}
    prefix_mask = torch.ones((1, S + slot_width), dtype=torch.long)
    for word in eligible:
        logits = query_logits(model, layouts[word]["query_ids"], exact_prefix[word], prefix_mask)
        top, scores = top_word(tok, logits, eligible)
        exact_correct[word] = top == word
        exact_rows.append({
            "word": word,
            "top": top,
            "correct": top == word,
            "target_score": scores[word],
            "top_score": scores[top],
        })

    # Deterministic future-world split: basis is learned only from earlier words.
    split = max(6, int(len(eligible) * 0.70))
    split = min(split, len(eligible) - 2)
    train_words = eligible[:split]
    holdout_words = eligible[split:]
    x_train = torch.stack([vectors[w] for w in train_words], dim=0)
    mean = x_train.mean(dim=0)
    centered = x_train - mean
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    max_rank = max(1, min(vh.shape[0], len(train_words) - 1))
    ranks = sorted({r for r in (1, 2, 4, 8, max_rank) if r <= max_rank})

    flat_dim = int(mean.numel())
    full_capsule_bytes = cache_bytes(capsules[eligible[0]])
    rank_reports = {}
    rows = []

    for rank in ranks:
        # Storage-aware codec: shared mean/basis bfloat16, per-fact coefficients fp16.
        mean_q = mean.to(torch.bfloat16).float()
        basis_q = vh[:rank].to(torch.bfloat16).float()
        top_matches = []
        correct_all = []
        correct_where_exact = []
        rel_errors = []
        logit_deltas = []
        for word in holdout_words:
            x = vectors[word]
            coeff = ((x - mean_q) @ basis_q.T).to(torch.float16).float()
            recon = mean_q + coeff @ basis_q
            rel_error = float(torch.linalg.vector_norm(recon - x) / torch.linalg.vector_norm(x))
            rel_errors.append(rel_error)
            reconstructed_capsule = unflatten_like(recon, capsules[word])
            cache = concat_cache(shared, reconstructed_capsule)
            comp_logits = query_logits(model, layouts[word]["query_ids"], cache, prefix_mask)
            exact_logits = query_logits(model, layouts[word]["query_ids"], exact_prefix[word], prefix_mask)
            comp_top, _ = top_word(tok, comp_logits, eligible)
            exact_top, _ = top_word(tok, exact_logits, eligible)
            top_matches.append(comp_top == exact_top)
            correct_all.append(comp_top == word)
            if exact_correct[word]:
                correct_where_exact.append(comp_top == word)
            delta = float(torch.max(torch.abs(comp_logits - exact_logits)).item())
            logit_deltas.append(delta)
            rows.append({
                "rank": rank,
                "word": word,
                "exact_top": exact_top,
                "compressed_top": comp_top,
                "exact_correct": exact_correct[word],
                "compressed_correct": comp_top == word,
                "top_match": comp_top == exact_top,
                "relative_capsule_l2_error": rel_error,
                "max_vocab_logit_delta": delta,
            })

        coeff_bytes = rank * 2
        shared_codec_bytes = (rank + 1) * flat_dim * 2
        def amortized_ratio(nfacts: int) -> float:
            return (shared_codec_bytes + nfacts * coeff_bytes) / (nfacts * full_capsule_bytes)

        rank_reports[str(rank)] = {
            "rank": rank,
            "holdout_count": len(holdout_words),
            "top_match_rate_vs_exact": sum(top_matches) / len(top_matches),
            "compressed_accuracy": sum(correct_all) / len(correct_all),
            "compressed_accuracy_conditioned_on_exact_baseline": (
                sum(correct_where_exact) / len(correct_where_exact)
                if correct_where_exact else None
            ),
            "mean_relative_capsule_l2_error": sum(rel_errors) / len(rel_errors),
            "max_vocab_logit_delta": max(logit_deltas),
            "per_fact_code_bytes": coeff_bytes,
            "full_capsule_bytes": full_capsule_bytes,
            "per_fact_code_ratio_excluding_shared_basis": coeff_bytes / full_capsule_bytes,
            "shared_codec_bytes": shared_codec_bytes,
            "amortized_storage_ratio_at_1k_facts": amortized_ratio(1_000),
            "amortized_storage_ratio_at_1m_facts": amortized_ratio(1_000_000),
        }

    best_rank = max(
        ranks,
        key=lambda r: (
            rank_reports[str(r)]["top_match_rate_vs_exact"],
            rank_reports[str(r)]["compressed_accuracy_conditioned_on_exact_baseline"] or -1,
            -r,
        ),
    )
    exact_accuracy = sum(exact_correct.values()) / len(exact_correct)
    holdout_exact_accuracy = sum(exact_correct[w] for w in holdout_words) / len(holdout_words)

    report = {
        "stage": "R288-SHARED-LATENT-PORT-CODEC",
        "architecture_candidate": "Shared Low-Rank Latent Port Codec",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_fact_text_tokens": 0,
        "write_path": "one frozen-model fact forward -> projection onto fixed shared basis -> fp16 coefficients",
        "read_path": "fp16 coefficients + shared bf16 basis -> reconstructed query-independent generation capsule",
        "eligible_words": eligible,
        "rejected_words": rejected,
        "stable_dynamic_slot_width": slot_width,
        "train_words_for_shared_basis": train_words,
        "future_holdout_words": holdout_words,
        "flat_capsule_dimension": flat_dim,
        "full_capsule_bytes": full_capsule_bytes,
        "exact_cache_accuracy_all": exact_accuracy,
        "exact_cache_accuracy_holdout": holdout_exact_accuracy,
        "ranks": rank_reports,
        "best_rank_by_top_match": best_rank,
        "exact_rows": exact_rows,
        "rows": rows,
        "scientific_scope": (
            "Real frozen Qwen2.5-0.5B. The shared codec basis is fit only on the train-word "
            "capsules; future held-out value words receive no basis retraining and no gradient updates. "
            "This probes whether model-native per-fact KV capsules occupy a low-dimensional shared "
            "subspace. It is a compactness experiment, not a novelty or RAG-superiority claim."
        ),
        "dod_status": "NOT_DOD; compact latent payload gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "eligible_count": len(eligible),
        "slot_width": slot_width,
        "train_count": len(train_words),
        "holdout_count": len(holdout_words),
        "exact_accuracy_all": exact_accuracy,
        "exact_accuracy_holdout": holdout_exact_accuracy,
        "best_rank": best_rank,
        "best": rank_reports[str(best_rank)],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
