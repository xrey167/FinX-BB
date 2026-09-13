from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
WEIGHT_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT = Path(os.environ.get("SO_R281_REPORT", "ci-qwen-r281/report.json"))
ENTITY = "Luma Harbor"
VALUES = ("red", "green", "blue", "yellow")

OPS = {
    "read": {
        "question": "What is Luma Harbor's registry color?",
        "candidates": VALUES,
        "expected": lambda v: v,
    },
    "is_red": {
        "question": "Is Luma Harbor's registry color red?",
        "candidates": ("Yes", "No"),
        "expected": lambda v: "Yes" if v == "red" else "No",
    },
    "membership": {
        "question": "Is Luma Harbor's registry color either red or yellow?",
        "candidates": ("Yes", "No"),
        "expected": lambda v: "Yes" if v in {"red", "yellow"} else "No",
    },
    "recode": {
        "question": "Encode Luma Harbor's registry color as A if it is red or green, otherwise B.",
        "candidates": ("A", "B"),
        "expected": lambda v: "A" if v in {"red", "green"} else "B",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def prefix_messages(value: str):
    return [{"role": "system", "content": f"Authoritative knowledge register: {ENTITY}'s registry color is {value}. Use this fact exactly."}]


def runtime_messages(question: str):
    return [{"role": "user", "content": question + " Answer only with the requested value, with no explanation."}]


def ids(tok, messages, *, generation: bool):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False)


def candidate_ids(tok, candidate: str):
    out = tok(candidate, add_special_tokens=False).input_ids
    if not out:
        raise RuntimeError(f"empty candidate tokenization: {candidate!r}")
    return out


def score_full(model, tok, full_prompt_ids, candidates):
    rows = {}
    for c in candidates:
        ci = candidate_ids(tok, c)
        x = torch.tensor([full_prompt_ids + ci], dtype=torch.long)
        mask = torch.ones_like(x)
        with torch.inference_mode():
            logits = model(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).logits[0]
        lp = F.log_softmax(logits.float(), dim=-1)
        base = len(full_prompt_ids)
        score = 0.0
        for j, token in enumerate(ci):
            score += float(lp[base + j - 1, token])
        rows[c] = score
    return rows


def score_cached(model, tok, query_ids, prefix_cache, prefix_len, candidates):
    rows = {}
    for c in candidates:
        ci = candidate_ids(tok, c)
        tail = query_ids + ci
        x = torch.tensor([tail], dtype=torch.long)
        mask = torch.ones((1, prefix_len + len(tail)), dtype=torch.long)
        pos = torch.arange(prefix_len, prefix_len + len(tail), dtype=torch.long).unsqueeze(0)
        cache_pos = torch.arange(prefix_len, prefix_len + len(tail), dtype=torch.long)
        cache = copy.deepcopy(prefix_cache)
        with torch.inference_mode():
            logits = model(
                input_ids=x,
                attention_mask=mask,
                position_ids=pos,
                cache_position=cache_pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            ).logits[0]
        lp = F.log_softmax(logits.float(), dim=-1)
        base = len(query_ids)
        score = 0.0
        for j, token in enumerate(ci):
            score += float(lp[base + j - 1, token])
        rows[c] = score
    return rows


def top(scores):
    return max(scores, key=scores.get)


def max_score_delta(a, b):
    return max(abs(a[k] - b[k]) for k in a)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    assert sha256_file(md / "model.safetensors") == WEIGHT_SHA
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    prefix_artifacts = {}
    for value in VALUES:
        pm = prefix_messages(value)
        pids = ids(tok, pm, generation=False)
        p = torch.tensor([pids], dtype=torch.long)
        pmask = torch.ones_like(p)
        with torch.inference_mode():
            po = model(input_ids=p, attention_mask=pmask, use_cache=True, return_dict=True)
        cache = po.past_key_values
        prefix_artifacts[value] = {"token_count": len(pids)}

        for op, spec in OPS.items():
            rm = runtime_messages(spec["question"])
            full_messages = pm + rm
            full_ids = ids(tok, full_messages, generation=True)
            if full_ids[: len(pids)] != pids:
                raise RuntimeError("chat-template prefix mismatch")
            qids = full_ids[len(pids):]
            expected = spec["expected"](value)
            text_scores = score_full(model, tok, full_ids, spec["candidates"])
            cache_scores = score_cached(model, tok, qids, cache, len(pids), spec["candidates"])
            text_top = top(text_scores)
            cache_top = top(cache_scores)
            rows.append({
                "revision_value": value,
                "operation": op,
                "expected": expected,
                "text_top": text_top,
                "cache_top": cache_top,
                "text_correct": text_top == expected,
                "cache_correct": cache_top == expected,
                "top_match": text_top == cache_top,
                "max_logprob_delta": max_score_delta(text_scores, cache_scores),
                "text_scores": text_scores,
                "cache_scores": cache_scores,
                "runtime_knowledge_text_tokens": 0,
            })

    # Counterfactual hot-swap audit: the same runtime query is scored under each compiled revision.
    by_op = {}
    for op in OPS:
        rr = [r for r in rows if r["operation"] == op]
        by_op[op] = {
            "n": len(rr),
            "text_accuracy": sum(r["text_correct"] for r in rr) / len(rr),
            "cache_accuracy": sum(r["cache_correct"] for r in rr) / len(rr),
            "text_cache_top_match": sum(r["top_match"] for r in rr) / len(rr),
            "max_logprob_delta": max(r["max_logprob_delta"] for r in rr),
        }
    all_text = sum(r["text_correct"] for r in rows) / len(rows)
    all_cache = sum(r["cache_correct"] for r in rows) / len(rows)
    all_match = sum(r["top_match"] for r in rows) / len(rows)
    report = {
        "stage": "R281-QWEN3-CAUSAL-KNOWLEDGE-REGISTER",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "knowledge_identity": ENTITY,
        "revision_values": list(VALUES),
        "prefix_artifacts": prefix_artifacts,
        "by_operation": by_op,
        "overall_text_accuracy": all_text,
        "overall_cache_accuracy": all_cache,
        "overall_text_cache_top_match": all_match,
        "rows": rows,
        "scientific_scope": "Real frozen Qwen3-0.6B. One query-independent causal prefix KV artifact is compiled per knowledge revision and reused unchanged across four different operations. Runtime receives no knowledge text and no per-fact gradients. This is an exact AOT-context reference interface, not a compact-memory novelty claim.",
        "dod_status": "NOT_DOD; exact operation-invariant causal register reference gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["overall_text_accuracy", "overall_cache_accuracy", "overall_text_cache_top_match", "by_operation"]}, indent=2))
    # Scientific gate: first require the privileged text context itself to solve at least 75%; cached must match text exactly at top-1 and scores closely.
    if all_text < 0.75:
        return 2
    if all_match != 1.0:
        return 3
    if max(r["max_logprob_delta"] for r in rows) > 1e-4:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
