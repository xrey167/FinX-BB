from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
WEIGHT_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT = Path(os.environ.get("SO_R286_REPORT", "ci-qwen-r286/report.json"))
ENTITY = "Luma Harbor"
VALUES = ("red", "green", "blue", "yellow")
OPS = {
    "read": ("What is Luma Harbor's registry color?", VALUES, lambda v: v),
    "is_red": ("Is Luma Harbor's registry color red?", ("Yes", "No"), lambda v: "Yes" if v == "red" else "No"),
    "membership": ("Is Luma Harbor's registry color either red or yellow?", ("Yes", "No"), lambda v: "Yes" if v in {"red", "yellow"} else "No"),
    "recode": ("Encode Luma Harbor's registry color as A if it is red or green, otherwise B.", ("A", "B"), lambda v: "A" if v in {"red", "green"} else "B"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def system_text(value: str) -> str:
    # Identity and relation are deliberately NOT duplicated into each neural page.
    # The trusted Selection Proof binds the page to the query's canonical slot.
    return (
        "Knowledge-VM contract: the authoritative value below belongs to the exact entity and relation "
        "selected for the user's next question. Treat it as current ground truth. Authoritative value: " + value
    )


def render_prefix(tok, value: str):
    msg = [{"role": "system", "content": system_text(value)}]
    return tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=False, enable_thinking=False)


def query_ids(tok, question: str):
    msg = [{"role": "user", "content": question + " Answer only with the requested value, with no explanation."}]
    return tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, enable_thinking=False)


def lcp(rows):
    n = min(len(r) for r in rows)
    i = 0
    while i < n and len({r[i] for r in rows}) == 1:
        i += 1
    return rows[0][:i]


def cache_kv(cache, idx):
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        return layer.keys, layer.values
    return cache.key_cache[idx], cache.value_cache[idx]


def slice_cache(cache, start: int, end: int | None = None):
    out = []
    for li in range(len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)):
        k, v = cache_kv(cache, li)
        out.append((k[:, :, start:end, :].contiguous(), v[:, :, start:end, :].contiguous()))
    return out


def assemble_cache(common_parts, tail_parts):
    out = DynamicCache()
    for li, ((kc, vc), (kt, vt)) in enumerate(zip(common_parts, tail_parts)):
        out.update(torch.cat([kc, kt], dim=2).contiguous(), torch.cat([vc, vt], dim=2).contiguous(), li)
    return out


def cache_bytes(parts):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in parts)


def candidate_token(tok, text):
    a = tok(text, add_special_tokens=False).input_ids
    if len(a) == 1:
        return int(a[0])
    b = tok(" " + text, add_special_tokens=False).input_ids
    if len(b) == 1:
        return int(b[0])
    raise RuntimeError((text, a, b))


def top_scores(tok, logits, candidates):
    scores = {c.casefold(): float(logits[candidate_token(tok, c)]) for c in candidates}
    return max(scores, key=scores.get), scores


def query_logits(model, qids, cache, past_len):
    q = torch.tensor([qids], dtype=torch.long)
    mask = torch.ones((1, past_len + len(qids)), dtype=torch.long)
    pos = torch.arange(past_len, past_len + len(qids), dtype=torch.long).unsqueeze(0)
    cache_pos = torch.arange(past_len, past_len + len(qids), dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=q, attention_mask=mask, position_ids=pos, cache_position=cache_pos,
                     past_key_values=copy.deepcopy(cache), use_cache=True, return_dict=True).logits[0, -1].float()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    assert sha256_file(md / "model.safetensors") == WEIGHT_SHA
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).eval()
    model.requires_grad_(False)

    prefixes = [render_prefix(tok, v) for v in VALUES]
    common_ids = lcp(prefixes)
    tails = [p[len(common_ids):] for p in prefixes]
    if not common_ids or any(not t for t in tails):
        raise RuntimeError({"common": len(common_ids), "tails": [len(t) for t in tails]})

    common = torch.tensor([common_ids], dtype=torch.long)
    with torch.inference_mode():
        co = model(input_ids=common, attention_mask=torch.ones_like(common), use_cache=True, return_dict=True)
    common_cache = co.past_key_values
    common_parts = slice_cache(common_cache, 0, len(common_ids))

    rows = []
    tail_bytes = []
    full_bytes = []
    for value, full_prefix, tail in zip(VALUES, prefixes, tails):
        # Reference full prefix from text.
        ft = torch.tensor([full_prefix], dtype=torch.long)
        with torch.inference_mode():
            fo = model(input_ids=ft, attention_mask=torch.ones_like(ft), use_cache=True, return_dict=True)
        full_cache = fo.past_key_values
        fp = slice_cache(full_cache, 0, len(full_prefix))
        full_bytes.append(cache_bytes(fp))

        # Compile only the fact-specific causal tail against the shared cache.
        tt = torch.tensor([tail], dtype=torch.long)
        plen = len(common_ids)
        mask = torch.ones((1, plen + len(tail)), dtype=torch.long)
        pos = torch.arange(plen, plen + len(tail), dtype=torch.long).unsqueeze(0)
        cpos = torch.arange(plen, plen + len(tail), dtype=torch.long)
        with torch.inference_mode():
            to = model(input_ids=tt, attention_mask=mask, position_ids=pos, cache_position=cpos,
                       past_key_values=copy.deepcopy(common_cache), use_cache=True, return_dict=True)
        full_from_tail = to.past_key_values
        tp = slice_cache(full_from_tail, plen, plen + len(tail))
        tail_bytes.append(cache_bytes(tp))
        assembled = assemble_cache(common_parts, tp)

        # Structural cache equality before semantic evaluation.
        max_cache_delta = 0.0
        for li in range(len(model.model.layers)):
            kref, vref = cache_kv(full_cache, li)
            kasm, vasm = cache_kv(assembled, li)
            max_cache_delta = max(max_cache_delta,
                                  float(torch.max(torch.abs(kref - kasm)).item()),
                                  float(torch.max(torch.abs(vref - vasm)).item()))

        for op, (question, candidates, expfn) in OPS.items():
            qids = query_ids(tok, question)
            ref = query_logits(model, qids, full_cache, len(full_prefix))
            got = query_logits(model, qids, assembled, len(full_prefix))
            rt, rs = top_scores(tok, ref, candidates)
            gt, gs = top_scores(tok, got, candidates)
            rows.append({
                "value": value, "operation": op, "expected": expfn(value),
                "full_top": rt, "assembled_top": gt,
                "full_correct": rt == expfn(value).casefold(),
                "assembled_correct": gt == expfn(value).casefold(),
                "top_match": rt == gt,
                "max_vocab_logit_delta": float(torch.max(torch.abs(ref - got)).item()),
                "max_cache_delta": max_cache_delta,
                "full_scores": rs, "assembled_scores": gs,
            })

    text_acc = sum(r["full_correct"] for r in rows) / len(rows)
    page_acc = sum(r["assembled_correct"] for r in rows) / len(rows)
    top_match = sum(r["top_match"] for r in rows) / len(rows)
    max_logit_delta = max(r["max_vocab_logit_delta"] for r in rows)
    max_cache_delta = max(r["max_cache_delta"] for r in rows)
    avg_full = sum(full_bytes) / len(full_bytes)
    avg_tail = sum(tail_bytes) / len(tail_bytes)
    shared = cache_bytes(common_parts)
    N = 100_000
    ratio_100k = (shared + N * avg_tail) / (N * avg_full)

    report = {
        "stage": "R286-QWEN3-SHARED-CAUSAL-NEURAL-PAGE",
        "model_id": MODEL_ID, "revision": REVISION, "weight_sha256": WEIGHT_SHA,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0, "runtime_knowledge_text_tokens": 0,
        "selection_binding": "identity_and_relation_external_authority_not_duplicated_in_neural_page",
        "common_prefix_tokens": len(common_ids),
        "fact_tail_tokens": [len(x) for x in tails],
        "shared_cache_bytes": shared,
        "mean_full_cache_bytes_per_fact": avg_full,
        "mean_fact_tail_bytes_per_fact": avg_tail,
        "amortized_cache_ratio_at_100k_facts": ratio_100k,
        "overall_full_accuracy": text_acc,
        "overall_page_accuracy": page_acc,
        "top_match_rate": top_match,
        "max_cache_delta": max_cache_delta,
        "max_vocab_logit_delta": max_logit_delta,
        "rows": rows,
        "scientific_scope": "Real frozen Qwen3-0.6B. Tests exact causal deduplication: one globally shared contract/template prefix cache plus only the fact-specific final causal KV tail per selected knowledge page. Identity/relation authority remains in the VM selection proof and is deliberately not redundantly encoded per page. No approximation and no per-fact training.",
        "dod_status": "NOT_DOD; exact shared-causal-page mechanism gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "common_prefix_tokens", "fact_tail_tokens", "shared_cache_bytes",
        "mean_full_cache_bytes_per_fact", "mean_fact_tail_bytes_per_fact",
        "amortized_cache_ratio_at_100k_facts", "overall_full_accuracy",
        "overall_page_accuracy", "top_match_rate", "max_cache_delta", "max_vocab_logit_delta")}, indent=2))
    if top_match != 1.0 or max_logit_delta > 1e-5 or max_cache_delta > 1e-5:
        return 2
    if ratio_100k >= 0.35:
        return 3
    if page_acc < 0.75:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
