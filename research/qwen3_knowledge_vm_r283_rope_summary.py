from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "a08cec3036ee1085a4863a0b730e5d3f1c2f8d04"
WEIGHT_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
OUT = Path(os.environ.get("SO_R283_REPORT", "ci-qwen-r283/report.json"))
ENTITY = "Luma Harbor"
VALUES = ("red", "green", "blue", "yellow")
SIZES = (1, 2, 4, 8, 16)
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


def ids(tok, messages, generation=False):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, enable_thinking=False)


def render(tok, value, question):
    pm = [{"role": "system", "content": f"Authoritative knowledge register: {ENTITY}'s registry color is {value}. Use this fact exactly."}]
    rm = [{"role": "user", "content": question + " Answer only with the requested value, with no explanation."}]
    pids = ids(tok, pm, False)
    fids = ids(tok, pm + rm, True)
    if fids[:len(pids)] != pids: raise RuntimeError("unstable prefix boundary")
    return pids, fids[len(pids):]


def cand_token(tok, text):
    a = tok(text, add_special_tokens=False).input_ids
    if len(a) == 1: return int(a[0])
    b = tok(" " + text, add_special_tokens=False).input_ids
    if len(b) == 1: return int(b[0])
    raise RuntimeError((text, a, b))


def top_scores(tok, logits, candidates):
    scores = {c.casefold(): float(logits[cand_token(tok, c)]) for c in candidates}
    return max(scores, key=scores.get), scores


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def cache_kv(cache, idx):
    if hasattr(cache, "layers"):
        layer = cache.layers[idx]
        return layer.keys, layer.values
    return cache.key_cache[idx], cache.value_cache[idx]


def rerotated_tail(model, full_cache, full_len, keep):
    keep = min(keep, full_len)
    old_pos = torch.arange(full_len - keep, full_len, dtype=torch.long).unsqueeze(0)
    new_pos = torch.arange(0, keep, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros((1, keep, model.config.hidden_size), dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        cos_old, sin_old = model.model.rotary_emb(dummy, old_pos)
        cos_new, sin_new = model.model.rotary_emb(dummy, new_pos)
    cos_old, sin_old = cos_old.unsqueeze(1), sin_old.unsqueeze(1)
    cos_new, sin_new = cos_new.unsqueeze(1), sin_new.unsqueeze(1)
    out = DynamicCache()
    for i in range(len(model.model.layers)):
        k, v = cache_kv(full_cache, i)
        kt = k[:, :, -keep:, :].contiguous()
        vt = v[:, :, -keep:, :].contiguous()
        unrot = kt * cos_old - rotate_half(kt) * sin_old
        knew = unrot * cos_new + rotate_half(unrot) * sin_new
        out.update(knew.contiguous(), vt, i)
    return out, keep


def query_logits(model, query_ids, cache, past_len):
    q = torch.tensor([query_ids], dtype=torch.long)
    mask = torch.ones((1, past_len + len(query_ids)), dtype=torch.long)
    pos = torch.arange(past_len, past_len + len(query_ids), dtype=torch.long).unsqueeze(0)
    cache_pos = torch.arange(past_len, past_len + len(query_ids), dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=q, attention_mask=mask, position_ids=pos, cache_position=cache_pos, past_key_values=copy.deepcopy(cache), use_cache=True, return_dict=True).logits[0, -1].float()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors"]))
    assert sha256_file(md / "model.safetensors") == WEIGHT_SHA
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager", trust_remote_code=False).eval()
    model.requires_grad_(False)

    rows = []
    for value in VALUES:
        pids, _ = render(tok, value, OPS["read"][0])
        p = torch.tensor([pids], dtype=torch.long)
        with torch.inference_mode():
            po = model(input_ids=p, attention_mask=torch.ones_like(p), use_cache=True, return_dict=True)
        full_cache = po.past_key_values
        L = len(pids)
        compact = {k: rerotated_tail(model, full_cache, L, k) for k in SIZES}
        for op, (question, candidates, expfn) in OPS.items():
            p2, qids = render(tok, value, question)
            assert p2 == pids
            ref_logits = query_logits(model, qids, full_cache, L)
            ref_top, ref_scores = top_scores(tok, ref_logits, candidates)
            for requested, (ccache, kk) in compact.items():
                logits = query_logits(model, qids, ccache, kk)
                top, scores = top_scores(tok, logits, candidates)
                rows.append({
                    "value": value,
                    "operation": op,
                    "expected": expfn(value),
                    "requested_slots": requested,
                    "kept_slots": kk,
                    "full_top": ref_top,
                    "compact_top": top,
                    "top_match": top == ref_top,
                    "compact_correct": top == expfn(value).casefold(),
                    "full_correct": ref_top == expfn(value).casefold(),
                    "max_vocab_logit_delta": float(torch.max(torch.abs(logits - ref_logits)).item()),
                    "full_scores": ref_scores,
                    "compact_scores": scores,
                })

    by_size = {}
    for k in SIZES:
        rr = [r for r in rows if r["requested_slots"] == k]
        by_size[str(k)] = {
            "n": len(rr),
            "top_match_rate": sum(r["top_match"] for r in rr) / len(rr),
            "compact_accuracy": sum(r["compact_correct"] for r in rr) / len(rr),
            "full_accuracy": sum(r["full_correct"] for r in rr) / len(rr),
            "median_logit_delta": float(torch.tensor([r["max_vocab_logit_delta"] for r in rr]).median().item()),
            "max_logit_delta": max(r["max_vocab_logit_delta"] for r in rr),
        }
    report = {
        "stage": "R283-QWEN3-ROPE-NORMALIZED-CAUSAL-SUMMARY",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "summary_by_slots": by_size,
        "rows": rows,
        "scientific_scope": "Real frozen Qwen3-0.6B. Tests whether the final causal fact-prefix KV states can act as compact query-independent summary slots after exact RoPE rebasing to a shorter virtual register. The reference is the full AOT fact-prefix cache; no training or answer-direction fitting occurs.",
        "dod_status": "NOT_DOD; compact causal summary mechanism gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(by_size, indent=2))
    # Mechanism passes only if some <=4-slot capsule preserves >=90% of full-cache top decisions.
    if max(by_size[str(k)]["top_match_rate"] for k in (1,2,4)) < 0.90:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
