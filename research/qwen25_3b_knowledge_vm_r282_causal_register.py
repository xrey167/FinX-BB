from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_WEIGHTS = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
OUT = Path(os.environ.get("SO_R282_REPORT", "ci-qwen-r282/report.json"))
COLORS = ("red", "blue", "green", "yellow")
IDENTITY = "ZN6MDGADZD"
SPECS = ("eq0", "map01", "not_red", "conjunction", "new_labels", "read")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def question(spec: str) -> str:
    noun = f"entity {IDENTITY}"
    if spec == "eq0":
        return f"Is {noun} red? Return only yes or no."
    if spec == "map01":
        return f"For {noun}, return A if it is red or blue, otherwise return B. Return only A or B."
    if spec == "not_red":
        return f"Is {noun} not red? Return only yes or no."
    if spec == "conjunction":
        return f"Is {noun} neither red nor blue? Return only yes or no."
    if spec == "new_labels":
        return f"For {noun}, return circle if it is red or yellow, otherwise return square. Return only circle or square."
    if spec == "read":
        return f"What color is {noun}? Return only the color word."
    raise ValueError(spec)


def expected(spec: str, value: int) -> str:
    if spec == "read": return COLORS[value]
    if spec == "eq0": return "yes" if value == 0 else "no"
    if spec == "map01": return "a" if value in (0, 1) else "b"
    if spec == "not_red": return "no" if value == 0 else "yes"
    if spec == "conjunction": return "yes" if value not in (0, 1) else "no"
    if spec == "new_labels": return "circle" if value in (0, 3) else "square"
    raise ValueError(spec)


def candidates(spec: str):
    if spec == "read": return COLORS
    if spec in ("eq0", "not_red", "conjunction"): return ("yes", "no")
    if spec == "map01": return ("A", "B")
    if spec == "new_labels": return ("circle", "square")
    raise ValueError(spec)


def candidate_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        # Try a leading space only as a tokenizer compatibility fallback.
        ids2 = tok.encode(" " + s, add_special_tokens=False)
        if len(ids2) == 1:
            return int(ids2[0])
        raise RuntimeError(f"candidate is not one token: {s!r} -> {ids} / {ids2}")
    return int(ids[0])


def canonical_name(s: str) -> str:
    return s.strip().casefold()


def render(tok, value: int, spec: str):
    fact = f"Stored fact: the color of item {IDENTITY} is {COLORS[value]}.\n"
    q = question(spec)
    messages = [
        {"role": "system", "content": "Follow the requested answer format. Give only the requested single word, without explanation."},
        {"role": "user", "content": fact + q},
    ]
    full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    idx = full_text.find(q)
    if idx < 0:
        raise RuntimeError("query substring missing from rendered prompt")
    prefix_text = full_text[:idx]
    full_ids = tok(full_text, add_special_tokens=False).input_ids
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("token boundary is not causal-prefix stable")
    return full_ids, prefix_ids, full_ids[len(prefix_ids):], full_text, prefix_text


def top_from_logits(tok, logits, spec: str):
    cs = candidates(spec)
    scores = {canonical_name(c): float(logits[candidate_token(tok, c)]) for c in cs}
    return max(scores, key=scores.get), scores


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    md = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"]))
    verified = {name: sha256_file(md / name) for name in EXPECTED_WEIGHTS}
    assert verified == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, padding_side="left", trust_remote_code=False)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(md, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=False).eval()
    model.requires_grad_(False)

    # Compile the four revision prefixes once. All six operations use these same caches unchanged.
    prefix_ids_by_value = []
    for v in range(4):
        _, pids, _, _, _ = render(tok, v, "read")
        prefix_ids_by_value.append(pids)
    lengths = {len(x) for x in prefix_ids_by_value}
    if len(lengths) != 1:
        raise RuntimeError(f"prefix lengths differ across value revisions: {lengths}")
    L = len(prefix_ids_by_value[0])
    prefix_batch = torch.tensor(prefix_ids_by_value, dtype=torch.long)
    prefix_mask = torch.ones_like(prefix_batch)
    with torch.inference_mode():
        prefix_out = model(input_ids=prefix_batch, attention_mask=prefix_mask, use_cache=True, return_dict=True)
    compiled_cache = prefix_out.past_key_values

    rows = []
    for spec in SPECS:
        full_rows, query_rows = [], []
        rendered = []
        for v in range(4):
            fids, pids, qids, full_text, _ = render(tok, v, spec)
            if pids != prefix_ids_by_value[v]:
                raise RuntimeError("fact prefix changed with operation")
            full_rows.append(fids)
            query_rows.append(qids)
            rendered.append(full_text)
        if len({len(x) for x in full_rows}) != 1 or len({len(x) for x in query_rows}) != 1:
            raise RuntimeError("operation batch token lengths differ across revisions")
        full = torch.tensor(full_rows, dtype=torch.long)
        fmask = torch.ones_like(full)
        with torch.inference_mode():
            text_logits = model(input_ids=full, attention_mask=fmask, use_cache=False, return_dict=True).logits[:, -1, :].float()

        query = torch.tensor(query_rows, dtype=torch.long)
        qmask = torch.ones((4, L + query.shape[1]), dtype=torch.long)
        pos = torch.arange(L, L + query.shape[1], dtype=torch.long).unsqueeze(0).expand(4, -1)
        cache_pos = torch.arange(L, L + query.shape[1], dtype=torch.long)
        with torch.inference_mode():
            cached_logits = model(
                input_ids=query,
                attention_mask=qmask,
                position_ids=pos,
                cache_position=cache_pos,
                past_key_values=copy.deepcopy(compiled_cache),
                use_cache=True,
                return_dict=True,
            ).logits[:, -1, :].float()

        for v in range(4):
            tt, ts = top_from_logits(tok, text_logits[v], spec)
            ct, cs = top_from_logits(tok, cached_logits[v], spec)
            exp = expected(spec, v)
            delta = float(torch.max(torch.abs(text_logits[v] - cached_logits[v])).item())
            rows.append({
                "spec": spec,
                "value": v,
                "color": COLORS[v],
                "expected": exp,
                "text_top": tt,
                "cache_top": ct,
                "text_correct": tt == exp,
                "cache_correct": ct == exp,
                "top_match": tt == ct,
                "max_vocab_logit_delta": delta,
                "text_candidate_scores": ts,
                "cache_candidate_scores": cs,
            })

    by_spec = {}
    for spec in SPECS:
        rr = [r for r in rows if r["spec"] == spec]
        by_spec[spec] = {
            "n": len(rr),
            "text_accuracy": sum(r["text_correct"] for r in rr) / len(rr),
            "cache_accuracy": sum(r["cache_correct"] for r in rr) / len(rr),
            "top_match": sum(r["top_match"] for r in rr) / len(rr),
            "max_vocab_logit_delta": max(r["max_vocab_logit_delta"] for r in rr),
        }
    text_acc = sum(r["text_correct"] for r in rows) / len(rows)
    cache_acc = sum(r["cache_correct"] for r in rows) / len(rows)
    match = sum(r["top_match"] for r in rows) / len(rows)
    max_delta = max(r["max_vocab_logit_delta"] for r in rows)
    report = {
        "stage": "R282-QWEN25-3B-CAUSAL-KNOWLEDGE-REGISTER",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": verified,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "knowledge_identity": IDENTITY,
        "compiled_revision_count": 4,
        "compiled_prefix_token_count": L,
        "operations": list(SPECS),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "overall_text_accuracy": text_acc,
        "overall_cache_accuracy": cache_acc,
        "overall_text_cache_top_match": match,
        "max_vocab_logit_delta": max_delta,
        "by_operation": by_spec,
        "rows": rows,
        "scientific_scope": "Real frozen Qwen2.5-3B-Instruct. Exact CR1 operation family. The fact-only causal prompt prefix is compiled once per revision into a query-independent KV artifact, then reused unchanged across six operations. No per-fact gradients; no runtime fact text. This is a reference interface, not compact-memory novelty.",
        "dod_status": "NOT_DOD; strong-teacher operation-invariant causal register reference gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall_text_accuracy": text_acc, "overall_cache_accuracy": cache_acc, "top_match": match, "max_delta": max_delta, "by_operation": by_spec}, indent=2))
    if text_acc < 0.90: return 2
    if match != 1.0: return 3
    if max_delta > 0.02: return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
