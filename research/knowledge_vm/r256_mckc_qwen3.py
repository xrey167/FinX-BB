from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
OUT = Path("r256_results.json")

FACTS = {
    "Luma Harbor": "red",
    "Neris Point": "green",
    "Vexa Port": "blue",
    "Tarin Quay": "yellow",
}


def queries(entity: str, value: str):
    direct = (f"What is {entity}'s registry color? Answer with only the color.", value)
    is_red = (f"Is {entity}'s registry color red? Answer only Yes or No.", "Yes" if value == "red" else "No")
    member = (f"Is {entity}'s registry color either red or yellow? Answer only Yes or No.", "Yes" if value in {"red", "yellow"} else "No")
    recode = (f"Encode {entity}'s registry color as A if it is red or green, otherwise B. Answer only A or B.", "A" if value in {"red", "green"} else "B")
    return {"read": direct, "is_red": is_red, "membership": member, "recode": recode}


def cache_from_selected(full_cache, indices):
    new = DynamicCache()
    for layer_idx, layer in enumerate(full_cache.layers):
        k = layer.keys[:, :, indices, :].contiguous()
        v = layer.values[:, :, indices, :].contiguous()
        new.update(k, v, layer_idx)
    return new


def greedy_with_cache(model, tokenizer, query_ids, cache, prefix_position_offset: int, max_new_tokens=6):
    device = next(model.parameters()).device
    q = torch.tensor([query_ids], dtype=torch.long, device=device)
    past_len = cache.get_seq_length()
    attention_mask = torch.ones((1, past_len + q.shape[1]), dtype=torch.long, device=device)
    position_ids = torch.arange(prefix_position_offset, prefix_position_offset + q.shape[1], device=device).unsqueeze(0)
    with torch.inference_mode():
        out = model(input_ids=q, attention_mask=attention_mask, position_ids=position_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1]
    generated = []
    next_pos = prefix_position_offset + q.shape[1]
    for _ in range(max_new_tokens):
        token = int(logits.argmax(-1).item())
        generated.append(token)
        if token in {tokenizer.eos_token_id, getattr(tokenizer, "im_end_id", -1)}:
            break
        inp = torch.tensor([[token]], device=device)
        attention_mask = torch.ones((1, cache.get_seq_length() + 1), dtype=torch.long, device=device)
        pos = torch.tensor([[next_pos]], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(input_ids=inp, attention_mask=attention_mask, position_ids=pos, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1]
        next_pos += 1
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated


def exact_answer(text: str, expected: str) -> bool:
    t = text.strip().strip(".!,;:").casefold()
    return t == expected.casefold()


def main():
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, torch_dtype=torch.float32, trust_remote_code=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = []
    cache_sizes = {}
    decoded_prefixes = {}

    for entity, value in FACTS.items():
        system = f"Authoritative knowledge register: {entity}'s registry color is {value}."
        system_msgs = [{"role": "system", "content": system}]
        prefix_ids = tokenizer.apply_chat_template(system_msgs, tokenize=True, add_generation_prompt=False)
        prefix = torch.tensor([prefix_ids], dtype=torch.long)
        with torch.inference_mode():
            pref_out = model(input_ids=prefix, use_cache=True)
        full_cache = pref_out.past_key_values
        L = len(prefix_ids)
        decoded_prefixes[entity] = [tokenizer.decode([x], skip_special_tokens=False) for x in prefix_ids]
        cache_sizes[entity] = L

        for op, (question, expected) in queries(entity, value).items():
            full_msgs = system_msgs + [{"role": "user", "content": question}]
            full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=True)
            if full_ids[:L] != prefix_ids:
                raise RuntimeError("chat template prefix is not stable")
            query_ids = full_ids[L:]

            # Full text reference.
            full_tensor = torch.tensor([full_ids], dtype=torch.long)
            with torch.inference_mode():
                ref = model.generate(full_tensor, max_new_tokens=6, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            ref_text = tokenizer.decode(ref[0, len(full_ids):].tolist(), skip_special_tokens=True).strip()
            rows.append({"entity": entity, "value": value, "op": op, "mode": "text", "text": ref_text, "expected": expected, "correct": exact_answer(ref_text, expected)})

            # Full AOT cache should reproduce the text prefix without runtime fact tokens.
            aot_text, _ = greedy_with_cache(model, tokenizer, query_ids, copy.deepcopy(full_cache), L)
            rows.append({"entity": entity, "value": value, "op": op, "mode": "kv_full", "text": aot_text, "expected": expected, "correct": exact_answer(aot_text, expected), "matches_text": aot_text == ref_text})

            for k in [1, 2, 4, 8, 16]:
                kk = min(k, L)
                indices = list(range(L - kk, L))
                compact = cache_from_selected(full_cache, indices)
                text, _ = greedy_with_cache(model, tokenizer, query_ids, compact, L)
                rows.append({"entity": entity, "value": value, "op": op, "mode": f"kv_tail_{kk}", "text": text, "expected": expected, "correct": exact_answer(text, expected), "matches_text": text == ref_text})

    summary = {}
    for mode in sorted({r["mode"] for r in rows}):
        rr = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "n": len(rr),
            "correct": sum(r["correct"] for r in rr),
            "accuracy": sum(r["correct"] for r in rr) / len(rr),
            "matches_text": sum(bool(r.get("matches_text")) for r in rr if "matches_text" in r),
        }
    result = {
        "model": MODEL_ID,
        "revision": REVISION,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "facts": FACTS,
        "prefix_lengths": cache_sizes,
        "summary": summary,
        "rows": rows,
        "decoded_prefix_tokens": decoded_prefixes,
        "claim_boundary": "Real pretrained causal decoder experiment. Full AOT KV is a context-cache baseline, not a novelty claim. Tail-KV is an exploratory compact operation-independent capsule test.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
