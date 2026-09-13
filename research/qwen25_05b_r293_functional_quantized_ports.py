from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    candidate_token,
    concat_cache,
    legacy_cache,
    query_logits,
    run_segment,
    sha256_file,
    slice_cache,
)
from research.qwen25_05b_r288_shared_latent_port_codec import (
    MODEL_ID,
    REVISION,
    EXPECTED_WEIGHTS,
    VALUE_POOL,
)
from research.qwen25_05b_r290_quantized_residual_ports import mean_cache, quantize_cache

OUT = Path(os.environ.get("SO_R293_REPORT", "ci-qwen-r293/report.json"))
PORT_STEM = "Memory-port value:"
QUESTION_MARKER = "\nQuestion: "
SYSTEM = (
    "The memory-port value is authoritative mutable world state. Use it to answer the "
    "question. Follow the requested output vocabulary exactly and output one token only."
)

SET_HALF = set(VALUE_POOL[:16])
SET_ALT = set(VALUE_POOL[::2])
SET_HEAD = set(VALUE_POOL[:8])
SET_TAIL = set(VALUE_POOL[-8:])
SET_EDGE = set(VALUE_POOL[:4] + VALUE_POOL[-4:])

OPS = {
    "read": "Return the memory-port value itself, and nothing else.",
    "half": "Return A if the value is one of [" + ", ".join(VALUE_POOL[:16]) + "]; otherwise return B.",
    "alternating": "Return A if the value is one of [" + ", ".join(VALUE_POOL[::2]) + "]; otherwise return B.",
    "head8": "Return yes if the value is one of [" + ", ".join(VALUE_POOL[:8]) + "]; otherwise return no.",
    "tail8": "Return yes if the value is one of [" + ", ".join(VALUE_POOL[-8:]) + "]; otherwise return no.",
    "edge8": "Return circle if the value is one of [" + ", ".join(VALUE_POOL[:4] + VALUE_POOL[-4:]) + "]; otherwise return square.",
}


def expected(op: str, word: str) -> str:
    if op == "read":
        return word.casefold()
    if op == "half":
        return "a" if word in SET_HALF else "b"
    if op == "alternating":
        return "a" if word in SET_ALT else "b"
    if op == "head8":
        return "yes" if word in SET_HEAD else "no"
    if op == "tail8":
        return "yes" if word in SET_TAIL else "no"
    if op == "edge8":
        return "circle" if word in SET_EDGE else "square"
    raise ValueError(op)


def candidates(op: str, eligible):
    if op == "read":
        return tuple(eligible)
    if op in ("half", "alternating"):
        return ("A", "B")
    if op in ("head8", "tail8"):
        return ("yes", "no")
    if op == "edge8":
        return ("circle", "square")
    raise ValueError(op)


def boundary(offsets, char_pos: int) -> int:
    for i, (start, end) in enumerate(offsets):
        if start >= char_pos:
            return i
        if start < char_pos < end:
            return i
    return len(offsets)


def render(tok, word: str, op: str):
    dynamic = f" {word}."
    user = PORT_STEM + dynamic + QUESTION_MARKER + OPS[op]
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    stem_i = text.find(PORT_STEM)
    q_i = text.find(QUESTION_MARKER, stem_i)
    if stem_i < 0 or q_i < 0:
        raise RuntimeError("prompt markers missing")
    dyn_char = stem_i + len(PORT_STEM)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)
    d0 = boundary(offsets, dyn_char)
    q0 = boundary(offsets, q_i)
    if not (0 < d0 < q0 < len(ids)):
        raise RuntimeError(f"bad boundaries {d0=} {q0=} total={len(ids)}")
    return {
        "static_ids": ids[:d0],
        "dynamic_ids": ids[d0:q0],
        "query_ids": ids[q0:],
    }


def top(tok, logits, op: str, eligible):
    scores = {}
    for c in candidates(op, eligible):
        try:
            tid = candidate_token(tok, c)
        except RuntimeError:
            continue
        scores[c.casefold()] = float(logits[tid])
    if not scores:
        raise RuntimeError(f"no one-token candidates for {op}")
    return max(scores, key=scores.get)


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
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    layouts = {(w, op): render(tok, w, op) for w in VALUE_POOL for op in OPS}
    eligible = []
    groups = {}
    for word in VALUE_POOL:
        try:
            candidate_token(tok, word)
        except RuntimeError:
            continue
        static_variants = {tuple(layouts[(word, op)]["static_ids"]) for op in OPS}
        dyn_variants = {tuple(layouts[(word, op)]["dynamic_ids"]) for op in OPS}
        if len(static_variants) != 1 or len(dyn_variants) != 1:
            continue
        key = (next(iter(static_variants)), len(next(iter(dyn_variants))))
        groups.setdefault(key, []).append(word)
    if not groups:
        raise RuntimeError("no stable query-independent port group")
    group_key, eligible = max(groups.items(), key=lambda kv: len(kv[1]))
    eligible = list(eligible)
    if len(eligible) < 16:
        raise RuntimeError(f"stable group too small: {eligible}")

    static_ids = list(group_key[0])
    D = int(group_key[1])
    S = len(static_ids)
    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        sout = model(input_ids=st, attention_mask=torch.ones_like(st), use_cache=True, return_dict=True)
    shared = legacy_cache(sout.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    capsules = {}
    prefixes = {}
    for word in eligible:
        dyn = layouts[(word, "read")]["dynamic_ids"]
        out = run_segment(model, dyn, shared, shared_mask)
        pcache = legacy_cache(out.past_key_values)
        prefixes[word] = pcache
        capsules[word] = slice_cache(pcache, S, S + D)

    split = max(8, int(len(eligible) * 0.70))
    split = min(split, len(eligible) - 4)
    train_words = eligible[:split]
    holdout_words = eligible[split:]
    anchor = mean_cache([capsules[w] for w in train_words])
    prefix_mask = torch.ones((1, S + D), dtype=torch.long)

    compressed = {}
    storage = {}
    relerr = {}
    for word in holdout_words:
        qc, nbytes, err = quantize_cache(
            capsules[word], k_bits=3, v_bits=3, group_size=32,
            anchor=anchor, correction_topk=0,
        )
        compressed[word] = concat_cache(shared, qc)
        storage[word] = nbytes
        relerr[word] = err

    rows = []
    exact_semantic = []
    compressed_semantic = []
    top_matches = []
    deltas = []
    op_summary = {}
    for op in OPS:
        om = []
        oe = []
        oc = []
        od = []
        for word in holdout_words:
            qids = layouts[(word, op)]["query_ids"]
            exact_logits = query_logits(model, qids, prefixes[word], prefix_mask)
            comp_logits = query_logits(model, qids, compressed[word], prefix_mask)
            exact_top = top(tok, exact_logits, op, eligible)
            comp_top = top(tok, comp_logits, op, eligible)
            exp = expected(op, word)
            delta = float(torch.max(torch.abs(exact_logits - comp_logits)).item())
            match = exact_top == comp_top
            exact_ok = exact_top == exp
            comp_ok = comp_top == exp
            top_matches.append(match)
            exact_semantic.append(exact_ok)
            compressed_semantic.append(comp_ok)
            deltas.append(delta)
            om.append(match); oe.append(exact_ok); oc.append(comp_ok); od.append(delta)
            rows.append({
                "op": op,
                "word": word,
                "expected": exp,
                "exact_top": exact_top,
                "compressed_top": comp_top,
                "top_match": match,
                "exact_correct": exact_ok,
                "compressed_correct": comp_ok,
                "max_vocab_logit_delta": delta,
                "mean_tensor_relative_l2_error": relerr[word],
                "per_fact_bytes": storage[word],
            })
        op_summary[op] = {
            "top_match_rate": sum(om) / len(om),
            "exact_semantic_accuracy": sum(oe) / len(oe),
            "compressed_semantic_accuracy": sum(oc) / len(oc),
            "max_vocab_logit_delta": max(od),
        }

    full_bf16_bytes = sum(
        k.numel() * k.element_size() + v.numel() * v.element_size()
        for k, v in legacy_cache(capsules[holdout_words[0]])
    )
    per_fact_bytes = max(storage.values())
    report = {
        "stage": "R293-FUNCTIONAL-QUANTIZED-PORTS",
        "architecture_candidate": "Query-Independent Quantized Temporal Ports",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "codec": "train-only shared mean anchor + groupwise residual K3/V3, group=32",
        "per_fact_gradient_steps": 0,
        "train_words_for_anchor": train_words,
        "future_holdout_words": holdout_words,
        "operations": list(OPS),
        "overall_top_match_rate_vs_exact": sum(top_matches) / len(top_matches),
        "overall_exact_semantic_accuracy": sum(exact_semantic) / len(exact_semantic),
        "overall_compressed_semantic_accuracy": sum(compressed_semantic) / len(compressed_semantic),
        "max_vocab_logit_delta": max(deltas),
        "full_bf16_capsule_bytes": full_bf16_bytes,
        "compressed_per_fact_bytes": per_fact_bytes,
        "per_fact_storage_ratio": per_fact_bytes / full_bf16_bytes,
        "op_summary": op_summary,
        "rows": rows,
        "gate": (
            "A single query-independent compressed capsule for each unseen value must preserve the exact-cache "
            "winner across multiple queries; the capsule is not recomputed or retrained per operation."
        ),
        "functional_gate_pass": sum(top_matches) / len(top_matches) >= 0.98,
        "dod_status": "NOT_DOD; functional compact-port generalization gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "eligible": len(eligible),
        "train": len(train_words),
        "holdout": len(holdout_words),
        "top_match": report["overall_top_match_rate_vs_exact"],
        "exact_accuracy": report["overall_exact_semantic_accuracy"],
        "compressed_accuracy": report["overall_compressed_semantic_accuracy"],
        "storage_ratio": report["per_fact_storage_ratio"],
        "max_logit_delta": report["max_vocab_logit_delta"],
        "op_summary": op_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
