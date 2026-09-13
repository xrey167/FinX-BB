from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    concat_cache,
    legacy_cache,
    mutated_lane,
    query_logits,
    sha256_file,
    slice_cache,
    candidate_token,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
EXPECTED_WEIGHTS = {
    "model.safetensors": "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342",
}
OUT = Path(os.environ.get("SO_R292_REPORT", "ci-qwen-r292/report.json"))
COLORS = ("red", "blue", "green", "yellow")
SYSTEM = (
    "A memory port provides authoritative mutable world state. Read it carefully. "
    "The user may include neutral memo slots after the port. At the final question, "
    "answer only with the authoritative color word."
)
QUERY = "What color is the authoritative memory-port value? Return only the color word."


def token_boundary(offsets, char_pos: int) -> int:
    for i, (start, end) in enumerate(offsets):
        if start >= char_pos:
            return i
        if start < char_pos < end:
            return i
    return len(offsets)


def render(tok, color: str, memo_count: int):
    stem = "Memory-port value:"
    dynamic = f" {color}."
    memo = (
        " Memorize the authoritative value above internally. Do not restate it. "
        + " ".join(["[MEMO]"] * memo_count)
        + " "
    )
    user = stem + dynamic + memo + QUERY
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    stem_i = text.find(stem)
    if stem_i < 0:
        raise RuntimeError("stem missing")
    dynamic_start_char = stem_i + len(stem)
    memo_start_char = dynamic_start_char + len(dynamic)
    query_start_char = text.find(QUERY)
    if query_start_char < memo_start_char:
        raise RuntimeError("query boundary missing")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)
    d0 = token_boundary(offsets, dynamic_start_char)
    m0 = token_boundary(offsets, memo_start_char)
    q0 = token_boundary(offsets, query_start_char)
    if not (0 < d0 < m0 < q0 < len(ids)):
        raise RuntimeError(f"bad boundaries {d0=} {m0=} {q0=} total={len(ids)}")
    return {
        "ids": ids,
        "static_ids": ids[:d0],
        "dynamic_ids": ids[d0:m0],
        "memo_ids": ids[m0:q0],
        "query_ids": ids[q0:],
        "static_n": d0,
        "dynamic_n": m0 - d0,
        "memo_n": q0 - m0,
        "prefix_n": q0,
    }


def top_color(tok, logits):
    scores = {c: float(logits[candidate_token(tok, c)]) for c in COLORS}
    return max(scores, key=scores.get), scores


def full_prefix_cache(model, prefix_ids):
    x = torch.tensor([prefix_ids], dtype=torch.long)
    with torch.inference_mode():
        out = model(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            use_cache=True,
            return_dict=True,
        )
    return legacy_cache(out.past_key_values)


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

    rows = []
    stale_effect_rows = []
    masked_mutation_deltas = []
    closure_oracle_deltas = []
    naive_oracle_deltas = []
    closure_top_matches = []
    naive_top_matches = []
    oracle_correct = []
    memo_counts = (4, 16, 48)
    transitions = (("red", "blue"), ("blue", "green"), ("green", "yellow"), ("yellow", "red"))

    for memo_count in memo_counts:
        layouts = {c: render(tok, c, memo_count) for c in COLORS}
        ref = layouts[COLORS[0]]
        for c in COLORS[1:]:
            x = layouts[c]
            if x["static_ids"] != ref["static_ids"]:
                raise RuntimeError("static tokenization changed across colors")
            if x["dynamic_n"] != ref["dynamic_n"] or x["memo_ids"] != ref["memo_ids"]:
                raise RuntimeError("slot or memo tokenization changed across colors")
            if x["query_ids"] != ref["query_ids"]:
                raise RuntimeError("query tokenization changed across colors")
        S = ref["static_n"]
        D = ref["dynamic_n"]
        M = ref["memo_n"]
        P = ref["prefix_n"]

        caches = {}
        for color in COLORS:
            caches[color] = full_prefix_cache(model, layouts[color]["ids"][:P])

        for old_color, new_color in transitions:
            old = caches[old_color]
            new = caches[new_color]
            static = slice_cache(new, 0, S)
            new_field = slice_cache(new, S, S + D)
            stale_memo = slice_cache(old, S + D, S + D + M)
            oracle_memo = slice_cache(new, S + D, S + D + M)

            # Naive edit: only authoritative field K/V is replaced; downstream memo K/V
            # written while the old generation was live remains untouched.
            naive = concat_cache(static, new_field, stale_memo)
            all_on = torch.ones((1, P), dtype=torch.long)

            # Lifetime closure: stale memo positions are retained physically, but tagged
            # with dependency (pod,g_old) and masked when that generation is no longer live.
            closure_mask = torch.cat([
                torch.ones((1, S + D), dtype=torch.long),
                torch.zeros((1, M), dtype=torch.long),
            ], dim=1)
            stale_memo_mut = mutated_lane(stale_memo, seed=1000 + memo_count * 10 + COLORS.index(old_color))
            closure_mut = concat_cache(static, new_field, stale_memo_mut)

            # Oracle full reprefill carries the freshly recomputed memo. We also compute a
            # structural oracle with memo masked; this isolates lifecycle semantics from the
            # model's optional use of memo-slot representations.
            oracle = new
            oracle_masked = concat_cache(static, new_field, oracle_memo)

            qids = ref["query_ids"]
            oracle_logits = query_logits(model, qids, oracle, all_on)
            naive_logits = query_logits(model, qids, naive, all_on)
            closure_logits = query_logits(model, qids, naive, closure_mask)
            closure_mut_logits = query_logits(model, qids, closure_mut, closure_mask)
            oracle_masked_logits = query_logits(model, qids, oracle_masked, closure_mask)

            oracle_top, _ = top_color(tok, oracle_logits)
            naive_top, _ = top_color(tok, naive_logits)
            closure_top, _ = top_color(tok, closure_logits)
            oracle_masked_top, _ = top_color(tok, oracle_masked_logits)

            naive_delta = float(torch.max(torch.abs(naive_logits - oracle_logits)).item())
            closure_delta = float(torch.max(torch.abs(closure_logits - oracle_masked_logits)).item())
            mutation_delta = float(torch.max(torch.abs(closure_logits - closure_mut_logits)).item())
            naive_oracle_deltas.append(naive_delta)
            closure_oracle_deltas.append(closure_delta)
            masked_mutation_deltas.append(mutation_delta)
            oracle_correct.append(oracle_top == new_color)
            naive_top_matches.append(naive_top == oracle_top)
            closure_top_matches.append(closure_top == oracle_masked_top)

            stale_effect = naive_top != oracle_top or naive_delta > 0.25
            if stale_effect:
                stale_effect_rows.append((memo_count, old_color, new_color, naive_delta, oracle_top, naive_top))
            rows.append({
                "memo_count": memo_count,
                "memo_tokens": M,
                "old_color": old_color,
                "new_color": new_color,
                "oracle_top": oracle_top,
                "naive_field_swap_top": naive_top,
                "lifetime_closure_top": closure_top,
                "masked_fresh_oracle_top": oracle_masked_top,
                "oracle_correct": oracle_top == new_color,
                "naive_matches_oracle_top": naive_top == oracle_top,
                "closure_matches_masked_oracle_top": closure_top == oracle_masked_top,
                "naive_vs_full_recompute_max_vocab_delta": naive_delta,
                "closure_vs_fresh_masked_oracle_max_vocab_delta": closure_delta,
                "masked_stale_memo_randomization_max_vocab_delta": mutation_delta,
                "stale_downstream_effect_detected": stale_effect,
            })

    report = {
        "stage": "R292-TRANSITIVE-LIFETIME-CLOSURE",
        "architecture_candidate": "Lifetime-Carrying KV / Transitive Generation Closure",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "oracle_accuracy": sum(oracle_correct) / len(oracle_correct),
        "naive_field_swap_top_match_vs_full_recompute": sum(naive_top_matches) / len(naive_top_matches),
        "lifetime_closure_top_match_vs_fresh_masked_oracle": sum(closure_top_matches) / len(closure_top_matches),
        "max_naive_vs_full_recompute_vocab_delta": max(naive_oracle_deltas),
        "max_closure_vs_fresh_masked_oracle_vocab_delta": max(closure_oracle_deltas),
        "max_masked_stale_memo_randomization_vocab_delta": max(masked_mutation_deltas),
        "cases_with_detectable_stale_downstream_effect": len(stale_effect_rows),
        "total_cases": len(rows),
        "mechanism": (
            "Every cached state carries the generations it causally depended on. A cache entry is admissible "
            "only while all dependency generations remain authoritative. On update/revoke, downstream neural "
            "notes written under the old generation are masked transitively, not merely the source field K/V. "
            "This directly targets resurrection through downstream prefill notes."
        ),
        "dependency_rule": "valid(cache_entry)=AND_{(pod,g) in deps(entry)} authority[pod]==g AND live[pod]",
        "scientific_scope": (
            "Real frozen Qwen2.5-0.5B with neutral downstream memo slots. The gate separately compares naive "
            "source-only KV replacement, full reprefill, and transitive masking. Exact masked-byte mutation "
            "invariance is a causal property; broad natural-language quality and scalable dependency tracking "
            "remain future gates."
        ),
        "prior_art_pressure": (
            "Programmable-KV (arXiv:2606.17107) shows that downstream prefill notes can retain old field-conditioned "
            "conclusions after source KV replacement. R292 therefore treats dependency closure, not source KV editing, "
            "as the required lifecycle primitive."
        ),
        "rows": rows,
        "dod_status": "NOT_DOD; transitive anti-resurrection gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "oracle_accuracy": report["oracle_accuracy"],
        "naive_top_match": report["naive_field_swap_top_match_vs_full_recompute"],
        "closure_top_match": report["lifetime_closure_top_match_vs_fresh_masked_oracle"],
        "max_naive_delta": report["max_naive_vs_full_recompute_vocab_delta"],
        "max_closure_delta": report["max_closure_vs_fresh_masked_oracle_vocab_delta"],
        "max_masked_mutation_delta": report["max_masked_stale_memo_randomization_vocab_delta"],
        "stale_effect_cases": report["cases_with_detectable_stale_downstream_effect"],
        "total_cases": report["total_cases"],
    }, indent=2))

    if report["lifetime_closure_top_match_vs_fresh_masked_oracle"] != 1.0:
        return 2
    if report["max_closure_vs_fresh_masked_oracle_vocab_delta"] > 0.02:
        return 3
    if report["max_masked_stale_memo_randomization_vocab_delta"] > 0.02:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
