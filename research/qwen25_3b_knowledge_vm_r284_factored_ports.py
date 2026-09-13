from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_WEIGHTS = {
    "model-00001-of-00002.safetensors": "67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2",
    "model-00002-of-00002.safetensors": "a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d",
}
OUT = Path(os.environ.get("SO_R284_REPORT", "ci-qwen-r284/report.json"))
COLORS = ("red", "blue", "green", "yellow")
IDENTITY = "ZN6MDGADZD"
SPECS = ("eq0", "map01", "not_red", "conjunction", "new_labels", "read")
PORT_STEM = "Memory-port value:"
SYSTEM = (
    "An external resolver has already selected the authoritative memory port for "
    "the entity named in the user's question. Treat the memory-port value as that "
    "entity's color. Follow the requested answer format exactly and answer with "
    "only the requested single word."
)


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
    if spec == "read":
        return COLORS[value]
    if spec == "eq0":
        return "yes" if value == 0 else "no"
    if spec == "map01":
        return "a" if value in (0, 1) else "b"
    if spec == "not_red":
        return "no" if value == 0 else "yes"
    if spec == "conjunction":
        return "yes" if value not in (0, 1) else "no"
    if spec == "new_labels":
        return "circle" if value in (0, 3) else "square"
    raise ValueError(spec)


def candidates(spec: str):
    if spec == "read":
        return COLORS
    if spec in ("eq0", "not_red", "conjunction"):
        return ("yes", "no")
    if spec == "map01":
        return ("A", "B")
    if spec == "new_labels":
        return ("circle", "square")
    raise ValueError(spec)


def candidate_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    ids2 = tok.encode(" " + s, add_special_tokens=False)
    if len(ids2) == 1:
        return int(ids2[0])
    raise RuntimeError(f"candidate is not one token: {s!r} -> {ids} / {ids2}")


def canonical(s: str) -> str:
    return s.strip().casefold()


def top_from_logits(tok, logits, spec: str):
    scores = {
        canonical(c): float(logits[candidate_token(tok, c)])
        for c in candidates(spec)
    }
    return max(scores, key=scores.get), scores


def render(tok, value: int, spec: str):
    q = question(spec)
    dynamic_text = f" {COLORS[value]}. "
    user = PORT_STEM + dynamic_text + q
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    full_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    q_idx = full_text.find(q)
    if q_idx < 0:
        raise RuntimeError("query substring missing")
    stem_idx = full_text.find(PORT_STEM)
    if stem_idx < 0 or stem_idx >= q_idx:
        raise RuntimeError("port stem missing or misplaced")
    static_end = stem_idx + len(PORT_STEM)

    enc = tok(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = list(enc.input_ids)
    offsets = list(enc.offset_mapping)

    def boundary_at_or_after(char_pos: int) -> int:
        for i, (start, end) in enumerate(offsets):
            if start >= char_pos:
                return i
            if start < char_pos < end:
                return i
        return len(full_ids)

    static_tok = boundary_at_or_after(static_end)
    query_tok = boundary_at_or_after(q_idx)
    if not (0 < static_tok < query_tok < len(full_ids)):
        raise RuntimeError(
            f"invalid token boundaries static={static_tok} query={query_tok} total={len(full_ids)}"
        )

    static_ids = full_ids[:static_tok]
    prefix_ids = full_ids[:query_tok]
    dynamic_ids = full_ids[static_tok:query_tok]
    query_ids = full_ids[query_tok:]
    return {
        "full_text": full_text,
        "static_text": full_text[:static_end],
        "prefix_text": full_text[:q_idx],
        "full_ids": full_ids,
        "static_ids": static_ids,
        "dynamic_ids": dynamic_ids,
        "query_ids": query_ids,
        "dynamic_text": full_text[static_end:q_idx],
        "static_token_boundary": static_tok,
        "query_token_boundary": query_tok,
    }


def legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def slice_cache(cache, start: int, end: int):
    c = legacy_cache(cache)
    return tuple(
        (
            k[:, :, start:end, :].contiguous().clone(),
            v[:, :, start:end, :].contiguous().clone(),
        )
        for k, v in c
    )


def concat_cache(*caches):
    cs = [legacy_cache(c) for c in caches]
    n = len(cs[0])
    if any(len(c) != n for c in cs):
        raise RuntimeError("cache layer count mismatch")
    return tuple(
        (
            torch.cat([c[i][0] for c in cs], dim=2).contiguous(),
            torch.cat([c[i][1] for c in cs], dim=2).contiguous(),
        )
        for i in range(n)
    )


def cache_seq_len(cache) -> int:
    c = legacy_cache(cache)
    return int(c[0][0].shape[2])


def cache_bytes(cache) -> int:
    total = 0
    for k, v in legacy_cache(cache):
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return int(total)


def zero_lane_like(capsule):
    return tuple((torch.zeros_like(k), torch.zeros_like(v)) for k, v in legacy_cache(capsule))


def mutated_lane(capsule, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out = []
    for k, v in legacy_cache(capsule):
        rk = torch.randn(k.shape, generator=gen, dtype=torch.float32).to(k.dtype)
        rv = torch.randn(v.shape, generator=gen, dtype=torch.float32).to(v.dtype)
        out.append((rk, rv))
    return tuple(out)


def run_segment(model, token_ids, base_cache, past_mask):
    past_len = cache_seq_len(base_cache)
    seg = torch.tensor([token_ids], dtype=torch.long)
    seg_mask = torch.ones((1, len(token_ids)), dtype=torch.long)
    mask = torch.cat([past_mask, seg_mask], dim=1)
    pos = torch.arange(past_len, past_len + len(token_ids), dtype=torch.long).unsqueeze(0)
    cache_pos = torch.arange(past_len, past_len + len(token_ids), dtype=torch.long)
    with torch.inference_mode():
        out = model(
            input_ids=seg,
            attention_mask=mask,
            position_ids=pos,
            cache_position=cache_pos,
            past_key_values=copy.deepcopy(base_cache),
            use_cache=True,
            return_dict=True,
        )
    return out


def query_logits(model, query_ids, cache, past_mask):
    past_len = cache_seq_len(cache)
    q = torch.tensor([query_ids], dtype=torch.long)
    qmask = torch.ones((1, len(query_ids)), dtype=torch.long)
    mask = torch.cat([past_mask, qmask], dim=1)
    pos = torch.arange(past_len, past_len + len(query_ids), dtype=torch.long).unsqueeze(0)
    cache_pos = torch.arange(past_len, past_len + len(query_ids), dtype=torch.long)
    with torch.inference_mode():
        return model(
            input_ids=q,
            attention_mask=mask,
            position_ids=pos,
            cache_position=cache_pos,
            past_key_values=copy.deepcopy(cache),
            use_cache=True,
            return_dict=True,
        ).logits[0, -1, :].float()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    md = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            allow_patterns=[
                "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
                "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
            ],
        )
    )
    verified = {name: sha256_file(md / name) for name in EXPECTED_WEIGHTS}
    assert verified == EXPECTED_WEIGHTS

    tok = AutoTokenizer.from_pretrained(
        md, local_files_only=True, padding_side="left", trust_remote_code=False
    )
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    layouts = {(v, spec): render(tok, v, spec) for v in range(4) for spec in SPECS}
    static_variants = {tuple(x["static_ids"]) for x in layouts.values()}
    if len(static_variants) != 1:
        raise RuntimeError(f"static port prelude varied: {len(static_variants)} variants")
    static_ids = list(next(iter(static_variants)))
    S = len(static_ids)

    dyn_by_value = {}
    dyn_text_by_value = {}
    for v in range(4):
        variants = {tuple(layouts[(v, spec)]["dynamic_ids"]) for spec in SPECS}
        if len(variants) != 1:
            raise RuntimeError(f"dynamic payload varied by operation for value {v}")
        dyn_by_value[v] = list(next(iter(variants)))
        dyn_text_by_value[v] = layouts[(v, "read")]["dynamic_text"]
    dyn_lengths = {len(x) for x in dyn_by_value.values()}
    if len(dyn_lengths) != 1:
        raise RuntimeError(f"dynamic slot width differs across revisions: {dyn_lengths}")
    D = next(iter(dyn_lengths))
    if D > 4:
        raise RuntimeError(f"dynamic port too wide for R284 gate: {D} > 4")

    static_tensor = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        static_out = model(
            input_ids=static_tensor,
            attention_mask=torch.ones_like(static_tensor),
            use_cache=True,
            return_dict=True,
        )
    shared_cache = legacy_cache(static_out.past_key_values)
    assert cache_seq_len(shared_cache) == S
    shared_mask = torch.ones((1, S), dtype=torch.long)

    lane0 = {}
    full_prefix_cache = {}
    for v in range(4):
        seg_out = run_segment(model, dyn_by_value[v], shared_cache, shared_mask)
        pcache = legacy_cache(seg_out.past_key_values)
        assert cache_seq_len(pcache) == S + D
        full_prefix_cache[v] = pcache
        lane0[v] = slice_cache(pcache, S, S + D)

    zero0 = zero_lane_like(lane0[0])
    lane1_base = concat_cache(shared_cache, zero0)
    lane1_base_mask = torch.cat(
        [shared_mask, torch.zeros((1, D), dtype=torch.long)], dim=1
    )
    lane1 = {}
    for v in range(4):
        seg_out = run_segment(model, dyn_by_value[v], lane1_base, lane1_base_mask)
        pcache = legacy_cache(seg_out.past_key_values)
        lane1[v] = slice_cache(pcache, S + D, S + 2 * D)

    rows = []
    exact_deltas = []
    stale_invariance_deltas = []
    revoked_invariance_deltas = []
    lane_correct = []
    full_text_correct = []
    exact_top_match = []

    for v in range(4):
        recomposed = concat_cache(shared_cache, lane0[v])
        recomposed_mask = torch.ones((1, S + D), dtype=torch.long)
        for spec in SPECS:
            layout = layouts[(v, spec)]
            full = torch.tensor([layout["full_ids"]], dtype=torch.long)
            with torch.inference_mode():
                text_logits = model(
                    input_ids=full,
                    attention_mask=torch.ones_like(full),
                    use_cache=False,
                    return_dict=True,
                ).logits[0, -1, :].float()

            exact_logits = query_logits(
                model, layout["query_ids"], recomposed, recomposed_mask
            )
            reference_logits = query_logits(
                model,
                layout["query_ids"],
                full_prefix_cache[v],
                torch.ones((1, S + D), dtype=torch.long),
            )
            exact_delta = float(torch.max(torch.abs(exact_logits - reference_logits)).item())
            text_cache_delta = float(torch.max(torch.abs(text_logits - exact_logits)).item())
            exact_deltas.append(exact_delta)
            tt, _ = top_from_logits(tok, text_logits, spec)
            et, _ = top_from_logits(tok, exact_logits, spec)
            exp = expected(spec, v)
            full_text_correct.append(tt == exp)
            exact_top_match.append(tt == et)
            rows.append({
                "phase": "exact_factorization",
                "value": v,
                "color": COLORS[v],
                "spec": spec,
                "expected": exp,
                "text_top": tt,
                "port_top": et,
                "text_correct": tt == exp,
                "port_correct": et == exp,
                "top_match": tt == et,
                "max_prefix_cache_delta": exact_delta,
                "max_text_port_delta": text_cache_delta,
            })

    transitions = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for old_v, new_v in transitions:
        cache = concat_cache(shared_cache, lane0[old_v], lane1[new_v])
        mutated = concat_cache(
            shared_cache, mutated_lane(lane0[old_v], 1000 + old_v), lane1[new_v]
        )
        active_mask = torch.cat([
            shared_mask,
            torch.zeros((1, D), dtype=torch.long),
            torch.ones((1, D), dtype=torch.long),
        ], dim=1)
        for spec in SPECS:
            qids = layouts[(new_v, spec)]["query_ids"]
            a = query_logits(model, qids, cache, active_mask)
            b = query_logits(model, qids, mutated, active_mask)
            delta = float(torch.max(torch.abs(a - b)).item())
            stale_invariance_deltas.append(delta)
            top, _ = top_from_logits(tok, a, spec)
            ok = top == expected(spec, new_v)
            lane_correct.append(ok)
            rows.append({
                "phase": "update_lane1_live",
                "old_value": old_v,
                "new_value": new_v,
                "spec": spec,
                "expected": expected(spec, new_v),
                "top": top,
                "correct": ok,
                "stale_lane_mutation_max_delta": delta,
            })

        cache2 = concat_cache(shared_cache, lane0[new_v], lane1[old_v])
        mutated2 = concat_cache(
            shared_cache, lane0[new_v], mutated_lane(lane1[old_v], 2000 + old_v)
        )
        active_mask2 = torch.cat([
            shared_mask,
            torch.ones((1, D), dtype=torch.long),
            torch.zeros((1, D), dtype=torch.long),
        ], dim=1)
        for spec in SPECS:
            qids = layouts[(new_v, spec)]["query_ids"]
            a = query_logits(model, qids, cache2, active_mask2)
            b = query_logits(model, qids, mutated2, active_mask2)
            delta = float(torch.max(torch.abs(a - b)).item())
            stale_invariance_deltas.append(delta)
            top, _ = top_from_logits(tok, a, spec)
            ok = top == expected(spec, new_v)
            lane_correct.append(ok)
            rows.append({
                "phase": "update_lane0_live",
                "old_value": old_v,
                "new_value": new_v,
                "spec": spec,
                "expected": expected(spec, new_v),
                "top": top,
                "correct": ok,
                "stale_lane_mutation_max_delta": delta,
            })

    revoked_mask = torch.cat([
        shared_mask,
        torch.zeros((1, D), dtype=torch.long),
        torch.zeros((1, D), dtype=torch.long),
    ], dim=1)
    for v in range(4):
        stale = concat_cache(shared_cache, lane0[v], lane1[(v + 1) % 4])
        random_stale = concat_cache(
            shared_cache,
            mutated_lane(lane0[v], 3000 + v),
            mutated_lane(lane1[(v + 1) % 4], 4000 + v),
        )
        for spec in SPECS:
            qids = layouts[(v, spec)]["query_ids"]
            a = query_logits(model, qids, stale, revoked_mask)
            b = query_logits(model, qids, random_stale, revoked_mask)
            delta = float(torch.max(torch.abs(a - b)).item())
            revoked_invariance_deltas.append(delta)
            rows.append({
                "phase": "revoked_both_lanes",
                "stale_value": v,
                "spec": spec,
                "all_stale_mutation_max_delta": delta,
            })

    capsule_bytes = cache_bytes(lane0[0])
    shared_bytes = cache_bytes(shared_cache)
    full_prefix_bytes = cache_bytes(full_prefix_cache[0])
    double_lane_bytes = 2 * capsule_bytes
    full_text_acc = sum(full_text_correct) / len(full_text_correct)
    exact_match = sum(exact_top_match) / len(exact_top_match)
    lane_acc = sum(lane_correct) / len(lane_correct)

    report = {
        "stage": "R284-FACTORIZED-GENERATION-GATED-LATENT-PORTS",
        "architecture_candidate": "Generation-Gated Latent Port Fabric (GGLPF)",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": verified,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "per_fact_gradient_steps": 0,
        "runtime_knowledge_text_tokens": 0,
        "routing_in_payload": False,
        "static_shared_tokens": S,
        "dynamic_slots_per_generation": D,
        "dynamic_text_by_value": dyn_text_by_value,
        "shared_cache_bytes": shared_bytes,
        "single_generation_capsule_bytes": capsule_bytes,
        "double_generation_capsule_bytes_per_port": double_lane_bytes,
        "conventional_full_prefix_cache_bytes_per_port": full_prefix_bytes,
        "single_generation_storage_ratio_vs_full_prefix": capsule_bytes / full_prefix_bytes,
        "double_generation_storage_ratio_vs_full_prefix": double_lane_bytes / full_prefix_bytes,
        "full_text_accuracy": full_text_acc,
        "exact_port_top_match": exact_match,
        "max_exact_factorization_logit_delta": max(exact_deltas),
        "dual_lane_expected_accuracy": lane_acc,
        "max_masked_stale_mutation_logit_delta": max(stale_invariance_deltas),
        "max_revoked_all_stale_mutation_logit_delta": max(revoked_invariance_deltas),
        "authority_semantics": {
            "live": "exactly one generation lane mask bit is 1",
            "update": "compile into inactive lane, atomically flip authority bit, old lane stays materialized but mask=0",
            "revoke": "both generation lane mask bits are 0",
            "no_resurrection_certificate": "mutating masked stale K/V bytes must leave full vocabulary logits unchanged",
        },
        "scientific_scope": (
            "Real frozen Qwen2.5-3B-Instruct. External routing is separated from storage. "
            "A static model-native port prelude is shared globally; each fact contributes only "
            "a small query-independent K/V capsule. Two fixed generation lanes implement "
            "neural double-buffer lifecycle control. Masked stale generations remain physically "
            "materialized while a causal attention mask makes their bytes provably irrelevant "
            "to logits. This is a mechanism gate, not yet a claim of broad RAG superiority."
        ),
        "dod_status": "NOT_DOD; architecture mechanism gate",
        "rows": rows,
    }
    canonical_report = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical_report).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "static_tokens": S,
        "dynamic_slots": D,
        "full_text_accuracy": full_text_acc,
        "exact_top_match": exact_match,
        "max_exact_delta": max(exact_deltas),
        "dual_lane_accuracy": lane_acc,
        "max_stale_delta": max(stale_invariance_deltas),
        "max_revoked_delta": max(revoked_invariance_deltas),
        "single_storage_ratio": capsule_bytes / full_prefix_bytes,
        "double_storage_ratio": double_lane_bytes / full_prefix_bytes,
    }, indent=2))

    if full_text_acc < 0.90:
        return 2
    if exact_match != 1.0 or max(exact_deltas) > 0.02:
        return 3
    if max(stale_invariance_deltas) > 0.02:
        return 4
    if max(revoked_invariance_deltas) > 0.02:
        return 5
    if lane_acc < 0.90:
        return 6
    if D > 4:
        return 7
    if capsule_bytes >= full_prefix_bytes:
        return 8
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
