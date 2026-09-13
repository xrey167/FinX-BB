from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    MODEL_ID,
    REVISION,
    EXPECTED_WEIGHTS,
    COLORS,
    SPECS,
    sha256_file,
    render,
    expected,
    top_from_logits,
    legacy_cache,
    slice_cache,
    concat_cache,
    zero_lane_like,
    mutated_lane,
    query_logits,
    run_segment,
)

OUT = Path(os.environ.get("SO_R286_REPORT", "ci-qwen-r286/report.json"))


def zero_values(capsule):
    return tuple((k.clone(), torch.zeros_like(v)) for k, v in legacy_cache(capsule))


def random_keys_zero_values(capsule, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out = []
    for k, v in legacy_cache(capsule):
        rk = torch.randn(k.shape, generator=gen, dtype=torch.float32).to(k.dtype)
        out.append((rk, torch.zeros_like(v)))
    return tuple(out)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors",
            "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json",
        ],
    ))
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

    layouts = {(v, s): render(tok, v, s) for v in range(4) for s in SPECS}
    static_ids = layouts[(0, "read")]["static_ids"]
    assert all(x["static_ids"] == static_ids for x in layouts.values())
    S = len(static_ids)

    dyn = {}
    for v in range(4):
        ids = layouts[(v, "read")]["dynamic_ids"]
        assert all(layouts[(v, s)]["dynamic_ids"] == ids for s in SPECS)
        dyn[v] = ids
    widths = {len(x) for x in dyn.values()}
    assert len(widths) == 1
    D = next(iter(widths))

    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        so = model(
            input_ids=st,
            attention_mask=torch.ones_like(st),
            use_cache=True,
            return_dict=True,
        )
    shared = legacy_cache(so.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    lane0 = {}
    for v in range(4):
        o = run_segment(model, dyn[v], shared, shared_mask)
        c = legacy_cache(o.past_key_values)
        lane0[v] = slice_cache(c, S, S + D)

    zero0 = zero_lane_like(lane0[0])
    lane1_base = concat_cache(shared, zero0)
    lane1_base_mask = torch.cat(
        [shared_mask, torch.zeros((1, D), dtype=torch.long)], dim=1
    )
    lane1 = {}
    for v in range(4):
        o = run_segment(model, dyn[v], lane1_base, lane1_base_mask)
        c = legacy_cache(o.past_key_values)
        lane1[v] = slice_cache(c, S + D, S + 2 * D)

    both_on = torch.cat([
        shared_mask,
        torch.ones((1, D), dtype=torch.long),
        torch.ones((1, D), dtype=torch.long),
    ], dim=1)
    authority_new = torch.cat([
        shared_mask,
        torch.zeros((1, D), dtype=torch.long),
        torch.ones((1, D), dtype=torch.long),
    ], dim=1)

    transitions = [(0, 1), (1, 2), (2, 3), (3, 0)]
    rows = []
    acc = {
        "conflict_both_live": [],
        "value_zero_stale": [],
        "kv_zero_stale": [],
        "authority_mask": [],
    }
    authority_mutation_deltas = []
    value_zero_key_mutation_deltas = []
    kv_zero_vs_authority_deltas = []

    for old_v, new_v in transitions:
        conflict = concat_cache(shared, lane0[old_v], lane1[new_v])
        value_zero = concat_cache(shared, zero_values(lane0[old_v]), lane1[new_v])
        value_zero_key_mut = concat_cache(
            shared,
            random_keys_zero_values(lane0[old_v], 7000 + old_v),
            lane1[new_v],
        )
        kv_zero = concat_cache(shared, zero_lane_like(lane0[old_v]), lane1[new_v])
        authority_mut = concat_cache(
            shared,
            mutated_lane(lane0[old_v], 8000 + old_v),
            lane1[new_v],
        )

        for spec in SPECS:
            qids = layouts[(new_v, spec)]["query_ids"]
            logits_conflict = query_logits(model, qids, conflict, both_on)
            logits_vzero = query_logits(model, qids, value_zero, both_on)
            logits_vzero_keymut = query_logits(model, qids, value_zero_key_mut, both_on)
            logits_kvzero = query_logits(model, qids, kv_zero, both_on)
            logits_authority = query_logits(model, qids, conflict, authority_new)
            logits_authority_mut = query_logits(model, qids, authority_mut, authority_new)

            tops = {
                "conflict_both_live": top_from_logits(tok, logits_conflict, spec)[0],
                "value_zero_stale": top_from_logits(tok, logits_vzero, spec)[0],
                "kv_zero_stale": top_from_logits(tok, logits_kvzero, spec)[0],
                "authority_mask": top_from_logits(tok, logits_authority, spec)[0],
            }
            exp = expected(spec, new_v)
            for k, top in tops.items():
                acc[k].append(top == exp)

            auth_delta = float(
                torch.max(torch.abs(logits_authority - logits_authority_mut)).item()
            )
            vzero_key_delta = float(
                torch.max(torch.abs(logits_vzero - logits_vzero_keymut)).item()
            )
            kvzero_auth_delta = float(
                torch.max(torch.abs(logits_kvzero - logits_authority)).item()
            )
            authority_mutation_deltas.append(auth_delta)
            value_zero_key_mutation_deltas.append(vzero_key_delta)
            kv_zero_vs_authority_deltas.append(kvzero_auth_delta)

            rows.append({
                "old_value": old_v,
                "old_color": COLORS[old_v],
                "new_value": new_v,
                "new_color": COLORS[new_v],
                "spec": spec,
                "expected": exp,
                "tops": tops,
                "authority_mask_stale_randomization_max_vocab_delta": auth_delta,
                "value_zero_randomized_stale_keys_max_vocab_delta": vzero_key_delta,
                "kv_zero_vs_authority_max_vocab_delta": kvzero_auth_delta,
            })

    accuracy = {k: sum(v) / len(v) for k, v in acc.items()}
    report = {
        "stage": "R286-AUTHORITY-MASK-VS-CACHE-REPAIR",
        "architecture_candidate": "Causal Authority Mask for Generation-Gated Latent Ports",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": verified,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "dynamic_slots_per_lane": D,
        "policy_accuracy": accuracy,
        "max_authority_mask_stale_randomization_logit_delta": max(authority_mutation_deltas),
        "median_authority_mask_stale_randomization_logit_delta": float(
            torch.tensor(authority_mutation_deltas).median().item()
        ),
        "max_value_zero_randomized_stale_keys_logit_delta": max(value_zero_key_mutation_deltas),
        "median_value_zero_randomized_stale_keys_logit_delta": float(
            torch.tensor(value_zero_key_mutation_deltas).median().item()
        ),
        "max_kv_zero_vs_authority_logit_delta": max(kv_zero_vs_authority_deltas),
        "mechanism_claim": (
            "A stale generation whose attention-mask bit is zero is causally absent: "
            "randomizing every stale K/V element must not alter vocabulary logits. "
            "By contrast, zeroing only stale values leaves stale keys in the softmax "
            "addressing denominator; randomizing those keys can alter logits even though "
            "the stale values are all zero."
        ),
        "scientific_scope": (
            "Real frozen Qwen2.5-3B-Instruct, synthetic authoritative color ports, "
            "four sequential update transitions and six operations per transition. "
            "This compares causal mechanisms, not external systems or full benchmarks."
        ),
        "rows": rows,
        "dod_status": "NOT_DOD; anti-resurrection mechanism comparison",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "dynamic_slots_per_lane": D,
        "policy_accuracy": accuracy,
        "max_authority_mask_stale_randomization_delta": max(authority_mutation_deltas),
        "max_value_zero_randomized_stale_keys_delta": max(value_zero_key_mutation_deltas),
        "max_kv_zero_vs_authority_delta": max(kv_zero_vs_authority_deltas),
    }, indent=2))

    if accuracy["authority_mask"] < 0.90:
        return 2
    if max(authority_mutation_deltas) > 0.02:
        return 3
    if max(value_zero_key_mutation_deltas) <= 0.02:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
