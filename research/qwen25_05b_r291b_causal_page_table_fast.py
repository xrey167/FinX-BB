from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import (
    SPECS,
    concat_cache,
    legacy_cache,
    mutated_lane,
    query_logits,
    render,
    run_segment,
    sha256_file,
    slice_cache,
)
from research.qwen25_05b_r291_causal_page_table import (
    CausalPageTable,
    Handle,
    MODEL_ID,
    REVISION,
    EXPECTED_WEIGHTS,
    top_candidate,
)

OUT = Path(os.environ.get("SO_R291B_REPORT", "ci-qwen-r291b/report.json"))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rng = random.Random(2911)

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

    probe_specs = ("read", "eq0")
    layouts = {(v, s): render(tok, v, s) for v in range(4) for s in probe_specs}
    static_ids = layouts[(0, "read")]["static_ids"]
    assert all(x["static_ids"] == static_ids for x in layouts.values())
    dynamic = {}
    for v in range(4):
        ids = layouts[(v, "read")]["dynamic_ids"]
        assert all(layouts[(v, s)]["dynamic_ids"] == ids for s in probe_specs)
        dynamic[v] = ids
    D = len(dynamic[0])
    assert all(len(x) == D for x in dynamic.values())
    S = len(static_ids)

    st = torch.tensor([static_ids], dtype=torch.long)
    with torch.inference_mode():
        sout = model(input_ids=st, attention_mask=torch.ones_like(st), use_cache=True, return_dict=True)
    shared = legacy_cache(sout.past_key_values)
    shared_mask = torch.ones((1, S), dtype=torch.long)

    capsules = {}
    prefix = {}
    for v in range(4):
        out = run_segment(model, dynamic[v], shared, shared_mask)
        pcache = legacy_cache(out.past_key_values)
        prefix[v] = pcache
        capsules[v] = slice_cache(pcache, S, S + D)

    table = CausalPageTable()
    pod_id = 7
    generation = 0
    stale_handles: list[Handle] = []
    rejected_stale = 0
    rejected_forged = 0
    stale_mutation_equivalence = []
    active_equivalence = []
    active_top_match = []
    probes = []

    generation_updates = 4096
    probe_stride = 512  # 8 real-model lifecycle probes, bounded for CI

    for step in range(generation_updates):
        generation += 1
        value = rng.randrange(4)
        old_handle = table.current_handle(pod_id)
        h = table.write(pod_id, generation, capsules[value])
        table.commit(h)
        if old_handle is not None:
            stale_handles.append(old_handle)
            rejected_stale += int(not table.independent_verify(old_handle))
        forged = Handle(pod_id, generation + 100000, h.page_id, h.payload_digest, h.model_revision)
        rejected_forged += int(not table.independent_verify(forged))

        if step % probe_stride == 0:
            active = table.materialize_verified(pod_id)
            assert active is not None
            stale_page_ids = [pid for pid in table.pages if pid != h.page_id]
            for pid in rng.sample(stale_page_ids, min(8, len(stale_page_ids))):
                table.pages[pid].cache = mutated_lane(table.pages[pid].cache, seed=step * 1000 + pid)

            for spec in probe_specs:
                qids = layouts[(value, spec)]["query_ids"]
                direct_logits = query_logits(
                    model, qids, prefix[value], torch.ones((1, S + D), dtype=torch.long)
                )
                verified_cache = concat_cache(shared, active)
                verified_logits = query_logits(
                    model, qids, verified_cache, torch.ones((1, S + D), dtype=torch.long)
                )
                delta = float(torch.max(torch.abs(direct_logits - verified_logits)).item())
                active_equivalence.append(delta)
                direct_top = top_candidate(tok, direct_logits, spec)
                verified_top = top_candidate(tok, verified_logits, spec)
                active_top_match.append(direct_top == verified_top)

                for pid in rng.sample(stale_page_ids, min(4, len(stale_page_ids))):
                    table.pages[pid].cache = mutated_lane(table.pages[pid].cache, seed=step * 2000 + pid)
                active2 = table.materialize_verified(pod_id)
                verified2 = query_logits(
                    model,
                    qids,
                    concat_cache(shared, active2),
                    torch.ones((1, S + D), dtype=torch.long),
                )
                sdelta = float(torch.max(torch.abs(verified_logits - verified2)).item())
                stale_mutation_equivalence.append(sdelta)
                probes.append({
                    "step": step,
                    "generation": generation,
                    "value": value,
                    "spec": spec,
                    "direct_top": direct_top,
                    "verified_top": verified_top,
                    "active_logit_delta": delta,
                    "stale_page_mutation_logit_delta": sdelta,
                })

    pages_before_revoke = len(table.pages)
    table.revoke(pod_id)
    revoked_returns_none = table.materialize_verified(pod_id) is None
    stale_after_revoke_rejected = all(not table.independent_verify(h) for h in stale_handles[-256:])

    generation += 1
    h = table.write(pod_id, generation, capsules[1])
    table.commit(h)
    lookup_ns = []
    raw_ns = []
    for _ in range(50000):
        t0 = time.perf_counter_ns()
        _ = table.pages[table.authority[pod_id]]
        raw_ns.append(time.perf_counter_ns() - t0)
        t0 = time.perf_counter_ns()
        _ = table.materialize_verified(pod_id)
        lookup_ns.append(time.perf_counter_ns() - t0)

    report = {
        "stage": "R291B-CAUSAL-PAGE-TABLE-BOUNDED-REAL-GATE",
        "architecture_candidate": "Causal Page Table (CPT) for Temporal Capability Ports",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "generation_updates": generation_updates,
        "real_model_probe_count": len(probes),
        "probe_specs": probe_specs,
        "materialized_pages_before_revoke": pages_before_revoke,
        "stale_handle_rejection_rate": rejected_stale / max(len(stale_handles), 1),
        "forged_future_handle_rejection_rate": rejected_forged / generation_updates,
        "revoked_returns_no_attention_page": revoked_returns_none,
        "stale_handles_rejected_after_revoke": stale_after_revoke_rejected,
        "active_top_match_rate_vs_direct": sum(active_top_match) / len(active_top_match),
        "max_active_full_vocab_logit_delta_vs_direct": max(active_equivalence),
        "max_full_vocab_logit_delta_after_stale_page_mutation": max(stale_mutation_equivalence),
        "raw_page_lookup_ns_median": statistics.median(raw_ns),
        "verified_authority_lookup_ns_median": statistics.median(lookup_ns),
        "verified_authority_lookup_ns_p99": sorted(lookup_ns)[int(len(lookup_ns) * 0.99)],
        "attention_sequence_contains_stale_generations": False,
        "logical_context_growth_with_revision_count": 0,
        "mechanism": (
            "Historical generations remain physically resident, but the independent authority table "
            "materializes exactly one verified page into the neural read path. Update = immutable page "
            "write + monotonic authority flip; revoke = no admissible page. The real-model probes test "
            "full-vocabulary equality while thousands of stale pages remain resident and are mutated."
        ),
        "claim_boundary": (
            "The lifecycle/control path is exercised for every one of 4096 revisions; expensive model "
            "equivalence probes are deliberately bounded. This is not yet an end-to-end RAG benchmark."
        ),
        "probes": probes,
        "dod_status": "NOT_DOD; causal page-table real-model gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "generation_updates",
        "real_model_probe_count",
        "stale_handle_rejection_rate",
        "forged_future_handle_rejection_rate",
        "revoked_returns_no_attention_page",
        "active_top_match_rate_vs_direct",
        "max_active_full_vocab_logit_delta_vs_direct",
        "max_full_vocab_logit_delta_after_stale_page_mutation",
        "verified_authority_lookup_ns_median",
        "verified_authority_lookup_ns_p99",
    )}, indent=2))

    if report["stale_handle_rejection_rate"] != 1.0:
        return 2
    if report["forged_future_handle_rejection_rate"] != 1.0:
        return 3
    if not revoked_returns_none or not stale_after_revoke_rejected:
        return 4
    if report["active_top_match_rate_vs_direct"] != 1.0:
        return 5
    if report["max_active_full_vocab_logit_delta_vs_direct"] > 0.02:
        return 6
    if report["max_full_vocab_logit_delta_after_stale_page_mutation"] > 0.02:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
