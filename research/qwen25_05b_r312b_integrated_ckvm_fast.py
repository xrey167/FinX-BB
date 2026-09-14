from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS, VALUE_STRINGS
from research.qwen25_05b_r312_integrated_ckvm_transaction import World, LifetimeDAG, execute, decode_prefix, run_suffix


OUT = Path(os.environ.get("SO_R312B_REPORT", "ci-qwen-r312b/report.json"))
OPS = ("copy_a", "if_same")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    model_dir = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {name: sha256_file(model_dir / name) for name in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    value_ids = [tuple(tok.encode(x, add_special_tokens=False)) for x in VALUE_STRINGS]
    a, b = "HO-A-FAST", "HO-B-FAST"
    pa, pb = 101, 102
    world = World(value_ids)
    world.add(pa, 0, True)
    world.add(pb, 1, False)
    world.bind_alias(a, pa)
    world.bind_alias(b, pb)
    world.bind_alias("late-a-fast", pa)
    world.bind_alias("late-b-fast", pb)
    dag = LifetimeDAG()

    checkpoints = {}
    for op in OPS:
        prefix = decode_prefix(tok, op, a, b)
        prefix_ids = tok(prefix, add_special_tokens=False).input_ids
        t = torch.tensor([prefix_ids], dtype=torch.long)
        with torch.inference_mode():
            out = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=True, return_dict=True)
        checkpoints[op] = (prefix, prefix_ids, legacy_cache(out.past_key_values))

    rows = []
    deltas = []
    invalidation_checks = []
    same_value_rewrite_checks = []
    full_times = []
    splice_times = []
    update_ns = []

    for step in range(4):
        op = OPS[step % len(OPS)]
        ca, cb = world.capability(pa), world.capability(pb)
        state = execute(world, dag, op, ca, cb)
        assert state.output_ids
        prefix, prefix_ids, prefix_cache = checkpoints[op]
        suffix = list(state.output_ids) + tok.encode("\nStatus:", add_special_tokens=False)

        full_ids = tok(prefix, add_special_tokens=False).input_ids + suffix
        ft = torch.tensor([full_ids], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.inference_mode():
            full_out = model(input_ids=ft, attention_mask=torch.ones_like(ft), use_cache=True, return_dict=True)
        full_times.append(time.perf_counter() - t0)
        full_logits = full_out.logits[0, -1].float()

        t0 = time.perf_counter()
        splice_out = run_suffix(model, suffix, prefix_cache, len(prefix_ids))
        splice_times.append(time.perf_counter() - t0)
        splice_logits = splice_out.logits[0, -1].float()
        delta = float((full_logits - splice_logits).abs().max())
        deltas.append(delta)

        # Force an update on a dependency that was actually read.
        target_pid = state.deps[0][0]
        old_generation = world.pages[target_pid].generation
        old_value_idx = world.pages[target_pid].value_idx
        same_value = (step % 2 == 0)
        new_value_idx = old_value_idx if same_value else (old_value_idx + 1) % len(value_ids)
        t0 = time.perf_counter_ns()
        world.update(target_pid, new_value_idx, world.pages[target_pid].flag)
        dag.invalidate((target_pid, old_generation))
        update_ns.append(time.perf_counter_ns() - t0)

        old_factor_dead = not dag.is_valid(state.factor_id)
        invalidation_checks.append(old_factor_dead)
        if same_value:
            same_value_rewrite_checks.append(old_factor_dead and world.pages[target_pid].generation != old_generation and world.pages[target_pid].value_idx == old_value_idx)

        # Re-execute current world from current capabilities. No retraining and no
        # reuse of the invalidated J state.
        current = execute(world, dag, op, world.capability(pa), world.capability(pb))
        assert current.output_ids
        rows.append({
            "step": step,
            "op": op,
            "old_deps": state.deps,
            "old_factor_dead_after_update": old_factor_dead,
            "same_value_generation_rewrite": same_value,
            "current_factor_live": dag.is_valid(current.factor_id),
            "full_vs_splice_max_logit_delta": delta,
        })

    alias_identity = (
        world.resolve(a) == world.resolve("late-a-fast") == pa
        and world.resolve(b) == world.resolve("late-b-fast") == pb
    )

    report = {
        "stage": "R312B-INTEGRATED-CKVM-FAST",
        "architecture_candidate": "CKCA + CKVM + CLFD + late-bound real-model decode",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "operations": list(OPS),
        "transitions": len(rows),
        "rows": rows,
        "max_full_vs_splice_logit_delta": max(deltas),
        "all_old_factors_dead_after_dependency_update": all(invalidation_checks),
        "same_value_generation_rewrite_checks_passed": all(same_value_rewrite_checks) if same_value_rewrite_checks else False,
        "canonical_alias_identity": alias_identity,
        "median_full_recompute_seconds": sorted(full_times)[len(full_times) // 2],
        "median_late_splice_seconds": sorted(splice_times)[len(splice_times) // 2],
        "median_full_over_splice_speedup": sorted(full_times)[len(full_times) // 2] / sorted(splice_times)[len(splice_times) // 2],
        "median_authority_update_plus_invalidation_ns": sorted(update_ns)[len(update_ns) // 2],
        "updates_require_gradient_steps": False,
        "dod_status": "NOT_DOD; reduced real-model integration gate",
        "claim_boundary": (
            "R312b intentionally removes the learned planner from the integrated timing path to isolate real-model late binding, "
            "generation lifetime invalidation and same-value rewrite semantics. Planner/compiler generalization is tested separately."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "max_full_vs_splice_logit_delta",
        "all_old_factors_dead_after_dependency_update",
        "same_value_generation_rewrite_checks_passed",
        "canonical_alias_identity",
        "median_full_recompute_seconds",
        "median_late_splice_seconds",
        "median_full_over_splice_speedup",
        "median_authority_update_plus_invalidation_ns",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
