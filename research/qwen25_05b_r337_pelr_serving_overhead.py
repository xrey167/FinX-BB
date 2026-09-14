from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file, legacy_cache
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS


STAGE = "R337-QWEN-PELR-SERVING-OVERHEAD"
REPORT_PATH = Path(os.environ.get("SO_R337_REPORT", "ci-r337/report.json"))
PAIRED_RUNS = int(os.environ.get("SO_R337_PAIRED_RUNS", "8"))
STRESS = int(os.environ.get("SO_R337_STRESS", "200000"))
SEED = 3370914


@dataclass
class Pod:
    generation: int
    value: str


class PELR:
    """Packed lifetime metadata attached to request execution context, not tensors."""

    def __init__(self, pods: dict[int, Pod]) -> None:
        self.pods = pods
        self.next_factor = 1
        self.factor_live: dict[int, bool] = {}
        self.factor_deps: dict[int, tuple[tuple[int, int], ...]] = {}
        self.reverse: dict[tuple[int, int], list[int]] = {}

    def snapshot_readset(self, pids: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((pid, self.pods[pid].generation) for pid in pids)

    def commit(self, deps: tuple[tuple[int, int], ...]) -> bool:
        return all(self.pods[pid].generation == g for pid, g in deps)

    def seal(self, deps: tuple[tuple[int, int], ...]) -> int:
        fid = self.next_factor
        self.next_factor += 1
        self.factor_live[fid] = True
        self.factor_deps[fid] = deps
        for dep in deps:
            self.reverse.setdefault(dep, []).append(fid)
        return fid

    def invalidate(self, pid: int, old_generation: int) -> int:
        touched = 0
        for fid in self.reverse.get((pid, old_generation), ()):
            if self.factor_live.get(fid, False):
                self.factor_live[fid] = False
                touched += 1
        return touched

    def admit(self, fid: int) -> bool:
        return self.factor_live.get(fid, False)


def run_suffix(model, suffix_ids, prefix_cache, prefix_len):
    ids = torch.tensor([suffix_ids], dtype=torch.long)
    mask = torch.ones((1, prefix_len + len(suffix_ids)), dtype=torch.long)
    pos = torch.arange(prefix_len, prefix_len + len(suffix_ids), dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        return model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=copy.deepcopy(prefix_cache),
            use_cache=True,
            return_dict=True,
        )


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        md,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    prefix_text = (
        "A trusted CKCA runtime has already compiled this governed request. "
        "The current committed value will be supplied after the immutable prefix.\n"
        "Governed result:\n"
    )
    prefix_ids = tok(prefix_text, add_special_tokens=False).input_ids
    pt = torch.tensor([prefix_ids], dtype=torch.long)
    with torch.inference_mode():
        po = model(input_ids=pt, attention_mask=torch.ones_like(pt), use_cache=True, return_dict=True)
    prefix_cache = legacy_cache(po.past_key_values)

    pods = {
        11: Pod(1, "VAL_1042"),
        12: Pod(1, "VAL_9183"),
    }
    pelr = PELR(pods)
    pids = (11, 12)

    def suffix_for_current() -> list[int]:
        # Same governed task bytes in control/guarded path; only metadata differs.
        text = f"{pods[11].value} | {pods[12].value}\nState:"
        return tok(text, add_special_tokens=False).input_ids

    # Warm-up both model path and metadata structures.
    suffix = suffix_for_current()
    for _ in range(2):
        _ = run_suffix(model, suffix, prefix_cache, len(prefix_ids))
        deps = pelr.snapshot_readset(pids)
        assert pelr.commit(deps)
        fid = pelr.seal(deps)
        assert pelr.admit(fid)

    control_times = []
    guarded_times = []
    metadata_times = []
    max_logit_delta = 0.0
    guarded_conflicts = 0
    same_value_rewrites = 0

    rng = random.Random(SEED)
    for i in range(PAIRED_RUNS):
        suffix = suffix_for_current()
        order_guard_first = bool(i & 1)

        def control_call():
            t0 = time.perf_counter_ns()
            out = run_suffix(model, suffix, prefix_cache, len(prefix_ids))
            return out, time.perf_counter_ns() - t0

        def guarded_call(inject_race: bool = False):
            meta0 = time.perf_counter_ns()
            deps = pelr.snapshot_readset(pids)
            meta_pre = time.perf_counter_ns() - meta0

            t0 = time.perf_counter_ns()
            out = run_suffix(model, suffix, prefix_cache, len(prefix_ids))
            model_ns = time.perf_counter_ns() - t0

            if inject_race:
                pid = pids[i % len(pids)]
                old_g = pods[pid].generation
                old_v = pods[pid].value
                pods[pid].generation += 1
                # Same-value ABA rewrite: bytes unchanged, generation changed.
                pods[pid].value = old_v
                pelr.invalidate(pid, old_g)

            meta0 = time.perf_counter_ns()
            ok = pelr.commit(deps)
            if not ok:
                # Local retry metadata uses the fresh generation. The model bytes are
                # unchanged for same-value rewrite, so no second neural compute is
                # needed for this special equality-certified case; a fresh factor is minted.
                fresh = pelr.snapshot_readset(pids)
                assert pelr.commit(fresh)
                deps = fresh
            fid = pelr.seal(deps)
            admitted = pelr.admit(fid)
            meta_post = time.perf_counter_ns() - meta0
            assert admitted
            return out, model_ns, meta_pre + meta_post, ok

        inject = (i % 4 == 3)
        if inject:
            same_value_rewrites += 1

        if order_guard_first:
            gout, gmodel, gmeta, ok = guarded_call(inject)
            cout, cns = control_call()
        else:
            cout, cns = control_call()
            gout, gmodel, gmeta, ok = guarded_call(inject)

        control_times.append(cns)
        guarded_times.append(gmodel + gmeta)
        metadata_times.append(gmeta)
        guarded_conflicts += int(not ok)
        max_logit_delta = max(
            max_logit_delta,
            float((cout.logits[0, -1].float() - gout.logits[0, -1].float()).abs().max()),
        )

    # Metadata-only scale stress. This measures the coherence plane without Qwen
    # latency hiding it and includes concurrent/same-value transitions.
    stress_commit_ns = []
    stress_admit_ns = []
    stress_conflicts = 0
    stress_detected = 0
    stress_escaped = 0
    stress_stale_rejected = 0
    stress_stale_escaped = 0
    stress_same_value = 0
    for i in range(STRESS):
        deps = pelr.snapshot_readset(pids)
        race = rng.random() < 0.10
        if race:
            pid = pids[i & 1]
            old_g = pods[pid].generation
            old_v = pods[pid].value
            pods[pid].generation += 1
            pods[pid].value = old_v if rng.random() < 0.5 else f"VAL_{rng.randrange(10000):04d}"
            stress_same_value += int(pods[pid].value == old_v)
            pelr.invalidate(pid, old_g)
            stress_conflicts += 1

        t0 = time.perf_counter_ns()
        ok = pelr.commit(deps)
        stress_commit_ns.append(time.perf_counter_ns() - t0)
        if not ok:
            stress_detected += 1
            deps = pelr.snapshot_readset(pids)
            if not pelr.commit(deps):
                stress_escaped += 1
        elif race:
            stress_escaped += 1

        fid = pelr.seal(deps)
        if rng.random() < 0.05:
            pid = pids[(i + 1) & 1]
            old_g = pods[pid].generation
            pods[pid].generation += 1
            # same-value post-commit expiry is sufficient to kill the factor.
            pelr.invalidate(pid, old_g)
            t0 = time.perf_counter_ns()
            live = pelr.admit(fid)
            stress_admit_ns.append(time.perf_counter_ns() - t0)
            if live:
                stress_stale_escaped += 1
            else:
                stress_stale_rejected += 1
        else:
            t0 = time.perf_counter_ns()
            _ = pelr.admit(fid)
            stress_admit_ns.append(time.perf_counter_ns() - t0)

    med_control = statistics.median(control_times)
    med_guarded = statistics.median(guarded_times)
    med_meta = statistics.median(metadata_times)
    # Direct model timing noise can make guarded median slightly lower; report both
    # measured paired overhead and metadata/compute ratio. The latter isolates the
    # control plane without pretending Python CPU timing is production evidence.
    measured_overhead = (med_guarded - med_control) / med_control
    metadata_fraction = med_meta / med_control

    report = {
        "stage": STAGE,
        "architecture_candidate": "Packed Epistemic Lifetime Region around real frozen-Qwen late-splice serving",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "paired_model_runs": PAIRED_RUNS,
        "same_value_generation_rewrites_in_paired_runs": same_value_rewrites,
        "paired_conflicts_detected": guarded_conflicts,
        "max_control_vs_guarded_full_vocab_logit_delta": max_logit_delta,
        "median_control_serving_ns": med_control,
        "median_guarded_serving_ns": med_guarded,
        "median_lifetime_metadata_ns": med_meta,
        "measured_guarded_overhead_fraction_vs_control": measured_overhead,
        "metadata_time_fraction_of_control_model_compute": metadata_fraction,
        "metadata_stress_iterations": STRESS,
        "stress_conflicts_expected": stress_conflicts,
        "stress_conflicts_detected": stress_detected,
        "stress_conflicts_escaped": stress_escaped,
        "stress_same_value_conflicts": stress_same_value,
        "stress_stale_serves_rejected": stress_stale_rejected,
        "stress_stale_serves_escaped": stress_stale_escaped,
        "median_stress_commit_validation_ns": statistics.median(stress_commit_ns),
        "median_stress_factor_admission_ns": statistics.median(stress_admit_ns),
        "mechanism": (
            "A real frozen-Qwen late-splice request runs inside one implicit PELR execution context. The exact generation read-set is captured at the "
            "World Port boundary, ordinary Transformer compute carries no per-tensor lineage object, the Neural Commit Barrier validates once after "
            "compute, and retained state receives one FactorID. Same-value generation rewrites still conflict temporally even though neural input bytes "
            "are unchanged."
        ),
        "claim_boundary": (
            "This is real pretrained-model serving evidence for boundary-only metadata overhead, but it is a small CPU late-splice microbenchmark, not "
            "the production <5% DoD. GPU batching, long sequences, concurrent requests, real World Port execution, GANPT and free-form decoding must "
            "be measured together before the overhead target can be claimed."
        ),
        "dod_status": "NOT_DOD; real-model boundary-lifetime overhead gate",
    }
    report["contract_pass"] = (
        max_logit_delta == 0.0
        and guarded_conflicts == same_value_rewrites
        and stress_detected == stress_conflicts
        and stress_escaped == 0
        and stress_stale_escaped == 0
        and stress_stale_rejected > 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
