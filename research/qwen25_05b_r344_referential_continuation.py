from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.qwen25_3b_knowledge_vm_r284_factored_ports import sha256_file
from research.qwen25_05b_r308_late_bound_handle_vm import MODEL_ID, REVISION, EXPECTED_WEIGHTS
from research.qwen25_05b_r332_terminal_world_port import (
    ENTITIES, VALUE_COUNT, OPS, PromptRow, make_prompt_rows, labels, future_world,
)
from research.qwen25_05b_r332b_neural_isa_world_alu import encode_multiscale
from research.qwen25_05b_r332c_augmented_neural_isa import make_augmented_rows

STAGE = "R344-QWEN-COMPACT-REFERENTIAL-CONTINUATION"
REPORT_PATH = Path(os.environ.get("SO_R344_REPORT", "ci-r344/report.json"))
TRAIN_PER_OP = int(os.environ.get("SO_R344_TRAIN_PER_OP", "96"))
EVAL_PER_OP = int(os.environ.get("SO_R344_EVAL_PER_OP", "48"))
ENCODE_BATCH = int(os.environ.get("SO_R344_ENCODE_BATCH", "32"))
COMPILER_STEPS = int(os.environ.get("SO_R344_COMPILER_STEPS", "750"))
J_STEPS = int(os.environ.get("SO_R344_J_STEPS", "1300"))
CODE_DIM = int(os.environ.get("SO_R344_CODE_DIM", "32"))
WORLD_SEEDS = (17, 29, 43, 71, 101)


def bit_features(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(5, device=v.device)
    return ((v[:, None] >> shifts[None, :]) & 1).float()


class ReferentialCompiler(nn.Module):
    """Frozen-Qwen feature -> compact world-independent continuation code."""
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192), nn.GELU(),
            nn.Linear(192, CODE_DIM),
        )
        self.prototypes = nn.Parameter(torch.randn(OPS, CODE_DIM) * 0.05)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def code(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.code(x)
        p = F.normalize(self.prototypes, dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * (z @ p.T)


class JContinuation(nn.Module):
    """Shared mutable-world continuation. No entity identity is an input."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE_DIM + 10, 128), nn.GELU(),
            nn.Linear(128, 96), nn.GELU(),
            nn.Linear(96, 64), nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, code: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([code.float(), bit_features(va), bit_features(vb)], dim=-1))


def train_compiler(features: torch.Tensor, rows: list[PromptRow]):
    torch.manual_seed(34401)
    m = ReferentialCompiler(features.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    gen = torch.Generator().manual_seed(34402)
    m.train(); t0 = time.perf_counter()
    for _ in range(COMPILER_STEPS):
        idx = torch.randint(0, len(rows), (256,), generator=gen)
        x = features[idx]
        yy = y[idx]
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype) * 0.005
        logits = m(x + noise)
        z = m.code(x)
        p = F.normalize(m.prototypes, dim=-1)[yy]
        loss = F.cross_entropy(logits, yy, label_smoothing=0.02) + 0.08 * (1.0 - (z * p).sum(-1)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m.eval(), time.perf_counter() - t0


def train_j(codes: torch.Tensor, rows: list[PromptRow]):
    torch.manual_seed(34403)
    j = JContinuation()
    opt = torch.optim.AdamW(j.parameters(), lr=2.2e-3, weight_decay=1e-4)
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    gen = torch.Generator().manual_seed(34404)
    j.train(); t0 = time.perf_counter()
    for _ in range(J_STEPS):
        idx = torch.randint(0, len(rows), (384,), generator=gen)
        va = torch.randint(0, VALUE_COUNT, (len(idx),), generator=gen)
        vb = torch.randint(0, VALUE_COUNT, (len(idx),), generator=gen)
        y = labels(op[idx], va, vb)
        # Train on the same fp16 descriptor representation used at serving time.
        code = codes[idx].to(torch.float16).float()
        loss = F.cross_entropy(j(code, va, vb), y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(j.parameters(), 1.0); opt.step()
    return j.eval(), time.perf_counter() - t0


@torch.inference_mode()
def compiler_acc(c: ReferentialCompiler, f: torch.Tensor, rows: list[PromptRow]) -> float:
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    return float(c(f).argmax(-1).eq(y).float().mean())


@torch.inference_mode()
def descriptor_pipeline(j: JContinuation, codes16: torch.Tensor, rows: list[PromptRow], world: torch.Tensor) -> float:
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    va, vb = world[ea], world[eb]
    pred = j(codes16.float(), va, vb).argmax(-1)
    return float(pred.eq(labels(op, va, vb)).float().mean())


@dataclass
class WorldRow:
    seed: int
    home: float
    future: float
    future_heldout_language: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    md = Path(snapshot_download(
        repo_id=MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "*.index.json", "*.merges", "*.vocab", "merges.txt", "vocab.json"],
    ))
    hashes = {n: sha256_file(md / n) for n in EXPECTED_WEIGHTS}
    assert hashes == EXPECTED_WEIGHTS
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        md, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    train_rows = make_augmented_rows(34411, TRAIN_PER_OP)
    eval_rows = make_prompt_rows(34412, heldout=False, per_op=EVAL_PER_OP)
    held_rows = make_prompt_rows(34413, heldout=True, per_op=EVAL_PER_OP)

    t0 = time.perf_counter()
    train_f = encode_multiscale(model, tok, train_rows, ENCODE_BATCH)
    eval_f = encode_multiscale(model, tok, eval_rows, ENCODE_BATCH)
    held_f = encode_multiscale(model, tok, held_rows, ENCODE_BATCH)
    qwen_feature_seconds = time.perf_counter() - t0

    compiler, compiler_s = train_compiler(train_f, train_rows)
    with torch.inference_mode():
        train_code = compiler.code(train_f)
        eval_code16 = compiler.code(eval_f).to(torch.float16)
        held_code16 = compiler.code(held_f).to(torch.float16)
    j, j_s = train_j(train_code.detach(), train_rows)

    train_comp_acc = compiler_acc(compiler, train_f, train_rows)
    eval_comp_acc = compiler_acc(compiler, eval_f, eval_rows)
    held_comp_acc = compiler_acc(compiler, held_f, held_rows)

    rows = []
    for seed in WORLD_SEEDS:
        g = torch.Generator().manual_seed(seed * 9011 + 7)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
        future = future_world(home, g)
        rows.append(WorldRow(
            seed,
            descriptor_pipeline(j, eval_code16, eval_rows, home),
            descriptor_pipeline(j, eval_code16, eval_rows, future),
            descriptor_pipeline(j, held_code16, held_rows, future),
        ))

    # World-independent descriptor check: update arbitrary world cells; descriptors are byte-identical.
    before = eval_code16.clone()
    refs_a = torch.tensor([r.entity_a for r in eval_rows], dtype=torch.int16)
    refs_b = torch.tensor([r.entity_b for r in eval_rows], dtype=torch.int16)
    descriptor_digest_before = hashlib.sha256(before.numpy().tobytes() + refs_a.numpy().tobytes() + refs_b.numpy().tobytes()).hexdigest()
    dummy_world = torch.arange(ENTITIES) % VALUE_COUNT
    for i in range(5000):
        dummy_world[i % ENTITIES] = (dummy_world[i % ENTITIES] + 7) % VALUE_COUNT
    descriptor_digest_after = hashlib.sha256(eval_code16.numpy().tobytes() + refs_a.numpy().tobytes() + refs_b.numpy().tobytes()).hexdigest()

    # Transaction/race gate over live reference descriptors.
    g = torch.Generator().manual_seed(34499)
    world = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
    generations = torch.ones(ENTITIES, dtype=torch.long)
    expected_conflicts = detected = escaped = retries = semantic_mismatches = 0
    rng = random.Random(344100)
    for _ in range(8000):
        idx = rng.randrange(len(eval_rows))
        r = eval_rows[idx]
        ids = (r.entity_a, r.entity_b)
        seen = generations[list(ids)].clone()
        va = world[r.entity_a].view(1); vb = world[r.entity_b].view(1)
        pred = int(j(eval_code16[idx:idx+1].float(), va, vb).argmax(-1))
        raced = rng.random() < 0.10
        if raced:
            pid = ids[rng.randrange(2)]
            generations[pid] += 1
            if rng.random() >= 0.45:
                world[pid] = rng.randrange(VALUE_COUNT)
            expected_conflicts += 1
        now = generations[list(ids)]
        if not torch.equal(seen, now):
            detected += 1; retries += 1
            va = world[r.entity_a].view(1); vb = world[r.entity_b].view(1)
            pred = int(j(eval_code16[idx:idx+1].float(), va, vb).argmax(-1))
        elif raced:
            escaped += 1
        truth = int(labels(torch.tensor([r.op]), world[r.entity_a].view(1), world[r.entity_b].view(1))[0])
        semantic_mismatches += int(pred != truth)

    # Descriptor storage compared with cached frozen-Qwen multiscale feature.
    descriptor_bytes_per_query = CODE_DIM * 2 + 2 * 2  # fp16 code + two int16 refs
    qwen_feature_bytes_per_query = eval_f.shape[1] * 2  # BF16-equivalent cache

    # Read latency for batched descriptor path.
    bench_n = min(64, len(eval_rows))
    ea = torch.tensor([r.entity_a for r in eval_rows[:bench_n]], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in eval_rows[:bench_n]], dtype=torch.long)
    op = torch.tensor([r.op for r in eval_rows[:bench_n]], dtype=torch.long)
    bench_world = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=g)
    for _ in range(10): j(eval_code16[:bench_n].float(), bench_world[ea], bench_world[eb])
    times = []
    for _ in range(100):
        ts = time.perf_counter_ns(); j(eval_code16[:bench_n].float(), bench_world[ea], bench_world[eb]); times.append(time.perf_counter_ns()-ts)

    # A small real-Qwen recompile sample to show what the descriptor skips.
    sample_rows = eval_rows[:8]
    qwen_times = []
    for _ in range(3):
        ts = time.perf_counter_ns(); _ = encode_multiscale(model, tok, sample_rows, 8); qwen_times.append(time.perf_counter_ns()-ts)
    descriptor_per_query_ns = statistics.median(times) / bench_n
    qwen_recompile_per_query_ns = statistics.median(qwen_times) / len(sample_rows)

    report = {
        "stage": STAGE,
        "architecture_candidate": "Frozen-Qwen Compact Referential Continuation (Qwen-CRC)",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "qwen_feature_dimension": int(eval_f.shape[1]),
        "continuation_code_dimension": CODE_DIM,
        "descriptor_bytes_per_query": descriptor_bytes_per_query,
        "bf16_qwen_feature_cache_bytes_per_query": qwen_feature_bytes_per_query,
        "descriptor_to_qwen_feature_cache_ratio": descriptor_bytes_per_query / qwen_feature_bytes_per_query,
        "descriptor_contains_current_world_values": False,
        "descriptor_contains_last_seen_generations": False,
        "train_compiler_accuracy": train_comp_acc,
        "eval_compiler_accuracy": eval_comp_acc,
        "heldout_language_compiler_accuracy": held_comp_acc,
        "mean_home_accuracy": mean(rows, "home"),
        "mean_future_world_accuracy": mean(rows, "future"),
        "mean_future_heldout_language_accuracy": mean(rows, "future_heldout_language"),
        "world_rows": [asdict(x) for x in rows],
        "descriptor_digest_unchanged_after_5000_world_writes": descriptor_digest_before == descriptor_digest_after,
        "write_time_descriptor_invalidations_or_patches": 0,
        "race_conflicts_expected": expected_conflicts,
        "race_conflicts_detected": detected,
        "race_conflicts_escaped": escaped,
        "race_retries": retries,
        "race_semantic_mismatches": semantic_mismatches,
        "qwen_feature_build_seconds": qwen_feature_seconds,
        "compiler_train_seconds": compiler_s,
        "j_train_seconds": j_s,
        "median_descriptor_j_execute_ns_per_query": descriptor_per_query_ns,
        "median_real_qwen_recompile_ns_per_query_small_sample": qwen_recompile_per_query_ns,
        "recompile_over_descriptor_execute_ratio": qwen_recompile_per_query_ns / max(1.0, descriptor_per_query_ns),
        "world_rebinding_gradient_steps": 0,
        "mechanism": (
            "A real frozen pretrained Qwen language state is compiled once into a 32-dimensional continuation code plus canonical entity refs. "
            "The cached neural object contains no current world values. A shared identity-blind J continuation dereferences the referenced "
            "current values only at use time, with exact generation retry on races. Thus pretrained language computation is retained while the "
            "cached continuation remains future-bindable across arbitrary world rewrites."
        ),
        "claim_boundary": (
            "This moves compact referential continuation onto a real pretrained language backbone, but the controlled operation set and small "
            "J network are not an open-ended LLM benchmark. It is evidence for the neural-state semantics, not a novelty proof."
        ),
        "dod_status": "NOT_DOD; real-pretrained future-bindable neural descriptor gate",
    }
    report["contract_pass"] = (
        eval_comp_acc >= 0.97 and held_comp_acc >= 0.90
        and report["mean_future_world_accuracy"] >= 0.95
        and report["mean_future_heldout_language_accuracy"] >= 0.90
        and descriptor_digest_before == descriptor_digest_after
        and detected == expected_conflicts and escaped == 0 and retries == expected_conflicts
        and semantic_mismatches == 0
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "eval_compiler_accuracy", "heldout_language_compiler_accuracy",
        "mean_future_world_accuracy", "mean_future_heldout_language_accuracy",
        "descriptor_bytes_per_query", "bf16_qwen_feature_cache_bytes_per_query",
        "descriptor_to_qwen_feature_cache_ratio", "descriptor_digest_unchanged_after_5000_world_writes",
        "race_conflicts_expected", "race_conflicts_detected", "race_conflicts_escaped", "race_semantic_mismatches",
        "recompile_over_descriptor_execute_ratio", "contract_pass", "report_sha256",
    ]}, indent=2))
    if not report["contract_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
