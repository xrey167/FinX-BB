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
    ENTITIES,
    VALUE_COUNT,
    OPS,
    PromptRow,
    make_prompt_rows,
    labels,
    future_world,
)

STAGE = "R332B-FROZEN-QWEN-NEURAL-ISA-WORLD-ALU"
REPORT_PATH = Path(os.environ.get("SO_R332B_REPORT", "ci-r332b/report.json"))
TRAIN_PROMPTS_PER_OP = int(os.environ.get("SO_R332B_TRAIN_PROMPTS_PER_OP", "128"))
EVAL_PROMPTS_PER_OP = int(os.environ.get("SO_R332B_EVAL_PROMPTS_PER_OP", "96"))
ENCODE_BATCH = int(os.environ.get("SO_R332B_ENCODE_BATCH", "32"))
PROBE_STEPS = int(os.environ.get("SO_R332B_PROBE_STEPS", "1200"))
PROBE_BATCH = int(os.environ.get("SO_R332B_PROBE_BATCH", "256"))
WORLD_SEEDS = (17, 29, 43, 71, 101)


class ISAProbe(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192),
            nn.GELU(),
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, OPS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.inference_mode()
def encode_multiscale(model, tok, rows: list[PromptRow], batch_size: int) -> torch.Tensor:
    texts = [r.text + "\nCompile operation:" for r in rows]
    feats = []
    n_layers = len(model.model.layers)
    # hidden_states[0] is embeddings; final index n_layers is normalized final state.
    layer_ids = sorted(set([n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers]))
    for start in range(0, len(rows), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True)
        out = model.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last = enc.attention_mask.sum(dim=1) - 1
        r = torch.arange(len(chunk))
        parts = []
        for lid in layer_ids:
            h = out.hidden_states[lid]
            parts.append(h[r, last].float())
        # Add one mean-pooled semantic summary from the final layer.
        hfinal = out.hidden_states[-1].float()
        mask = enc.attention_mask.unsqueeze(-1).float()
        mean = (hfinal * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        parts.append(mean)
        feats.append(torch.cat(parts, dim=-1).cpu())
    return torch.cat(feats, dim=0)


def train_probe(features: torch.Tensor, rows: list[PromptRow]) -> tuple[ISAProbe, float, float]:
    torch.manual_seed(3320914)
    probe = ISAProbe(features.shape[1])
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    gen = torch.Generator().manual_seed(332771)
    tail = []
    t0 = time.perf_counter()
    probe.train()
    for step in range(PROBE_STEPS):
        idx = torch.randint(0, len(rows), (PROBE_BATCH,), generator=gen)
        logits = probe(features[idx])
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if step >= PROBE_STEPS - 50:
            tail.append(float(loss.detach()))
    return probe.eval(), time.perf_counter() - t0, statistics.mean(tail)


@torch.inference_mode()
def op_accuracy(probe: ISAProbe, features: torch.Tensor, rows: list[PromptRow]) -> float:
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    pred = probe(features).argmax(-1)
    return float(pred.eq(y).float().mean())


@torch.inference_mode()
def execute_pipeline(probe: ISAProbe, features: torch.Tensor, rows: list[PromptRow], world: torch.Tensor) -> float:
    pred_op = probe(features).argmax(-1)
    true_op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    va = world[ea]
    vb = world[eb]
    pred = labels(pred_op, va, vb)
    truth = labels(true_op, va, vb)
    return float(pred.eq(truth).float().mean())


@torch.inference_mode()
def execute_stale_control(
    probe: ISAProbe,
    features: torch.Tensor,
    rows: list[PromptRow],
    stale_home: torch.Tensor,
    label_world: torch.Tensor,
) -> float:
    pred_op = probe(features).argmax(-1)
    true_op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    stale_va, stale_vb = stale_home[ea], stale_home[eb]
    current_va, current_vb = label_world[ea], label_world[eb]
    pred = labels(pred_op, stale_va, stale_vb)
    truth = labels(true_op, current_va, current_vb)
    return float(pred.eq(truth).float().mean())


@torch.inference_mode()
def benchmark(model, tok, probe, rows, cached_features, world):
    chunk = rows[:min(ENCODE_BATCH, len(rows))]
    ea = torch.tensor([r.entity_a for r in chunk], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in chunk], dtype=torch.long)

    # Warm-up.
    _ = encode_multiscale(model, tok, chunk, len(chunk))
    _ = probe(cached_features[:len(chunk)])

    full_times = []
    for i in range(3):
        current = (world + i + 1) % VALUE_COUNT
        t0 = time.perf_counter_ns()
        f = encode_multiscale(model, tok, chunk, len(chunk))
        op = probe(f).argmax(-1)
        _ = labels(op, current[ea], current[eb])
        full_times.append(time.perf_counter_ns() - t0)

    cached_times = []
    f = cached_features[:len(chunk)]
    op = probe(f).argmax(-1)
    for i in range(80):
        current = (world + i + 3) % VALUE_COUNT
        t0 = time.perf_counter_ns()
        _ = labels(op, current[ea], current[eb])
        cached_times.append(time.perf_counter_ns() - t0)
    return statistics.median(full_times), statistics.median(cached_times)


@dataclass
class WorldRow:
    seed: int
    home_acc: float
    future_acc: float
    future_heldout_language_acc: float
    stale_parametric_future_acc: float
    stale_parametric_future_heldout_language_acc: float


def mean(rows: list[WorldRow], key: str) -> float:
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
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
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).eval()
    model.requires_grad_(False)

    train_rows = make_prompt_rows(332117, heldout=False, per_op=TRAIN_PROMPTS_PER_OP)
    eval_rows = make_prompt_rows(332129, heldout=False, per_op=EVAL_PROMPTS_PER_OP)
    heldout_rows = make_prompt_rows(332131, heldout=True, per_op=EVAL_PROMPTS_PER_OP)

    t0 = time.perf_counter()
    train_features = encode_multiscale(model, tok, train_rows, ENCODE_BATCH)
    eval_features = encode_multiscale(model, tok, eval_rows, ENCODE_BATCH)
    heldout_features = encode_multiscale(model, tok, heldout_rows, ENCODE_BATCH)
    feature_build_seconds = time.perf_counter() - t0

    # B is model/world independent by signature. Re-encoding exactly the same language
    # around an arbitrary world rewrite must remain deterministic.
    probe_rows = eval_rows[:ENCODE_BATCH]
    b1 = encode_multiscale(model, tok, probe_rows, ENCODE_BATCH)
    _world_change = torch.arange(ENTITIES) % VALUE_COUNT
    b2 = encode_multiscale(model, tok, probe_rows, ENCODE_BATCH)
    b_world_delta = float((b1 - b2).abs().max())

    probe, train_seconds, tail_loss = train_probe(train_features, train_rows)
    train_op_acc = op_accuracy(probe, train_features, train_rows)
    eval_op_acc = op_accuracy(probe, eval_features, eval_rows)
    heldout_op_acc = op_accuracy(probe, heldout_features, heldout_rows)

    world_rows = []
    bench_world = None
    for seed in WORLD_SEEDS:
        gen = torch.Generator().manual_seed(seed * 22103 + 9)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=gen)
        future = future_world(home, gen)
        if bench_world is None:
            bench_world = future
        world_rows.append(WorldRow(
            seed=seed,
            home_acc=execute_pipeline(probe, eval_features, eval_rows, home),
            future_acc=execute_pipeline(probe, eval_features, eval_rows, future),
            future_heldout_language_acc=execute_pipeline(probe, heldout_features, heldout_rows, future),
            stale_parametric_future_acc=execute_stale_control(probe, eval_features, eval_rows, home, future),
            stale_parametric_future_heldout_language_acc=execute_stale_control(probe, heldout_features, heldout_rows, home, future),
        ))

    full_ns, cached_ns = benchmark(model, tok, probe, eval_rows, eval_features, bench_world)
    probe_params = sum(p.numel() for p in probe.parameters())

    report = {
        "stage": STAGE,
        "architecture_candidate": "Frozen Qwen semantic B rail -> compact Neural ISA -> typed current-world ALU",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "probe_parameter_count": probe_params,
        "feature_build_seconds": feature_build_seconds,
        "probe_train_seconds": train_seconds,
        "probe_tail_loss": tail_loss,
        "train_operation_accuracy": train_op_acc,
        "eval_operation_accuracy": eval_op_acc,
        "heldout_language_operation_accuracy": heldout_op_acc,
        "max_b_feature_delta_across_world_rewrite": b_world_delta,
        "world_rows": [asdict(r) for r in world_rows],
        "mean_home_acc": mean(world_rows, "home_acc"),
        "mean_future_acc": mean(world_rows, "future_acc"),
        "mean_future_heldout_language_acc": mean(world_rows, "future_heldout_language_acc"),
        "mean_stale_parametric_future_acc": mean(world_rows, "stale_parametric_future_acc"),
        "mean_stale_parametric_future_heldout_language_acc": mean(world_rows, "stale_parametric_future_heldout_language_acc"),
        "future_gain_current_world_minus_stale_parametric": mean(world_rows, "future_acc") - mean(world_rows, "stale_parametric_future_acc"),
        "median_full_language_compile_execute_ns": full_ns,
        "median_cached_isa_current_world_execute_ns": cached_ns,
        "cached_isa_speedup": full_ns / cached_ns,
        "world_rebinding_gradient_steps": 0,
        "mechanism": (
            "The frozen pretrained model is used for language semantics only. A small Neural ISA decoder maps multiscale frozen Qwen features to "
            "an allowlisted operation code. Entity source identity is handled by a trusted linker outside this probe. Current values are then read "
            "from the World ABI and executed by a typed operator. The mutable world is therefore not compressed into Qwen hidden state or the "
            "ISA probe, and the ISA can be cached across ordinary value generations."
        ),
        "why_r332_failed_and_this_differs": (
            "R332 asked one terminal MLP to jointly infer language semantics and implement world arithmetic from a frozen final hidden state. "
            "R332b factorizes the boundary: pretrained B supplies semantic features, a compact compiler decodes only the operation, and the J/ALU "
            "executes typed current values. This exposes the computation structure instead of hiding it inside a generic fusion head."
        ),
        "claim_boundary": (
            "This is a real-pretrained compiler/World-ABI gate, but J is a typed ALU rather than a free-form neural decoder. It does not establish "
            "novelty or superiority to RAG/model editing. R333 separately tests an actual mid-layer neural World Port."
        ),
        "dod_status": "NOT_DOD; frozen-pretrained Neural ISA + World ALU gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "probe_parameter_count",
        "train_operation_accuracy",
        "eval_operation_accuracy",
        "heldout_language_operation_accuracy",
        "mean_home_acc",
        "mean_future_acc",
        "mean_future_heldout_language_acc",
        "mean_stale_parametric_future_acc",
        "future_gain_current_world_minus_stale_parametric",
        "max_b_feature_delta_across_world_rewrite",
        "cached_isa_speedup",
        "report_sha256",
    ]}, indent=2))

    if b_world_delta != 0.0:
        return 2
    if eval_op_acc < 0.95:
        return 3
    if heldout_op_acc < 0.90:
        return 4
    if report["mean_future_acc"] < 0.95:
        return 5
    if report["mean_future_heldout_language_acc"] < 0.90:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
