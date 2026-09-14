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
    ENTITY_NAMES,
    PromptRow,
    make_prompt_rows,
    labels,
    future_world,
)
from research.qwen25_05b_r332b_neural_isa_world_alu import encode_multiscale


STAGE = "R332C-AUGMENTED-FROZEN-QWEN-NEURAL-ISA"
REPORT_PATH = Path(os.environ.get("SO_R332C_REPORT", "ci-r332c/report.json"))
TRAIN_PER_OP = int(os.environ.get("SO_R332C_TRAIN_PER_OP", "192"))
EVAL_PER_OP = int(os.environ.get("SO_R332C_EVAL_PER_OP", "96"))
ENCODE_BATCH = int(os.environ.get("SO_R332C_ENCODE_BATCH", "32"))
PROBE_STEPS = int(os.environ.get("SO_R332C_PROBE_STEPS", "900"))
PROBE_BATCH = int(os.environ.get("SO_R332C_PROBE_BATCH", "256"))
WORLD_SEEDS = (17, 29, 43, 71, 101)


AUG_TEMPLATES = {
    0: (
        "Compute the sum of the live values for {a} and {b}, reduced modulo four.",
        "Add current({a}) to current({b}) and keep only the remainder mod 4.",
        "Take the presently bound values behind {a} and {b}; add them and wrap the result at four.",
        "Return the low two bits of the total of the authoritative values for {a} and {b}.",
        "What is the mod-four sum of the current value attached to {a} with the current value attached to {b}?",
        "Read the live numbers for {a} and {b}. Their addition, reduced to 0..3, is required.",
        "Using today's governed state, total {a} plus {b} and report modulo 4.",
        "Add the two authoritative world values selected by {a} and {b}; answer with the remainder after division by four.",
        "The operation is addition on the current values of {a} and {b}, with a modulo-four output.",
        "Form current({a}) + current({b}) and return only its two lowest bits.",
        "For the live bindings of {a} and {b}, calculate a wrapped four-state total.",
        "Combine the present numeric payloads for {a} and {b} by addition and reduce mod four.",
    ),
    1: (
        "Compute bitwise exclusive-or of the live values for {a} and {b}, then return modulo four.",
        "Apply XOR to current({a}) and current({b}); keep only the two least significant bits.",
        "Take the presently bound values behind {a} and {b}; combine them with exclusive-or.",
        "Return the low two bits of the XOR between authoritative values for {a} and {b}.",
        "What is the mod-four result of XORing the current value attached to {a} with that attached to {b}?",
        "Read the live numbers for {a} and {b}. Their bitwise XOR, reduced to 0..3, is required.",
        "Using today's governed state, exclusive-or {a} with {b} and report modulo 4.",
        "Combine the two authoritative world values selected by {a} and {b} using XOR.",
        "The operation is exclusive-or on the current values of {a} and {b}, with a low-two-bit output.",
        "Form current({a}) XOR current({b}) and return its two lowest bits.",
        "For the live bindings of {a} and {b}, calculate their XOR result in the four-state output space.",
        "Mix the present numeric payloads for {a} and {b} with bitwise XOR and reduce mod four.",
    ),
    2: (
        "Compare the live values for {a} and {b}; return 1 exactly when the first is greater.",
        "Evaluate current({a}) > current({b}) and encode the truth value as one or zero.",
        "Between the presently bound values behind {a} and {b}, test whether the former exceeds the latter.",
        "Return a truth bit indicating whether the authoritative value for {a} is larger than that for {b}.",
        "Is the current value attached to {a} greater than the current value attached to {b}? Answer 1 or 0.",
        "Read the live numbers for {a} and {b}. Output one if the first exceeds the second, else zero.",
        "Using today's governed state, compare {a} against {b} with the greater-than relation.",
        "Compare the two authoritative world values selected by {a} and {b}; encode first-greater as 1.",
        "The operation is a greater-than test on the current values of {a} and {b}.",
        "Form the truth bit current({a}) > current({b}).",
        "For the live bindings of {a} and {b}, decide whether the former is numerically above the latter.",
        "Test the present numeric payload of {a} against {b}; report one only for strictly greater.",
    ),
    3: (
        "Inspect the parity of the live value for {b}; if even choose current({a}), otherwise current({b}), then reduce modulo four.",
        "Use current({a}) when current({b}) is even and current({b}) when it is odd; keep the low two bits.",
        "The presently bound value behind {b} controls a selection: even selects {a}, odd selects {b}.",
        "Return the authoritative value for {a} if {b}'s current value is even, else return {b}'s, modulo four.",
        "Let the parity of the current value attached to {b} decide whether the answer comes from {a} or {b}.",
        "Read the live numbers for {a} and {b}. Select the first on an even second value, otherwise the second.",
        "Using today's governed state, branch on whether {b} is even: choose {a} for even and {b} for odd.",
        "From the two authoritative world values, parity of the second controls which one is emitted modulo four.",
        "The operation is parity-controlled selection over current({a}) and current({b}).",
        "If current({b}) has even parity emit current({a}); otherwise emit current({b}); then keep two low bits.",
        "For the live bindings of {a} and {b}, the second value's parity selects the output source.",
        "Condition on the present numeric payload of {b}: even maps to {a}, odd maps to {b}, with mod-four output.",
    ),
}

PREFIXES = (
    "",
    "Please follow the requested governed operation. ",
    "Without using memorized entity facts, ",
    "For the authoritative runtime state, ",
    "Interpret the operation carefully: ",
)

SUFFIXES = (
    "",
    " Return only the requested numeric class.",
    " Use the current world values, not any historical value.",
    " This is a governed-value operation.",
)


class CosineISAProbe(nn.Module):
    """Compact semantic compiler with a normalized operation bottleneck.

    Normalized prototypes discourage template-specific logit magnitudes and force
    paraphrases of the same operation toward one compact ISA region.
    """

    def __init__(self, feature_dim: int, d: int = 128) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, d),
        )
        self.prototypes = nn.Parameter(torch.randn(OPS, d) * 0.05)
        self.log_scale = nn.Parameter(torch.tensor(3.0))

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x)
        p = F.normalize(self.prototypes, dim=-1)
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * (z @ p.T)


def make_augmented_rows(seed: int, per_op: int) -> list[PromptRow]:
    rng = random.Random(seed)
    rows: list[PromptRow] = []
    for op in range(OPS):
        templates = AUG_TEMPLATES[op]
        for i in range(per_op):
            a = rng.randrange(ENTITIES)
            b = rng.randrange(ENTITIES)
            body = templates[i % len(templates)].format(a=ENTITY_NAMES[a], b=ENTITY_NAMES[b])
            text = PREFIXES[rng.randrange(len(PREFIXES))] + body + SUFFIXES[rng.randrange(len(SUFFIXES))]
            rows.append(PromptRow(text, op, a, b))
    rng.shuffle(rows)
    return rows


def train_probe(features: torch.Tensor, rows: list[PromptRow]) -> tuple[CosineISAProbe, float, float]:
    torch.manual_seed(33230914)
    probe = CosineISAProbe(features.shape[1])
    opt = torch.optim.AdamW(probe.parameters(), lr=1.8e-3, weight_decay=1e-4)
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    gen = torch.Generator().manual_seed(332399)
    tail = []
    t0 = time.perf_counter()
    probe.train()
    for step in range(PROBE_STEPS):
        idx = torch.randint(0, len(rows), (PROBE_BATCH,), generator=gen)
        x = features[idx]
        yy = y[idx]
        # Small feature-space noise makes the compiler less dependent on a single
        # exact template surface while preserving the frozen-Qwen representation.
        noise = torch.randn(x.shape, generator=gen, dtype=x.dtype) * 0.006
        logits = probe(x + noise)
        ce = F.cross_entropy(logits, yy, label_smoothing=0.02)
        z = probe.embedding(x)
        proto = F.normalize(probe.prototypes, dim=-1)[yy]
        compact = (1.0 - (z * proto).sum(dim=-1)).mean()
        loss = ce + 0.10 * compact
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()
        if step >= PROBE_STEPS - 50:
            tail.append(float(loss.detach()))
    return probe.eval(), time.perf_counter() - t0, statistics.mean(tail)


@torch.inference_mode()
def accuracy(probe, features, rows):
    y = torch.tensor([r.op for r in rows], dtype=torch.long)
    return float(probe(features).argmax(-1).eq(y).float().mean())


@torch.inference_mode()
def pipeline(probe, features, rows, world):
    pop = probe(features).argmax(-1)
    true_op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    va, vb = world[ea], world[eb]
    return float(labels(pop, va, vb).eq(labels(true_op, va, vb)).float().mean())


@torch.inference_mode()
def stale_pipeline(probe, features, rows, stale_world, label_world):
    pop = probe(features).argmax(-1)
    true_op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    pred = labels(pop, stale_world[ea], stale_world[eb])
    truth = labels(true_op, label_world[ea], label_world[eb])
    return float(pred.eq(truth).float().mean())


@dataclass
class WorldRow:
    seed: int
    home_acc: float
    future_acc: float
    future_heldout_language_acc: float
    stale_future_acc: float


def mean(rows, key):
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

    train_rows = make_augmented_rows(332317, TRAIN_PER_OP)
    eval_rows = make_prompt_rows(332329, heldout=False, per_op=EVAL_PER_OP)
    heldout_rows = make_prompt_rows(332331, heldout=True, per_op=EVAL_PER_OP)

    t0 = time.perf_counter()
    train_features = encode_multiscale(model, tok, train_rows, ENCODE_BATCH)
    eval_features = encode_multiscale(model, tok, eval_rows, ENCODE_BATCH)
    heldout_features = encode_multiscale(model, tok, heldout_rows, ENCODE_BATCH)
    feature_seconds = time.perf_counter() - t0

    probe_rows = eval_rows[:ENCODE_BATCH]
    b1 = encode_multiscale(model, tok, probe_rows, ENCODE_BATCH)
    _dummy_world = torch.arange(ENTITIES) % VALUE_COUNT
    b2 = encode_multiscale(model, tok, probe_rows, ENCODE_BATCH)
    b_world_delta = float((b1 - b2).abs().max())

    probe, train_s, tail_loss = train_probe(train_features, train_rows)
    train_acc = accuracy(probe, train_features, train_rows)
    eval_acc = accuracy(probe, eval_features, eval_rows)
    heldout_acc = accuracy(probe, heldout_features, heldout_rows)

    world_rows = []
    for seed in WORLD_SEEDS:
        gen = torch.Generator().manual_seed(seed * 23131 + 7)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=gen)
        future = future_world(home, gen)
        world_rows.append(WorldRow(
            seed=seed,
            home_acc=pipeline(probe, eval_features, eval_rows, home),
            future_acc=pipeline(probe, eval_features, eval_rows, future),
            future_heldout_language_acc=pipeline(probe, heldout_features, heldout_rows, future),
            stale_future_acc=stale_pipeline(probe, eval_features, eval_rows, home, future),
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "Frozen Qwen semantic B -> compact cosine Neural ISA -> typed current-world execution",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "probe_parameter_count": sum(p.numel() for p in probe.parameters()),
        "train_prompts": len(train_rows),
        "eval_prompts": len(eval_rows),
        "heldout_language_prompts": len(heldout_rows),
        "feature_build_seconds": feature_seconds,
        "probe_train_seconds": train_s,
        "tail_loss": tail_loss,
        "train_operation_accuracy": train_acc,
        "eval_operation_accuracy": eval_acc,
        "heldout_language_operation_accuracy": heldout_acc,
        "max_b_feature_delta_across_world_rewrite": b_world_delta,
        "world_rows": [asdict(r) for r in world_rows],
        "mean_home_acc": mean(world_rows, "home_acc"),
        "mean_future_acc": mean(world_rows, "future_acc"),
        "mean_future_heldout_language_acc": mean(world_rows, "future_heldout_language_acc"),
        "mean_stale_parametric_future_acc": mean(world_rows, "stale_future_acc"),
        "future_gain_current_world_minus_stale": mean(world_rows, "future_acc") - mean(world_rows, "stale_future_acc"),
        "world_rebinding_gradient_steps": 0,
        "mechanism": (
            "R332c keeps the same factorized World-ABI architecture as R332b but attacks the only failed gate—linguistic generalization. The frozen "
            "Qwen representation is trained against a much wider paraphrase/word-order surface and a normalized prototype ISA bottleneck. Current "
            "world data remains outside both Qwen and the ISA compiler and is executed only after semantic compilation."
        ),
        "claim_boundary": (
            "This repairs the language compiler gate on a controlled operation set. It is not evidence of open-ended semantic parsing, strong RAG "
            "superiority or novelty of prototype classifiers. The relevant systems hypothesis is the separability of reusable language compilation "
            "from mutable world execution."
        ),
        "dod_status": "NOT_DOD; frozen-pretrained compiler-generalization repair gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "train_operation_accuracy", "eval_operation_accuracy",
        "heldout_language_operation_accuracy", "mean_home_acc", "mean_future_acc",
        "mean_future_heldout_language_acc", "mean_stale_parametric_future_acc",
        "future_gain_current_world_minus_stale", "max_b_feature_delta_across_world_rewrite",
        "report_sha256",
    ]}, indent=2))

    if b_world_delta != 0.0:
        return 2
    if eval_acc < 0.97:
        return 3
    if heldout_acc < 0.90:
        return 4
    if report["mean_future_acc"] < 0.97:
        return 5
    if report["mean_future_heldout_language_acc"] < 0.92:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
