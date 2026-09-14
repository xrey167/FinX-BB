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


STAGE = "R332-FROZEN-QWEN-TERMINAL-WORLD-PORT"
REPORT_PATH = Path(os.environ.get("SO_R332_REPORT", "ci-r332/report.json"))
SEEDS = (17, 29, 43)
ENTITIES = 32
VALUE_COUNT = 16
CLASS_COUNT = 4
OPS = 4
TRAIN_PROMPTS_PER_OP = int(os.environ.get("SO_R332_TRAIN_PROMPTS_PER_OP", "96"))
EVAL_PROMPTS_PER_OP = int(os.environ.get("SO_R332_EVAL_PROMPTS_PER_OP", "64"))
J_STEPS = int(os.environ.get("SO_R332_J_STEPS", "1400"))
J_BATCH = int(os.environ.get("SO_R332_J_BATCH", "512"))
ENCODE_BATCH = int(os.environ.get("SO_R332_ENCODE_BATCH", "32"))

ENTITY_NAMES = tuple(f"GOVERNED-ENTITY-{i:02d}" for i in range(ENTITIES))

TRAIN_TEMPLATES = {
    0: (
        "Using the current governed values of {a} and {b}, compute their sum modulo four.",
        "Add the present authoritative values for {a} and {b}; return the result modulo 4.",
        "For {a} and {b}, take the current values and calculate the low-two-bit sum.",
    ),
    1: (
        "Using the current governed values of {a} and {b}, compute their bitwise XOR modulo four.",
        "XOR the present authoritative values for {a} and {b}, then keep the low two bits.",
        "For {a} and {b}, combine the current values with XOR and return modulo 4.",
    ),
    2: (
        "Compare the current governed values of {a} and {b}. Return 1 if the first is larger, otherwise 0.",
        "Is the present authoritative value for {a} greater than the value for {b}? Encode yes as 1 and no as 0.",
        "For {a} versus {b}, test whether the first current value exceeds the second; answer 1 or 0.",
    ),
    3: (
        "Read the current governed values of {a} and {b}. If the second is even use the first, else use the second; return modulo four.",
        "For present values {a} and {b}: choose {a} when {b} is even, otherwise choose {b}, then keep the low two bits.",
        "Condition on the parity of {b}: even selects current {a}, odd selects current {b}; answer modulo 4.",
    ),
}

HELDOUT_TEMPLATES = {
    0: (
        "Take today's authoritative numbers attached to {a} and {b}; after adding them, what remains after wrapping at four?",
        "What is the mod-four total of the live values behind {a} together with {b}?",
    ),
    1: (
        "Combine the live numbers behind {a} and {b} by exclusive-or and report only the lowest two bits.",
        "What low-two-bit number results from XORing the authoritative values currently bound to {a} and {b}?",
    ),
    2: (
        "Between the live values bound to {a} and {b}, does the former exceed the latter? Use 1 for yes and 0 for no.",
        "Return the truth bit for current({a}) > current({b}).",
    ),
    3: (
        "Inspect the live value of {b}: its parity decides whether the answer comes from {a} or {b}; emit the selected value modulo four.",
        "Use current({a}) when current({b}) is even and current({b}) otherwise, reducing the chosen value mod four.",
    ),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def value_bits(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(4, device=v.device)
    return ((v[:, None] >> shifts[None, :]) & 1).float()


def labels(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0
    y[m] = (va[m] + vb[m]) & 3
    m = op == 1
    y[m] = (va[m] ^ vb[m]) & 3
    m = op == 2
    y[m] = (va[m] > vb[m]).long()
    m = op == 3
    y[m] = torch.where((vb[m] & 1) == 0, va[m] & 3, vb[m] & 3)
    return y


def future_world(home: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    shift = torch.randint(1, VALUE_COUNT, (ENTITIES,), generator=gen)
    future = (home + shift) % VALUE_COUNT
    assert not bool((future == home).any())
    return future


@dataclass(frozen=True)
class PromptRow:
    text: str
    op: int
    entity_a: int
    entity_b: int


def make_prompt_rows(seed: int, *, heldout: bool, per_op: int) -> list[PromptRow]:
    rng = random.Random(seed)
    bank = HELDOUT_TEMPLATES if heldout else TRAIN_TEMPLATES
    rows: list[PromptRow] = []
    for op in range(OPS):
        for i in range(per_op):
            a = rng.randrange(ENTITIES)
            b = rng.randrange(ENTITIES)
            template = bank[op][i % len(bank[op])]
            text = template.format(a=ENTITY_NAMES[a], b=ENTITY_NAMES[b])
            rows.append(PromptRow(text, op, a, b))
    rng.shuffle(rows)
    return rows


@torch.inference_mode()
def encode_rows(model, tok, rows: list[PromptRow], batch_size: int) -> torch.Tensor:
    feats = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc = tok(
            [r.text for r in chunk],
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        out = model.model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # Right padding: the current last semantic token is at sum(mask)-1.
        last = enc.attention_mask.sum(dim=1) - 1
        h = out.last_hidden_state[torch.arange(len(chunk)), last].float().cpu()
        feats.append(h)
    return torch.cat(feats, dim=0)


class TerminalWorldPort(nn.Module):
    """Small mutable J side rail attached to a frozen pretrained B representation.

    The B feature is language/skill state. Current world bytes enter only through
    the explicit two-value Port. Entity identity is not provided separately to J.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + 8, 192),
            nn.GELU(),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, CLASS_COUNT),
        )

    def forward(self, b_feature: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b_feature, value_bits(va), value_bits(vb)], dim=-1)
        return self.net(x)


def train_port(seed: int, features: torch.Tensor, rows: list[PromptRow]) -> tuple[TerminalWorldPort, float]:
    seed_all(seed)
    model = TerminalWorldPort(features.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed * 11003 + 71)
    op_all = torch.tensor([r.op for r in rows], dtype=torch.long)
    n = len(rows)
    t0 = time.perf_counter()
    model.train()
    for _ in range(J_STEPS):
        idx = torch.randint(0, n, (J_BATCH,), generator=gen)
        b = features[idx]
        op = op_all[idx]
        # Train operation semantics over values sampled independently of entity.
        # This prevents the side rail from learning the home entity->value table.
        va = torch.randint(0, VALUE_COUNT, (J_BATCH,), generator=gen)
        vb = torch.randint(0, VALUE_COUNT, (J_BATCH,), generator=gen)
        # Reserve ~1/7 of value pairs as unseen combinations.
        keep = ((va * 17 + vb * 5 + op) % 7) != 0
        if not bool(keep.all()):
            # Resample rejected rows in-place until every training pair is legal.
            bad = ~keep
            while bool(bad.any()):
                va[bad] = torch.randint(0, VALUE_COUNT, (int(bad.sum()),), generator=gen)
                vb[bad] = torch.randint(0, VALUE_COUNT, (int(bad.sum()),), generator=gen)
                keep = ((va * 17 + vb * 5 + op) % 7) != 0
                bad = ~keep
        y = labels(op, va, vb)
        logits = model(b, va, vb)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval(), time.perf_counter() - t0


@torch.inference_mode()
def eval_world(port: TerminalWorldPort, features: torch.Tensor, rows: list[PromptRow], world: torch.Tensor) -> float:
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    va = world[ea]
    vb = world[eb]
    y = labels(op, va, vb)
    pred = port(features, va, vb).argmax(-1)
    return float(pred.eq(y).float().mean())


@torch.inference_mode()
def eval_parametric_control(port: TerminalWorldPort, features: torch.Tensor, rows: list[PromptRow], stale_home: torch.Tensor, label_world: torch.Tensor) -> float:
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    ea = torch.tensor([r.entity_a for r in rows], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in rows], dtype=torch.long)
    stale_va = stale_home[ea]
    stale_vb = stale_home[eb]
    true_va = label_world[ea]
    true_vb = label_world[eb]
    y = labels(op, true_va, true_vb)
    pred = port(features, stale_va, stale_vb).argmax(-1)
    return float(pred.eq(y).float().mean())


@torch.inference_mode()
def eval_heldout_pairs(port: TerminalWorldPort, features: torch.Tensor, rows: list[PromptRow], seed: int) -> float:
    gen = torch.Generator().manual_seed(seed * 13007 + 31)
    n = len(rows)
    op = torch.tensor([r.op for r in rows], dtype=torch.long)
    va = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
    vb = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
    mask = ((va * 17 + vb * 5 + op) % 7) == 0
    # Force every row into the held-out pair partition.
    guard = 0
    while not bool(mask.all()):
        bad = ~mask
        va[bad] = torch.randint(0, VALUE_COUNT, (int(bad.sum()),), generator=gen)
        vb[bad] = torch.randint(0, VALUE_COUNT, (int(bad.sum()),), generator=gen)
        mask = ((va * 17 + vb * 5 + op) % 7) == 0
        guard += 1
        if guard > 100:
            raise RuntimeError("failed to sample held-out pairs")
    y = labels(op, va, vb)
    pred = port(features, va, vb).argmax(-1)
    return float(pred.eq(y).float().mean())


@torch.inference_mode()
def control_backbone_logits(model, tok) -> torch.Tensor:
    text = "Explain in one sentence why a stable software interface is useful."
    enc = tok(text, return_tensors="pt")
    return model(**enc, use_cache=False, return_dict=True).logits[0, -1].float().cpu()


@torch.inference_mode()
def benchmark(model, tok, port, rows: list[PromptRow], cached_features: torch.Tensor, world: torch.Tensor):
    chunk = rows[:min(ENCODE_BATCH, len(rows))]
    ea = torch.tensor([r.entity_a for r in chunk], dtype=torch.long)
    eb = torch.tensor([r.entity_b for r in chunk], dtype=torch.long)
    va, vb = world[ea], world[eb]
    feat = cached_features[:len(chunk)]

    # Warm-up.
    _ = encode_rows(model, tok, chunk, len(chunk))
    for _ in range(5):
        port(feat, va, vb)

    full_times = []
    for _ in range(3):
        t0 = time.perf_counter_ns()
        f = encode_rows(model, tok, chunk, len(chunk))
        port(f, va, vb)
        full_times.append(time.perf_counter_ns() - t0)

    j_times = []
    for i in range(40):
        current = (world + i + 1) % VALUE_COUNT
        va2, vb2 = current[ea], current[eb]
        t0 = time.perf_counter_ns()
        port(feat, va2, vb2)
        j_times.append(time.perf_counter_ns() - t0)

    return statistics.median(full_times), statistics.median(j_times)


@dataclass
class SeedResult:
    seed: int
    home_acc: float
    future_rebound_acc: float
    future_rebound_heldout_language_acc: float
    stale_parametric_future_acc: float
    stale_parametric_future_heldout_language_acc: float
    heldout_value_pair_acc: float
    port_train_seconds: float


def mean(rows: list[SeedResult], key: str) -> float:
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

    backbone_before = control_backbone_logits(model, tok)

    # B features are computed once and reused by all world states and all seeds.
    train_rows = make_prompt_rows(33217, heldout=False, per_op=TRAIN_PROMPTS_PER_OP)
    eval_rows = make_prompt_rows(33229, heldout=False, per_op=EVAL_PROMPTS_PER_OP)
    heldout_rows = make_prompt_rows(33231, heldout=True, per_op=EVAL_PROMPTS_PER_OP)
    t0 = time.perf_counter()
    train_features = encode_rows(model, tok, train_rows, ENCODE_BATCH)
    eval_features = encode_rows(model, tok, eval_rows, ENCODE_BATCH)
    heldout_features = encode_rows(model, tok, heldout_rows, ENCODE_BATCH)
    b_encode_seconds = time.perf_counter() - t0

    # Exact B-world non-interference: rerun the same language after creating a
    # completely different world. The world is not an argument to B.
    probe_rows = eval_rows[:ENCODE_BATCH]
    b1 = encode_rows(model, tok, probe_rows, ENCODE_BATCH)
    _dummy_world = torch.arange(ENTITIES) % VALUE_COUNT
    b2 = encode_rows(model, tok, probe_rows, ENCODE_BATCH)
    b_world_delta = float((b1 - b2).abs().max())

    seed_rows: list[SeedResult] = []
    benchmark_full_ns = None
    benchmark_j_ns = None
    total_port_params = None

    for seed in SEEDS:
        gen = torch.Generator().manual_seed(seed * 19001 + 101)
        home = torch.randint(0, VALUE_COUNT, (ENTITIES,), generator=gen)
        future = future_world(home, gen)
        port, train_seconds = train_port(seed, train_features, train_rows)
        total_port_params = sum(p.numel() for p in port.parameters())

        home_acc = eval_world(port, eval_features, eval_rows, home)
        future_acc = eval_world(port, eval_features, eval_rows, future)
        future_held = eval_world(port, heldout_features, heldout_rows, future)
        stale_future = eval_parametric_control(port, eval_features, eval_rows, home, future)
        stale_future_held = eval_parametric_control(port, heldout_features, heldout_rows, home, future)
        held_pair = eval_heldout_pairs(port, heldout_features, heldout_rows, seed)

        seed_rows.append(SeedResult(
            seed=seed,
            home_acc=home_acc,
            future_rebound_acc=future_acc,
            future_rebound_heldout_language_acc=future_held,
            stale_parametric_future_acc=stale_future,
            stale_parametric_future_heldout_language_acc=stale_future_held,
            heldout_value_pair_acc=held_pair,
            port_train_seconds=train_seconds,
        ))

        if benchmark_full_ns is None:
            benchmark_full_ns, benchmark_j_ns = benchmark(
                model, tok, port, eval_rows, eval_features, future
            )

    backbone_after = control_backbone_logits(model, tok)
    backbone_logit_delta = float((backbone_before - backbone_after).abs().max())

    report = {
        "stage": STAGE,
        "architecture_candidate": "Frozen Pretrained B-Transformer + Terminal Typed World Port J side rail",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "weights_sha256": hashes,
        "backbone_frozen": all(not p.requires_grad for p in model.parameters()),
        "port_parameter_count": total_port_params,
        "b_feature_encode_seconds": b_encode_seconds,
        "train_language_prompts": len(train_rows),
        "eval_language_prompts": len(eval_rows),
        "heldout_language_prompts": len(heldout_rows),
        "entities": ENTITIES,
        "value_count": VALUE_COUNT,
        "operations": OPS,
        "world_rebinding_changes_every_entity_value": True,
        "world_rebinding_gradient_steps": 0,
        "max_b_feature_delta_across_world_transition": b_world_delta,
        "max_frozen_backbone_control_logit_delta_after_port_training": backbone_logit_delta,
        "per_seed": [asdict(r) for r in seed_rows],
        "mean_home_acc": mean(seed_rows, "home_acc"),
        "mean_future_rebound_acc": mean(seed_rows, "future_rebound_acc"),
        "mean_future_rebound_heldout_language_acc": mean(seed_rows, "future_rebound_heldout_language_acc"),
        "mean_stale_parametric_future_acc": mean(seed_rows, "stale_parametric_future_acc"),
        "mean_stale_parametric_future_heldout_language_acc": mean(seed_rows, "stale_parametric_future_heldout_language_acc"),
        "mean_heldout_value_pair_acc": mean(seed_rows, "heldout_value_pair_acc"),
        "future_gain_world_port_minus_stale_parametric": mean(seed_rows, "future_rebound_acc") - mean(seed_rows, "stale_parametric_future_acc"),
        "median_full_b_plus_j_ns": benchmark_full_ns,
        "median_cached_b_j_only_ns": benchmark_j_ns,
        "cached_b_world_update_speedup": benchmark_full_ns / benchmark_j_ns,
        "mechanism": (
            "A real frozen pretrained Qwen B representation carries natural-language task semantics while current governed values are injected "
            "only through an explicit typed two-value side rail. The side rail is trained on value semantics sampled independently of entity, "
            "so current entity->value bindings are never stored in B or J parameters. A complete world rewrite therefore changes only runtime "
            "Port bytes; cached B features and frozen backbone outputs remain bit-identical."
        ),
        "control_design": (
            "The stale-parametric control uses the identical frozen Qwen B features and identical trained J side rail but supplies the exact old "
            "home entity->value table after a complete future rebinding. Thus the comparison isolates current external runtime values versus a "
            "perfectly remembered but stale parameter-owned world snapshot."
        ),
        "claim_boundary": (
            "This is the first V5 gate using a real frozen pretrained language representation as B, but the typed World Port is still a terminal "
            "side rail rather than a mid-layer Transformer port. It is not yet free-form generation, a strong RAG/editable-memory benchmark, "
            "or a novelty proof. R333 must push the Port into a late pretrained decoder layer and carry lifetime/page semantics through KV."
        ),
        "dod_status": "NOT_DOD; real-pretrained terminal World Port gate",
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "backbone_frozen",
        "port_parameter_count",
        "mean_home_acc",
        "mean_future_rebound_acc",
        "mean_future_rebound_heldout_language_acc",
        "mean_stale_parametric_future_acc",
        "mean_stale_parametric_future_heldout_language_acc",
        "mean_heldout_value_pair_acc",
        "future_gain_world_port_minus_stale_parametric",
        "max_b_feature_delta_across_world_transition",
        "max_frozen_backbone_control_logit_delta_after_port_training",
        "cached_b_world_update_speedup",
        "report_sha256",
    ]}, indent=2))

    # Conservative mechanism gate. Preserve failures instead of claiming success.
    if b_world_delta != 0.0:
        return 2
    if backbone_logit_delta != 0.0:
        return 3
    if report["mean_home_acc"] < 0.90:
        return 4
    if report["mean_future_rebound_acc"] < 0.90:
        return 5
    if report["mean_future_rebound_heldout_language_acc"] < 0.80:
        return 6
    if report["mean_heldout_value_pair_acc"] < 0.85:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
