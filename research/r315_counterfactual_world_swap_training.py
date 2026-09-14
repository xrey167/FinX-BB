from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE = "R315-COUNTERFACTUAL-WORLD-SWAP-TRAINING"
ENTITY_COUNT = 64
VALUE_COUNT = 64
OP_COUNT = 4
CLASS_COUNT = 4
VALUE_BITS = 6
SEEDS = (17, 29, 43)
TRAIN_STEPS = int(os.environ.get("SO_R315_STEPS", "900"))
BATCH = int(os.environ.get("SO_R315_BATCH", "256"))
EVAL_WORLDS = int(os.environ.get("SO_R315_EVAL_WORLDS", "64"))
EVAL_BATCH = int(os.environ.get("SO_R315_EVAL_BATCH", "512"))
REPORT_PATH = Path(os.environ.get("SO_R315_REPORT", "r315_report.json"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def new_world(generator: torch.Generator) -> torch.Tensor:
    # A world is a mutable canonical Pod->value binding. A fresh permutation
    # guarantees that entity identity is statistically useless for current value.
    return torch.randperm(VALUE_COUNT, generator=generator)


def value_features(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(VALUE_BITS, device=v.device)
    bits = ((v[:, None] >> shifts[None, :]) & 1).float()
    scalar = (v.float() / float(VALUE_COUNT - 1)).unsqueeze(1)
    centered = ((v.float() - (VALUE_COUNT - 1) / 2.0) / VALUE_COUNT).unsqueeze(1)
    return torch.cat([bits, scalar, centered], dim=1)


def labels(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0
    y[m] = (va[m] < vb[m]).long()  # classes {0,1}
    m = op == 1
    y[m] = (va[m] + vb[m]) & 3
    m = op == 2
    y[m] = torch.abs(va[m] - vb[m]) & 3
    m = op == 3
    y[m] = (va[m] ^ vb[m]) & 3
    return y


def sample_queries(world: torch.Tensor, batch: int, generator: torch.Generator):
    a = torch.randint(0, ENTITY_COUNT, (batch,), generator=generator)
    b = torch.randint(0, ENTITY_COUNT, (batch,), generator=generator)
    op = torch.randint(0, OP_COUNT, (batch,), generator=generator)
    va = world[a]
    vb = world[b]
    return a, b, op, va, vb, labels(op, va, vb)


class FutureBindableNet(nn.Module):
    """Two-plane mechanism probe.

    B-plane sees only query structure/entity identity. Mutable values enter only
    through an explicit late Port. A learnable B shortcut is intentionally kept:
    a static-world learner can exploit it, while counterfactual world swapping
    should make the shortcut unprofitable and force value use through the Port.
    """

    def __init__(self) -> None:
        super().__init__()
        self.entity = nn.Embedding(ENTITY_COUNT, 20)
        self.operation = nn.Embedding(OP_COUNT, 12)
        self.b_mlp = nn.Sequential(
            nn.Linear(52, 80), nn.GELU(), nn.Linear(80, 64), nn.GELU()
        )
        self.port = nn.Sequential(nn.Linear(VALUE_BITS + 2, 32), nn.GELU(), nn.Linear(32, 24), nn.GELU())
        self.j_mlp = nn.Sequential(
            nn.Linear(64 + 24 + 24, 96), nn.GELU(), nn.Linear(96, 64), nn.GELU()
        )
        self.j_head = nn.Linear(64, CLASS_COUNT)
        self.b_head = nn.Linear(64, CLASS_COUNT)
        self.shortcut_alpha = nn.Parameter(torch.tensor(1.0))

    def encode_b(self, a: torch.Tensor, b: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.entity(a), self.entity(b), self.operation(op)], dim=-1)
        return self.b_mlp(x)

    def forward(self, a, b, op, va_feat, vb_feat):
        h = self.encode_b(a, b, op)
        pa = self.port(va_feat)
        pb = self.port(vb_feat)
        j = self.j_mlp(torch.cat([h, pa, pb], dim=-1))
        b_logits = self.b_head(h)
        j_logits = self.j_head(j)
        logits = j_logits + self.shortcut_alpha * b_logits
        return logits, b_logits, j_logits


def train_model(seed: int, mode: str):
    assert mode in {"static", "world_swap"}
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed * 1009 + 7)
    home = new_world(gen)
    model = FutureBindableNet()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model.train()
    loss_tail = []
    t0 = time.perf_counter()
    for step in range(TRAIN_STEPS):
        world = home if mode == "static" else new_world(gen)
        a, b, op, va, vb, y = sample_queries(world, BATCH, gen)
        logits, _, _ = model(a, b, op, value_features(va), value_features(vb))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step >= TRAIN_STEPS - 50:
            loss_tail.append(float(loss.detach()))
    return model.eval(), home, {
        "train_seconds": time.perf_counter() - t0,
        "tail_loss": sum(loss_tail) / max(1, len(loss_tail)),
        "shortcut_alpha": float(model.shortcut_alpha.detach()),
    }


@torch.inference_mode()
def evaluate_one(model: FutureBindableNet, label_world: torch.Tensor, port_world: torch.Tensor, generator: torch.Generator, batch: int):
    a, b, op, va_label, vb_label, y = sample_queries(label_world, batch, generator)
    va_port = port_world[a]
    vb_port = port_world[b]
    logits, b_logits, _ = model(a, b, op, value_features(va_port), value_features(vb_port))
    zero = torch.zeros(batch, VALUE_BITS + 2)
    zero_logits, _, _ = model(a, b, op, zero, zero)
    return {
        "full": float((logits.argmax(-1) == y).float().mean()),
        "b_only": float((b_logits.argmax(-1) == y).float().mean()),
        "zero_port": float((zero_logits.argmax(-1) == y).float().mean()),
    }


@torch.inference_mode()
def evaluate_future(model: FutureBindableNet, home: torch.Tensor, seed: int):
    gen = torch.Generator().manual_seed(seed * 9173 + 41)
    full, b_only, zero_port, stale_port = [], [], [], []
    mutated_full = []
    for _ in range(EVAL_WORLDS):
        world = new_world(gen)
        m = evaluate_one(model, world, world, gen, EVAL_BATCH)
        full.append(m["full"])
        b_only.append(m["b_only"])
        zero_port.append(m["zero_port"])
        stale_port.append(evaluate_one(model, world, home, gen, EVAL_BATCH)["full"])

    # Sparse online edit: rewrite 1/4 of canonical entities, no gradients.
    edited = home.clone()
    ids = torch.randperm(ENTITY_COUNT, generator=gen)[: ENTITY_COUNT // 4]
    replacement = torch.randint(0, VALUE_COUNT, (len(ids),), generator=gen)
    edited[ids] = replacement
    for _ in range(8):
        mutated_full.append(evaluate_one(model, edited, edited, gen, EVAL_BATCH)["full"])

    return {
        "future_full_acc": sum(full) / len(full),
        "future_b_only_acc": sum(b_only) / len(b_only),
        "future_zero_port_acc": sum(zero_port) / len(zero_port),
        "future_stale_port_acc": sum(stale_port) / len(stale_port),
        "sparse_edit_acc_without_gradient": sum(mutated_full) / len(mutated_full),
    }


@dataclass
class SeedResult:
    seed: int
    static_home_acc: float
    static_future_acc: float
    static_future_b_only_acc: float
    static_future_zero_port_acc: float
    static_future_stale_port_acc: float
    static_sparse_edit_acc_without_gradient: float
    static_shortcut_alpha: float
    swap_home_acc: float
    swap_future_acc: float
    swap_future_b_only_acc: float
    swap_future_zero_port_acc: float
    swap_future_stale_port_acc: float
    swap_sparse_edit_acc_without_gradient: float
    swap_shortcut_alpha: float
    static_train_seconds: float
    swap_train_seconds: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def home_accuracy(model, home, seed):
    gen = torch.Generator().manual_seed(seed * 3253 + 3)
    vals = [evaluate_one(model, home, home, gen, EVAL_BATCH)["full"] for _ in range(12)]
    return sum(vals) / len(vals)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows: list[SeedResult] = []
    for seed in SEEDS:
        static_model, static_home, sm = train_model(seed, "static")
        swap_model, swap_home, wm = train_model(seed, "world_swap")
        se = evaluate_future(static_model, static_home, seed)
        we = evaluate_future(swap_model, swap_home, seed + 10000)
        rows.append(
            SeedResult(
                seed=seed,
                static_home_acc=home_accuracy(static_model, static_home, seed),
                static_future_acc=se["future_full_acc"],
                static_future_b_only_acc=se["future_b_only_acc"],
                static_future_zero_port_acc=se["future_zero_port_acc"],
                static_future_stale_port_acc=se["future_stale_port_acc"],
                static_sparse_edit_acc_without_gradient=se["sparse_edit_acc_without_gradient"],
                static_shortcut_alpha=sm["shortcut_alpha"],
                swap_home_acc=home_accuracy(swap_model, swap_home, seed + 1),
                swap_future_acc=we["future_full_acc"],
                swap_future_b_only_acc=we["future_b_only_acc"],
                swap_future_zero_port_acc=we["future_zero_port_acc"],
                swap_future_stale_port_acc=we["future_stale_port_acc"],
                swap_sparse_edit_acc_without_gradient=we["sparse_edit_acc_without_gradient"],
                swap_shortcut_alpha=wm["shortcut_alpha"],
                static_train_seconds=sm["train_seconds"],
                swap_train_seconds=wm["train_seconds"],
            )
        )

    report = {
        "stage": STAGE,
        "architecture_candidate": "Counterfactual World-Swap Indirection Training (CWSIT)",
        "mechanism": (
            "Training repeatedly randomizes canonical entity->value bindings while keeping query semantics stable. "
            "Mutable values are supplied only through a late-bound Port, making the reusable B-plane statistically "
            "unable to encode the current world. The intended learned invariant is: B encodes operators/skills; "
            "current facts remain swappable runtime data."
        ),
        "claim_boundary": (
            "This is a synthetic mechanism gate for a future-bindable training objective, not evidence of novelty, "
            "real-language quality, or superiority to RAG/editable-memory systems."
        ),
        "dod_status": "NOT_DOD; future-bindable training mechanism gate",
        "entities": ENTITY_COUNT,
        "values": VALUE_COUNT,
        "operations": OP_COUNT,
        "training_steps": TRAIN_STEPS,
        "eval_worlds_per_seed": EVAL_WORLDS,
        "seeds": list(SEEDS),
        "per_seed": [asdict(r) for r in rows],
        "mean_static_home_acc": mean(rows, "static_home_acc"),
        "mean_static_future_acc": mean(rows, "static_future_acc"),
        "mean_static_sparse_edit_acc_without_gradient": mean(rows, "static_sparse_edit_acc_without_gradient"),
        "mean_world_swap_home_acc": mean(rows, "swap_home_acc"),
        "mean_world_swap_future_acc": mean(rows, "swap_future_acc"),
        "mean_world_swap_sparse_edit_acc_without_gradient": mean(rows, "swap_sparse_edit_acc_without_gradient"),
        "mean_world_swap_future_b_only_acc": mean(rows, "swap_future_b_only_acc"),
        "mean_world_swap_future_zero_port_acc": mean(rows, "swap_future_zero_port_acc"),
        "mean_world_swap_future_stale_port_acc": mean(rows, "swap_future_stale_port_acc"),
        "future_gain_world_swap_minus_static": mean(rows, "swap_future_acc") - mean(rows, "static_future_acc"),
        "updates_require_gradient_steps": False,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage",
        "mean_static_home_acc",
        "mean_static_future_acc",
        "mean_world_swap_home_acc",
        "mean_world_swap_future_acc",
        "mean_world_swap_sparse_edit_acc_without_gradient",
        "mean_world_swap_future_b_only_acc",
        "mean_world_swap_future_zero_port_acc",
        "mean_world_swap_future_stale_port_acc",
        "future_gain_world_swap_minus_static",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
