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


STAGE = "R317-BINDING-EQUIVARIANT-EXECUTION"
ENTITY_COUNT = 64
VALUE_COUNT = 64
OP_COUNT = 4
CLASS_COUNT = 8
SEEDS = (17, 29, 43)
TRAIN_STEPS = int(os.environ.get("SO_R317_STEPS", "1200"))
BATCH = int(os.environ.get("SO_R317_BATCH", "384"))
EVAL_WORLDS = int(os.environ.get("SO_R317_EVAL_WORLDS", "64"))
EVAL_BATCH = int(os.environ.get("SO_R317_EVAL_BATCH", "512"))
REPORT_PATH = Path(os.environ.get("SO_R317_REPORT", "r317_report.json"))


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def world(gen: torch.Generator) -> torch.Tensor:
    return torch.randperm(VALUE_COUNT, generator=gen)


def deranged_world(home: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    for _ in range(1000):
        w = world(gen)
        if not bool((w == home).any()):
            return w
    # Deterministic fallback: cyclic shift of the home value assignment.
    return torch.roll(home, 1)


def target(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0
    y[m] = (va[m] + vb[m]) & 7
    m = op == 1
    y[m] = (va[m] ^ vb[m]) & 7
    m = op == 2
    y[m] = torch.abs(va[m] - vb[m]) & 7
    m = op == 3
    y[m] = (3 * va[m] + 5 * vb[m]) & 7
    return y


def holdout_mask(a: torch.Tensor, b: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
    # Entity/operator combinations deliberately absent from training.
    return ((a + 3 * b + 5 * op) % 7) == 0


def sample(world_map: torch.Tensor, n: int, gen: torch.Generator, *, training: bool = False, heldout_only: bool = False):
    out = []
    need = n
    while need > 0:
        draw = max(need * 2, 256)
        a = torch.randint(0, ENTITY_COUNT, (draw,), generator=gen)
        b = torch.randint(0, ENTITY_COUNT, (draw,), generator=gen)
        op = torch.randint(0, OP_COUNT, (draw,), generator=gen)
        h = holdout_mask(a, b, op)
        keep = ~h if training else (h if heldout_only else torch.ones_like(h, dtype=torch.bool))
        a, b, op = a[keep], b[keep], op[keep]
        take = min(need, a.numel())
        if take:
            a, b, op = a[:take], b[:take], op[:take]
            va, vb = world_map[a], world_map[b]
            out.append((a, b, op, va, vb, target(op, va, vb)))
            need -= take
    return tuple(torch.cat([row[i] for row in out]) for i in range(6))


class MonolithicShortcut(nn.Module):
    """Allows an entity-identity shortcut and therefore can bind facts into weights."""

    def __init__(self) -> None:
        super().__init__()
        self.ent = nn.Embedding(ENTITY_COUNT, 24)
        self.op = nn.Embedding(OP_COUNT, 16)
        self.val = nn.Embedding(VALUE_COUNT, 24)
        self.core = nn.Sequential(nn.Linear(24 * 4 + 16, 112), nn.GELU(), nn.Linear(112, 80), nn.GELU())
        self.core_head = nn.Linear(80, CLASS_COUNT)
        self.short = nn.Sequential(nn.Linear(24 * 2 + 16, 64), nn.GELU(), nn.Linear(64, CLASS_COUNT))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, a, b, op, va, vb):
        x = torch.cat([self.ent(a), self.ent(b), self.op(op), self.val(va), self.val(vb)], -1)
        core = self.core_head(self.core(x))
        shortcut = self.short(torch.cat([self.ent(a), self.ent(b), self.op(op)], -1))
        return core + self.alpha * shortcut, shortcut


class BindingEquivariant(nn.Module):
    """Entity identities are used only by the authority plane to address Pods.

    After current values are read, the neural operator receives (opcode, value_a,
    value_b) and has no identity channel. Consequently arbitrary rebinding of
    entities to values cannot create a learned entity/value shortcut.
    """

    def __init__(self) -> None:
        super().__init__()
        self.op = nn.Embedding(OP_COUNT, 16)
        self.val = nn.Embedding(VALUE_COUNT, 28)
        self.net = nn.Sequential(
            nn.Linear(16 + 28 + 28, 112), nn.GELU(),
            nn.Linear(112, 96), nn.GELU(),
            nn.Linear(96, CLASS_COUNT),
        )

    def forward(self, a, b, op, va, vb):  # a/b intentionally ignored after address resolution
        x = torch.cat([self.op(op), self.val(va), self.val(vb)], -1)
        return self.net(x)


def typed_alu(op: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
    # Exact CKVM execution path: semantics is an opcode, world values are typed data.
    return target(op, va, vb)


def train(seed: int, cls):
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed * 10007 + 19)
    home = world(gen)
    model = cls()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        a, b, op, va, vb, y = sample(home, BATCH, gen, training=True)
        out = model(a, b, op, va, vb)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval(), home, time.perf_counter() - t0


@torch.inference_mode()
def accuracy(model, w: torch.Tensor, gen: torch.Generator, n: int, *, heldout_only: bool = False):
    a, b, op, va, vb, y = sample(w, n, gen, heldout_only=heldout_only)
    out = model(a, b, op, va, vb)
    logits = out[0] if isinstance(out, tuple) else out
    return float((logits.argmax(-1) == y).float().mean())


@torch.inference_mode()
def monolithic_shortcut_accuracy(model: MonolithicShortcut, w: torch.Tensor, gen: torch.Generator, n: int):
    a, b, op, va, vb, y = sample(w, n, gen)
    _, shortcut = model(a, b, op, va, vb)
    return float((shortcut.argmax(-1) == y).float().mean())


@torch.inference_mode()
def identity_channel_delta(model: BindingEquivariant, w: torch.Tensor, gen: torch.Generator, n: int):
    a, b, op, va, vb, _ = sample(w, n, gen)
    logits1 = model(a, b, op, va, vb)
    fake_a = torch.randint(0, ENTITY_COUNT, a.shape, generator=gen)
    fake_b = torch.randint(0, ENTITY_COUNT, b.shape, generator=gen)
    logits2 = model(fake_a, fake_b, op, va, vb)
    return float((logits1 - logits2).abs().max())


@dataclass
class Row:
    seed: int
    mono_home_acc: float
    mono_deranged_future_acc: float
    mono_heldout_entity_op_acc: float
    mono_future_shortcut_acc: float
    equiv_home_acc: float
    equiv_deranged_future_acc: float
    equiv_heldout_entity_op_acc: float
    equiv_identity_channel_max_logit_delta: float
    typed_alu_deranged_future_acc: float
    mono_train_seconds: float
    equiv_train_seconds: float


def mean(rows: list[Row], key: str) -> float:
    return sum(getattr(r, key) for r in rows) / len(rows)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows: list[Row] = []
    for seed in SEEDS:
        mono, home_m, tm = train(seed, MonolithicShortcut)
        equiv, home_e, te = train(seed, BindingEquivariant)
        assert torch.equal(home_m, home_e)
        gen = torch.Generator().manual_seed(seed * 3331 + 73)
        future_m = []
        future_e = []
        hold_m = []
        hold_e = []
        alu = []
        for _ in range(EVAL_WORLDS):
            w = deranged_world(home_m, gen)
            future_m.append(accuracy(mono, w, gen, EVAL_BATCH))
            future_e.append(accuracy(equiv, w, gen, EVAL_BATCH))
            hold_m.append(accuracy(mono, w, gen, EVAL_BATCH, heldout_only=True))
            hold_e.append(accuracy(equiv, w, gen, EVAL_BATCH, heldout_only=True))
            a, b, op, va, vb, y = sample(w, EVAL_BATCH, gen)
            alu.append(float((typed_alu(op, va, vb) == y).float().mean()))
        rows.append(Row(
            seed=seed,
            mono_home_acc=accuracy(mono, home_m, gen, EVAL_BATCH * 4),
            mono_deranged_future_acc=sum(future_m) / len(future_m),
            mono_heldout_entity_op_acc=sum(hold_m) / len(hold_m),
            mono_future_shortcut_acc=monolithic_shortcut_accuracy(mono, deranged_world(home_m, gen), gen, EVAL_BATCH * 4),
            equiv_home_acc=accuracy(equiv, home_e, gen, EVAL_BATCH * 4),
            equiv_deranged_future_acc=sum(future_e) / len(future_e),
            equiv_heldout_entity_op_acc=sum(hold_e) / len(hold_e),
            equiv_identity_channel_max_logit_delta=identity_channel_delta(equiv, deranged_world(home_e, gen), gen, EVAL_BATCH * 4),
            typed_alu_deranged_future_acc=sum(alu) / len(alu),
            mono_train_seconds=tm,
            equiv_train_seconds=te,
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "Binding-Equivariant CKVM / world-permutation firewall",
        "mechanism": (
            "Canonical identity is permitted only on the address path. After Pod resolution, value execution is a "
            "function of opcode and current typed values, with no entity-identity channel. This makes future "
            "entity->value rebinding an architectural symmetry rather than a learned pair-generalization problem."
        ),
        "formal_contract": "Exec(op,a,b,W) = R(op, W[a], W[b]); R has no access to a or b after address resolution",
        "all_future_worlds_deranged_from_training_binding": True,
        "heldout_entity_operator_combinations_removed_from_training": True,
        "seeds": list(SEEDS),
        "per_seed": [asdict(r) for r in rows],
        "mean_monolithic_home_acc": mean(rows, "mono_home_acc"),
        "mean_monolithic_deranged_future_acc": mean(rows, "mono_deranged_future_acc"),
        "mean_monolithic_heldout_entity_op_acc": mean(rows, "mono_heldout_entity_op_acc"),
        "mean_binding_equivariant_home_acc": mean(rows, "equiv_home_acc"),
        "mean_binding_equivariant_deranged_future_acc": mean(rows, "equiv_deranged_future_acc"),
        "mean_binding_equivariant_heldout_entity_op_acc": mean(rows, "equiv_heldout_entity_op_acc"),
        "max_binding_equivariant_identity_channel_logit_delta": max(r.equiv_identity_channel_max_logit_delta for r in rows),
        "mean_typed_alu_deranged_future_acc": mean(rows, "typed_alu_deranged_future_acc"),
        "future_gain_equivariant_minus_monolithic": mean(rows, "equiv_deranged_future_acc") - mean(rows, "mono_deranged_future_acc"),
        "updates_require_gradient_steps": False,
        "dod_status": "NOT_DOD; architectural binding-generalization gate",
        "claim_boundary": (
            "Permutation/equivariant architectures, semantic parsing, typed program execution and tool use are established. "
            "R317 tests a stricter CKCA information-flow contract that removes identity->fact shortcuts by construction; "
            "novelty of the complete architecture remains to be established."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "mean_monolithic_home_acc",
        "mean_monolithic_deranged_future_acc",
        "mean_binding_equivariant_home_acc",
        "mean_binding_equivariant_deranged_future_acc",
        "mean_binding_equivariant_heldout_entity_op_acc",
        "max_binding_equivariant_identity_channel_logit_delta",
        "mean_typed_alu_deranged_future_acc",
        "future_gain_equivariant_minus_monolithic",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
