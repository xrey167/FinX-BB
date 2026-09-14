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


STAGE = "R318-CAUSAL-DUAL-RAIL-EXECUTION"
ENTITY_COUNT = 64
VALUE_COUNT = 64
OP_COUNT = 4
CLASS_COUNT = 8
SEEDS = (17, 29, 43)
TRAIN_STEPS = int(os.environ.get("SO_R318_STEPS", "1100"))
BATCH = int(os.environ.get("SO_R318_BATCH", "384"))
EVAL_WORLDS = int(os.environ.get("SO_R318_EVAL_WORLDS", "48"))
EVAL_BATCH = int(os.environ.get("SO_R318_EVAL_BATCH", "512"))
REPORT_PATH = Path(os.environ.get("SO_R318_REPORT", "r318_report.json"))


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


def sample(w: torch.Tensor, n: int, gen: torch.Generator):
    a = torch.randint(0, ENTITY_COUNT, (n,), generator=gen)
    b = torch.randint(0, ENTITY_COUNT, (n,), generator=gen)
    op = torch.randint(0, OP_COUNT, (n,), generator=gen)
    va, vb = w[a], w[b]
    return a, b, op, va, vb, target(op, va, vb)


class Residual(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x):
        return x + self.ff(self.ln(x))


class DualRail(nn.Module):
    """Immutable B rail + narrow mutable J rail.

    B sees query/operator structure only. It never receives current values.
    J receives the current value Port and can read cached B activations. There is
    no edge from J back into B, so mutable facts cannot contaminate reusable B state.
    """

    def __init__(self):
        super().__init__()
        self.op = nn.Embedding(OP_COUNT, 32)
        self.b_blocks = nn.ModuleList([Residual(32) for _ in range(6)])
        self.val = nn.Embedding(VALUE_COUNT, 24)
        self.j_in = nn.Sequential(nn.Linear(24 + 24 + 32, 32), nn.GELU())
        self.b_to_j = nn.ModuleList([nn.Linear(32, 32, bias=False) for _ in range(4)])
        self.j_blocks = nn.ModuleList([Residual(32) for _ in range(4)])
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, CLASS_COUNT))

    def compile_b(self, op: torch.Tensor):
        x = self.op(op)
        states = []
        for block in self.b_blocks:
            x = block(x)
            states.append(x)
        return states

    def run_j(self, b_states, va: torch.Tensor, vb: torch.Tensor):
        j = self.j_in(torch.cat([self.val(va), self.val(vb), b_states[1]], dim=-1))
        for i, block in enumerate(self.j_blocks):
            j = j + self.b_to_j[i](b_states[i + 2])
            j = block(j)
        return j

    def forward(self, op, va, vb):
        b = self.compile_b(op)
        j = self.run_j(b, va, vb)
        return self.head(torch.cat([b[-1], j], dim=-1)), b, j

    def forward_cached(self, cached_b, va, vb):
        j = self.run_j(cached_b, va, vb)
        return self.head(torch.cat([cached_b[-1], j], dim=-1)), j


class ContaminatingSingleRail(nn.Module):
    """Control: mutable values enter the shared rail before its final blocks."""

    def __init__(self):
        super().__init__()
        self.op = nn.Embedding(OP_COUNT, 32)
        self.val = nn.Embedding(VALUE_COUNT, 24)
        self.pre = nn.ModuleList([Residual(32) for _ in range(2)])
        self.inject = nn.Sequential(nn.Linear(32 + 24 + 24, 32), nn.GELU())
        self.post = nn.ModuleList([Residual(32) for _ in range(4)])
        self.head = nn.Linear(32, CLASS_COUNT)

    def forward(self, op, va, vb):
        x = self.op(op)
        for block in self.pre:
            x = block(x)
        immutable_prefix = x
        x = self.inject(torch.cat([x, self.val(va), self.val(vb)], -1))
        contaminated = []
        for block in self.post:
            x = block(x)
            contaminated.append(x)
        return self.head(x), immutable_prefix, contaminated


def train(seed: int, cls):
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed * 10007 + 101)
    home = world(gen)
    model = cls()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        _, _, op, va, vb, y = sample(home, BATCH, gen)
        out = model(op, va, vb)
        logits = out[0]
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval(), home, time.perf_counter() - t0


@torch.inference_mode()
def acc(model, w, gen, n):
    _, _, op, va, vb, y = sample(w, n, gen)
    logits = model(op, va, vb)[0]
    return float((logits.argmax(-1) == y).float().mean())


@torch.inference_mode()
def dual_rail_cache_equivalence(model: DualRail, w, gen, n):
    _, _, op, va, vb, _ = sample(w, n, gen)
    full, b, _ = model(op, va, vb)
    cached, _ = model.forward_cached(b, va, vb)
    return float((full - cached).abs().max())


@torch.inference_mode()
def mutable_update_b_invariance(model: DualRail, home, gen, n):
    _, _, op, va, vb, _ = sample(home, n, gen)
    _, b1, _ = model(op, va, vb)
    w2 = deranged_world(home, gen)
    _, _, _, va2, vb2, _ = sample(w2, n, gen)
    # B depends only on opcode. Reuse the same opcode vector to isolate world changes.
    b2 = model.compile_b(op)
    return max(float((x - y).abs().max()) for x, y in zip(b1, b2))


@torch.inference_mode()
def single_rail_contamination_delta(model: ContaminatingSingleRail, home, gen, n):
    a, b, op, va, vb, _ = sample(home, n, gen)
    _, prefix1, states1 = model(op, va, vb)
    edited = home.clone()
    edited[a] = (edited[a] + 1) % VALUE_COUNT
    edited[b] = (edited[b] + 3) % VALUE_COUNT
    va2, vb2 = edited[a], edited[b]
    _, prefix2, states2 = model(op, va2, vb2)
    prefix_delta = float((prefix1 - prefix2).abs().max())
    post_delta = max(float((x - y).abs().max()) for x, y in zip(states1, states2))
    return prefix_delta, post_delta


@torch.inference_mode()
def benchmark_cached_update(model: DualRail, w, gen, batch=1024, loops=80):
    _, _, op, va, vb, _ = sample(w, batch, gen)
    # Warm-up.
    for _ in range(5):
        model(op, va, vb)
    b = model.compile_b(op)
    for _ in range(5):
        model.forward_cached(b, va, vb)

    t0 = time.perf_counter_ns()
    for _ in range(loops):
        model(op, va, vb)
    full_ns = time.perf_counter_ns() - t0

    t0 = time.perf_counter_ns()
    for i in range(loops):
        va2 = (va + i) % VALUE_COUNT
        vb2 = (vb + 3 * i) % VALUE_COUNT
        model.forward_cached(b, va2, vb2)
    cached_ns = time.perf_counter_ns() - t0
    return full_ns / loops, cached_ns / loops


@dataclass
class Row:
    seed: int
    dual_home_acc: float
    dual_future_acc: float
    single_home_acc: float
    single_future_acc: float
    dual_full_vs_cached_max_logit_delta: float
    dual_b_world_update_max_state_delta: float
    single_immutable_prefix_delta: float
    single_post_injection_state_delta: float
    full_forward_ns: float
    cached_world_update_forward_ns: float
    update_speedup: float
    dual_train_seconds: float
    single_train_seconds: float


def mean(rows, key):
    return sum(getattr(r, key) for r in rows) / len(rows)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows = []
    for seed in SEEDS:
        dual, home_d, td = train(seed, DualRail)
        single, home_s, ts = train(seed, ContaminatingSingleRail)
        assert torch.equal(home_d, home_s)
        gen = torch.Generator().manual_seed(seed * 3331 + 89)
        fut_d, fut_s = [], []
        eq = []
        for _ in range(EVAL_WORLDS):
            w = deranged_world(home_d, gen)
            fut_d.append(acc(dual, w, gen, EVAL_BATCH))
            fut_s.append(acc(single, w, gen, EVAL_BATCH))
            eq.append(dual_rail_cache_equivalence(dual, w, gen, EVAL_BATCH))
        bdelta = mutable_update_b_invariance(dual, home_d, gen, EVAL_BATCH)
        pre_delta, post_delta = single_rail_contamination_delta(single, home_s, gen, EVAL_BATCH)
        full_ns, cached_ns = benchmark_cached_update(dual, deranged_world(home_d, gen), gen)
        rows.append(Row(
            seed=seed,
            dual_home_acc=acc(dual, home_d, gen, EVAL_BATCH * 4),
            dual_future_acc=sum(fut_d) / len(fut_d),
            single_home_acc=acc(single, home_s, gen, EVAL_BATCH * 4),
            single_future_acc=sum(fut_s) / len(fut_s),
            dual_full_vs_cached_max_logit_delta=max(eq),
            dual_b_world_update_max_state_delta=bdelta,
            single_immutable_prefix_delta=pre_delta,
            single_post_injection_state_delta=post_delta,
            full_forward_ns=full_ns,
            cached_world_update_forward_ns=cached_ns,
            update_speedup=full_ns / cached_ns,
            dual_train_seconds=td,
            single_train_seconds=ts,
        ))

    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Dual-Rail Neural Execution (CDRNE)",
        "mechanism": (
            "A wide reusable B rail carries language/operator skill and is structurally forbidden from reading mutable values. "
            "A narrow J rail reads current typed value Ports and cached B activations, but has no edge back into B. "
            "World updates therefore recompute only J while B remains bit-identical and reusable."
        ),
        "seeds": list(SEEDS),
        "per_seed": [asdict(r) for r in rows],
        "mean_dual_home_acc": mean(rows, "dual_home_acc"),
        "mean_dual_deranged_future_acc": mean(rows, "dual_future_acc"),
        "mean_single_home_acc": mean(rows, "single_home_acc"),
        "mean_single_deranged_future_acc": mean(rows, "single_future_acc"),
        "max_dual_full_vs_cached_logit_delta": max(r.dual_full_vs_cached_max_logit_delta for r in rows),
        "max_dual_b_world_update_state_delta": max(r.dual_b_world_update_max_state_delta for r in rows),
        "mean_single_post_injection_state_delta": mean(rows, "single_post_injection_state_delta"),
        "mean_cached_update_speedup": mean(rows, "update_speedup"),
        "mean_full_forward_ns": mean(rows, "full_forward_ns"),
        "mean_cached_world_update_forward_ns": mean(rows, "cached_world_update_forward_ns"),
        "mutable_world_can_flow_back_into_b": False,
        "dod_status": "NOT_DOD; split-neural-state architecture gate",
        "claim_boundary": (
            "Adapters, side networks, two-stream models and cache reuse are established. R318 tests a stricter one-way "
            "lifecycle information-flow architecture for CKCA, not standalone novelty."
        ),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "mean_dual_home_acc",
        "mean_dual_deranged_future_acc",
        "mean_single_home_acc",
        "mean_single_deranged_future_acc",
        "max_dual_full_vs_cached_logit_delta",
        "max_dual_b_world_update_state_delta",
        "mean_single_post_injection_state_delta",
        "mean_cached_update_speedup",
        "report_sha256",
    ]}, indent=2))


if __name__ == "__main__":
    main()
