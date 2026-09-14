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

STAGE = "R318B-DIRECTIONAL-GATED-DUAL-RAIL"
ENTITY_COUNT = 64
VALUE_COUNT = 64
OP_COUNT = 4
CLASS_COUNT = 8
SEEDS = (17, 29, 43)
TRAIN_STEPS = int(os.environ.get("SO_R318B_STEPS", "1000"))
BATCH = int(os.environ.get("SO_R318B_BATCH", "512"))
EVAL_WORLDS = int(os.environ.get("SO_R318B_EVAL_WORLDS", "48"))
EVAL_BATCH = int(os.environ.get("SO_R318B_EVAL_BATCH", "1024"))
REPORT_PATH = Path(os.environ.get("SO_R318B_REPORT", "r318b_report.json"))


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def new_world(g: torch.Generator):
    return torch.randperm(VALUE_COUNT, generator=g)


def deranged(home, g):
    for _ in range(1000):
        w = new_world(g)
        if not bool((w == home).any()):
            return w
    return torch.roll(home, 1)


def target(op, a, b):
    y = torch.empty_like(op)
    m = op == 0; y[m] = (a[m] + b[m]) & 7
    m = op == 1; y[m] = (a[m] ^ b[m]) & 7
    m = op == 2; y[m] = torch.abs(a[m] - b[m]) & 7
    m = op == 3; y[m] = (3 * a[m] + 5 * b[m]) & 7
    return y


def sample(w, n, g):
    ea = torch.randint(0, ENTITY_COUNT, (n,), generator=g)
    eb = torch.randint(0, ENTITY_COUNT, (n,), generator=g)
    op = torch.randint(0, OP_COUNT, (n,), generator=g)
    va, vb = w[ea], w[eb]
    return ea, eb, op, va, vb, target(op, va, vb)


class Residual(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))
    def forward(self, x):
        return x + self.ff(self.ln(x))


class DirectionalGatedDualRail(nn.Module):
    """B is reusable and immutable with respect to current-world values.

    B compiles an operation into plan/control state. J receives only (current
    values, immutable control). Hard expert selection is equivalent to a typed
    CKVM opcode dispatch and cannot transmit mutable information into B.
    """
    def __init__(self):
        super().__init__()
        self.op = nn.Embedding(OP_COUNT, 48)
        self.b_blocks = nn.ModuleList([Residual(48) for _ in range(8)])
        self.plan_head = nn.Linear(48, 24)
        # Exact value identity is current runtime data. One-hot avoids forcing a
        # tiny latent to rediscover a lossy code for 64 arbitrary values.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(VALUE_COUNT * 2 + 24, 192), nn.GELU(),
                nn.Linear(192, 160), nn.GELU(),
                nn.Linear(160, CLASS_COUNT),
            ) for _ in range(OP_COUNT)
        ])

    def compile_b(self, op):
        x = self.op(op)
        states = []
        for block in self.b_blocks:
            x = block(x)
            states.append(x)
        plan = self.plan_head(x)
        return states, plan

    def run_j(self, op, plan, va, vb):
        x = torch.cat([
            F.one_hot(va, VALUE_COUNT).float(),
            F.one_hot(vb, VALUE_COUNT).float(),
            plan,
        ], dim=-1)
        out = torch.empty((len(op), CLASS_COUNT), dtype=x.dtype)
        for k, expert in enumerate(self.experts):
            mask = op == k
            if mask.any():
                out[mask] = expert(x[mask])
        return out

    def forward(self, op, va, vb):
        states, plan = self.compile_b(op)
        return self.run_j(op, plan, va, vb), states, plan

    def forward_cached(self, op, cached_plan, va, vb):
        return self.run_j(op, cached_plan, va, vb)


class SharedRailControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Embedding(OP_COUNT, 48)
        self.val = nn.Embedding(VALUE_COUNT, 40)
        self.net = nn.Sequential(
            nn.Linear(48 + 40 + 40, 192), nn.GELU(),
            nn.Linear(192, 160), nn.GELU(),
            nn.Linear(160, 96), nn.GELU(),
            nn.Linear(96, CLASS_COUNT),
        )
    def forward(self, op, va, vb):
        return self.net(torch.cat([self.op(op), self.val(va), self.val(vb)], -1))


def train(seed, cls):
    seed_all(seed)
    g = torch.Generator().manual_seed(seed * 10007 + 313)
    home = new_world(g)
    model = cls()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    t0 = time.perf_counter()
    for _ in range(TRAIN_STEPS):
        _, _, op, va, vb, y = sample(home, BATCH, g)
        out = model(op, va, vb)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval(), home, time.perf_counter() - t0


@torch.inference_mode()
def accuracy(model, w, g, n):
    _, _, op, va, vb, y = sample(w, n, g)
    out = model(op, va, vb)
    logits = out[0] if isinstance(out, tuple) else out
    return float((logits.argmax(-1) == y).float().mean())


@torch.inference_mode()
def equivalence_and_invariance(model: DirectionalGatedDualRail, w, g, n):
    _, _, op, va, vb, _ = sample(w, n, g)
    full, states1, plan1 = model(op, va, vb)
    cached = model.forward_cached(op, plan1, va, vb)
    full_cached_delta = float((full - cached).abs().max())

    # Change all values but reuse identical op. B must be exactly unchanged.
    w2 = deranged(w, g)
    ea = torch.randint(0, ENTITY_COUNT, (n,), generator=g)
    eb = torch.randint(0, ENTITY_COUNT, (n,), generator=g)
    va2, vb2 = w2[ea], w2[eb]
    states2, plan2 = model.compile_b(op)
    bdelta = max([float((a-b).abs().max()) for a,b in zip(states1, states2)] + [float((plan1-plan2).abs().max())])
    # Current output must change when current values change for at least some rows.
    current2 = model.forward_cached(op, plan1, va2, vb2)
    changed_fraction = float((full.argmax(-1) != current2.argmax(-1)).float().mean())
    return full_cached_delta, bdelta, changed_fraction


@torch.inference_mode()
def benchmark(model: DirectionalGatedDualRail, w, g, n=2048, loops=80):
    _, _, op, va, vb, _ = sample(w, n, g)
    for _ in range(5): model(op, va, vb)
    _, _, plan = model(op, va, vb)
    for _ in range(5): model.forward_cached(op, plan, va, vb)
    t0 = time.perf_counter_ns()
    for _ in range(loops): model(op, va, vb)
    full = (time.perf_counter_ns() - t0) / loops
    t0 = time.perf_counter_ns()
    for i in range(loops):
        model.forward_cached(op, plan, (va+i) % VALUE_COUNT, (vb+3*i) % VALUE_COUNT)
    cached = (time.perf_counter_ns() - t0) / loops
    return full, cached


@dataclass
class Row:
    seed: int
    dual_home_acc: float
    dual_future_acc: float
    control_home_acc: float
    control_future_acc: float
    max_full_vs_cached_logit_delta: float
    max_b_state_delta_on_world_change: float
    current_output_changed_fraction_after_world_swap: float
    full_ns: float
    cached_ns: float
    speedup: float
    dual_train_s: float
    control_train_s: float


def mean(rows, key): return sum(getattr(r, key) for r in rows) / len(rows)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    rows=[]
    for seed in SEEDS:
        dual, home, td = train(seed, DirectionalGatedDualRail)
        control, home2, tc = train(seed, SharedRailControl)
        assert torch.equal(home, home2)
        g = torch.Generator().manual_seed(seed * 3331 + 211)
        fd=[]; fc=[]; eq=[]; bd=[]; ch=[]
        for _ in range(EVAL_WORLDS):
            w=deranged(home,g)
            fd.append(accuracy(dual,w,g,EVAL_BATCH))
            fc.append(accuracy(control,w,g,EVAL_BATCH))
            a,b,c=equivalence_and_invariance(dual,w,g,EVAL_BATCH)
            eq.append(a);bd.append(b);ch.append(c)
        full,cached=benchmark(dual,deranged(home,g),g)
        rows.append(Row(
            seed=seed,
            dual_home_acc=accuracy(dual,home,g,EVAL_BATCH*2),
            dual_future_acc=sum(fd)/len(fd),
            control_home_acc=accuracy(control,home,g,EVAL_BATCH*2),
            control_future_acc=sum(fc)/len(fc),
            max_full_vs_cached_logit_delta=max(eq),
            max_b_state_delta_on_world_change=max(bd),
            current_output_changed_fraction_after_world_swap=sum(ch)/len(ch),
            full_ns=full,cached_ns=cached,speedup=full/cached,
            dual_train_s=td,control_train_s=tc,
        ))
    report={
        "stage":STAGE,
        "architecture_candidate":"Directional-Gated Causal Dual Rail (DG-CDRNE)",
        "mechanism":(
            "The B rail compiles immutable operator control and never reads mutable values. A typed hard gate selects a J expert; "
            "the J expert reads current value data plus the cached B plan. There is no J->B edge. This retains exact B cache "
            "invariance while restoring sufficient mutable-compute capacity."
        ),
        "per_seed":[asdict(r) for r in rows],
        "mean_dual_home_acc":mean(rows,"dual_home_acc"),
        "mean_dual_deranged_future_acc":mean(rows,"dual_future_acc"),
        "mean_control_home_acc":mean(rows,"control_home_acc"),
        "mean_control_deranged_future_acc":mean(rows,"control_future_acc"),
        "max_full_vs_cached_logit_delta":max(r.max_full_vs_cached_logit_delta for r in rows),
        "max_b_state_delta_on_world_change":max(r.max_b_state_delta_on_world_change for r in rows),
        "mean_output_change_fraction_after_world_swap":mean(rows,"current_output_changed_fraction_after_world_swap"),
        "mean_cached_world_update_speedup":mean(rows,"speedup"),
        "quality_gap_dual_minus_control_future":mean(rows,"dual_future_acc")-mean(rows,"control_future_acc"),
        "mutable_world_can_flow_back_into_b":False,
        "dod_status":"NOT_DOD; repaired split-neural-state mechanism gate",
        "claim_boundary":(
            "Mixture-of-experts, hard routing, side networks and one-way adapters are established components. The test concerns "
            "their CKCA-specific causal/lifetime information-flow contract, not standalone component novelty."
        ),
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode()
    report["report_sha256"]=hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in [
        "mean_dual_home_acc","mean_dual_deranged_future_acc","mean_control_home_acc","mean_control_deranged_future_acc",
        "max_full_vs_cached_logit_delta","max_b_state_delta_on_world_change","mean_cached_world_update_speedup",
        "quality_gap_dual_minus_control_future","report_sha256"]},indent=2))

if __name__ == "__main__": main()
