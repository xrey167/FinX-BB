from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE = "R350-PROSPECTIVE-REBINDING-LOSS"
REPORT_PATH = Path(os.environ.get("SO_R350_REPORT", "ci-r350/report.json"))
STEPS = int(os.environ.get("SO_R350_STEPS", "1400"))
BATCH = int(os.environ.get("SO_R350_BATCH", "512"))
CODE = int(os.environ.get("SO_R350_CODE", "12"))
SEM = 24
VALUE_COUNT = 32
OPS = 4
SEEDS = (17, 29, 43, 71, 101)


def bits(v: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(5, device=v.device)
    return ((v[:, None] >> shifts[None, :]) & 1).float()


def label(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(op)
    m = op == 0; y[m] = ((a[m] & 3) + (b[m] & 3)) & 3
    m = op == 1; y[m] = (a[m] ^ b[m]) & 3
    m = op == 2; y[m] = (a[m] > b[m]).long()
    m = op == 3; y[m] = torch.where((b[m] & 1) == 0, a[m] & 3, b[m] & 3)
    return y


class Compiler(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEM + 10, 96), nn.GELU(),
            nn.Linear(96, 64), nn.GELU(),
            nn.Linear(64, CODE),
        )

    def forward(self, semantic: torch.Tensor, va: torch.Tensor, vb: torch.Tensor):
        x = torch.cat([semantic, bits(va), bits(vb)], dim=-1)
        return F.normalize(self.net(x), dim=-1)


class JPlane(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CODE + 10, 96), nn.GELU(),
            nn.Linear(96, 72), nn.GELU(), nn.Linear(72, 4),
        )

    def forward(self, z, va, vb):
        return self.net(torch.cat([z, bits(va), bits(vb)], dim=-1))


def semantic_rows(proto, op, gen):
    # language/semantic representation with instance noise, but no entity value facts.
    return proto[op] + 0.08 * torch.randn((len(op), SEM), generator=gen)


def train(seed: int, prospective: bool):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed * 13007 + 5)
    proto = F.normalize(torch.randn(OPS, SEM, generator=gen), dim=-1)
    c = Compiler(); j = JPlane()
    opt = torch.optim.AdamW(list(c.parameters()) + list(j.parameters()), lr=2.5e-3, weight_decay=1e-4)
    c.train(); j.train()
    tail = []
    for step in range(STEPS):
        op = torch.randint(0, OPS, (BATCH,), generator=gen)
        sem = semantic_rows(proto, op, gen)
        a1 = torch.randint(0, VALUE_COUNT, (BATCH,), generator=gen)
        b1 = torch.randint(0, VALUE_COUNT, (BATCH,), generator=gen)
        z1 = c(sem, a1, b1)
        y1 = label(op, a1, b1)
        if prospective:
            # Counterfactual paired world for the same query semantics.
            a2 = torch.randint(0, VALUE_COUNT, (BATCH,), generator=gen)
            b2 = torch.randint(0, VALUE_COUNT, (BATCH,), generator=gen)
            z2 = c(sem, a2, b2)
            y2 = label(op, a2, b2)

            # Prospective Rebinding Loss (PRL): descriptor invariance plus cross-world
            # swap consistency. The compiler is allowed to see snapshot values, but a
            # useful descriptor must survive when its materialization world changes.
            loss_native = F.cross_entropy(j(z1, a1, b1), y1) + F.cross_entropy(j(z2, a2, b2), y2)
            loss_swap = F.cross_entropy(j(z1, a2, b2), y2) + F.cross_entropy(j(z2, a1, b1), y1)
            invariant = (1.0 - (z1 * z2).sum(-1)).mean()
            loss = 0.5 * loss_native + 0.5 * loss_swap + 0.35 * invariant
        else:
            # Ordinary current-snapshot training: compiler and executor always see the
            # same world, permitting the descriptor to encode snapshot-specific answers.
            loss = F.cross_entropy(j(z1, a1, b1), y1)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(c.parameters()) + list(j.parameters()), 1.0)
        opt.step()
        if step >= STEPS - 50:
            tail.append(float(loss.detach()))
    return c.eval(), j.eval(), proto, statistics.mean(tail)


@torch.inference_mode()
def evaluate(c, j, proto, seed):
    gen = torch.Generator().manual_seed(seed * 17011 + 13)
    n = 12000
    op = torch.randint(0, OPS, (n,), generator=gen)
    sem = semantic_rows(proto, op, gen)
    home_a = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
    home_b = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
    future_a = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
    future_b = torch.randint(0, VALUE_COUNT, (n,), generator=gen)

    z_home = c(sem, home_a, home_b)
    z_future_compile = c(sem, future_a, future_b)
    home_acc = float(j(z_home, home_a, home_b).argmax(-1).eq(label(op, home_a, home_b)).float().mean())
    rebound_acc = float(j(z_home, future_a, future_b).argmax(-1).eq(label(op, future_a, future_b)).float().mean())
    fresh_acc = float(j(z_future_compile, future_a, future_b).argmax(-1).eq(label(op, future_a, future_b)).float().mean())
    cos = (z_home * z_future_compile).sum(-1)
    mean_descriptor_drift = float((1.0 - cos).mean())
    p95_descriptor_drift = float(torch.quantile(1.0 - cos, 0.95))

    # Cross-world cycle: materialize one descriptor under 16 independently sampled
    # future worlds. No gradient steps or descriptor recompilation are permitted.
    future_accs = []
    for _ in range(16):
        fa = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
        fb = torch.randint(0, VALUE_COUNT, (n,), generator=gen)
        future_accs.append(float(j(z_home, fa, fb).argmax(-1).eq(label(op, fa, fb)).float().mean()))

    # Linear leakage probes: can a cheap classifier recover the home low-two-bit
    # value payload from the supposedly semantic descriptor?
    train_n = 8000
    xtr, xte = z_home[:train_n].detach(), z_home[train_n:].detach()
    ya = (home_a & 3).long(); yb = (home_b & 3).long()
    def probe(y):
        w = nn.Linear(CODE, 4)
        optim = torch.optim.AdamW(w.parameters(), lr=1e-2)
        for _ in range(250):
            logits = w(xtr)
            loss = F.cross_entropy(logits, y[:train_n])
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        with torch.inference_mode():
            return float(w(xte).argmax(-1).eq(y[train_n:]).float().mean())
    # temporarily re-enable grads for probe fitting
    with torch.enable_grad():
        leak_a = probe(ya)
        leak_b = probe(yb)

    return {
        "home_accuracy": home_acc,
        "future_rebind_accuracy_without_recompile": rebound_acc,
        "fresh_future_recompile_accuracy": fresh_acc,
        "mean_future_rebind_accuracy_16_worlds": statistics.mean(future_accs),
        "minimum_future_rebind_accuracy_16_worlds": min(future_accs),
        "mean_descriptor_cosine_drift_across_world_rewrite": mean_descriptor_drift,
        "p95_descriptor_cosine_drift_across_world_rewrite": p95_descriptor_drift,
        "home_value_low2_probe_accuracy_a": leak_a,
        "home_value_low2_probe_accuracy_b": leak_b,
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.set_num_interop_threads(1)

    rows = []
    for seed in SEEDS:
        bc, bj, bp, bloss = train(seed, prospective=False)
        pc, pj, pp, ploss = train(seed, prospective=True)
        b = evaluate(bc, bj, bp, seed + 1000)
        p = evaluate(pc, pj, pp, seed + 2000)
        rows.append({"seed": seed, "baseline_tail_loss": bloss, "prospective_tail_loss": ploss,
                     "baseline": b, "prospective_rebinding": p})

    def mean(path0, path1):
        return statistics.mean(r[path0][path1] for r in rows)
    report = {
        "stage": STAGE,
        "architecture_candidate": "Prospective Rebinding Loss (PRL) / Counterfactual World-Swap Training",
        "seeds": list(SEEDS),
        "rows": rows,
        "baseline_mean_future_rebind_accuracy": mean("baseline", "mean_future_rebind_accuracy_16_worlds"),
        "prl_mean_future_rebind_accuracy": mean("prospective_rebinding", "mean_future_rebind_accuracy_16_worlds"),
        "future_rebind_gain_prl_minus_baseline": mean("prospective_rebinding", "mean_future_rebind_accuracy_16_worlds") - mean("baseline", "mean_future_rebind_accuracy_16_worlds"),
        "baseline_mean_descriptor_drift": mean("baseline", "mean_descriptor_cosine_drift_across_world_rewrite"),
        "prl_mean_descriptor_drift": mean("prospective_rebinding", "mean_descriptor_cosine_drift_across_world_rewrite"),
        "baseline_mean_value_probe_accuracy": statistics.mean((mean("baseline", "home_value_low2_probe_accuracy_a"), mean("baseline", "home_value_low2_probe_accuracy_b"))),
        "prl_mean_value_probe_accuracy": statistics.mean((mean("prospective_rebinding", "home_value_low2_probe_accuracy_a"), mean("prospective_rebinding", "home_value_low2_probe_accuracy_b"))),
        "world_rebinding_gradient_steps_at_inference": 0,
        "mechanism": (
            "PRL trains the same semantic query under paired counterfactual world snapshots. Descriptor codes must be interchangeable across those worlds: "
            "the code compiled from W1 is executed with W2 values and vice versa, while an invariance term penalizes snapshot-dependent code drift. "
            "This turns future rebinding from an architectural assumption into an explicit training objective: semantics should remain in the retained "
            "state while mutable payload is forced into the late-bound channel."
        ),
        "claim_boundary": (
            "Invariant representation learning, counterfactual/domain randomization and consistency regularization are established. The unresolved claim "
            "is the specific cross-world swap objective for learning future-bindable neural activation state over mutable authoritative knowledge. "
            "This synthetic gate is not a novelty proof or pretrained-LLM result."
        ),
        "dod_status": "NOT_DOD; prospective-state training-objective gate",
    }
    prl_min = min(r["prospective_rebinding"]["minimum_future_rebind_accuracy_16_worlds"] for r in rows)
    report["minimum_prl_future_rebind_accuracy_across_seeds"] = prl_min
    report["contract_pass"] = (
        prl_min >= 0.97
        and report["prl_mean_descriptor_drift"] <= 0.02
        and report["prl_mean_future_rebind_accuracy"] >= report["baseline_mean_future_rebind_accuracy"] + 0.15
        and report["prl_mean_value_probe_accuracy"] <= report["baseline_mean_value_probe_accuracy"] - 0.10
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in [
        "stage", "baseline_mean_future_rebind_accuracy", "prl_mean_future_rebind_accuracy",
        "future_rebind_gain_prl_minus_baseline", "baseline_mean_descriptor_drift", "prl_mean_descriptor_drift",
        "baseline_mean_value_probe_accuracy", "prl_mean_value_probe_accuracy",
        "minimum_prl_future_rebind_accuracy_across_seeds", "contract_pass", "report_sha256"
    ]}, indent=2))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
