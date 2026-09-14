from __future__ import annotations

import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from research import r350_prospective_rebinding_loss as base


def evaluate(c, j, proto, seed):
    gen = torch.Generator().manual_seed(seed * 17011 + 13)
    n = 12000
    op = torch.randint(0, base.OPS, (n,), generator=gen)
    sem = base.semantic_rows(proto, op, gen)
    home_a = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)
    home_b = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)
    future_a = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)
    future_b = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)

    with torch.inference_mode():
        z_home = c(sem, home_a, home_b)
        z_future_compile = c(sem, future_a, future_b)
        home_acc = float(j(z_home, home_a, home_b).argmax(-1).eq(base.label(op, home_a, home_b)).float().mean())
        rebound_acc = float(j(z_home, future_a, future_b).argmax(-1).eq(base.label(op, future_a, future_b)).float().mean())
        fresh_acc = float(j(z_future_compile, future_a, future_b).argmax(-1).eq(base.label(op, future_a, future_b)).float().mean())
        cos = (z_home * z_future_compile).sum(-1)
        mean_descriptor_drift = float((1.0 - cos).mean())
        p95_descriptor_drift = float(torch.quantile(1.0 - cos, 0.95))
        future_accs = []
        for _ in range(16):
            fa = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)
            fb = torch.randint(0, base.VALUE_COUNT, (n,), generator=gen)
            future_accs.append(float(j(z_home, fa, fb).argmax(-1).eq(base.label(op, fa, fb)).float().mean()))

    # Probe fitting must run outside inference mode.  The descriptor itself stays detached.
    train_n = 8000
    xtr = z_home[:train_n].detach().clone()
    xte = z_home[train_n:].detach().clone()
    ya = (home_a & 3).long(); yb = (home_b & 3).long()

    def probe(y, probe_seed):
        torch.manual_seed(probe_seed)
        w = nn.Linear(base.CODE, 4)
        optim = torch.optim.AdamW(w.parameters(), lr=1e-2)
        for _ in range(250):
            logits = w(xtr)
            loss = F.cross_entropy(logits, y[:train_n])
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        with torch.inference_mode():
            return float(w(xte).argmax(-1).eq(y[train_n:]).float().mean())

    leak_a = probe(ya, seed + 31)
    leak_b = probe(yb, seed + 47)
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


base.evaluate = evaluate
base.STAGE = "R350B-PROSPECTIVE-REBINDING-LOSS"

if __name__ == "__main__":
    raise SystemExit(base.main())
