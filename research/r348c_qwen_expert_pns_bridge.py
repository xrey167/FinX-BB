from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from research import r348_qwen_fast_pns_bridge as base
from research import r348b_qwen_fast_pns_bridge as binding_fix

# Keep the canonical operand-role repair from R348b.
base.make_rows = binding_fix.make_rows
base.parse_refs = binding_fix.parse_refs
base.STAGE = "R348C-QWEN-PROSPECTIVE-NEURAL-STATE-BRIDGE"


class RoutedExpertJ(nn.Module):
    """Neural J-plane with a hard semantic route and one value expert per op.

    R348b showed that one monolithic J MLP coupled code-cluster variation to the
    value function and created avoidable execution errors despite perfect Qwen
    operation compilation.  The architecture already treats operation semantics
    as B/PNS state, so R348c makes that separation explicit: code chooses an expert;
    each expert learns only its typed current-value transformation.
    """

    def __init__(self, prototypes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prototypes", F.normalize(prototypes.detach().clone(), dim=-1))
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(10, 64), nn.GELU(),
                nn.Linear(64, 64), nn.GELU(),
                nn.Linear(64, 4),
            ) for _ in range(base.OPS)
        ])

    def route(self, code: torch.Tensor) -> torch.Tensor:
        return (F.normalize(code.float(), dim=-1) @ self.prototypes.T).argmax(-1)

    def forward(self, code: torch.Tensor, va: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        route = self.route(code)
        x = torch.cat([base.bit_features(va), base.bit_features(vb)], dim=-1)
        out = torch.empty((len(x), 4), dtype=x.dtype, device=x.device)
        for op, expert in enumerate(self.experts):
            mask = route == op
            if mask.any():
                out[mask] = expert(x[mask])
        return out


def train_routed_j(c):
    torch.manual_seed(34831)
    j = RoutedExpertJ(c.prototypes)
    opt = torch.optim.AdamW(j.parameters(), lr=3.0e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(34832)
    proto = F.normalize(c.prototypes.detach(), dim=-1)
    j.train()

    # Exhaustive value pairs are repeatedly sampled with balanced operations.
    # The actual held-out PNS code is used only to select the semantic expert.
    steps = max(base.J_STEPS, 1400)
    for _ in range(steps):
        per = 128
        op = torch.arange(base.OPS).repeat_interleave(per)
        va = torch.randint(0, base.VALUE_COUNT, (len(op),), generator=gen)
        vb = torch.randint(0, base.VALUE_COUNT, (len(op),), generator=gen)
        code = proto[op]
        loss = F.cross_entropy(j(code, va, vb), base.label(op, va, vb))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(j.parameters(), 1.0)
        opt.step()

    # Deterministic exhaustive typed-domain polish/check. If an expert is not yet
    # exact, continue briefly rather than lowering the correctness gate.
    all_va = torch.arange(base.VALUE_COUNT).repeat_interleave(base.VALUE_COUNT)
    all_vb = torch.arange(base.VALUE_COUNT).repeat(base.VALUE_COUNT)
    for _round in range(500):
        wrong = 0
        losses = []
        for op_i in range(base.OPS):
            op = torch.full_like(all_va, op_i)
            code = proto[op]
            logits = j(code, all_va, all_vb)
            y = base.label(op, all_va, all_vb)
            wrong += int((logits.argmax(-1) != y).sum())
            losses.append(F.cross_entropy(logits, y))
        if wrong == 0:
            break
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    return j.eval()


base.train_j = train_routed_j

if __name__ == "__main__":
    raise SystemExit(base.main())
