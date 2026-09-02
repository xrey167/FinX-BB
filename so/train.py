"""Resampled-world training for the Mini-Transformer.

Each step draws a fresh synthetic world with random lifecycle states and a
batch of queries.  Loss = answer cross-entropy + ``route_weight`` × routing
cross-entropy (the control plane knows which cell holds which fact, so the
routing target is available by construction; the ablation family measures
what happens without it).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Bank, bank_from_world, encode_queries, sample_training_queries
from .model import ModelConfig, MutableKnowledgeTransformer
from .world import World


@dataclass
class TrainConfig:
    seed: int = 0
    n_steps: int = 3000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: int = 200
    n_entities: int = 256
    n_relations: int = 4
    n_synonyms: int = 2
    n_cells_min: int = 700
    n_cells_max: int = 1000
    p_revoked: float = 0.10
    p_shred: float = 0.05
    p_stale: float = 0.05
    train_noise: float = 0.03
    route_weight: float = 0.5
    mix: Dict[str, float] = field(default_factory=lambda: {"fwd1": 0.40, "fwd2": 0.25, "fwd3": 0.20, "rev1": 0.15})
    fixed_world: bool = False        # E-000002: train on ONE world so that facts can be memorised
    log_every: int = 250

    def to_dict(self) -> Dict:
        return asdict(self)


def make_centre(seed: int, marker_dim: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    c = rng.normal(size=marker_dim)
    return c / np.linalg.norm(c)


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.n_steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def routing_loss(routing: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the routing distribution against the cell (or null) it should read."""
    B, H, C1 = routing.shape
    target = route.clone()
    ignore = target == -2
    target[target == -1] = C1 - 1           # null cell is the last column
    target[ignore] = 0
    logp = torch.log(routing.clamp_min(1e-9)).reshape(B * H, C1)
    nll = F.nll_loss(logp, target.reshape(-1), reduction="none").reshape(B, H)
    keep = ~ignore
    return (nll * keep).sum() / keep.sum().clamp_min(1)


def train(model_cfg: ModelConfig, cfg: TrainConfig, world_override: Optional[World] = None,
          verbose: bool = True) -> Dict:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MutableKnowledgeTransformer(model_cfg)
    centre = make_centre(cfg.seed, model_cfg.marker_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    fixed = world_override
    if cfg.fixed_world and fixed is None:
        fixed = World.sample(rng, cfg.n_entities, cfg.n_relations, cfg.n_cells_max, cfg.n_synonyms)
    fixed_bank: Optional[Bank] = None
    history: List[Dict] = []
    t0 = time.time()
    model.train()
    for step in range(cfg.n_steps):
        if fixed is not None:
            world = fixed
            if fixed_bank is None:
                fixed_bank = bank_from_world(rng, world, centre, p_revoked=0.0, p_shred=0.0, p_stale=0.0)
            bank = fixed_bank
        else:
            n_cells = int(rng.integers(cfg.n_cells_min, cfg.n_cells_max + 1))
            world = World.sample(rng, cfg.n_entities, cfg.n_relations, n_cells, cfg.n_synonyms)
            bank = bank_from_world(rng, world, centre, cfg.p_revoked, cfg.p_shred, cfg.p_stale)
        queries = sample_training_queries(rng, world, bank, cfg.batch_size, cfg.mix)
        batch = encode_queries(queries, bank, world, model_cfg.max_hops)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        logits, routing, _ = model(bank.tensors(), batch.mode, batch.start, batch.rels, batch.hop_valid,
                                   noise=cfg.train_noise)
        loss_ans = F.cross_entropy(logits, batch.target)
        loss = loss_ans
        if model_cfg.use_routing and cfg.route_weight > 0:
            loss = loss + cfg.route_weight * routing_loss(routing, batch.route)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % cfg.log_every == 0 or step == 0:
            acc = (logits.argmax(-1) == batch.target).float().mean().item()
            rec = {"step": step + 1, "loss": float(loss.item()), "answer_loss": float(loss_ans.item()),
                   "batch_acc": acc, "elapsed_s": time.time() - t0}
            history.append(rec)
            if verbose:
                print(f"  step {step + 1:5d}  loss {rec['loss']:.4f}  ans {rec['answer_loss']:.4f}  acc {acc:.3f}  "
                      f"{rec['elapsed_s']:.0f}s", flush=True)
    model.eval()
    return {"model": model, "centre": centre, "history": history, "train_config": cfg.to_dict(),
            "model_config": model_cfg.to_dict(), "fixed_world": fixed, "train_seconds": time.time() - t0}
