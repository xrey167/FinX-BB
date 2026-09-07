"""Reconstruction attacks against a deletion (ledger section 24).

    direct query, paraphrase, multi-hop, reverse, forced-choice,
    representation probe, activation probe, dependency reconstruction

"Context completion" does not apply to the symbolic query format and is
recorded as not applicable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def forced_choice(logits: np.ndarray, truth: Sequence[int], rng: np.random.Generator, n_entities: int) -> float:
    """Fraction of queries whose true object out-scores a random distractor entity (chance = 0.5)."""
    wins = 0
    for row, t in zip(logits, truth):
        d = int(rng.integers(0, n_entities - 1))
        d = d + 1 if d >= t else d
        wins += int(row[t] > row[d])
    return wins / len(truth)


def object_rank(logits: np.ndarray, truth: Sequence[int], n_entities: int) -> Dict[str, float]:
    """Rank statistics of the true object among entity logits (rank 0 = best; chance mean ≈ n/2)."""
    ranks = []
    for row, t in zip(logits, truth):
        ent = row[:n_entities]
        ranks.append(int((ent > ent[t]).sum()))
    ranks = np.asarray(ranks)
    return {"mean_rank": float(ranks.mean()), "top1": float((ranks == 0).mean()),
            "top5": float((ranks < 5).mean()), "chance_mean_rank": (n_entities - 1) / 2}


class LinearProbe:
    """Multinomial logistic-regression probe from hidden states to object ids."""

    def __init__(self, d: int, n_classes: int, seed: int = 0):
        torch.manual_seed(seed)
        self.w = torch.nn.Linear(d, n_classes)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-2, weight_decay: float = 1e-4) -> float:
        xt, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.w.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(self.w(xt), yt)
            loss.backward()
            opt.step()
        return float(loss.item())

    @torch.no_grad()
    def accuracy(self, x: np.ndarray, y: np.ndarray, topk: int = 1) -> float:
        xt = torch.as_tensor(x, dtype=torch.float32)
        pred = self.w(xt).topk(topk, dim=-1).indices.numpy()
        return float(np.mean([int(t in row) for t, row in zip(y, pred)]))


def routing_mass(routing: np.ndarray, positions: Sequence[int], hop: int = 0) -> np.ndarray:
    """Attention mass placed on the given bank positions at ``hop`` (one per query)."""
    return np.asarray([routing[i, hop, p] for i, p in enumerate(positions)])
