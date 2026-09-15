"""Shared helpers for the validation experiments (E-000002 and later)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from so.data import bank_from_store, encode_queries
from so.evaluation import build_eval_world, predict
from so.experiments.e000001b_mini_transformer import EVAL_CONFIG, train_or_load
from so.model import ModelConfig
from so.reference import ReferenceResolver
from so.train import TrainConfig
from so.world import Query, UNKNOWN, World


def load_base_model(seed: int, steps: int = 3000):
    """The E-000001-B model of ``seed`` (trained if the checkpoint is missing)."""
    return train_or_load("e000001b", seed, ModelConfig(), TrainConfig(seed=seed, n_steps=steps))


def fresh_world(seed: int, centre: np.ndarray, cfg: Dict[str, Any] = EVAL_CONFIG):
    rng, world, store, kids = build_eval_world(seed, cfg["n_entities"], cfg["n_relations"], cfg["n_synonyms"],
                                               cfg["n_cells"], cfg["n_alt_structures"], centre)
    return rng, world, store, kids, ReferenceResolver(store)


def answers(model, store, world, queries: Sequence[Query], **kw) -> np.ndarray:
    return predict(model, store, world, queries, **kw).answers


def all_paraphrases(world: World, subject: int, relation: int) -> List[Query]:
    return [Query("fwd", subject, (relation,), (world.surface_of(relation, k),)) for k in range(world.n_synonyms)]


@torch.no_grad()
def hidden_states(model, store, world, queries: Sequence[Query], cell_mask: Optional[np.ndarray] = None,
                  batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hidden (N, d), routing (N, H, C+1), logits (N, n_entities+1)) for ``queries``."""
    bank = bank_from_store(store)
    tensors = bank.tensors()
    mask_t = None if cell_mask is None else torch.as_tensor(cell_mask, dtype=torch.bool)
    hs, rs, ls = [], [], []
    for i in range(0, len(queries), batch_size):
        chunk = list(queries[i: i + batch_size])
        b = encode_queries(chunk, bank, world, model.cfg.max_hops)
        logits, routing, extras = model(tensors, b.mode, b.start, b.rels, b.hop_valid, cell_mask=mask_t)
        hs.append(extras["hidden"].numpy()); rs.append(routing.numpy()); ls.append(logits.numpy())
    return np.concatenate(hs), np.concatenate(rs), np.concatenate(ls)


def position_of_kid(store, kid: int) -> int:
    return int(np.where(store.bank()["kid"] == kid)[0][0])


def unknown_rate(a: np.ndarray) -> float:
    return float((a == UNKNOWN).mean())


def accuracy(a: np.ndarray, truth: Sequence[int]) -> float:
    return float((a == np.asarray(truth)).mean())
