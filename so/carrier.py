"""Shaping a fact's carrier so that removing it removes the fact, and nothing else.

WHY THIS EXISTS. ``so/workspace.py`` measured the closure of a fact in a frozen GPT-2 and returned a
number that is useless in exactly the way the collateral was built to expose: for the fact
France -> Paris, eight of eight candidate directions had to go, and by the time they had, bystander
accuracy had fallen from 1.0000 to 0.0000. The directions that carry one fact are not private to it.
Facts are superposed, and the model rebuilds what is projected away -- the Hydra effect.

So there is nothing in the residual stream whose removal is the fact and only the fact. A store has
such a thing: a record. A pod makes it canonical -- one object, many links, delete the object and
every access path dies. **A representation has no inode.** This module builds one.

TWO PROPERTIES, AND NEITHER IS SUFFICIENT ALONE.

  TIED     every access path to a fact delivers it on the SAME direction. Without this the closure is
           the number of phrasings, and a deletion misses the ones nobody trained on -- which is this
           programme's largest recorded failure (E-000017's fired kill criterion; E-000026's 0.8850
           propagation against a 0.90 bar; E-000025's alias reading at 0.3078 on the worst template).
  PRIVATE  different facts' carriers are mutually near-orthogonal, so removing one leaves the others.
           Without this the collateral is the 1.0000 -> 0.0000 measured above.

                    | not tied              | tied
      not private   | closure k, collateral high | closure 1, collateral high   <- where we are
      private       | closure k, collateral low  | closure 1, collateral low    <- a deletion primitive

THE CAPACITY ARITHMETIC, BECAUSE IT DECIDES WHETHER PRIVACY IS POSSIBLE AT ALL. Exact mutual
orthogonality of n carriers needs d >= n, and the synthetic model has n_entities = 256 in d_model =
128. Exact is therefore impossible and near-orthogonal is what is on offer: the Welch bound gives a
minimum achievable maximum coherence of sqrt((n - d) / (d (n - 1))) = sqrt(128 / (128 * 255)) = 0.0626
for those numbers. That is enough. Removing a carrier at coherence c leaves a fraction c^2 of any
other carrier removed with it -- about 0.4 percent here -- so the collateral bound is a consequence of
the coherence, and the coherence is what the loss reports. The bound is computed rather than assumed,
and a loss that pushed below it would be optimising against a theorem.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def welch_bound(n: int, d: int) -> float:
    """The smallest maximum coherence any n unit vectors in d dimensions can have.

    Zero when n <= d, where exact orthogonality exists. Above that it is the floor a privacy loss must
    not be asked to beat, and the number the achieved coherence has to be read against.
    """
    if n <= d:
        return 0.0
    return math.sqrt((n - d) / (d * (n - 1)))


def carriers(model) -> torch.Tensor:
    """The direction the model reads for each object: ``v_fwd(ent_emb(o))``, one row per object.

    This is the tensor that matters rather than the embedding, because it is what ``encode_bank``
    puts into ``v_f`` and therefore what the readout sees. Shaping the embedding instead would leave
    the projection free to undo it.
    """
    return model.v_fwd(model.ent_emb.weight)


def privacy_loss(model, sample: Optional[int] = None, generator: Optional[torch.Generator] = None
                 ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Push the carriers towards mutual near-orthogonality, and no further than the Welch bound.

    The penalty is the mean squared off-diagonal cosine, hinged at the bound: a set of carriers that
    has reached the bound contributes nothing, so the loss cannot fight the answer loss over a margin
    that provably does not exist. ``sample`` takes a random subset of rows per step, which is what
    makes this affordable when the object vocabulary is large; the full Gram is n^2 and is fine at 256.
    """
    v = carriers(model)
    n_all, d = v.shape
    if sample is not None and sample < n_all:
        idx = torch.randperm(n_all, generator=generator)[:sample]
        v = v[idx]
    n = v.shape[0]
    u = F.normalize(v, dim=-1)
    gram = u @ u.t()
    off = gram - torch.diag(torch.diagonal(gram))
    bound = welch_bound(n, d)
    excess = (off.abs() - bound).clamp(min=0.0)
    loss = (excess ** 2).sum() / max(n * (n - 1), 1)
    with torch.no_grad():
        stats = {"coherence_max": float(off.abs().max()), "coherence_rms": float((off ** 2).mean().sqrt()),
                 "welch_bound": bound, "n": n, "d": d}
    return loss, stats


def ablate_carrier(h: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
    """Project the readout state out of a carrier direction. One vector per row of ``h``."""
    u = F.normalize(carrier, dim=-1)
    return h - (h * u).sum(-1, keepdim=True) * u


def ablation_loss(model, hidden: torch.Tensor, target: torch.Tensor, unknown_index: int
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train against the certificate: with the carrier removed, the model must not answer.

    This is the tying property stated as the thing it is FOR. A loss that tied representations across
    phrasings would be a proxy; this optimises the quantity the certificate measures -- if the answer
    survives the removal of the carrier, the carrier was not where the fact was.

    The obvious objection is that it teaches the model to detect the ablation and play dead. That is
    why it must never be reported without the collateral: the same ablation is applied to a batch in
    which most rows are about OTHER facts, and those rows keep their ordinary answer loss. A model
    that plays dead whenever a projection has happened pays for it there, immediately, in the same
    step. The two terms are returned separately so a caller can see which one is moving.
    """
    obj = target.clamp(max=model.ent_emb.weight.shape[0] - 1)
    c = model.v_fwd(model.ent_emb(obj))                       # each row's own carrier
    logits = model.readout(ablate_carrier(hidden, c))
    tgt = torch.full_like(target, unknown_index)
    loss = F.cross_entropy(logits, tgt)
    with torch.no_grad():
        survives = float((logits.argmax(-1) == target).float().mean())
    return loss, {"answer_survives_ablation": survives}
