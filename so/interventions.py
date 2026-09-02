"""Causal interventions on knowledge cells (ledger section 25).

    disable C  -> routing to the cell is blocked (cell_mask)
    swap A, B  -> payloads exchanged
    restore    -> the inverse operation
    replace C  -> payload overwritten in place

A biomarker is only causal if intervening on the cell the marker points at
changes the answer predictably, and intervening elsewhere does not.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def disable_mask(store, kid: int) -> np.ndarray:
    b = store.bank()
    mask = np.ones(b["kid"].shape[0], dtype=bool)
    mask[b["kid"] == kid] = False
    return mask


def routed_position(routing_row: np.ndarray, hop: int = 0) -> Optional[int]:
    """Bank position receiving the most mass at ``hop`` (None if it is the null cell)."""
    p = routing_row[hop]
    k = int(p.argmax())
    return None if k == p.shape[0] - 1 else k
