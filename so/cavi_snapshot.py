"""Forward-atomic neural-memory consumption for CAVI.

E-000075 falsified the weaker per-read interpretation of the CAVI consumption
boundary: when an adapter reads the same logical memory at multiple transformer
layers, a lifecycle mutation can commit between those reads and one inference can
observe a torn mix of authority generations.

This module implements the corrected reference contract.  One live-authority
snapshot is taken at the first memory read and the authority lock remains held
through the last memory read of the same forward.  Snapshot isolation and locks
are established systems mechanisms; the research question is the neural-memory
composition, not those primitives themselves.
"""
from __future__ import annotations

import threading
from typing import Callable

import numpy as np
import torch

from so.llm_adapter import transformer_blocks


class ForwardSnapshotConsumptionGuard:
    """Validate once and consume one authority generation across all read sites.

    The guard is intentionally scoped only to the adapter's memory-consuming
    region, from the first configured read-layer pre-hook through the last
    configured read-layer post-hook.  Mutations using the same authority lock
    cannot linearize between neural read sites.

    ``mask_fn`` must return a one-dimensional live-row mask aligned with the
    current Bank snapshot.  The serialized Bank remains data, never authority.
    """

    def __init__(self, adapter, mask_fn: Callable[[], np.ndarray | torch.Tensor], lock):
        if lock is None:
            raise ValueError("ForwardSnapshotConsumptionGuard requires the live-authority lock")
        reads = tuple(int(x) for x in adapter.cfg.read_layers)
        if not reads:
            raise ValueError("adapter has no memory read layers")
        if tuple(sorted(reads)) != reads:
            raise ValueError("read_layers must be in forward execution order")

        self.adapter = adapter
        self.mask_fn = mask_fn
        self.lock = lock
        self._tls = threading.local()
        blocks = transformer_blocks(adapter.lm)
        self._first = reads[0]
        self._last = reads[-1]
        # The terminal block is never entered when an earlier block raises.
        # Therefore its always_call hook is not a request-finally boundary.
        # Register a separate outer cleanup hook, including for the fast-head
        # adapter which calls lm.transformer rather than lm.forward.
        root = adapter if isinstance(adapter, torch.nn.Module) else adapter.lm
        self._handles = []
        try:
            self._handles.append(blocks[self._first].register_forward_pre_hook(self._begin))
            self._handles.append(blocks[self._last].register_forward_hook(self._end, always_call=True))
            self._handles.append(root.register_forward_pre_hook(self._root_begin, prepend=True))
            self._handles.append(root.register_forward_hook(self._root_end, always_call=True))
        except Exception:
            for handle in self._handles:
                handle.remove()
            raise

    def _root_begin(self, module, inputs):
        self._tls.depth = getattr(self._tls, "depth", 0) + 1

    def _root_end(self, module, inputs, output):
        depth = max(0, getattr(self._tls, "depth", 1) - 1)
        self._tls.depth = depth
        # A rejected nested forward must not release an outer forward's lock.
        if depth == 0:
            self._release()

    def _release(self):
        if getattr(self._tls, "held", False):
            self._tls.held = False
            self.lock.release()

    def _begin(self, module, inputs):
        ctx = self.adapter._ctx
        if ctx is None:
            return None
        if getattr(self._tls, "held", False):
            raise RuntimeError("nested memory-consuming forward on the same thread is not supported")

        self.lock.acquire()
        self._tls.held = True
        try:
            base = ctx.get("_cavi_base_allowed")
            if base is None:
                base = ctx["allowed"].clone()
                ctx["_cavi_base_allowed"] = base
            live = self.mask_fn()
            if not torch.is_tensor(live):
                live = torch.as_tensor(live, dtype=torch.bool, device=base.device)
            else:
                live = live.to(device=base.device, dtype=torch.bool)
            if live.ndim != 1 or live.numel() != base.numel():
                raise ValueError(f"live mask shape {tuple(live.shape)} does not match {tuple(base.shape)}")
            # Freeze one authorization decision for every memory read in this forward.
            ctx["allowed"] = base & live
            ctx["_cavi_forward_snapshot"] = True
            return None
        except Exception:
            self._tls.held = False
            self.lock.release()
            raise

    def _end(self, module, inputs, output):
        self._release()
        return None

    def close(self) -> None:
        """Detach on the installing thread with no forwards concurrently in flight."""
        self._release()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
