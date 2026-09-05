"""CAVI live indirection authority.

This module intentionally uses no cryptographic novelty.  It models the minimum live state needed to
falsify the CAVI composition: a cached resolver result is data, not authority.  At neural-memory
consumption time both the alias binding incarnation and the canonical pod incarnation/reachability
are checked against independent live state.

The distinction matters because a pod-only version check can reject stale payload snapshots after a
pod update/delete, but cannot reject a cached OLD ALIAS RESOLUTION when the alias has been relinked to
a different still-live pod.  CAVI therefore treats both the reference and the referent as versioned
objects and revalidates the pair adjacent to consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

import numpy as np
import torch


class Scope(str, Enum):
    BYPASS = "BYPASS"      # query is outside memory scope: exact base-model path
    RESOLVE = "RESOLVE"    # alias/pod witness is current and live
    UNKNOWN = "UNKNOWN"    # in memory scope, but reference/referent is absent, stale or dead


@dataclass(frozen=True)
class ResolveWitness:
    alias_id: int
    alias_incarnation: int
    pod_id: int
    pod_incarnation: int


@dataclass(frozen=True)
class RowManifest:
    """Authority metadata captured when a Bank/tensor snapshot was serialized.

    Each row belongs either to a canonical pod (alias_id=-1) or to an alias that resolves to a pod.
    The arrays line up with Bank rows.  They are claims from the snapshot, never trusted by themselves.
    """
    pod_id: np.ndarray
    pod_incarnation: np.ndarray
    alias_id: np.ndarray
    alias_incarnation: np.ndarray


@dataclass
class _Pod:
    incarnation: int = 1
    live: bool = True


@dataclass
class _Alias:
    incarnation: int
    pod_id: int
    live: bool = True


class CAVIAuthority:
    def __init__(self) -> None:
        self.pods: Dict[int, _Pod] = {}
        self.aliases: Dict[int, _Alias] = {}

    def create_pod(self, pod_id: int) -> None:
        if pod_id in self.pods:
            raise ValueError(f"pod {pod_id} already exists")
        self.pods[pod_id] = _Pod()

    def create_alias(self, alias_id: int, pod_id: int) -> None:
        if alias_id in self.aliases:
            raise ValueError(f"alias {alias_id} already exists")
        self._require_live_pod(pod_id)
        self.aliases[alias_id] = _Alias(1, pod_id, True)

    def pod_incarnation(self, pod_id: int) -> int:
        return self.pods[pod_id].incarnation

    def alias_incarnation(self, alias_id: int) -> int:
        return self.aliases[alias_id].incarnation

    def update_pod(self, pod_id: int) -> None:
        p = self._require_live_pod(pod_id)
        p.incarnation += 1

    def shred_pod(self, pod_id: int) -> None:
        p = self._require_live_pod(pod_id)
        p.incarnation += 1
        p.live = False

    def restore_pod(self, pod_id: int) -> None:
        p = self.pods[pod_id]
        p.incarnation += 1
        p.live = True

    def delete_pod(self, pod_id: int) -> None:
        p = self._require_live_pod(pod_id)
        p.incarnation += 1
        p.live = False

    def recreate_pod_same_id(self, pod_id: int) -> None:
        """ABA control: same logical id, strictly newer incarnation."""
        p = self.pods[pod_id]
        p.incarnation += 1
        p.live = True

    def relink_alias(self, alias_id: int, pod_id: int) -> None:
        self._require_live_pod(pod_id)
        a = self.aliases[alias_id]
        a.incarnation += 1
        a.pod_id = pod_id
        a.live = True

    def revoke_alias(self, alias_id: int) -> None:
        a = self.aliases[alias_id]
        a.incarnation += 1
        a.live = False

    def restore_alias(self, alias_id: int) -> None:
        a = self.aliases[alias_id]
        a.incarnation += 1
        a.live = True

    def witness(self, alias_id: int) -> ResolveWitness:
        a = self.aliases[alias_id]
        p = self.pods[a.pod_id]
        if not a.live or not p.live:
            raise PermissionError("cannot resolve a dead alias/pod")
        return ResolveWitness(alias_id, a.incarnation, a.pod_id, p.incarnation)

    def validate_witness(self, w: ResolveWitness) -> bool:
        a = self.aliases.get(w.alias_id)
        p = self.pods.get(w.pod_id)
        return bool(
            a is not None and p is not None and a.live and p.live
            and a.incarnation == w.alias_incarnation
            and a.pod_id == w.pod_id
            and p.incarnation == w.pod_incarnation
        )

    def validate_pod_only(self, w: ResolveWitness) -> bool:
        """Baseline: consume-time check of only the referent generation."""
        p = self.pods.get(w.pod_id)
        return bool(p is not None and p.live and p.incarnation == w.pod_incarnation)

    def row_mask(self, manifest: RowManifest, *, full: bool = True) -> np.ndarray:
        """Revalidate every serialized row against live authority.

        `full=False` is the pod-only baseline.  Full CAVI also checks alias incarnation AND binding.
        Bystander rows remain usable when another pod/alias is stale.
        """
        n = int(manifest.pod_id.shape[0])
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            pid = int(manifest.pod_id[i])
            pi = int(manifest.pod_incarnation[i])
            p = self.pods.get(pid)
            ok = p is not None and p.live and p.incarnation == pi
            aid = int(manifest.alias_id[i])
            if full and aid >= 0:
                ai = int(manifest.alias_incarnation[i])
                a = self.aliases.get(aid)
                ok = ok and a is not None and a.live and a.incarnation == ai and a.pod_id == pid
            out[i] = bool(ok)
        return out

    def scope(self, *, in_scope: bool, witness: Optional[ResolveWitness]) -> Scope:
        if not in_scope:
            return Scope.BYPASS
        if witness is not None and self.validate_witness(witness):
            return Scope.RESOLVE
        return Scope.UNKNOWN

    def _require_live_pod(self, pod_id: int) -> _Pod:
        p = self.pods.get(pod_id)
        if p is None or not p.live:
            raise KeyError(f"pod {pod_id} is not live")
        return p


class NeuralConsumptionGuard:
    """Refresh live row authority at each neural-memory read site.

    A caller-computed ``cell_mask`` still has a resolve->consume TOCTOU gap.  This guard attaches a
    *pre-hook* to the same transformer blocks used by ``KnowledgeAdapterLM``.  Immediately before each
    adapter read hook executes, it asks independent live authority for a new mask and replaces the
    model context's allowed rows with ``base_allowed & live_mask``.  Cached resolver outputs and cached
    export-time masks therefore cannot authorize a later read.

    The callback returns a bool vector aligned with the already-materialized Bank rows.  It may be a
    full CAVI alias+pod check or a baseline such as pod-only checking.  The mechanism is intentionally
    generic so experiments can compare policies without changing trained weights.
    """
    def __init__(self, adapter, mask_fn: Callable[[], np.ndarray | torch.Tensor]):
        # Local import avoids making the authority layer depend on a particular model family at import time.
        from so.llm_adapter import transformer_blocks
        self.adapter = adapter
        self.mask_fn = mask_fn
        blocks = transformer_blocks(adapter.lm)
        self._handles = [blocks[l].register_forward_pre_hook(self._pre_hook) for l in adapter.cfg.read_layers]

    def _pre_hook(self, module, inputs):
        ctx = self.adapter._ctx
        if ctx is None:
            return None
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
        ctx["allowed"] = base & live
        return None

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
