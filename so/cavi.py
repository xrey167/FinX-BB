"""CAVI live indirection authority.

This module intentionally uses no cryptographic novelty. It models the minimum live state needed to
falsify the CAVI composition: a cached resolver result is data, not authority. At neural-memory
consumption time both the alias binding incarnation and the canonical pod incarnation/reachability
are checked against independent live state.

The distinction matters because a pod-only version check can reject stale payload snapshots after a
pod update/delete, but cannot reject a cached OLD ALIAS RESOLUTION when the alias has been relinked to
a different still-live pod. CAVI therefore treats both the reference and the referent as versioned
objects and revalidates the pair adjacent to consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional
import threading

import numpy as np
import torch


class Scope(str, Enum):
    BYPASS = "BYPASS"
    RESOLVE = "RESOLVE"
    UNKNOWN = "UNKNOWN"


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
    The arrays line up with Bank rows. They are claims from the snapshot, never trusted by themselves.
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
    """Independent live authority for references and canonical pods.

    The RLock is not claimed as novel; it is the reference implementation of the atomicity boundary.
    Mutation and validation share the same lock so a concurrent relink/update cannot commit in the
    interval between the final live validation and the neural read that consumes the serialized row.
    Production implementations can replace the lock with an equivalent atomic/versioned protocol.
    """
    def __init__(self) -> None:
        self.pods: Dict[int, _Pod] = {}
        self.aliases: Dict[int, _Alias] = {}
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    def create_pod(self, pod_id: int) -> None:
        with self._lock:
            if pod_id in self.pods:
                raise ValueError(f"pod {pod_id} already exists")
            self.pods[pod_id] = _Pod()

    def create_alias(self, alias_id: int, pod_id: int) -> None:
        with self._lock:
            if alias_id in self.aliases:
                raise ValueError(f"alias {alias_id} already exists")
            self._require_live_pod(pod_id)
            self.aliases[alias_id] = _Alias(1, pod_id, True)

    def pod_incarnation(self, pod_id: int) -> int:
        with self._lock:
            return self.pods[pod_id].incarnation

    def alias_incarnation(self, alias_id: int) -> int:
        with self._lock:
            return self.aliases[alias_id].incarnation

    def update_pod(self, pod_id: int) -> None:
        with self._lock:
            p = self._require_live_pod(pod_id)
            p.incarnation += 1

    def shred_pod(self, pod_id: int) -> None:
        with self._lock:
            p = self._require_live_pod(pod_id)
            p.incarnation += 1
            p.live = False

    def restore_pod(self, pod_id: int) -> None:
        with self._lock:
            p = self.pods[pod_id]
            p.incarnation += 1
            p.live = True

    def delete_pod(self, pod_id: int) -> None:
        with self._lock:
            p = self._require_live_pod(pod_id)
            p.incarnation += 1
            p.live = False

    def recreate_pod_same_id(self, pod_id: int) -> None:
        with self._lock:
            p = self.pods[pod_id]
            p.incarnation += 1
            p.live = True

    def relink_alias(self, alias_id: int, pod_id: int) -> None:
        with self._lock:
            self._require_live_pod(pod_id)
            a = self.aliases[alias_id]
            a.incarnation += 1
            a.pod_id = pod_id
            a.live = True

    def revoke_alias(self, alias_id: int) -> None:
        with self._lock:
            a = self.aliases[alias_id]
            a.incarnation += 1
            a.live = False

    def restore_alias(self, alias_id: int) -> None:
        with self._lock:
            a = self.aliases[alias_id]
            a.incarnation += 1
            a.live = True

    def witness(self, alias_id: int) -> ResolveWitness:
        with self._lock:
            a = self.aliases[alias_id]
            p = self.pods[a.pod_id]
            if not a.live or not p.live:
                raise PermissionError("cannot resolve a dead alias/pod")
            return ResolveWitness(alias_id, a.incarnation, a.pod_id, p.incarnation)

    def validate_witness(self, w: ResolveWitness) -> bool:
        with self._lock:
            a = self.aliases.get(w.alias_id)
            p = self.pods.get(w.pod_id)
            return bool(
                a is not None and p is not None and a.live and p.live
                and a.incarnation == w.alias_incarnation
                and a.pod_id == w.pod_id
                and p.incarnation == w.pod_incarnation
            )

    def validate_pod_only(self, w: ResolveWitness) -> bool:
        with self._lock:
            p = self.pods.get(w.pod_id)
            return bool(p is not None and p.live and p.incarnation == w.pod_incarnation)

    def row_mask(self, manifest: RowManifest, *, full: bool = True) -> np.ndarray:
        """Revalidate serialized rows against live authority under one authority snapshot."""
        with self._lock:
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
    """Refresh authority immediately before each neural-memory read and make it atomic with that read.

    `KnowledgeAdapterLM` consumes memory in a forward hook registered on each read-layer transformer
    block. The guard registers a PRE-hook that acquires the same live-authority lock used by mutations,
    computes a fresh row mask, and then deliberately keeps that lock held while the transformer block
    runs and while the adapter's already-registered forward hook consumes memory. A second forward hook,
    registered later than the adapter hook, releases the lock afterwards.

    This closes the otherwise real pre-hook -> block -> memory-hook TOCTOU interval. The lock is a
    reference implementation, not a novelty claim; any equivalent atomic/versioned consume protocol
    can implement the same contract.
    """
    def __init__(self, adapter, mask_fn: Callable[[], np.ndarray | torch.Tensor], lock=None):
        from so.llm_adapter import transformer_blocks
        self.adapter = adapter
        self.mask_fn = mask_fn
        self.lock = lock
        self._tls = threading.local()
        blocks = transformer_blocks(adapter.lm)
        self._pre_handles = [blocks[l].register_forward_pre_hook(self._pre_hook) for l in adapter.cfg.read_layers]
        self._post_handles = []
        for l in adapter.cfg.read_layers:
            try:
                h = blocks[l].register_forward_hook(self._post_hook, always_call=True)
            except TypeError:  # older torch fallback
                h = blocks[l].register_forward_hook(self._post_hook)
            self._post_handles.append(h)

    def _pre_hook(self, module, inputs):
        ctx = self.adapter._ctx
        if ctx is None:
            return None
        held = False
        if self.lock is not None:
            self.lock.acquire()
            held = True
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
            ctx["allowed"] = base & live
            self._tls.held = held
            return None
        except Exception:
            if held:
                self.lock.release()
            self._tls.held = False
            raise

    def _post_hook(self, module, inputs, output):
        if getattr(self._tls, "held", False):
            self._tls.held = False
            self.lock.release()
        return None

    def close(self) -> None:
        for h in self._pre_handles + self._post_handles:
            h.remove()
        self._pre_handles.clear()
        self._post_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
