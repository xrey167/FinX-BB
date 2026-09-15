"""Live incarnation authority for version-qualified neural-memory access.

This module is a research prototype for the CAVI trust boundary. It intentionally lives OUTSIDE a
serialized Bank: replay protection is impossible if the attacker can replay the authority state too.

Nothing cryptographic here is claimed as novel. HMACs, nonces, one-time capabilities, monotonically
increasing generations and revocable capabilities are established systems/security techniques. The
research question is whether binding them to one canonical LLM memory identity closes the stale-bank
replay gap while preserving symlink fan-out and lifecycle semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os
from typing import Dict, Optional, Set


@dataclass(frozen=True)
class Capability:
    pod_id: int
    incarnation: int
    nonce: bytes
    mac: bytes


@dataclass
class PodAuthorityState:
    incarnation: int = 1
    active: bool = True
    deleted: bool = False


class IncarnationAuthority:
    """Trusted live authority whose state is not part of a cached neural-memory snapshot.

    Contract:
      * every lifecycle transition that can invalidate old memory material BUMPS incarnation;
      * deleted pod IDs are never reactivated;
      * a capability is valid only for the current active incarnation and caller-provided nonce;
      * capabilities are one-use, so replay inside the same incarnation is rejected;
      * rollback/restore means selecting/restoring payload in the store THEN bumping to a new
        incarnation; an old incarnation never becomes current again (ABA prevention).
    """

    def __init__(self, secret: Optional[bytes] = None):
        self._secret = secret or os.urandom(32)
        self._pods: Dict[int, PodAuthorityState] = {}
        self._used: Set[bytes] = set()

    def create(self, pod_id: int) -> int:
        if pod_id in self._pods:
            raise ValueError(f"pod {pod_id} already exists")
        self._pods[pod_id] = PodAuthorityState()
        return 1

    def state(self, pod_id: int) -> PodAuthorityState:
        if pod_id not in self._pods:
            raise KeyError(pod_id)
        s = self._pods[pod_id]
        return PodAuthorityState(s.incarnation, s.active, s.deleted)

    def _bump(self, pod_id: int, *, active: bool) -> int:
        s = self._pods[pod_id]
        if s.deleted:
            raise KeyError(f"pod {pod_id} deleted")
        s.incarnation += 1
        s.active = active
        return s.incarnation

    # Any payload/lifecycle transition invalidates old exported material. Even an UPDATE bumps.
    def update(self, pod_id: int) -> int: return self._bump(pod_id, active=True)
    def rollback(self, pod_id: int) -> int: return self._bump(pod_id, active=True)
    def revoke(self, pod_id: int) -> int: return self._bump(pod_id, active=False)
    def restore(self, pod_id: int) -> int: return self._bump(pod_id, active=True)
    def shred(self, pod_id: int) -> int: return self._bump(pod_id, active=False)
    def resign(self, pod_id: int) -> int: return self._bump(pod_id, active=True)
    def evict(self, pod_id: int) -> int: return self._bump(pod_id, active=False)

    def delete(self, pod_id: int) -> int:
        s = self._pods[pod_id]
        if s.deleted:
            raise KeyError(f"pod {pod_id} deleted")
        s.incarnation += 1
        s.active = False
        s.deleted = True
        return s.incarnation

    def _message(self, pod_id: int, incarnation: int, nonce: bytes) -> bytes:
        return pod_id.to_bytes(8, "big", signed=False) + incarnation.to_bytes(8, "big", signed=False) + nonce

    def issue(self, pod_id: int, nonce: bytes) -> Capability:
        if not isinstance(nonce, (bytes, bytearray)) or len(nonce) < 16:
            raise ValueError("nonce must be >=16 bytes")
        s = self._pods[pod_id]
        if s.deleted or not s.active:
            raise PermissionError("pod is not active")
        n = bytes(nonce)
        msg = self._message(pod_id, s.incarnation, n)
        mac = hmac.new(self._secret, msg, hashlib.sha256).digest()
        return Capability(pod_id, s.incarnation, n, mac)

    def verify_and_consume(self, cap: Capability, expected_nonce: bytes) -> bool:
        s = self._pods.get(cap.pod_id)
        if s is None or s.deleted or not s.active:
            return False
        if cap.incarnation != s.incarnation or cap.nonce != bytes(expected_nonce):
            return False
        msg = self._message(cap.pod_id, cap.incarnation, cap.nonce)
        want = hmac.new(self._secret, msg, hashlib.sha256).digest()
        if not hmac.compare_digest(cap.mac, want):
            return False
        fingerprint = hashlib.sha256(msg + cap.mac).digest()
        if fingerprint in self._used:
            return False
        self._used.add(fingerprint)
        return True
