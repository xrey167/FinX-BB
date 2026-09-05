"""Consume-time authorization boundary for exported neural-memory snapshots.

The key design rule is that an exported Bank is data, not authority. A model call may consume that
snapshot only after an independent LIVE authority verifies a one-use capability for the canonical
pod incarnation and the current request nonce.

This is intentionally small and boring. The primitives are established security techniques. The
research value, if any, has to come from the end-to-end memory lifecycle/attestation composition,
not from calling an HMAC a new invention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from so.incarnation import Capability, IncarnationAuthority


@dataclass(frozen=True)
class AuthorizedSnapshot:
    pod_id: int
    bank: Any
    capability: Capability


def consume(authority: IncarnationAuthority, snapshot: AuthorizedSnapshot, request_nonce: bytes) -> Optional[Any]:
    """Return the Bank exactly once iff live authority says this pod incarnation is current.

    Rejection returns None, which is the existing adapter API's exact base-model/bypass path.
    """
    if snapshot.capability.pod_id != snapshot.pod_id:
        return None
    return snapshot.bank if authority.verify_and_consume(snapshot.capability, request_nonce) else None
