"""Authoritative, stdlib-only lifecycle reference kernel.

This is a bounded executable contract for a later transactional implementation, not exact wire-
schema conformance with the broader architecture-v1 document. In particular, the in-memory MVP
does not add tenant/commit-sequence fields to every head, request IDs to resolution/verifier
receipts, or principal/time/use fields to path witnesses; those bindings live in capabilities and
receipts here. It does not claim to delete knowledge from model weights. The in-memory lock is the
linearization boundary; serialized payloads, witnesses, capabilities, artifacts, and audit records
never become live authority.

The audit uses HMAC-SHA256 with explicit domain separation. It is intentionally a one-key MVP, not
Ed25519 or RFC 8785. A production adapter must durably protect/rotate keys and persist a trusted
terminal checkpoint. Publication calls a sink while holding the authority lock; sinks must be
short, non-blocking, and must not re-enter this kernel. Durable outbox/transactional publication is
required when an external side effect cannot share the authority transaction.

Lifecycle semantics are explicit: REVOKE preserves the selected immutable payload revision,
ROLLBACK is allowed only while ACTIVE and selects an existing revision under a new authority
generation, and RESTORE is allowed only from REVOKED and may select an explicit existing revision.
The legacy ``secret`` constructor argument is domain-derived into distinct keys; new callers should
provide separate ``capability_secret`` and ``audit_secret`` values. ``clock`` and
``allow_time_override`` are trusted recovery/test boundaries: the default rejects caller-supplied
times, and a recovery clock must not move behind the authenticated issuance-time floor.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Callable, Iterable, Mapping, Protocol, Sequence, TypeAlias, TypeVar


class LifecycleError(RuntimeError):
    """Base class for rejected lifecycle operations."""


class GenerationConflict(LifecycleError):
    """The caller's compare-and-swap generation is stale."""


class IdempotencyConflict(LifecycleError):
    """An idempotency key was already used for a different command body."""


class AuditIntegrityError(LifecycleError):
    """An audit stream failed authentication, chaining, or replay validation."""


class CapabilityError(PermissionError):
    """A consume capability is invalid, expired, stale, or already spent."""


class PublicationError(PermissionError):
    """A derived artifact cannot cross the publication boundary."""


class AuthorityStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    DELETED = "DELETED"


class TargetKind(str, Enum):
    POD = "POD"
    ALIAS = "ALIAS"


class RouteKind(str, Enum):
    BYPASS = "BYPASS"
    RESOLVE = "RESOLVE"
    UNKNOWN = "UNKNOWN"


class ReceiptKind(str, Enum):
    MUTATION = "MUTATION"
    RESOLUTION = "RESOLUTION"
    DERIVATION = "DERIVATION"
    CAPABILITY_ISSUANCE = "CAPABILITY_ISSUANCE"
    CONSUMPTION = "CONSUMPTION"
    PUBLICATION = "PUBLICATION"
    VERIFIER_RUN = "VERIFIER_RUN"


class VerifierVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    # Accepted as a legacy input only; aggregate receipts always emit INCONCLUSIVE.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AliasTarget:
    kind: TargetKind
    object_id: str

    @classmethod
    def pod(cls, pod_id: str) -> "AliasTarget":
        return cls(TargetKind.POD, pod_id)

    @classmethod
    def alias(cls, alias_id: str) -> "AliasTarget":
        return cls(TargetKind.ALIAS, alias_id)


@dataclass(frozen=True)
class PolicyHead:
    authority_generation: int
    exact_scope: frozenset[str]


@dataclass(frozen=True)
class PodHead:
    pod_id: str
    authority_generation: int
    payload_revision: int
    status: AuthorityStatus
    payload_hash: str
    policy_generation: int


@dataclass(frozen=True)
class AliasHead:
    alias_id: str
    authority_generation: int
    status: AuthorityStatus
    target: AliasTarget


@dataclass(frozen=True)
class AliasPathNode:
    alias_id: str
    authority_generation: int
    target: AliasTarget


@dataclass(frozen=True)
class PathWitness:
    namespace: str
    root_alias_id: str
    alias_path: tuple[AliasPathNode, ...]
    pod_id: str
    pod_generation: int
    payload_revision: int
    payload_hash: str
    policy_generation: int


@dataclass(frozen=True)
class MissingAliasDependency:
    namespace: str
    alias_id: str
    policy_generation: int


@dataclass(frozen=True)
class OutOfScopeDependency:
    namespace: str
    alias_id: str
    policy_generation: int


@dataclass(frozen=True)
class UnresolvedDependency:
    """Fail-closed marker for an error that is not a stable negative lookup."""

    namespace: str
    alias_id: str
    reason: str
    policy_generation: int


LineageDependency: TypeAlias = (
    PathWitness | MissingAliasDependency | OutOfScopeDependency | UnresolvedDependency
)


@dataclass(frozen=True)
class RouteDecision:
    kind: RouteKind
    witness: PathWitness | None = None
    dependency: LineageDependency | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ArtifactLineage:
    dependencies: tuple[LineageDependency, ...]

    @classmethod
    def union(cls, groups: Iterable[Iterable[LineageDependency]]) -> "ArtifactLineage":
        ordered: dict[LineageDependency, None] = {}
        for group in groups:
            for dependency in group:
                ordered.setdefault(dependency, None)
        return cls(tuple(ordered))


@dataclass(frozen=True)
class DerivedArtifact:
    artifact_id: str
    payload: bytes
    lineage: ArtifactLineage


@dataclass(frozen=True)
class PreparedPublication:
    publication_id: str
    artifact_id: str
    namespace: str
    payload_hash: str
    lineage_hash: str
    mac: str


class PublicationSink(Protocol):
    """Idempotent sink: repeated commits for one ID must have one durable effect."""

    def commit(self, publication_id: str, payload: bytes) -> bool: ...


@dataclass(frozen=True)
class VerifierResult:
    verifier: str
    verdict: VerifierVerdict
    positive_control_passed: bool


@dataclass(frozen=True)
class ConsumeCapability:
    witness: PathWitness
    principal: str
    namespace: str
    nonce: str
    issued_at: float
    expires_at: float
    mac: str


@dataclass(frozen=True)
class AuditRecord:
    sequence: int
    previous_hash: str
    event_json: str
    record_hash: str
    mac: str
    namespace: str
    receipt_kind: ReceiptKind
    max_hops: int
    recorded_at: float
    engine_id: str
    code_version: str
    key_id: str


@dataclass(frozen=True)
class AuditCheckpoint:
    length: int
    terminal_hash: str


_CommandResult: TypeAlias = PodHead | AliasHead | PolicyHead
_Result = TypeVar("_Result", PodHead, AliasHead, PolicyHead)
_GENESIS_HASH = "0" * 64
_AUDIT_HASH_DOMAIN = b"FINX-LIFECYCLE-AUDIT-HASH-V1\x00"
_AUDIT_MAC_DOMAIN = b"FINX-LIFECYCLE-AUDIT-MAC-V1\x00"
_CAPABILITY_DOMAIN = b"FINX-LIFECYCLE-CAPABILITY-V1\x00"
_CAPABILITY_COMMITMENT_DOMAIN = b"FINX-LIFECYCLE-CAPABILITY-COMMITMENT-V1\x00"
_PUBLICATION_DOMAIN = b"FINX-LIFECYCLE-PUBLICATION-V1\x00"
_CAPABILITY_KEY_DOMAIN = b"FINX-LIFECYCLE-CAPABILITY-KEY-V1\x00"
_AUDIT_KEY_DOMAIN = b"FINX-LIFECYCLE-AUDIT-KEY-V1\x00"


def _canonical_json(value: Mapping[str, object]) -> str:
    """Stable encoding for this prototype; deliberately not claimed as RFC 8785."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _audit_material(
    *,
    sequence: int,
    previous_hash: str,
    event_json: str,
    namespace: str,
    receipt_kind: ReceiptKind,
    max_hops: int,
    recorded_at: float,
    engine_id: str,
    code_version: str,
    key_id: str,
) -> bytes:
    return _canonical_json(
        {
            "sequence": sequence,
            "previous_hash": previous_hash,
            "event_json": event_json,
            "namespace": namespace,
            "receipt_kind": receipt_kind.value,
            "max_hops": max_hops,
            "recorded_at": recorded_at,
            "engine_id": engine_id,
            "code_version": code_version,
            "key_id": key_id,
        }
    ).encode("utf-8")


class LifecycleKernel:
    """Thread-safe authority for exact registered aliases and their derived state."""

    def __init__(
        self,
        *,
        namespace: str,
        max_hops: int = 8,
        secret: bytes | None = None,
        capability_secret: bytes | None = None,
        audit_secret: bytes | None = None,
        clock: Callable[[], float] | None = None,
        allow_time_override: bool = False,
        engine_id: str = "finx-lifecycle-kernel",
        code_version: str = "reference-v3",
        key_id: str = "local-v1",
    ) -> None:
        if type(namespace) is not str or not namespace:
            raise ValueError("namespace must be a non-empty string")
        if not self._max_hops_valid(max_hops):
            raise ValueError("max_hops must be an integer between 1 and 1024")
        if any(
            type(value) is not str or not value
            for value in (engine_id, code_version, key_id)
        ):
            raise ValueError(
                "audit engine_id, code_version, and key_id must be non-empty strings"
            )
        if type(allow_time_override) is not bool:
            raise ValueError("allow_time_override must be a bool")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._namespace = namespace
        self._max_hops = max_hops
        if secret is not None and (capability_secret is not None or audit_secret is not None):
            raise ValueError("legacy secret cannot be combined with explicit separated secrets")
        if secret is not None:
            master = self._secret_bytes(secret, "secret")
            capability_secret = hmac.new(
                master, _CAPABILITY_KEY_DOMAIN, hashlib.sha256
            ).digest()
            audit_secret = hmac.new(master, _AUDIT_KEY_DOMAIN, hashlib.sha256).digest()
        self._capability_secret = self._secret_bytes(
            secrets.token_bytes(32) if capability_secret is None else capability_secret,
            "capability_secret",
        )
        self._audit_secret = self._secret_bytes(
            secrets.token_bytes(32) if audit_secret is None else audit_secret,
            "audit_secret",
        )
        if hmac.compare_digest(self._capability_secret, self._audit_secret):
            raise ValueError("capability_secret and audit_secret must be distinct")
        self._clock = time.time if clock is None else clock
        self._allow_time_override = allow_time_override
        self._engine_id = engine_id
        self._code_version = code_version
        self._key_id = key_id
        self._last_audit_time = -math.inf
        self._last_security_time = -math.inf
        self._lock = threading.RLock()
        self._thread_state = threading.local()
        self._pods: dict[str, PodHead] = {}
        self._payloads: dict[str, dict[int, bytes]] = {}
        self._aliases: dict[str, AliasHead] = {}
        self._identities: dict[str, TargetKind] = {}
        self._policy = PolicyHead(1, frozenset())
        self._artifacts: dict[str, DerivedArtifact] = {}
        self._idempotency: dict[str, tuple[str, _CommandResult]] = {}
        self._audit: list[AuditRecord] = []
        self._issued_capabilities: dict[
            str, tuple[str, PathWitness, float, float, str]
        ] = {}
        self._spent_nonces: set[str] = set()
        self._published: set[str] = set()
        self._published_artifacts: set[str] = set()
        self._publication_bindings: dict[str, str] = {}
        self._prepared_publications: dict[str, str] = {}

    @property
    def namespace(self) -> str:
        self._require_not_publishing()
        return self._namespace

    @property
    def max_hops(self) -> int:
        self._require_not_publishing()
        return self._max_hops

    @property
    def audit_records(self) -> tuple[AuditRecord, ...]:
        self._require_not_publishing()
        with self._lock:
            return tuple(self._audit)

    def audit_checkpoint(self) -> AuditCheckpoint:
        self._require_not_publishing()
        with self._lock:
            terminal = self._audit[-1].record_hash if self._audit else _GENESIS_HASH
            return AuditCheckpoint(len(self._audit), terminal)

    def policy_head(self) -> PolicyHead:
        self._require_not_publishing()
        with self._lock:
            return self._policy

    def pod_head(self, pod_id: str) -> PodHead:
        self._require_not_publishing()
        pod_id = self._object_id(pod_id, "pod_id")
        with self._lock:
            return self._pods[pod_id]

    def alias_head(self, alias_id: str) -> AliasHead:
        self._require_not_publishing()
        alias_id = self._object_id(alias_id, "alias_id")
        with self._lock:
            return self._aliases[alias_id]

    def read_pod(self, pod_id: str) -> bytes:
        """Privileged authority inspection; not a product or neural authorization path."""

        self._require_not_publishing()
        pod_id = self._object_id(pod_id, "pod_id")
        with self._lock:
            head = self._pods[pod_id]
            if head.status is not AuthorityStatus.ACTIVE:
                raise PermissionError(f"pod {pod_id!r} is not active")
            return self._payloads[pod_id][head.payload_revision]

    def create_pod(
        self,
        pod_id: str,
        payload: bytes,
        *,
        idempotency_key: str,
        expected_generation: int = 0,
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        payload = self._payload(payload)
        self._require_create_generation(expected_generation)
        body = {
            "pod_id": pod_id,
            "payload_hex": payload.hex(),
            "expected_generation": expected_generation,
        }

        def apply() -> tuple[PodHead, dict[str, object]]:
            self._require_unused_identity(pod_id)
            head = PodHead(
                pod_id, 1, 1, AuthorityStatus.ACTIVE, _payload_hash(payload),
                self._policy.authority_generation,
            )
            self._payloads[pod_id] = {1: payload}
            self._pods[pod_id] = head
            self._identities[pod_id] = TargetKind.POD
            return head, self._pod_event("CREATE_POD", head, payload)

        return self._command("CREATE_POD", idempotency_key, body, apply)

    def edit_pod(
        self, pod_id: str, payload: bytes, *, expected_generation: int, idempotency_key: str
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        payload = self._payload(payload)
        self._require_positive_generation(expected_generation)
        body = {
            "pod_id": pod_id, "payload_hex": payload.hex(),
            "expected_generation": expected_generation,
        }

        def apply() -> tuple[PodHead, dict[str, object]]:
            current = self._active_pod_for_mutation(pod_id, expected_generation)
            revision = max(self._payloads[pod_id]) + 1
            head = PodHead(
                pod_id, current.authority_generation + 1, revision, AuthorityStatus.ACTIVE,
                _payload_hash(payload), self._policy.authority_generation,
            )
            self._payloads[pod_id][revision] = payload
            self._pods[pod_id] = head
            return head, self._pod_event("EDIT_POD", head, payload)

        return self._command("EDIT_POD", idempotency_key, body, apply)

    def revoke_pod(
        self, pod_id: str, *, expected_generation: int, idempotency_key: str
    ) -> PodHead:
        return self._change_pod_status(
            "REVOKE_POD", pod_id, expected_generation=expected_generation,
            idempotency_key=idempotency_key, expected_status=AuthorityStatus.ACTIVE,
            new_status=AuthorityStatus.REVOKED,
        )

    def restore_pod(
        self,
        pod_id: str,
        *,
        expected_generation: int,
        idempotency_key: str,
        payload_revision: int | None = None,
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        self._require_positive_generation(expected_generation)
        if payload_revision is not None:
            self._require_positive_revision(payload_revision)
        body = {
            "pod_id": pod_id, "expected_generation": expected_generation,
            "payload_revision": payload_revision,
        }

        def apply() -> tuple[PodHead, dict[str, object]]:
            current = self._pod_for_cas(pod_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"pod {pod_id!r} is permanently deleted")
            if current.status is not AuthorityStatus.REVOKED:
                raise PermissionError(f"pod {pod_id!r} must be REVOKED for restore")
            revision = current.payload_revision if payload_revision is None else payload_revision
            try:
                payload = self._payloads[pod_id][revision]
            except KeyError as exc:
                raise KeyError(f"pod {pod_id!r} has no payload revision {revision}") from exc
            head = PodHead(
                pod_id, current.authority_generation + 1, revision, AuthorityStatus.ACTIVE,
                _payload_hash(payload), self._policy.authority_generation,
            )
            self._pods[pod_id] = head
            return head, self._pod_event("RESTORE_POD", head)

        return self._command("RESTORE_POD", idempotency_key, body, apply)

    def delete_pod(
        self, pod_id: str, *, expected_generation: int, idempotency_key: str
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        self._require_positive_generation(expected_generation)
        body = {"pod_id": pod_id, "expected_generation": expected_generation}

        def apply() -> tuple[PodHead, dict[str, object]]:
            current = self._pod_for_cas(pod_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"pod {pod_id!r} is permanently deleted")
            head = replace(
                current, authority_generation=current.authority_generation + 1,
                status=AuthorityStatus.DELETED,
                policy_generation=self._policy.authority_generation,
            )
            self._pods[pod_id] = head
            return head, self._pod_event("DELETE_POD", head)

        return self._command("DELETE_POD", idempotency_key, body, apply)

    def rollback_pod(
        self,
        pod_id: str,
        *,
        to_payload_revision: int,
        expected_generation: int,
        idempotency_key: str,
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        self._require_positive_generation(expected_generation)
        self._require_positive_revision(to_payload_revision)
        body = {
            "pod_id": pod_id, "to_payload_revision": to_payload_revision,
            "expected_generation": expected_generation,
        }

        def apply() -> tuple[PodHead, dict[str, object]]:
            current = self._active_pod_for_mutation(pod_id, expected_generation)
            try:
                payload = self._payloads[pod_id][to_payload_revision]
            except KeyError as exc:
                raise KeyError(
                    f"pod {pod_id!r} has no payload revision {to_payload_revision}"
                ) from exc
            head = PodHead(
                pod_id, current.authority_generation + 1, to_payload_revision,
                AuthorityStatus.ACTIVE, _payload_hash(payload),
                self._policy.authority_generation,
            )
            self._pods[pod_id] = head
            return head, self._pod_event("ROLLBACK_POD", head)

        return self._command("ROLLBACK_POD", idempotency_key, body, apply)

    def create_alias(
        self,
        alias_id: str,
        target: AliasTarget,
        *,
        idempotency_key: str,
        expected_generation: int = 0,
    ) -> AliasHead:
        alias_id = self._object_id(alias_id, "alias_id")
        target = self._target(target)
        self._require_create_generation(expected_generation)
        body = {
            "alias_id": alias_id, "target": self._target_json(target),
            "expected_generation": expected_generation,
        }

        def apply() -> tuple[AliasHead, dict[str, object]]:
            self._require_unused_identity(alias_id)
            self._validate_target_path_locked(alias_id, target)
            head = AliasHead(alias_id, 1, AuthorityStatus.ACTIVE, target)
            self._aliases[alias_id] = head
            self._identities[alias_id] = TargetKind.ALIAS
            return head, self._alias_event("CREATE_ALIAS", head)

        return self._command("CREATE_ALIAS", idempotency_key, body, apply)

    def relink_alias(
        self,
        alias_id: str,
        target: AliasTarget,
        *,
        expected_generation: int,
        idempotency_key: str,
    ) -> AliasHead:
        alias_id = self._object_id(alias_id, "alias_id")
        target = self._target(target)
        self._require_positive_generation(expected_generation)
        body = {
            "alias_id": alias_id, "target": self._target_json(target),
            "expected_generation": expected_generation,
        }

        def apply() -> tuple[AliasHead, dict[str, object]]:
            current = self._active_alias_for_mutation(alias_id, expected_generation)
            self._validate_target_path_locked(alias_id, target)
            head = replace(
                current, authority_generation=current.authority_generation + 1, target=target
            )
            self._aliases[alias_id] = head
            return head, self._alias_event("RELINK_ALIAS", head)

        return self._command("RELINK_ALIAS", idempotency_key, body, apply)

    def revoke_alias(
        self, alias_id: str, *, expected_generation: int, idempotency_key: str
    ) -> AliasHead:
        return self._change_alias_status(
            "REVOKE_ALIAS", alias_id, expected_generation=expected_generation,
            idempotency_key=idempotency_key, expected_status=AuthorityStatus.ACTIVE,
            new_status=AuthorityStatus.REVOKED,
        )

    def restore_alias(
        self, alias_id: str, *, expected_generation: int, idempotency_key: str
    ) -> AliasHead:
        alias_id = self._object_id(alias_id, "alias_id")
        self._require_positive_generation(expected_generation)
        body = {"alias_id": alias_id, "expected_generation": expected_generation}

        def apply() -> tuple[AliasHead, dict[str, object]]:
            current = self._alias_for_cas(alias_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"alias {alias_id!r} is permanently deleted")
            if current.status is not AuthorityStatus.REVOKED:
                raise PermissionError(f"alias {alias_id!r} must be REVOKED for restore")
            self._validate_target_path_locked(alias_id, current.target)
            head = replace(
                current, authority_generation=current.authority_generation + 1,
                status=AuthorityStatus.ACTIVE,
            )
            self._aliases[alias_id] = head
            return head, self._alias_event("RESTORE_ALIAS", head)

        return self._command("RESTORE_ALIAS", idempotency_key, body, apply)

    def delete_alias(
        self, alias_id: str, *, expected_generation: int, idempotency_key: str
    ) -> AliasHead:
        alias_id = self._object_id(alias_id, "alias_id")
        self._require_positive_generation(expected_generation)
        body = {"alias_id": alias_id, "expected_generation": expected_generation}

        def apply() -> tuple[AliasHead, dict[str, object]]:
            current = self._alias_for_cas(alias_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"alias {alias_id!r} is permanently deleted")
            head = replace(
                current, authority_generation=current.authority_generation + 1,
                status=AuthorityStatus.DELETED,
            )
            self._aliases[alias_id] = head
            return head, self._alias_event("DELETE_ALIAS", head)

        return self._command("DELETE_ALIAS", idempotency_key, body, apply)

    def set_policy_scope(
        self,
        alias_ids: Iterable[str],
        *,
        expected_generation: int,
        idempotency_key: str,
    ) -> PolicyHead:
        self._require_positive_generation(expected_generation)
        exact_scope = frozenset(self._object_id(value, "alias_id") for value in alias_ids)
        body = {
            "exact_scope": sorted(exact_scope), "expected_generation": expected_generation,
        }

        def apply() -> tuple[PolicyHead, dict[str, object]]:
            if self._policy.authority_generation != expected_generation:
                raise GenerationConflict(
                    f"policy generation is {self._policy.authority_generation}, "
                    f"expected {expected_generation}"
                )
            policy = PolicyHead(expected_generation + 1, exact_scope)
            self._policy = policy
            for pod_id, head in tuple(self._pods.items()):
                self._pods[pod_id] = replace(
                    head, authority_generation=head.authority_generation + 1,
                    policy_generation=policy.authority_generation,
                )
            return policy, self._policy_event("SET_POLICY_SCOPE", policy)

        return self._command("SET_POLICY_SCOPE", idempotency_key, body, apply)

    def route(self, alias_id: str, *, in_scope: bool | None = None) -> RouteDecision:
        """Route by authoritative exact scope; ``in_scope`` is accepted but never trusted."""

        self._require_not_publishing()
        del in_scope
        alias_id = self._object_id(alias_id, "alias_id")
        try:
            with self._lock:
                registered = alias_id in self._aliases or alias_id in self._policy.exact_scope
                decision = self._resolve_locked(alias_id) if registered else RouteDecision(
                    RouteKind.BYPASS,
                    dependency=OutOfScopeDependency(
                        self.namespace, alias_id, self._policy.authority_generation
                    ),
                    reason="OUT_OF_SCOPE",
                )
                self._append_audit(
                    self._resolution_event(alias_id, decision),
                    ReceiptKind.RESOLUTION,
                )
                return decision
        except Exception:
            with self._lock:
                decision = RouteDecision(
                    RouteKind.UNKNOWN,
                    dependency=UnresolvedDependency(
                        self.namespace,
                        alias_id,
                        "AUTHORITY_ERROR",
                        self._policy.authority_generation,
                    ),
                    reason="AUTHORITY_ERROR",
                )
                try:
                    self._append_audit(
                        self._resolution_event(alias_id, decision),
                        ReceiptKind.RESOLUTION,
                    )
                except Exception as audit_exc:
                    raise AuditIntegrityError(
                        "route failed closed and its UNKNOWN receipt could not be audited"
                    ) from audit_exc
            return decision

    def _resolve_locked(self, alias_id: str) -> RouteDecision:
        path: list[AliasPathNode] = []
        visited: set[str] = set()
        current_id = alias_id
        while True:
            if current_id in visited:
                return self._unresolved(alias_id, "CYCLE")
            if len(path) >= self.max_hops:
                return self._unresolved(alias_id, "MAX_HOPS")
            visited.add(current_id)
            head = self._aliases.get(current_id)
            if head is None:
                dependency: LineageDependency
                if not path:
                    dependency = MissingAliasDependency(
                        self.namespace, alias_id, self._policy.authority_generation
                    )
                else:
                    dependency = UnresolvedDependency(
                        self.namespace, alias_id, "MISSING_TARGET",
                        self._policy.authority_generation,
                    )
                return RouteDecision(
                    RouteKind.UNKNOWN, dependency=dependency, reason="MISSING_ALIAS"
                )
            if head.status is not AuthorityStatus.ACTIVE:
                return self._unresolved(alias_id, "DEAD_ALIAS")
            path.append(AliasPathNode(head.alias_id, head.authority_generation, head.target))
            if head.target.kind is TargetKind.ALIAS:
                current_id = head.target.object_id
                continue
            pod = self._pods.get(head.target.object_id)
            if pod is None or pod.status is not AuthorityStatus.ACTIVE:
                return self._unresolved(alias_id, "MISSING_OR_DEAD_POD")
            if pod.policy_generation != self._policy.authority_generation:
                return self._unresolved(alias_id, "STALE_POLICY_HEAD")
            witness = PathWitness(
                self.namespace, alias_id, tuple(path), pod.pod_id, pod.authority_generation,
                pod.payload_revision, pod.payload_hash, self._policy.authority_generation,
            )
            return RouteDecision(RouteKind.RESOLVE, witness=witness, dependency=witness)

    def validate_witness(self, witness: PathWitness | None) -> bool:
        self._require_not_publishing()
        if witness is None:
            return False
        try:
            witness = self._snapshot_witness(witness)
            with self._lock:
                return self._validate_witness_locked(witness)
        except Exception:
            return False

    def _validate_witness_locked(self, witness: PathWitness) -> bool:
        if (
            not self._witness_well_formed(witness)
            or witness.policy_generation != self._policy.authority_generation
        ):
            return False
        alias_ids = [node.alias_id for node in witness.alias_path]
        if witness.root_alias_id != alias_ids[0] or len(alias_ids) != len(set(alias_ids)):
            return False
        authoritative = self._resolve_locked(witness.root_alias_id)
        return authoritative.kind is RouteKind.RESOLVE and authoritative.witness == witness

    def derive_artifact(
        self,
        artifact_id: str,
        payload: bytes,
        *,
        decisions: Sequence[RouteDecision] = (),
        parents: Sequence[str] = (),
    ) -> DerivedArtifact:
        self._require_not_publishing()
        artifact_id = self._object_id(artifact_id, "artifact_id")
        payload = self._payload(payload)
        try:
            decisions = tuple(self._snapshot_route_decision(item) for item in decisions)
            parents = tuple(self._object_id(item, "parent artifact_id") for item in parents)
        except (TypeError, ValueError) as exc:
            raise TypeError("decisions or parents contain a noncanonical value") from exc
        with self._lock:
            artifacts_before = dict(self._artifacts)
            audit_before = list(self._audit)
            audit_time_before = self._last_audit_time
            try:
                artifact = self._derive_artifact_locked(
                    artifact_id, payload, decisions, parents
                )
                self._append_audit(
                    {
                        "operation": "DERIVE_ARTIFACT",
                        "artifact_id": artifact.artifact_id,
                        "payload_hex": artifact.payload.hex(),
                        "lineage": self._lineage_json(artifact.lineage),
                    },
                    ReceiptKind.DERIVATION,
                )
                return artifact
            except BaseException:
                self._artifacts = artifacts_before
                self._audit = audit_before
                self._last_audit_time = audit_time_before
                raise

    def _derive_artifact_locked(
        self,
        artifact_id: str,
        payload: bytes,
        decisions: Sequence[RouteDecision],
        parents: Sequence[str],
    ) -> DerivedArtifact:
        if artifact_id in self._artifacts:
            raise ValueError(f"artifact {artifact_id!r} already exists")
        groups: list[Iterable[LineageDependency]] = []
        for parent_id in parents:
            parent = self._artifacts[parent_id]
            if not self._artifact_current_locked(parent):
                raise ValueError("parent artifact lineage is not current")
            groups.append(parent.lineage.dependencies)
        decision_dependencies: list[LineageDependency] = []
        for decision in decisions:
            if type(decision) is not RouteDecision or not isinstance(
                decision.kind, RouteKind
            ):
                raise TypeError("decision must be a typed RouteDecision")
            dependency = decision.dependency
            valid_kind = (
                decision.kind is RouteKind.RESOLVE
                and type(dependency) is PathWitness
                and decision.witness is dependency
            ) or (
                decision.kind is RouteKind.BYPASS
                and isinstance(dependency, OutOfScopeDependency)
                and decision.witness is None
            ) or (
                decision.kind is RouteKind.UNKNOWN
                and isinstance(
                    dependency, (MissingAliasDependency, UnresolvedDependency)
                )
                and decision.witness is None
            )
            if (
                not valid_kind
                or not self._dependency_well_formed(dependency)
                or not self._dependency_current_locked(dependency)
            ):
                raise ValueError(
                    "decision dependency is not current well-formed authority lineage"
                )
            decision_dependencies.append(dependency)
        groups.append(decision_dependencies)
        lineage = ArtifactLineage.union(groups)
        if not lineage.dependencies:
            raise ValueError("artifact lineage must contain an authority dependency")
        artifact = DerivedArtifact(artifact_id, payload, lineage)
        if not self._artifact_current_locked(artifact):
            raise ValueError("artifact lineage is not current at derivation event")
        self._artifacts[artifact_id] = artifact
        return artifact

    def artifact_current(self, artifact_id: str) -> bool:
        self._require_not_publishing()
        try:
            artifact_id = self._object_id(artifact_id, "artifact_id")
            with self._lock:
                return self._artifact_current_locked(self._artifacts[artifact_id])
        except Exception:
            return False

    def issue_capability(
        self,
        witness: PathWitness,
        *,
        principal: str,
        namespace: str,
        ttl_seconds: float,
        now: float | None = None,
    ) -> ConsumeCapability:
        self._require_not_publishing()
        if type(principal) is not str or not principal:
            raise CapabilityError("principal must be a non-empty string")
        if type(namespace) is not str or namespace != self.namespace:
            raise CapabilityError("namespace mismatch")
        if not self._finite_numeric(ttl_seconds) or ttl_seconds <= 0:
            raise CapabilityError("ttl_seconds must be finite positive numeric")
        try:
            witness = self._snapshot_witness(witness)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("witness is not canonical") from exc
        with self._lock:
            if not self._validate_witness_locked(witness):
                raise CapabilityError("witness is not current")
            security_time_before = self._last_security_time
            issued_before = dict(self._issued_capabilities)
            audit_before = list(self._audit)
            audit_time_before = self._last_audit_time
            try:
                issued_at = self._security_now(now)
                expires_at = issued_at + ttl_seconds
                if not math.isfinite(expires_at):
                    raise CapabilityError("capability expiry must be finite")
                if expires_at <= issued_at:
                    raise CapabilityError(
                        "ttl_seconds is too small to produce a later expiry"
                    )
                nonce = secrets.token_hex(16)
                if not self._random_id_valid(nonce):
                    raise CapabilityError(
                        "capability nonce generator returned a noncanonical ID"
                    )
                if nonce in self._issued_capabilities:
                    raise CapabilityError("capability nonce collision")
                witness_hash = hashlib.sha256(
                    _canonical_json(self._witness_json(witness)).encode("utf-8")
                ).hexdigest()
                mac = self._capability_mac(
                    witness, principal, namespace, nonce, issued_at, expires_at
                )
                capability = ConsumeCapability(
                    witness, principal, namespace, nonce, issued_at, expires_at, mac
                )
                self._append_audit(
                    {
                        "operation": "ISSUE_CAPABILITY",
                        "principal": principal,
                        "nonce": nonce,
                        "witness_hash": witness_hash,
                        "witness": self._witness_json(witness),
                        "capability_commitment": self._capability_commitment(mac),
                        "security_time": issued_at,
                        "expires_at": expires_at,
                    },
                    ReceiptKind.CAPABILITY_ISSUANCE,
                )
                self._issued_capabilities[nonce] = (
                    principal, witness, issued_at, expires_at, mac
                )
                return capability
            except BaseException:
                self._last_security_time = security_time_before
                self._issued_capabilities = issued_before
                self._audit = audit_before
                self._last_audit_time = audit_time_before
                raise

    def consume(
        self,
        capability: ConsumeCapability,
        *,
        principal: str,
        namespace: str,
        now: float | None = None,
    ) -> bytes:
        self._require_not_publishing()
        try:
            capability = self._snapshot_capability(capability)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("capability is not canonical") from exc
        self._validate_public_capability_fields(capability, principal, namespace)
        with self._lock:
            security_time_before = self._last_security_time
            spent_before = set(self._spent_nonces)
            audit_before = list(self._audit)
            audit_time_before = self._last_audit_time
            try:
                checked_at = self._security_now(now)
                payload = self._validate_capability_locked(
                    capability, principal, namespace, checked_at
                )
                self._append_audit(
                    self._consumption_event(
                        capability, principal, checked_at, artifact_id=None
                    ),
                    ReceiptKind.CONSUMPTION,
                )
                self._spent_nonces.add(capability.nonce)
                return payload
            except BaseException:
                self._last_security_time = security_time_before
                self._spent_nonces = spent_before
                self._audit = audit_before
                self._last_audit_time = audit_time_before
                raise

    def consume_to_artifact(
        self,
        capability: ConsumeCapability,
        *,
        artifact_id: str,
        principal: str,
        namespace: str,
        now: float | None = None,
    ) -> DerivedArtifact:
        self._require_not_publishing()
        artifact_id = self._object_id(artifact_id, "artifact_id")
        try:
            capability = self._snapshot_capability(capability)
        except (TypeError, ValueError) as exc:
            raise CapabilityError("capability is not canonical") from exc
        self._validate_public_capability_fields(capability, principal, namespace)
        with self._lock:
            security_time_before = self._last_security_time
            audit_before = list(self._audit)
            audit_time_before = self._last_audit_time
            artifacts_before = dict(self._artifacts)
            spent_before = set(self._spent_nonces)
            try:
                checked_at = self._security_now(now)
                payload = self._validate_capability_locked(
                    capability, principal, namespace, checked_at
                )
                if artifact_id in self._artifacts:
                    raise ValueError(f"artifact {artifact_id!r} already exists")
                record = self._prepare_audit(
                    self._consumption_event(
                        capability, principal, checked_at, artifact_id=artifact_id
                    ),
                    ReceiptKind.CONSUMPTION,
                )
                artifact = DerivedArtifact(
                    artifact_id, payload, ArtifactLineage((capability.witness,))
                )
                self._artifacts[artifact_id] = artifact
                self._spent_nonces.add(capability.nonce)
                self._commit_audit(record)
                return artifact
            except BaseException:
                self._artifacts = artifacts_before
                self._spent_nonces = spent_before
                self._audit = audit_before
                self._last_audit_time = audit_time_before
                self._last_security_time = security_time_before
                raise

    def prepare_publication(self, artifact_id: str) -> PreparedPublication:
        """Create a signed publication intent; no external effect or success receipt occurs."""

        self._require_not_publishing()
        artifact_id = self._object_id(artifact_id, "artifact_id")
        with self._lock:
            artifact = self._artifacts[artifact_id]
            if artifact_id in self._published_artifacts:
                raise PublicationError(f"artifact {artifact_id!r} was already published")
            if not self._artifact_current_locked(artifact):
                raise PublicationError(f"artifact {artifact_id!r} has stale authority lineage")
            publication_id = secrets.token_hex(16)
            if not self._random_id_valid(publication_id):
                raise PublicationError(
                    "publication ID generator returned a noncanonical random ID"
                )
            bound = self._prepared_publications.get(publication_id)
            published_bound = self._publication_bindings.get(publication_id)
            if bound is not None or published_bound is not None:
                raise PublicationError("publication ID collision")
            payload_hash = _payload_hash(artifact.payload)
            lineage_hash = self._lineage_hash(artifact.lineage)
            mac = self._publication_mac(
                publication_id, artifact_id, payload_hash, lineage_hash
            )
            prepared = PreparedPublication(
                publication_id,
                artifact_id,
                self.namespace,
                payload_hash,
                lineage_hash,
                mac,
            )
            self._prepared_publications[publication_id] = artifact_id
            return prepared

    def commit_publication(
        self, prepared: PreparedPublication, sink: PublicationSink
    ) -> None:
        """Commit through an idempotent sink, then append one success receipt.

        A sink exception is deliberately ambiguous: the caller retries the same signed
        ``PreparedPublication`` and the sink must deduplicate by ``publication_id``. This reference
        does not claim atomicity with an arbitrary external service.
        """

        self._require_not_publishing()
        try:
            prepared = self._snapshot_prepared_publication(prepared)
        except (TypeError, ValueError) as exc:
            raise PublicationError("prepared publication is not canonical") from exc
        if (
            not self._random_id_valid(prepared.publication_id)
            or type(prepared.artifact_id) is not str
            or not prepared.artifact_id
            or type(prepared.namespace) is not str
            or not self._digest_valid(prepared.payload_hash)
            or not self._digest_valid(prepared.lineage_hash)
            or not self._digest_valid(prepared.mac)
        ):
            raise PublicationError("prepared publication contains invalid fields")
        commit = getattr(sink, "commit", None)
        if not callable(commit):
            raise TypeError("sink must implement idempotent commit(publication_id, payload)")
        with self._lock:
            expected_mac = self._publication_mac(
                prepared.publication_id,
                prepared.artifact_id,
                prepared.payload_hash,
                prepared.lineage_hash,
            )
            if prepared.namespace != self.namespace or not hmac.compare_digest(
                prepared.mac, expected_mac
            ):
                raise PublicationError("prepared publication authentication failed")
            if prepared.publication_id in self._published:
                if self._publication_bindings.get(
                    prepared.publication_id
                ) != prepared.artifact_id:
                    raise PublicationError(
                        "publication ID is bound to another artifact"
                    )
                return
            prepared_bound = self._prepared_publications.get(prepared.publication_id)
            if prepared_bound is not None and prepared_bound != prepared.artifact_id:
                raise PublicationError("publication ID is bound to another artifact")
            if prepared.artifact_id in self._published_artifacts:
                raise PublicationError("artifact was already published")
            artifact = self._artifacts.get(prepared.artifact_id)
            if artifact is None or not self._artifact_current_locked(artifact):
                raise PublicationError("prepared publication has stale or missing lineage")
            payload_hash = _payload_hash(artifact.payload)
            lineage_hash = self._lineage_hash(artifact.lineage)
            if (
                prepared.payload_hash != payload_hash
                or prepared.lineage_hash != lineage_hash
            ):
                raise PublicationError("prepared publication authentication failed")
            self._thread_state.publishing = True
            try:
                committed = commit(prepared.publication_id, artifact.payload)
            finally:
                self._thread_state.publishing = False
            if committed is not True:
                raise PublicationError("idempotent sink did not confirm durable commit")
            audit_before = list(self._audit)
            audit_time_before = self._last_audit_time
            published_before = set(self._published)
            published_artifacts_before = set(self._published_artifacts)
            publication_bindings_before = dict(self._publication_bindings)
            try:
                record = self._prepare_audit(
                    {
                        "operation": "PUBLISH",
                        "publication_id": prepared.publication_id,
                        "artifact_id": prepared.artifact_id,
                        "payload_hash": payload_hash,
                        "lineage_hash": lineage_hash,
                    },
                    ReceiptKind.PUBLICATION,
                )
                self._commit_audit(record)
                self._published.add(prepared.publication_id)
                self._published_artifacts.add(prepared.artifact_id)
                self._publication_bindings[prepared.publication_id] = prepared.artifact_id
            except BaseException:
                self._audit = audit_before
                self._last_audit_time = audit_time_before
                self._published = published_before
                self._published_artifacts = published_artifacts_before
                self._publication_bindings = publication_bindings_before
                raise

    @staticmethod
    def combine_verifier_results(results: Sequence[VerifierResult]) -> VerifierVerdict:
        source = tuple(results)
        if any(type(result) is not VerifierResult for result in source):
            raise TypeError("results must contain exact VerifierResult values")
        normalized = tuple(
            VerifierResult(
                result.verifier, result.verdict, result.positive_control_passed
            )
            for result in source
        )
        if any(
            type(result.verifier) is not str
            or not result.verifier
            or not isinstance(result.verdict, VerifierVerdict)
            or type(result.positive_control_passed) is not bool
            for result in normalized
        ):
            raise TypeError("results must contain typed VerifierResult values")
        by_role = {result.verifier: result for result in normalized}
        if (
            len(normalized) != 2
            or len(by_role) != 2
            or set(by_role) != {"control", "workspace"}
        ):
            return VerifierVerdict.INCONCLUSIVE
        control = by_role["control"]
        workspace = by_role["workspace"]
        if (
            control.verdict is not VerifierVerdict.PASS
            or not control.positive_control_passed
            or not workspace.positive_control_passed
        ):
            return VerifierVerdict.INCONCLUSIVE
        if workspace.verdict is VerifierVerdict.FAIL:
            return VerifierVerdict.FAIL
        if workspace.verdict is VerifierVerdict.PASS:
            return VerifierVerdict.PASS
        return VerifierVerdict.INCONCLUSIVE

    def record_verifier_run(
        self, result: VerifierResult | Sequence[VerifierResult]
    ) -> AuditRecord:
        self._require_not_publishing()
        try:
            results = (result,) if isinstance(result, VerifierResult) else tuple(result)
        except TypeError as exc:
            raise TypeError("result must be a VerifierResult or a sequence of them") from exc
        if not results:
            raise ValueError("verifier run requires at least one result")
        normalized: list[VerifierResult] = []
        for item in results:
            normalized.append(self._snapshot_verifier_result(item))
        effective = self.combine_verifier_results(normalized)
        input_events = [
            {
                "verifier": item.verifier,
                "verdict": item.verdict.value,
                "positive_control_passed": item.positive_control_passed,
            }
            for item in normalized
        ]
        with self._lock:
            return self._append_audit(
                {
                    "operation": "VERIFIER_RUN",
                    "verifier": normalized[0].verifier,
                    "verdict": effective.value,
                    "positive_control_passed": all(
                        item.positive_control_passed for item in normalized
                    ),
                    "inputs": input_events,
                },
                ReceiptKind.VERIFIER_RUN,
            )

    @staticmethod
    def _receipt_event_schema_valid(
        receipt_kind: ReceiptKind, event: Mapping[str, object]
    ) -> bool:
        """Total, fail-closed wrapper around receipt schema validation."""

        try:
            return LifecycleKernel._receipt_event_schema_valid_unchecked(
                receipt_kind, event
            )
        except Exception:
            return False

    @staticmethod
    def _receipt_event_schema_valid_unchecked(
        receipt_kind: ReceiptKind, event: Mapping[str, object]
    ) -> bool:
        """Bind each authenticated receipt kind to one exact outer event schema."""

        if (
            type(receipt_kind) is not ReceiptKind
            or type(event) is not dict
            or not LifecycleKernel._plain_json_tree_valid(event)
        ):
            return False
        operation = event.get("operation")
        if type(operation) is not str:
            return False
        keys = set(event)
        if receipt_kind is ReceiptKind.MUTATION:
            kind = event.get("kind")
            pod_with_revision = {"CREATE_POD", "EDIT_POD"}
            pod_without_revision = {
                "REVOKE_POD", "RESTORE_POD", "ROLLBACK_POD", "DELETE_POD"
            }
            alias_operations = {
                "CREATE_ALIAS", "RELINK_ALIAS", "REVOKE_ALIAS",
                "RESTORE_ALIAS", "DELETE_ALIAS",
            }
            common = {
                "operation", "kind", "head", "idempotency_key",
                "command_fingerprint", "command_body",
            }
            if operation in pod_with_revision:
                expected = common | {"new_revision"}
                expected_kind = "pod"
            elif operation in pod_without_revision:
                expected = common
                expected_kind = "pod"
            elif operation in alias_operations:
                expected = common
                expected_kind = "alias"
            elif operation == "SET_POLICY_SCOPE":
                expected = common | {"pod_heads"}
                expected_kind = "policy"
            else:
                return False
            return bool(
                keys == expected
                and kind == expected_kind
                and isinstance(event.get("head"), dict)
                and isinstance(event.get("command_body"), dict)
                and type(event.get("idempotency_key")) is str
                and bool(event.get("idempotency_key"))
                and LifecycleKernel._digest_valid(event.get("command_fingerprint"))
            )
        schemas: dict[ReceiptKind, tuple[str, set[str]]] = {
            ReceiptKind.RESOLUTION: (
                "ROUTE",
                {
                    "operation", "alias_id", "decision", "reason", "policy_generation",
                    "witness", "dependency",
                },
            ),
            ReceiptKind.DERIVATION: (
                "DERIVE_ARTIFACT",
                {"operation", "artifact_id", "payload_hex", "lineage"},
            ),
            ReceiptKind.CAPABILITY_ISSUANCE: (
                "ISSUE_CAPABILITY",
                {
                    "operation", "principal", "nonce", "witness_hash",
                    "witness", "capability_commitment", "security_time", "expires_at",
                },
            ),
            ReceiptKind.PUBLICATION: (
                "PUBLISH",
                {
                    "operation", "publication_id", "artifact_id",
                    "payload_hash", "lineage_hash",
                },
            ),
            ReceiptKind.VERIFIER_RUN: (
                "VERIFIER_RUN",
                {
                    "operation", "verifier", "verdict", "positive_control_passed",
                    "inputs",
                },
            ),
        }
        if receipt_kind is ReceiptKind.CONSUMPTION:
            base = {
                "operation", "principal", "nonce", "witness_hash",
                "capability_commitment", "artifact_id", "security_time",
            }
            if operation == "CONSUME":
                expected_keys = base
            elif operation == "CONSUME_TO_ARTIFACT":
                expected_keys = base | {"payload_hex", "lineage"}
            else:
                return False
            expected_operation = operation
        else:
            expected_operation, expected_keys = schemas[receipt_kind]
        if operation != expected_operation or keys != expected_keys:
            return False
        nonempty_string = lambda value: type(value) is str and bool(value)
        finite_number = LifecycleKernel._finite_numeric
        if receipt_kind is ReceiptKind.RESOLUTION:
            decision = event.get("decision")
            return bool(
                nonempty_string(event.get("alias_id"))
                and type(decision) is str
                and decision in {member.value for member in RouteKind}
                and (
                    event.get("reason") is None
                    or nonempty_string(event.get("reason"))
                )
                and LifecycleKernel._positive_int(event.get("policy_generation"))
                and (
                    event.get("witness") is None
                    or isinstance(event.get("witness"), dict)
                )
                and isinstance(event.get("dependency"), dict)
            )
        if receipt_kind is ReceiptKind.CAPABILITY_ISSUANCE:
            issued_at = event.get("security_time")
            expires_at = event.get("expires_at")
            return bool(
                nonempty_string(event.get("principal"))
                and LifecycleKernel._random_id_valid(event.get("nonce"))
                and LifecycleKernel._digest_valid(event.get("witness_hash"))
                and isinstance(event.get("witness"), dict)
                and LifecycleKernel._digest_valid(event.get("capability_commitment"))
                and finite_number(issued_at)
                and finite_number(expires_at)
                and expires_at > issued_at  # type: ignore[operator]
            )
        if receipt_kind is ReceiptKind.DERIVATION:
            return bool(
                nonempty_string(event.get("artifact_id"))
                and type(event.get("payload_hex")) is str
                and isinstance(event.get("lineage"), list)
            )
        if receipt_kind is ReceiptKind.CONSUMPTION:
            artifact_id = event.get("artifact_id")
            valid = bool(
                nonempty_string(event.get("principal"))
                and LifecycleKernel._random_id_valid(event.get("nonce"))
                and LifecycleKernel._digest_valid(event.get("witness_hash"))
                and LifecycleKernel._digest_valid(event.get("capability_commitment"))
                and (artifact_id is None or nonempty_string(artifact_id))
                and finite_number(event.get("security_time"))
            )
            if operation == "CONSUME":
                return valid and artifact_id is None
            return bool(
                valid
                and nonempty_string(artifact_id)
                and type(event.get("payload_hex")) is str
                and isinstance(event.get("lineage"), list)
            )
        if receipt_kind is ReceiptKind.PUBLICATION:
            return bool(
                LifecycleKernel._random_id_valid(event.get("publication_id"))
                and nonempty_string(event.get("artifact_id"))
                and LifecycleKernel._digest_valid(event.get("payload_hash"))
                and LifecycleKernel._digest_valid(event.get("lineage_hash"))
            )
        verdict = event.get("verdict")
        inputs = event.get("inputs")
        return bool(
            nonempty_string(event.get("verifier"))
            and type(verdict) is str
            and verdict in {member.value for member in VerifierVerdict}
            and isinstance(event.get("positive_control_passed"), bool)
            and type(inputs) is list
            and bool(inputs)
            and all(
                type(item) is dict
                and set(item) == {
                    "verifier", "verdict", "positive_control_passed"
                }
                and nonempty_string(item.get("verifier"))
                and type(item.get("verdict")) is str
                and item.get("verdict") in {
                    member.value for member in VerifierVerdict
                }
                and type(item.get("positive_control_passed")) is bool
                for item in inputs
            )
        )

    @classmethod
    def verify_audit(
        cls,
        records: Sequence[AuditRecord],
        *,
        audit_secret: bytes,
        namespace: str,
        expected_terminal_hash: str,
        expected_length: int,
    ) -> bool:
        if (
            type(audit_secret) is not bytes
            or len(audit_secret) < 32
            or type(namespace) is not str
            or not namespace
            or not cls._digest_valid(expected_terminal_hash)
            or type(expected_length) is not int
            or expected_length < 0
        ):
            return False
        try:
            records = cls._snapshot_audit_records(records)
        except (TypeError, ValueError):
            return False
        if len(records) != expected_length:
            return False
        actual_terminal = records[-1].record_hash if records else _GENESIS_HASH
        if not cls._digest_valid(actual_terminal):
            return False
        if not hmac.compare_digest(actual_terminal, expected_terminal_hash):
            return False
        previous_hash = _GENESIS_HASH
        previous_time = -math.inf
        key_id = records[0].key_id if records else None
        engine_id = records[0].engine_id if records else None
        code_version = records[0].code_version if records else None
        audit_max_hops = records[0].max_hops if records else None
        for expected_sequence, record in enumerate(records, start=1):
            if (
                type(record.sequence) is not int
                or record.sequence != expected_sequence
                or not cls._digest_valid(record.previous_hash)
                or record.previous_hash != previous_hash
                or type(record.namespace) is not str
                or record.namespace != namespace
                or type(record.receipt_kind) is not ReceiptKind
                or not cls._max_hops_valid(record.max_hops)
                or record.max_hops != audit_max_hops
                or type(record.event_json) is not str
                or type(record.engine_id) is not str
                or not record.engine_id
                or record.engine_id != engine_id
                or type(record.code_version) is not str
                or not record.code_version
                or record.code_version != code_version
                or type(record.key_id) is not str
                or not record.key_id
                or record.key_id != key_id
                or not cls._finite_numeric(record.recorded_at)
                or record.recorded_at < previous_time
                or not cls._digest_valid(record.record_hash)
                or not cls._digest_valid(record.mac)
            ):
                return False
            try:
                material = _audit_material(
                    sequence=record.sequence, previous_hash=record.previous_hash,
                    event_json=record.event_json, namespace=record.namespace,
                    receipt_kind=record.receipt_kind, max_hops=record.max_hops,
                    recorded_at=record.recorded_at,
                    engine_id=record.engine_id, code_version=record.code_version,
                    key_id=record.key_id,
                )
                record_hash = hashlib.sha256(_AUDIT_HASH_DOMAIN + material).hexdigest()
                expected_mac = hmac.new(
                    audit_secret,
                    _AUDIT_MAC_DOMAIN + record_hash.encode("ascii"),
                    hashlib.sha256,
                ).hexdigest()
            except Exception:
                return False
            if not hmac.compare_digest(record.record_hash, record_hash) or not hmac.compare_digest(
                record.mac, expected_mac
            ):
                return False
            try:
                event = json.loads(record.event_json)
                valid_event = bool(
                    type(event) is dict
                    and cls._receipt_event_schema_valid(record.receipt_kind, event)
                    and record.event_json == _canonical_json(event)
                )
            except Exception:
                return False
            if not valid_event:
                return False
            previous_hash = record.record_hash
            previous_time = record.recorded_at
        return True

    @classmethod
    def replay(
        cls,
        records: Sequence[AuditRecord],
        *,
        namespace: str,
        audit_secret: bytes,
        capability_secret: bytes,
        expected_terminal_hash: str,
        expected_length: int,
        max_hops: int = 8,
        allow_time_override: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> "LifecycleKernel":
        try:
            return cls._replay_impl(
                records,
                namespace=namespace,
                audit_secret=audit_secret,
                capability_secret=capability_secret,
                expected_terminal_hash=expected_terminal_hash,
                expected_length=expected_length,
                max_hops=max_hops,
                allow_time_override=allow_time_override,
                clock=clock,
            )
        except AuditIntegrityError:
            raise
        except Exception as exc:
            raise AuditIntegrityError(
                "audit replay failed closed during semantic validation"
            ) from exc

    @classmethod
    def _replay_impl(
        cls,
        records: Sequence[AuditRecord],
        *,
        namespace: str,
        audit_secret: bytes,
        capability_secret: bytes,
        expected_terminal_hash: str,
        expected_length: int,
        max_hops: int = 8,
        allow_time_override: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> "LifecycleKernel":
        try:
            records = cls._snapshot_audit_records(records)
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("audit records could not be snapshotted") from exc
        if not cls.verify_audit(
            records, audit_secret=audit_secret, namespace=namespace,
            expected_terminal_hash=expected_terminal_hash,
            expected_length=expected_length,
        ):
            raise AuditIntegrityError("audit records failed authenticated checkpoint verification")
        metadata = records[0] if records else None
        if metadata is not None and metadata.max_hops != max_hops:
            raise AuditIntegrityError("replay max_hops configuration differs from audit")
        kernel = cls(
            namespace=namespace, max_hops=max_hops,
            audit_secret=audit_secret, capability_secret=capability_secret,
            allow_time_override=allow_time_override,
            clock=clock,
            engine_id=metadata.engine_id if metadata else "finx-lifecycle-kernel",
            code_version=metadata.code_version if metadata else "reference-v3",
            key_id=metadata.key_id if metadata else "local-v1",
        )
        with kernel._lock:
            for record in records:
                event = json.loads(record.event_json)
                if record.receipt_kind is ReceiptKind.RESOLUTION:
                    kernel._replay_resolution(event)
                    continue
                if record.receipt_kind is ReceiptKind.DERIVATION:
                    kernel._replay_artifact(event)
                    continue
                if record.receipt_kind is ReceiptKind.CAPABILITY_ISSUANCE:
                    nonce = event.get("nonce")
                    security_time = event.get("security_time")
                    expires_at = event.get("expires_at")
                    if not isinstance(nonce, str) or not nonce:
                        raise AuditIntegrityError("capability issuance lacks a nonce")
                    if nonce in kernel._issued_capabilities:
                        raise AuditIntegrityError("duplicate capability issuance nonce in audit")
                    if (
                        not cls._finite_numeric(security_time)
                        or not cls._finite_numeric(expires_at)
                        or expires_at <= security_time
                    ):
                        raise AuditIntegrityError("capability issuance has invalid security time")
                    if security_time < kernel._last_security_time:
                        raise AuditIntegrityError(
                            "capability security time moved backwards during replay"
                        )
                    principal = event.get("principal")
                    witness_hash = event.get("witness_hash")
                    if not isinstance(principal, str) or not isinstance(witness_hash, str):
                        raise AuditIntegrityError("capability issuance binding is invalid")
                    witness = kernel._witness_from_json(event.get("witness"))  # type: ignore[arg-type]
                    if (
                        hashlib.sha256(
                            _canonical_json(kernel._witness_json(witness)).encode("utf-8")
                        ).hexdigest()
                        != witness_hash
                        or not kernel._validate_witness_locked(witness)
                    ):
                        raise AuditIntegrityError(
                            "capability issuance contains a stale or mismatched witness"
                        )
                    capability_commitment = event.get("capability_commitment")
                    expected_capability_mac = kernel._capability_mac(
                        witness, principal, namespace, nonce,
                        float(security_time), float(expires_at),
                    )
                    if not isinstance(capability_commitment, str) or not hmac.compare_digest(
                        capability_commitment,
                        kernel._capability_commitment(expected_capability_mac),
                    ):
                        raise AuditIntegrityError(
                            "capability issuance MAC does not match its full binding"
                        )
                    kernel._issued_capabilities[nonce] = (
                        principal, witness, float(security_time), float(expires_at),
                        expected_capability_mac,
                    )
                    kernel._last_security_time = max(
                        kernel._last_security_time, float(security_time)
                    )
                    continue
                if record.receipt_kind is ReceiptKind.CONSUMPTION:
                    nonce = event.get("nonce")
                    if not isinstance(nonce, str) or not nonce:
                        raise AuditIntegrityError("consumption receipt lacks a nonce")
                    issuance = kernel._issued_capabilities.get(nonce)
                    if issuance is None:
                        raise AuditIntegrityError("consumption receipt lacks authenticated issuance")
                    if nonce in kernel._spent_nonces:
                        raise AuditIntegrityError("duplicate consumption nonce in audit")
                    security_time = event.get("security_time")
                    if not cls._finite_numeric(security_time):
                        raise AuditIntegrityError("consumption receipt has invalid security time")
                    if security_time < kernel._last_security_time:
                        raise AuditIntegrityError(
                            "consumption security time moved backwards during replay"
                        )
                    if (
                        event.get("principal") != issuance[0]
                        or event.get("witness_hash")
                        != hashlib.sha256(
                            _canonical_json(
                                kernel._witness_json(issuance[1])
                            ).encode("utf-8")
                        ).hexdigest()
                        or security_time < issuance[2]
                        or security_time >= issuance[3]
                        or event.get("capability_commitment")
                        != kernel._capability_commitment(issuance[4])
                        or not kernel._validate_witness_locked(issuance[1])
                    ):
                        raise AuditIntegrityError("consumption receipt violates issuance binding")
                    kernel._spent_nonces.add(nonce)
                    kernel._last_security_time = max(
                        kernel._last_security_time, float(security_time)
                    )
                    artifact_id = event.get("artifact_id")
                    if artifact_id is not None:
                        artifact = kernel._replay_artifact(event)
                        if (
                            len(artifact.lineage.dependencies) != 1
                            or not isinstance(
                                artifact.lineage.dependencies[0], PathWitness
                            )
                            or hashlib.sha256(
                                _canonical_json(
                                    kernel._witness_json(
                                        artifact.lineage.dependencies[0]
                                    )
                                ).encode("utf-8")
                            ).hexdigest()
                            != event.get("witness_hash")
                            or not kernel._validate_witness_locked(
                                artifact.lineage.dependencies[0]
                            )
                        ):
                            raise AuditIntegrityError(
                                "consumed artifact lineage violates capability witness"
                            )
                        witness = artifact.lineage.dependencies[0]
                        authoritative_payload = kernel._payloads[witness.pod_id][
                            witness.payload_revision
                        ]
                        if artifact.payload != authoritative_payload:
                            raise AuditIntegrityError(
                                "consumed artifact payload violates capability witness"
                            )
                    continue
                if record.receipt_kind is ReceiptKind.PUBLICATION:
                    publication_id = event.get("publication_id")
                    if not isinstance(publication_id, str) or not publication_id:
                        raise AuditIntegrityError("publication receipt lacks an ID")
                    if publication_id in kernel._published:
                        raise AuditIntegrityError("duplicate publication receipt")
                    artifact_id = event.get("artifact_id")
                    if not isinstance(artifact_id, str):
                        raise AuditIntegrityError("publication receipt lacks an artifact")
                    artifact = kernel._artifacts.get(artifact_id)
                    if artifact is None:
                        raise AuditIntegrityError("publication references a missing artifact")
                    if artifact_id in kernel._published_artifacts:
                        raise AuditIntegrityError("artifact was already published")
                    if not kernel._artifact_current_locked(artifact):
                        raise AuditIntegrityError("publication references a stale artifact")
                    if (
                        event.get("payload_hash") != _payload_hash(artifact.payload)
                        or event.get("lineage_hash")
                        != kernel._lineage_hash(artifact.lineage)
                    ):
                        raise AuditIntegrityError("publication artifact hashes do not match")
                    kernel._published.add(publication_id)
                    kernel._published_artifacts.add(artifact_id)
                    kernel._publication_bindings[publication_id] = artifact_id
                    continue
                if record.receipt_kind is ReceiptKind.VERIFIER_RUN:
                    kernel._replay_verifier_run(event)
                    continue
                if record.receipt_kind is not ReceiptKind.MUTATION:
                    continue
                operation = event.get("operation")
                command_body = event.get("command_body")
                fingerprint = event.get("command_fingerprint")
                if (
                    not isinstance(operation, str)
                    or not isinstance(command_body, dict)
                    or not isinstance(fingerprint, str)
                ):
                    raise AuditIntegrityError("mutation has invalid command fingerprint metadata")
                kernel._validate_replay_command_body(event, command_body)
                expected_fingerprint = hashlib.sha256(
                    _canonical_json(
                        {"operation": operation, "body": command_body}
                    ).encode("utf-8")
                ).hexdigest()
                if not hmac.compare_digest(fingerprint, expected_fingerprint):
                    raise AuditIntegrityError(
                        "illegal mutation: command fingerprint does not match body"
                    )
                result = kernel._replay_mutation(event)
                key = event.get("idempotency_key")
                if not isinstance(key, str) or not key or not isinstance(fingerprint, str):
                    raise AuditIntegrityError("mutation has invalid idempotency metadata")
                if key in kernel._idempotency:
                    raise AuditIntegrityError("duplicate idempotency key in audit")
                kernel._idempotency[key] = (fingerprint, result)
            kernel._audit = list(records)
            kernel._last_audit_time = records[-1].recorded_at if records else -math.inf
        return kernel

    def _replay_resolution(self, event: Mapping[str, object]) -> None:
        try:
            alias_id = self._object_id(event.get("alias_id"), "alias_id")  # type: ignore[arg-type]
            raw_witness = event.get("witness")
            if raw_witness is not None:
                if type(raw_witness) is not dict:
                    raise TypeError("resolution witness must be an object")
                self._witness_from_json(raw_witness)
            lineage = self._lineage_from_json([event.get("dependency")])
            if len(lineage.dependencies) != 1:
                raise TypeError("resolution dependency must be singular")
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError(
                "resolution receipt has invalid typed witness or dependency fields"
            ) from exc
        reason = event.get("reason")
        if reason == "AUTHORITY_ERROR":
            decision = RouteDecision(
                RouteKind.UNKNOWN,
                dependency=UnresolvedDependency(
                    self.namespace, alias_id, "AUTHORITY_ERROR",
                    self._policy.authority_generation,
                ),
                reason=reason,
            )
            if _canonical_json(dict(event)) != _canonical_json(
                self._resolution_event(alias_id, decision)
            ):
                raise AuditIntegrityError(
                    "authority-error resolution receipt is not tightly fail-closed"
                )
            return
        registered = alias_id in self._aliases or alias_id in self._policy.exact_scope
        decision = self._resolve_locked(alias_id) if registered else RouteDecision(
            RouteKind.BYPASS,
            dependency=OutOfScopeDependency(
                self.namespace, alias_id, self._policy.authority_generation
            ),
            reason="OUT_OF_SCOPE",
        )
        if _canonical_json(dict(event)) != _canonical_json(
            self._resolution_event(alias_id, decision)
        ):
            raise AuditIntegrityError(
                "resolution receipt does not match authoritative route decision or witness"
            )

    def _replay_verifier_run(self, event: Mapping[str, object]) -> None:
        raw_inputs = event.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise AuditIntegrityError("verifier receipt has no typed inputs")
        inputs: list[VerifierResult] = []
        try:
            for raw in raw_inputs:
                if not isinstance(raw, dict) or set(raw) != {
                    "verifier", "verdict", "positive_control_passed"
                }:
                    raise TypeError("invalid verifier input schema")
                verifier = self._object_id(raw["verifier"], "verifier")  # type: ignore[arg-type]
                if not isinstance(raw["verdict"], str) or type(
                    raw["positive_control_passed"]
                ) is not bool:
                    raise TypeError("invalid verifier input types")
                inputs.append(
                    VerifierResult(
                        verifier,
                        VerifierVerdict(raw["verdict"]),
                        raw["positive_control_passed"],
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("verifier receipt contains invalid inputs") from exc
        expected = self.combine_verifier_results(inputs)
        if (
            event.get("verdict") != expected.value
            or event.get("verifier") != inputs[0].verifier
            or event.get("positive_control_passed")
            is not all(item.positive_control_passed for item in inputs)
        ):
            raise AuditIntegrityError("verifier verdict does not match audited inputs")

    def _replay_artifact(self, event: Mapping[str, object]) -> DerivedArtifact:
        try:
            artifact_id = self._object_id(event.get("artifact_id"), "artifact_id")  # type: ignore[arg-type]
            payload_hex = event.get("payload_hex")
            if not isinstance(payload_hex, str):
                raise TypeError("artifact payload must be hexadecimal")
            payload = bytes.fromhex(payload_hex)
            if payload.hex() != payload_hex:
                raise TypeError("artifact payload hexadecimal is not canonical")
            lineage = self._lineage_from_json(event.get("lineage"))
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid artifact state in audit") from exc
        if artifact_id in self._artifacts:
            raise AuditIntegrityError("duplicate artifact identifier in audit")
        artifact = DerivedArtifact(artifact_id, payload, lineage)
        if not self._artifact_current_locked(artifact):
            raise AuditIntegrityError("artifact lineage is not current at derivation event")
        self._artifacts[artifact_id] = artifact
        return artifact

    def _validate_replay_command_body(
        self, event: Mapping[str, object], body: Mapping[str, object]
    ) -> None:
        """Validate the audited command domain before deriving its replay fingerprint."""

        operation = event.get("operation")
        head = event.get("head")
        if not isinstance(operation, str) or not isinstance(head, dict):
            raise AuditIntegrityError("mutation command has no typed operation or head")
        try:
            if event.get("kind") == "pod":
                object_id = self._object_id(body.get("pod_id"), "pod_id")  # type: ignore[arg-type]
                if head.get("pod_id") != object_id:
                    raise AuditIntegrityError("command fingerprint body targets another pod head")
                base_keys = {"pod_id", "expected_generation"}
                extra: set[str]
                if operation in {"CREATE_POD", "EDIT_POD"}:
                    extra = {"payload_hex"}
                    raw_revision = event.get("new_revision")
                    if not isinstance(raw_revision, dict):
                        raise AuditIntegrityError("command fingerprint lacks payload revision")
                    payload_hex = body.get("payload_hex")
                    if not isinstance(payload_hex, str) or raw_revision.get(
                        "payload_hex"
                    ) != payload_hex:
                        raise AuditIntegrityError("command fingerprint payload body disagrees")
                    bytes.fromhex(payload_hex)
                elif operation == "RESTORE_POD":
                    extra = {"payload_revision"}
                    selection = body.get("payload_revision")
                    if selection is not None and not self._positive_int(selection):
                        raise AuditIntegrityError("invalid restore payload revision in command body")
                    previous = self._pods.get(object_id)
                    expected_revision = (
                        previous.payload_revision if selection is None and previous else selection
                    )
                    if head.get("payload_revision") != expected_revision:
                        raise AuditIntegrityError("command fingerprint restore revision disagrees")
                elif operation == "ROLLBACK_POD":
                    extra = {"to_payload_revision"}
                    selection = body.get("to_payload_revision")
                    if not self._positive_int(selection) or head.get(
                        "payload_revision"
                    ) != selection:
                        raise AuditIntegrityError("command fingerprint rollback revision disagrees")
                else:
                    extra = set()
                if set(body) != base_keys | extra:
                    raise AuditIntegrityError("command fingerprint pod body schema is invalid")
            elif event.get("kind") == "alias":
                object_id = self._object_id(body.get("alias_id"), "alias_id")  # type: ignore[arg-type]
                if head.get("alias_id") != object_id:
                    raise AuditIntegrityError("command fingerprint body targets another alias head")
                expected_keys = {"alias_id", "expected_generation"}
                if operation in {"CREATE_ALIAS", "RELINK_ALIAS"}:
                    expected_keys.add("target")
                    if body.get("target") != head.get("target"):
                        raise AuditIntegrityError("command fingerprint alias target disagrees")
                if set(body) != expected_keys:
                    raise AuditIntegrityError("command fingerprint alias body schema is invalid")
            elif event.get("kind") == "policy":
                if set(body) != {"exact_scope", "expected_generation"}:
                    raise AuditIntegrityError("command fingerprint policy body schema is invalid")
                raw_scope = body.get("exact_scope")
                if (
                    not isinstance(raw_scope, list)
                    or len(raw_scope) != len(set(raw_scope))
                    or any(not isinstance(item, str) or not item for item in raw_scope)
                    or head.get("exact_scope") != raw_scope
                ):
                    raise AuditIntegrityError("command fingerprint policy scope disagrees")
            else:
                raise AuditIntegrityError("command fingerprint has unknown mutation domain")
            expected_generation = body.get("expected_generation")
            if operation in {"CREATE_POD", "CREATE_ALIAS"}:
                generation_valid = type(expected_generation) is int and expected_generation == 0
            else:
                generation_valid = self._positive_int(expected_generation)
            if not generation_valid:
                raise AuditIntegrityError("command fingerprint expected generation is invalid")
            authority_generation = head.get("authority_generation")
            if (
                not self._positive_int(authority_generation)
                or authority_generation != expected_generation + 1  # type: ignore[operator]
            ):
                raise AuditIntegrityError("command fingerprint generation disagrees with head")
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid object identifier or command body in audit") from exc

    def _command(
        self,
        operation: str,
        idempotency_key: str,
        body: Mapping[str, object],
        apply: Callable[[], tuple[_Result, dict[str, object]]],
    ) -> _Result:
        self._require_not_publishing()
        if type(idempotency_key) is not str or not idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        command_json = _canonical_json({"operation": operation, "body": dict(body)})
        fingerprint = hashlib.sha256(command_json.encode("utf-8")).hexdigest()
        with self._lock:
            prior = self._idempotency.get(idempotency_key)
            if prior is not None:
                prior_fingerprint, prior_result = prior
                if not hmac.compare_digest(fingerprint, prior_fingerprint):
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was used for another command"
                    )
                return prior_result  # type: ignore[return-value]
            snapshot = self._mutation_snapshot()
            try:
                result, event = apply()
                event["idempotency_key"] = idempotency_key
                event["command_fingerprint"] = fingerprint
                event["command_body"] = dict(body)
                self._append_audit(event, ReceiptKind.MUTATION)
                self._idempotency[idempotency_key] = (fingerprint, result)
                return result
            except BaseException:
                self._restore_mutation_snapshot(snapshot)
                raise

    def _change_pod_status(
        self,
        operation: str,
        pod_id: str,
        *,
        expected_generation: int,
        idempotency_key: str,
        expected_status: AuthorityStatus,
        new_status: AuthorityStatus,
    ) -> PodHead:
        pod_id = self._object_id(pod_id, "pod_id")
        self._require_positive_generation(expected_generation)
        body = {"pod_id": pod_id, "expected_generation": expected_generation}

        def apply() -> tuple[PodHead, dict[str, object]]:
            current = self._pod_for_cas(pod_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"pod {pod_id!r} is permanently deleted")
            if current.status is not expected_status:
                raise PermissionError(
                    f"pod {pod_id!r} must be {expected_status.value} for {operation}"
                )
            head = replace(
                current, authority_generation=current.authority_generation + 1,
                status=new_status, policy_generation=self._policy.authority_generation,
            )
            self._pods[pod_id] = head
            return head, self._pod_event(operation, head)

        return self._command(operation, idempotency_key, body, apply)

    def _change_alias_status(
        self,
        operation: str,
        alias_id: str,
        *,
        expected_generation: int,
        idempotency_key: str,
        expected_status: AuthorityStatus,
        new_status: AuthorityStatus,
    ) -> AliasHead:
        alias_id = self._object_id(alias_id, "alias_id")
        self._require_positive_generation(expected_generation)
        body = {"alias_id": alias_id, "expected_generation": expected_generation}

        def apply() -> tuple[AliasHead, dict[str, object]]:
            current = self._alias_for_cas(alias_id, expected_generation)
            if current.status is AuthorityStatus.DELETED:
                raise PermissionError(f"alias {alias_id!r} is permanently deleted")
            if current.status is not expected_status:
                raise PermissionError(
                    f"alias {alias_id!r} must be {expected_status.value} for {operation}"
                )
            head = replace(
                current, authority_generation=current.authority_generation + 1,
                status=new_status,
            )
            self._aliases[alias_id] = head
            return head, self._alias_event(operation, head)

        return self._command(operation, idempotency_key, body, apply)

    def _validate_target_path_locked(self, owner_alias_id: str, target: AliasTarget) -> None:
        if target.kind is TargetKind.POD:
            pod = self._pods.get(target.object_id)
            if pod is None:
                raise KeyError(f"unknown target pod {target.object_id!r}")
            if pod.status is not AuthorityStatus.ACTIVE:
                raise PermissionError(f"target pod {target.object_id!r} is not active")
            return
        if target.object_id == owner_alias_id:
            raise ValueError("alias target would create a self-cycle")
        alias = self._aliases.get(target.object_id)
        if alias is None:
            raise KeyError(f"unknown target alias {target.object_id!r}")
        if alias.status is not AuthorityStatus.ACTIVE:
            raise PermissionError(f"target alias {target.object_id!r} is not active")
        resolution = self._resolve_locked(target.object_id)
        if resolution.kind is not RouteKind.RESOLVE or resolution.witness is None:
            raise ValueError(f"target alias path is not resolvable: {resolution.reason}")
        path_ids = [node.alias_id for node in resolution.witness.alias_path]
        if owner_alias_id in path_ids:
            raise ValueError("alias target would create an indirect cycle")
        if len(path_ids) + 1 > self.max_hops:
            raise ValueError("alias target path exceeds max_hops")

    def _validate_public_capability_fields(
        self,
        capability: ConsumeCapability,
        principal: str,
        namespace: str,
    ) -> None:
        if type(principal) is not str or not principal:
            raise CapabilityError("principal must be a non-empty string")
        if type(namespace) is not str or namespace != self.namespace:
            raise CapabilityError("namespace mismatch")
        if type(capability) is not ConsumeCapability:
            raise CapabilityError("capability has an invalid type")
        if (
            type(capability.principal) is not str
            or not capability.principal
            or type(capability.namespace) is not str
            or capability.namespace != self.namespace
            or not self._random_id_valid(capability.nonce)
            or not self._digest_valid(capability.mac)
            or not self._witness_well_formed(capability.witness)
        ):
            raise CapabilityError("capability fields are not canonical")
        if not self._finite_numeric(capability.issued_at) or not self._finite_numeric(
            capability.expires_at
        ):
            raise CapabilityError("capability times must be finite numeric values")
        if capability.expires_at <= capability.issued_at:
            raise CapabilityError("capability time interval is invalid")

    def _validate_capability_locked(
        self,
        capability: ConsumeCapability,
        principal: str,
        namespace: str,
        checked_at: float,
    ) -> bytes:
        self._validate_public_capability_fields(capability, principal, namespace)
        if capability.principal != principal or capability.namespace != namespace:
            raise CapabilityError("capability principal or namespace mismatch")
        if namespace != self.namespace:
            raise CapabilityError("authority namespace mismatch")
        expected_mac = self._capability_mac(
            capability.witness, capability.principal, capability.namespace,
            capability.nonce, capability.issued_at, capability.expires_at,
        )
        if not hmac.compare_digest(capability.mac, expected_mac):
            raise CapabilityError("capability authentication failed")
        issuance = self._issued_capabilities.get(capability.nonce)
        if issuance != (
            capability.principal, capability.witness,
            capability.issued_at, capability.expires_at, capability.mac,
        ):
            raise CapabilityError("capability issuance is absent from authenticated authority")
        if capability.nonce in self._spent_nonces:
            raise CapabilityError("capability was already consumed")
        if checked_at < capability.issued_at:
            raise CapabilityError("capability cannot be consumed before issuance")
        if checked_at >= capability.expires_at:
            raise CapabilityError("capability expired")
        if not self._validate_witness_locked(capability.witness):
            raise CapabilityError("capability witness is stale")
        pod = self._pods[capability.witness.pod_id]
        return self._payloads[pod.pod_id][pod.payload_revision]

    def _artifact_current_locked(self, artifact: DerivedArtifact) -> bool:
        return all(
            self._dependency_current_locked(dependency)
            for dependency in artifact.lineage.dependencies
        )

    def _dependency_current_locked(self, dependency: LineageDependency) -> bool:
        if not self._dependency_well_formed(dependency):
            return False
        if type(dependency) is PathWitness:
            return self._validate_witness_locked(dependency)
        if type(dependency) is MissingAliasDependency:
            return bool(
                dependency.namespace == self.namespace
                and dependency.policy_generation == self._policy.authority_generation
                and dependency.alias_id not in self._aliases
                and dependency.alias_id in self._policy.exact_scope
            )
        if type(dependency) is OutOfScopeDependency:
            return bool(
                dependency.namespace == self.namespace
                and dependency.policy_generation == self._policy.authority_generation
                and dependency.alias_id not in self._aliases
                and dependency.alias_id not in self._policy.exact_scope
            )
        return False

    def _dependency_well_formed(self, dependency: object) -> bool:
        if type(dependency) is PathWitness:
            return self._witness_well_formed(dependency)
        if type(dependency) not in {
            MissingAliasDependency, OutOfScopeDependency, UnresolvedDependency
        }:
            return False
        if (
            type(dependency.namespace) is not str
            or dependency.namespace != self.namespace
            or type(dependency.alias_id) is not str
            or not dependency.alias_id
            or not self._positive_int(dependency.policy_generation)
        ):
            return False
        return type(dependency) is not UnresolvedDependency or bool(
            type(dependency.reason) is str and dependency.reason
        )

    def _witness_well_formed(self, witness: object) -> bool:
        if type(witness) is not PathWitness:
            return False
        if (
            type(witness.namespace) is not str
            or witness.namespace != self.namespace
            or type(witness.root_alias_id) is not str
            or not witness.root_alias_id
            or type(witness.pod_id) is not str
            or not witness.pod_id
            or type(witness.alias_path) is not tuple
            or not 1 <= len(witness.alias_path) <= self.max_hops
            or not self._positive_int(witness.pod_generation)
            or not self._positive_int(witness.payload_revision)
            or not self._positive_int(witness.policy_generation)
            or not self._digest_valid(witness.payload_hash)
        ):
            return False
        for node in witness.alias_path:
            if (
                type(node) is not AliasPathNode
                or type(node.alias_id) is not str
                or not node.alias_id
                or not self._positive_int(node.authority_generation)
                or type(node.target) is not AliasTarget
                or not isinstance(node.target.kind, TargetKind)
                or type(node.target.object_id) is not str
                or not node.target.object_id
            ):
                return False
        alias_ids = [node.alias_id for node in witness.alias_path]
        return bool(
            witness.root_alias_id == alias_ids[0]
            and len(alias_ids) == len(set(alias_ids))
        )

    def _capability_mac(
        self, witness: PathWitness, principal: str, namespace: str,
        nonce: str, issued_at: float, expires_at: float,
    ) -> str:
        message = _canonical_json(
            {
                "witness": self._witness_json(witness), "principal": principal,
                "namespace": namespace, "nonce": nonce, "issued_at": issued_at,
                "expires_at": expires_at,
            }
        ).encode("utf-8")
        return hmac.new(
            self._capability_secret, _CAPABILITY_DOMAIN + message, hashlib.sha256
        ).hexdigest()

    def _publication_mac(
        self,
        publication_id: str,
        artifact_id: str,
        payload_hash: str,
        lineage_hash: str,
    ) -> str:
        message = _canonical_json(
            {
                "publication_id": publication_id,
                "artifact_id": artifact_id,
                "namespace": self.namespace,
                "payload_hash": payload_hash,
                "lineage_hash": lineage_hash,
            }
        ).encode("utf-8")
        return hmac.new(
            self._capability_secret, _PUBLICATION_DOMAIN + message, hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _capability_commitment(capability_mac: str) -> str:
        return hashlib.sha256(
            _CAPABILITY_COMMITMENT_DOMAIN + capability_mac.encode("ascii")
        ).hexdigest()

    def _append_audit(
        self, event: Mapping[str, object], kind: ReceiptKind = ReceiptKind.MUTATION
    ) -> AuditRecord:
        audit_before = list(self._audit)
        audit_time_before = self._last_audit_time
        try:
            record = self._prepare_audit(event, kind)
            self._commit_audit(record)
            return record
        except BaseException:
            self._audit = audit_before
            self._last_audit_time = audit_time_before
            raise

    def _commit_audit(self, record: AuditRecord) -> None:
        expected_sequence = len(self._audit) + 1
        expected_previous = self._audit[-1].record_hash if self._audit else _GENESIS_HASH
        if record.sequence != expected_sequence or record.previous_hash != expected_previous:
            raise AuditIntegrityError("audit commit sequence changed before append")
        self._audit.append(record)

    def _prepare_audit(self, event: Mapping[str, object], kind: ReceiptKind) -> AuditRecord:
        if not isinstance(kind, ReceiptKind) or not self._receipt_event_schema_valid(kind, event):
            raise AuditIntegrityError("receipt kind and event schema do not match")
        event_json = _canonical_json(event)
        recorded_at = self._audit_now()
        sequence = len(self._audit) + 1
        previous_hash = self._audit[-1].record_hash if self._audit else _GENESIS_HASH
        material = _audit_material(
            sequence=sequence, previous_hash=previous_hash, event_json=event_json,
            namespace=self.namespace, receipt_kind=kind, max_hops=self.max_hops,
            recorded_at=recorded_at,
            engine_id=self._engine_id, code_version=self._code_version, key_id=self._key_id,
        )
        record_hash = hashlib.sha256(_AUDIT_HASH_DOMAIN + material).hexdigest()
        mac = hmac.new(
            self._audit_secret,
            _AUDIT_MAC_DOMAIN + record_hash.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return AuditRecord(
            sequence, previous_hash, event_json, record_hash, mac, self.namespace, kind,
            self.max_hops, recorded_at, self._engine_id, self._code_version, self._key_id,
        )

    def _audit_now(self) -> float:
        if getattr(self._thread_state, "sampling_clock", False):
            raise AuditIntegrityError("audit clock must not re-enter the kernel")
        self._thread_state.sampling_clock = True
        try:
            raw_value = self._clock()
        finally:
            self._thread_state.sampling_clock = False
        if not self._finite_numeric(raw_value):
            raise AuditIntegrityError("audit clock must return finite numeric time")
        value = float(raw_value)
        if value < self._last_audit_time:
            raise AuditIntegrityError("audit clock must be non-decreasing")
        self._last_audit_time = value
        return value

    def _security_now(self, explicit: float | None) -> float:
        if explicit is not None and not self._allow_time_override:
            raise CapabilityError("explicit now is disabled; inject a trusted clock")
        if explicit is None:
            if getattr(self._thread_state, "sampling_clock", False):
                raise CapabilityError("security clock must not re-enter the kernel")
            self._thread_state.sampling_clock = True
            try:
                raw_value = self._clock()
            finally:
                self._thread_state.sampling_clock = False
        else:
            raw_value = explicit
        if not self._finite_numeric(raw_value):
            raise CapabilityError("security time must be finite numeric")
        value = float(raw_value)
        if value < self._last_security_time:
            raise CapabilityError("security time moved backwards")
        self._last_security_time = value
        return value

    def _mutation_snapshot(self) -> tuple[object, ...]:
        return (
            dict(self._pods),
            {pod_id: dict(revisions) for pod_id, revisions in self._payloads.items()},
            dict(self._aliases), dict(self._identities), self._policy,
            dict(self._idempotency), list(self._audit), self._last_audit_time,
        )

    def _restore_mutation_snapshot(self, snapshot: tuple[object, ...]) -> None:
        (
            self._pods, self._payloads, self._aliases, self._identities, self._policy,
            self._idempotency, self._audit, self._last_audit_time,
        ) = snapshot  # type: ignore[assignment]

    def _replay_mutation(self, event: Mapping[str, object]) -> _CommandResult:
        kind = event.get("kind")
        if kind == "pod":
            raw_head = event.get("head")
            if not isinstance(raw_head, dict):
                raise AuditIntegrityError("pod event lacks a head")
            head = self._pod_head_from_json(raw_head)
            previous = self._pods.get(head.pod_id)
            self._check_replay_generation(previous, head)
            operation = event.get("operation")
            if not isinstance(operation, str):
                raise AuditIntegrityError("pod event lacks a typed operation")
            new_revision = event.get("new_revision")
            self._validate_pod_replay_transition(
                operation, previous, head, new_revision is not None
            )
            if head.policy_generation != self._policy.authority_generation:
                raise AuditIntegrityError("pod head is bound to the wrong policy generation")
            self._claim_replay_identity(head.pod_id, TargetKind.POD)
            if new_revision is not None:
                if not isinstance(new_revision, dict) or set(new_revision) != {
                    "payload_revision", "payload_hex"
                }:
                    raise AuditIntegrityError("invalid payload revision in audit")
                revision = new_revision["payload_revision"]
                payload_hex = new_revision["payload_hex"]
                if not self._positive_int(revision) or not isinstance(payload_hex, str):
                    raise AuditIntegrityError("invalid payload revision in audit")
                try:
                    payload = bytes.fromhex(payload_hex)
                except ValueError as exc:
                    raise AuditIntegrityError("invalid payload revision in audit") from exc
                if payload.hex() != payload_hex:
                    raise AuditIntegrityError("payload revision encoding is not canonical")
                if revision != head.payload_revision or _payload_hash(payload) != head.payload_hash:
                    raise AuditIntegrityError("payload revision does not match replayed head")
                revisions = self._payloads.setdefault(head.pod_id, {})
                if revision in revisions:
                    raise AuditIntegrityError("immutable payload revision was overwritten")
                revisions[revision] = payload
            if head.pod_id not in self._payloads or head.payload_revision not in self._payloads[head.pod_id]:
                raise AuditIntegrityError("pod head references an absent payload revision")
            if _payload_hash(self._payloads[head.pod_id][head.payload_revision]) != head.payload_hash:
                raise AuditIntegrityError("pod head hash does not match its immutable revision")
            self._pods[head.pod_id] = head
            return head
        if kind == "alias":
            raw_head = event.get("head")
            if not isinstance(raw_head, dict):
                raise AuditIntegrityError("alias event lacks a head")
            head = self._alias_head_from_json(raw_head)
            previous = self._aliases.get(head.alias_id)
            self._check_replay_generation(previous, head)
            self._claim_replay_identity(head.alias_id, TargetKind.ALIAS)
            operation = event.get("operation")
            if not isinstance(operation, str):
                raise AuditIntegrityError("alias event lacks a typed operation")
            self._validate_alias_replay_transition(operation, previous, head)
            if head.status is AuthorityStatus.ACTIVE and operation in {
                "CREATE_ALIAS",
                "RELINK_ALIAS",
                "RESTORE_ALIAS",
            }:
                self._validate_target_path_locked(head.alias_id, head.target)
            self._aliases[head.alias_id] = head
            return head
        if kind == "policy":
            if event.get("operation") != "SET_POLICY_SCOPE":
                raise AuditIntegrityError("illegal policy operation")
            raw_head = event.get("head")
            raw_pods = event.get("pod_heads")
            if not isinstance(raw_head, dict) or not isinstance(raw_pods, list):
                raise AuditIntegrityError("policy event is incomplete")
            if len(raw_pods) != len(self._pods) or any(
                not isinstance(value, dict) for value in raw_pods
            ):
                raise AuditIntegrityError("policy event has an invalid pod head list")
            policy = self._policy_head_from_json(raw_head)
            if policy.authority_generation != self._policy.authority_generation + 1:
                raise AuditIntegrityError("policy generation is not monotone")
            parsed_pods = [self._pod_head_from_json(value) for value in raw_pods]
            if len({head.pod_id for head in parsed_pods}) != len(parsed_pods):
                raise AuditIntegrityError("policy event contains duplicate pod heads")
            if [head.pod_id for head in parsed_pods] != sorted(
                head.pod_id for head in parsed_pods
            ):
                raise AuditIntegrityError("policy event pod heads are not canonically ordered")
            replayed_pods = {head.pod_id: head for head in parsed_pods}
            if set(replayed_pods) != set(self._pods):
                raise AuditIntegrityError("policy event does not cover every pod")
            for pod_id, head in replayed_pods.items():
                previous = self._pods[pod_id]
                if (
                    head.authority_generation != previous.authority_generation + 1
                    or head.policy_generation != policy.authority_generation
                    or head.payload_revision != previous.payload_revision
                    or head.payload_hash != previous.payload_hash
                    or head.status is not previous.status
                ):
                    raise AuditIntegrityError("policy event contains an invalid pod transition")
            self._policy = policy
            self._pods = replayed_pods
            return policy
        raise AuditIntegrityError(f"unsupported mutation kind {kind!r}")

    @staticmethod
    def _check_replay_generation(
        previous: PodHead | AliasHead | None, current: PodHead | AliasHead
    ) -> None:
        expected = 1 if previous is None else previous.authority_generation + 1
        if current.authority_generation != expected:
            raise AuditIntegrityError("replayed authority generation is not contiguous")
        if previous is not None and previous.status is AuthorityStatus.DELETED:
            raise AuditIntegrityError("terminal identity was mutated after deletion")

    def _validate_pod_replay_transition(
        self,
        operation: str,
        previous: PodHead | None,
        current: PodHead,
        has_new_revision: bool,
    ) -> None:
        if operation == "CREATE_POD":
            valid = (
                previous is None
                and current.status is AuthorityStatus.ACTIVE
                and current.payload_revision == 1
                and has_new_revision
            )
        elif previous is None:
            valid = False
        elif operation == "EDIT_POD":
            valid = (
                previous.status is AuthorityStatus.ACTIVE
                and current.status is AuthorityStatus.ACTIVE
                and current.payload_revision == max(self._payloads[current.pod_id]) + 1
                and has_new_revision
            )
        elif operation == "REVOKE_POD":
            valid = (
                previous.status is AuthorityStatus.ACTIVE
                and current.status is AuthorityStatus.REVOKED
                and current.payload_revision == previous.payload_revision
                and current.payload_hash == previous.payload_hash
                and not has_new_revision
            )
        elif operation == "RESTORE_POD":
            valid = (
                previous.status is AuthorityStatus.REVOKED
                and current.status is AuthorityStatus.ACTIVE
                and not has_new_revision
            )
        elif operation == "ROLLBACK_POD":
            valid = (
                previous.status is AuthorityStatus.ACTIVE
                and current.status is AuthorityStatus.ACTIVE
                and not has_new_revision
            )
        elif operation == "DELETE_POD":
            valid = (
                previous.status in {AuthorityStatus.ACTIVE, AuthorityStatus.REVOKED}
                and current.status is AuthorityStatus.DELETED
                and current.payload_revision == previous.payload_revision
                and current.payload_hash == previous.payload_hash
                and not has_new_revision
            )
        else:
            valid = False
        if not valid:
            raise AuditIntegrityError(f"illegal pod transition for {operation!r}")

    @staticmethod
    def _validate_alias_replay_transition(
        operation: str, previous: AliasHead | None, current: AliasHead
    ) -> None:
        if operation == "CREATE_ALIAS":
            valid = previous is None and current.status is AuthorityStatus.ACTIVE
        elif previous is None:
            valid = False
        elif operation == "RELINK_ALIAS":
            valid = (
                previous.status is AuthorityStatus.ACTIVE
                and current.status is AuthorityStatus.ACTIVE
            )
        elif operation == "REVOKE_ALIAS":
            valid = (
                previous.status is AuthorityStatus.ACTIVE
                and current.status is AuthorityStatus.REVOKED
                and current.target == previous.target
            )
        elif operation == "RESTORE_ALIAS":
            valid = (
                previous.status is AuthorityStatus.REVOKED
                and current.status is AuthorityStatus.ACTIVE
                and current.target == previous.target
            )
        elif operation == "DELETE_ALIAS":
            valid = (
                previous.status in {AuthorityStatus.ACTIVE, AuthorityStatus.REVOKED}
                and current.status is AuthorityStatus.DELETED
                and current.target == previous.target
            )
        else:
            valid = False
        if not valid:
            raise AuditIntegrityError(f"illegal alias transition for {operation!r}")

    def _claim_replay_identity(self, object_id: str, kind: TargetKind) -> None:
        previous = self._identities.get(object_id)
        if previous is not None and previous is not kind:
            raise AuditIntegrityError("cross-kind identity collision in audit")
        self._identities[object_id] = kind

    def _pod_for_cas(self, pod_id: str, expected_generation: int) -> PodHead:
        self._require_positive_generation(expected_generation)
        try:
            current = self._pods[pod_id]
        except KeyError as exc:
            raise KeyError(f"unknown pod {pod_id!r}") from exc
        if current.authority_generation != expected_generation:
            raise GenerationConflict(
                f"pod {pod_id!r} generation is {current.authority_generation}, "
                f"expected {expected_generation}"
            )
        return current

    def _active_pod_for_mutation(self, pod_id: str, expected_generation: int) -> PodHead:
        current = self._pod_for_cas(pod_id, expected_generation)
        if current.status is not AuthorityStatus.ACTIVE:
            raise PermissionError(f"pod {pod_id!r} is not active")
        return current

    def _alias_for_cas(self, alias_id: str, expected_generation: int) -> AliasHead:
        self._require_positive_generation(expected_generation)
        try:
            current = self._aliases[alias_id]
        except KeyError as exc:
            raise KeyError(f"unknown alias {alias_id!r}") from exc
        if current.authority_generation != expected_generation:
            raise GenerationConflict(
                f"alias {alias_id!r} generation is {current.authority_generation}, "
                f"expected {expected_generation}"
            )
        return current

    def _active_alias_for_mutation(self, alias_id: str, expected_generation: int) -> AliasHead:
        current = self._alias_for_cas(alias_id, expected_generation)
        if current.status is not AuthorityStatus.ACTIVE:
            raise PermissionError(f"alias {alias_id!r} is not active")
        return current

    def _require_unused_identity(self, object_id: str) -> None:
        if object_id in self._identities:
            kind = self._identities[object_id].value.lower()
            raise ValueError(f"identifier {object_id!r} is already a terminally reserved {kind}")

    def _require_not_publishing(self) -> None:
        if getattr(self._thread_state, "publishing", False):
            raise PublicationError("publication sink must not re-enter the kernel")
        if getattr(self._thread_state, "sampling_clock", False):
            raise AuditIntegrityError("clock callback must not re-enter the kernel")

    @staticmethod
    def _require_create_generation(expected_generation: int) -> None:
        if type(expected_generation) is not int:
            raise ValueError("expected_generation must be an integer")
        if expected_generation != 0:
            raise GenerationConflict(
                f"create requires expected_generation=0, got {expected_generation}"
            )

    @staticmethod
    def _require_positive_generation(expected_generation: int) -> None:
        if type(expected_generation) is not int or expected_generation < 1:
            raise ValueError("expected_generation must be a positive integer")

    @staticmethod
    def _require_positive_revision(payload_revision: int) -> None:
        if type(payload_revision) is not int or payload_revision < 1:
            raise ValueError("payload revision must be a positive integer")

    @staticmethod
    def _pod_event(
        operation: str, head: PodHead, new_payload: bytes | None = None
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "operation": operation, "kind": "pod",
            "head": LifecycleKernel._pod_head_json(head),
        }
        if new_payload is not None:
            event["new_revision"] = {
                "payload_revision": head.payload_revision, "payload_hex": new_payload.hex(),
            }
        return event

    @staticmethod
    def _pod_head_json(head: PodHead) -> dict[str, object]:
        return {
            "pod_id": head.pod_id, "authority_generation": head.authority_generation,
            "payload_revision": head.payload_revision, "status": head.status.value,
            "payload_hash": head.payload_hash, "policy_generation": head.policy_generation,
        }

    @classmethod
    def _alias_event(cls, operation: str, head: AliasHead) -> dict[str, object]:
        return {
            "operation": operation, "kind": "alias",
            "head": {
                "alias_id": head.alias_id,
                "authority_generation": head.authority_generation,
                "status": head.status.value, "target": cls._target_json(head.target),
            },
        }

    def _policy_event(self, operation: str, head: PolicyHead) -> dict[str, object]:
        return {
            "operation": operation, "kind": "policy",
            "head": {
                "authority_generation": head.authority_generation,
                "exact_scope": sorted(head.exact_scope),
            },
            "pod_heads": [
                self._pod_head_json(self._pods[pod_id]) for pod_id in sorted(self._pods)
            ],
        }

    @classmethod
    def _pod_head_from_json(cls, value: Mapping[str, object]) -> PodHead:
        try:
            if set(value) != {
                "pod_id", "authority_generation", "payload_revision", "status",
                "payload_hash", "policy_generation",
            }:
                raise TypeError("pod head keys are not exact")
            pod_id = cls._object_id(value["pod_id"], "pod_id")  # type: ignore[arg-type]
            authority_generation = value["authority_generation"]
            payload_revision = value["payload_revision"]
            policy_generation = value["policy_generation"]
            payload_hash = value["payload_hash"]
            if (
                not cls._positive_int(authority_generation)
                or not cls._positive_int(payload_revision)
                or not cls._positive_int(policy_generation)
                or not cls._digest_valid(payload_hash)
                or not isinstance(value["status"], str)
            ):
                raise TypeError("pod head field type is invalid")
            return PodHead(
                pod_id, authority_generation, payload_revision,
                AuthorityStatus(value["status"]), payload_hash, policy_generation,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid pod head in audit") from exc

    @classmethod
    def _alias_head_from_json(cls, value: Mapping[str, object]) -> AliasHead:
        try:
            if set(value) != {"alias_id", "authority_generation", "status", "target"}:
                raise TypeError("alias head keys are not exact")
            raw_target = value["target"]
            if not isinstance(raw_target, dict) or set(raw_target) != {"kind", "object_id"}:
                raise TypeError("target must be an object")
            if not isinstance(raw_target["kind"], str):
                raise TypeError("target kind must be a string")
            target = AliasTarget(
                TargetKind(raw_target["kind"]),
                cls._object_id(raw_target["object_id"], "target object_id"),  # type: ignore[arg-type]
            )
            alias_id = cls._object_id(value["alias_id"], "alias_id")  # type: ignore[arg-type]
            authority_generation = value["authority_generation"]
            if not cls._positive_int(authority_generation) or not isinstance(
                value["status"], str
            ):
                raise TypeError("alias head field type is invalid")
            return AliasHead(
                alias_id, authority_generation, AuthorityStatus(value["status"]), target,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid alias head in audit") from exc

    @classmethod
    def _policy_head_from_json(cls, value: Mapping[str, object]) -> PolicyHead:
        try:
            if set(value) != {"authority_generation", "exact_scope"}:
                raise TypeError("policy head keys are not exact")
            raw_scope = value["exact_scope"]
            if not isinstance(raw_scope, list) or not all(
                isinstance(item, str) and item for item in raw_scope
            ):
                raise TypeError("exact_scope must be a string list")
            if len(raw_scope) != len(set(raw_scope)):
                raise TypeError("exact_scope must not contain duplicates")
            if raw_scope != sorted(raw_scope):
                raise TypeError("exact_scope must be canonically ordered")
            for item in raw_scope:
                cls._object_id(item, "alias_id")
            authority_generation = value["authority_generation"]
            if not cls._positive_int(authority_generation):
                raise TypeError("policy generation must be a positive integer")
            return PolicyHead(authority_generation, frozenset(raw_scope))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid policy head in audit") from exc

    @staticmethod
    def _target_json(target: AliasTarget) -> dict[str, str]:
        return {"kind": target.kind.value, "object_id": target.object_id}

    @classmethod
    def _witness_json(cls, witness: PathWitness) -> dict[str, object]:
        return {
            "namespace": witness.namespace,
            "root_alias_id": witness.root_alias_id,
            "alias_path": [
                {
                    "alias_id": node.alias_id,
                    "authority_generation": node.authority_generation,
                    "target": cls._target_json(node.target),
                }
                for node in witness.alias_path
            ],
            "pod_id": witness.pod_id, "pod_generation": witness.pod_generation,
            "payload_revision": witness.payload_revision, "payload_hash": witness.payload_hash,
            "policy_generation": witness.policy_generation,
        }

    @classmethod
    def _lineage_json(cls, lineage: ArtifactLineage) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for dependency in lineage.dependencies:
            if isinstance(dependency, PathWitness):
                values.append({"kind": "PATH_WITNESS", "value": cls._witness_json(dependency)})
            elif isinstance(dependency, MissingAliasDependency):
                values.append({"kind": "MISSING_ALIAS", "value": dependency.__dict__})
            elif isinstance(dependency, OutOfScopeDependency):
                values.append({"kind": "OUT_OF_SCOPE", "value": dependency.__dict__})
            elif isinstance(dependency, UnresolvedDependency):
                values.append({"kind": "UNRESOLVED", "value": dependency.__dict__})
            else:
                raise TypeError("unsupported artifact lineage dependency")
        return values

    def _lineage_from_json(self, raw: object) -> ArtifactLineage:
        if not isinstance(raw, list) or not raw:
            raise AuditIntegrityError("artifact lineage must be a non-empty list")
        dependencies: list[LineageDependency] = []
        try:
            for entry in raw:
                if not isinstance(entry, dict) or set(entry) != {"kind", "value"}:
                    raise TypeError("invalid lineage entry")
                kind = entry["kind"]
                value = entry["value"]
                if not isinstance(kind, str) or not isinstance(value, dict):
                    raise TypeError("invalid lineage entry type")
                if kind == "PATH_WITNESS":
                    dependency: LineageDependency = self._witness_from_json(value)
                else:
                    expected = {"namespace", "alias_id", "policy_generation"}
                    if kind == "UNRESOLVED":
                        expected.add("reason")
                    if set(value) != expected:
                        raise TypeError("invalid negative lineage keys")
                    namespace = self._object_id(value["namespace"], "namespace")  # type: ignore[arg-type]
                    alias_id = self._object_id(value["alias_id"], "alias_id")  # type: ignore[arg-type]
                    generation = value["policy_generation"]
                    if namespace != self.namespace or not self._positive_int(generation):
                        raise TypeError("invalid negative lineage authority")
                    if kind == "MISSING_ALIAS":
                        dependency = MissingAliasDependency(namespace, alias_id, generation)
                    elif kind == "OUT_OF_SCOPE":
                        dependency = OutOfScopeDependency(namespace, alias_id, generation)
                    elif kind == "UNRESOLVED":
                        reason = self._object_id(value["reason"], "reason")  # type: ignore[arg-type]
                        dependency = UnresolvedDependency(
                            namespace, alias_id, reason, generation
                        )
                    else:
                        raise TypeError("unknown lineage kind")
                dependencies.append(dependency)
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid artifact lineage in audit") from exc
        lineage = ArtifactLineage.union((dependencies,))
        if len(lineage.dependencies) != len(dependencies):
            raise AuditIntegrityError("artifact lineage contains duplicate dependencies")
        return lineage

    def _witness_from_json(self, value: Mapping[str, object]) -> PathWitness:
        expected = {
            "namespace", "root_alias_id", "alias_path", "pod_id", "pod_generation",
            "payload_revision", "payload_hash", "policy_generation",
        }
        try:
            if set(value) != expected or not isinstance(value["alias_path"], list):
                raise TypeError("witness keys are not exact")
            namespace = self._object_id(value["namespace"], "namespace")  # type: ignore[arg-type]
            root = self._object_id(value["root_alias_id"], "root_alias_id")  # type: ignore[arg-type]
            pod_id = self._object_id(value["pod_id"], "pod_id")  # type: ignore[arg-type]
            nodes: list[AliasPathNode] = []
            for raw_node in value["alias_path"]:  # type: ignore[union-attr]
                if not isinstance(raw_node, dict) or set(raw_node) != {
                    "alias_id", "authority_generation", "target"
                }:
                    raise TypeError("invalid alias path node")
                raw_target = raw_node["target"]
                if not isinstance(raw_target, dict) or set(raw_target) != {
                    "kind", "object_id"
                }:
                    raise TypeError("invalid alias path target")
                generation = raw_node["authority_generation"]
                if not self._positive_int(generation) or not isinstance(
                    raw_target["kind"], str
                ):
                    raise TypeError("invalid alias path authority")
                nodes.append(
                    AliasPathNode(
                        self._object_id(raw_node["alias_id"], "alias_id"),  # type: ignore[arg-type]
                        generation,
                        AliasTarget(
                            TargetKind(raw_target["kind"]),
                            self._object_id(raw_target["object_id"], "target object_id"),  # type: ignore[arg-type]
                        ),
                    )
                )
            pod_generation = value["pod_generation"]
            payload_revision = value["payload_revision"]
            policy_generation = value["policy_generation"]
            if (
                namespace != self.namespace
                or not self._positive_int(pod_generation)
                or not self._positive_int(payload_revision)
                or not self._positive_int(policy_generation)
                or not self._digest_valid(value["payload_hash"])
            ):
                raise TypeError("invalid witness authority")
            witness = PathWitness(
                namespace, root, tuple(nodes), pod_id, pod_generation,
                payload_revision, value["payload_hash"], policy_generation,  # type: ignore[arg-type]
            )
            if not nodes or len(nodes) > self.max_hops:
                raise TypeError("invalid witness path length")
            return witness
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditIntegrityError("invalid path witness in audit") from exc

    def _consumption_event(
        self,
        capability: ConsumeCapability,
        principal: str,
        security_time: float,
        *,
        artifact_id: str | None,
    ) -> dict[str, object]:
        event: dict[str, object] = {
            "operation": "CONSUME" if artifact_id is None else "CONSUME_TO_ARTIFACT",
            "principal": principal, "nonce": capability.nonce,
            "capability_commitment": self._capability_commitment(capability.mac),
            "witness_hash": hashlib.sha256(
                _canonical_json(self._witness_json(capability.witness)).encode("utf-8")
            ).hexdigest(),
            "artifact_id": artifact_id,
            "security_time": security_time,
        }
        if artifact_id is not None:
            pod = self._pods[capability.witness.pod_id]
            payload = self._payloads[pod.pod_id][pod.payload_revision]
            lineage = ArtifactLineage((capability.witness,))
            event["payload_hex"] = payload.hex()
            event["lineage"] = self._lineage_json(lineage)
        return event

    def _resolution_event(
        self, alias_id: str, decision: RouteDecision
    ) -> dict[str, object]:
        if decision.dependency is None:
            raise AuditIntegrityError("route decision lacks an authority dependency")
        return {
            "operation": "ROUTE",
            "alias_id": alias_id,
            "decision": decision.kind.value,
            "reason": decision.reason,
            "policy_generation": self._policy.authority_generation,
            "witness": (
                self._witness_json(decision.witness)
                if decision.witness is not None else None
            ),
            "dependency": self._lineage_json(
                ArtifactLineage((decision.dependency,))
            )[0],
        }

    def _lineage_hash(self, lineage: ArtifactLineage) -> str:
        return hashlib.sha256(
            json.dumps(
                self._lineage_json(lineage), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

    def _unresolved(self, alias_id: str, reason: str) -> RouteDecision:
        dependency = UnresolvedDependency(
            self.namespace, alias_id, reason, self._policy.authority_generation
        )
        return RouteDecision(RouteKind.UNKNOWN, dependency=dependency, reason=reason)

    @staticmethod
    def _object_id(value: str, name: str) -> str:
        if type(value) is not str or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _positive_int(value: object) -> bool:
        return type(value) is int and value > 0

    @staticmethod
    def _max_hops_valid(value: object) -> bool:
        return type(value) is int and 1 <= value <= 1024

    @staticmethod
    def _nonnegative_int(value: object) -> bool:
        return type(value) is int and value >= 0

    @staticmethod
    def _digest_valid(value: object) -> bool:
        return bool(
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _random_id_valid(value: object) -> bool:
        return bool(
            type(value) is str
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _finite_numeric(value: object) -> bool:
        if type(value) not in (int, float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    @staticmethod
    def _plain_json_tree_valid(value: object) -> bool:
        """Reject hostile/noncanonical scalar types and malformed enum fields, iteratively."""

        stack = [value]
        visited = 0
        enum_string_fields = {"operation", "kind", "status", "decision", "verdict"}
        while stack:
            item = stack.pop()
            visited += 1
            if visited > 100_000:
                return False
            if type(item) is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        return False
                    if key in enum_string_fields and type(child) is not str:
                        return False
                    stack.append(child)
            elif type(item) is list:
                stack.extend(item)
            elif item is None or type(item) in (str, bool):
                continue
            elif type(item) in (int, float):
                if not LifecycleKernel._finite_numeric(item):
                    return False
            else:
                return False
        return True

    @classmethod
    def _snapshot_alias_target(cls, value: AliasTarget) -> AliasTarget:
        if type(value) is not AliasTarget:
            raise TypeError("target must be an exact AliasTarget")
        kind = value.kind
        object_id = value.object_id
        if not isinstance(kind, TargetKind):
            raise TypeError("target must contain a typed TargetKind")
        return AliasTarget(kind, cls._object_id(object_id, "target.object_id"))

    @classmethod
    def _snapshot_witness(cls, value: PathWitness) -> PathWitness:
        if type(value) is not PathWitness:
            raise TypeError("witness must be an exact PathWitness")
        raw_path = value.alias_path
        if type(raw_path) is not tuple:
            raise TypeError("witness alias_path must be an exact tuple")
        nodes: list[AliasPathNode] = []
        for raw_node in raw_path:
            if type(raw_node) is not AliasPathNode:
                raise TypeError("witness nodes must be exact AliasPathNode values")
            nodes.append(
                AliasPathNode(
                    raw_node.alias_id,
                    raw_node.authority_generation,
                    cls._snapshot_alias_target(raw_node.target),
                )
            )
        return PathWitness(
            value.namespace,
            value.root_alias_id,
            tuple(nodes),
            value.pod_id,
            value.pod_generation,
            value.payload_revision,
            value.payload_hash,
            value.policy_generation,
        )

    @classmethod
    def _snapshot_dependency(cls, value: LineageDependency) -> LineageDependency:
        if type(value) is PathWitness:
            return cls._snapshot_witness(value)
        if type(value) is MissingAliasDependency:
            return MissingAliasDependency(
                value.namespace, value.alias_id, value.policy_generation
            )
        if type(value) is OutOfScopeDependency:
            return OutOfScopeDependency(
                value.namespace, value.alias_id, value.policy_generation
            )
        if type(value) is UnresolvedDependency:
            return UnresolvedDependency(
                value.namespace, value.alias_id, value.reason, value.policy_generation
            )
        raise TypeError("lineage dependency must have an exact supported type")

    @classmethod
    def _snapshot_route_decision(cls, value: RouteDecision) -> RouteDecision:
        if type(value) is not RouteDecision:
            raise TypeError("decision must be an exact RouteDecision")
        kind = value.kind
        raw_witness = value.witness
        raw_dependency = value.dependency
        reason = value.reason
        witness = (
            None if raw_witness is None else cls._snapshot_witness(raw_witness)
        )
        if raw_dependency is None:
            dependency = None
        elif raw_dependency is raw_witness and witness is not None:
            dependency = witness
        else:
            dependency = cls._snapshot_dependency(raw_dependency)
        return RouteDecision(kind, witness, dependency, reason)

    @classmethod
    def _snapshot_capability(cls, value: ConsumeCapability) -> ConsumeCapability:
        if type(value) is not ConsumeCapability:
            raise TypeError("capability must be an exact ConsumeCapability")
        return ConsumeCapability(
            cls._snapshot_witness(value.witness),
            value.principal,
            value.namespace,
            value.nonce,
            value.issued_at,
            value.expires_at,
            value.mac,
        )

    @staticmethod
    def _snapshot_prepared_publication(
        value: PreparedPublication,
    ) -> PreparedPublication:
        if type(value) is not PreparedPublication:
            raise TypeError("prepared publication must be an exact value")
        return PreparedPublication(
            value.publication_id,
            value.artifact_id,
            value.namespace,
            value.payload_hash,
            value.lineage_hash,
            value.mac,
        )

    @classmethod
    def _snapshot_verifier_result(cls, value: VerifierResult) -> VerifierResult:
        if type(value) is not VerifierResult:
            raise TypeError("verifier result must be an exact VerifierResult")
        verifier = value.verifier
        verdict = value.verdict
        positive_control_passed = value.positive_control_passed
        if not isinstance(verdict, VerifierVerdict) or type(
            positive_control_passed
        ) is not bool:
            raise TypeError("verifier result fields are not typed")
        return VerifierResult(
            cls._object_id(verifier, "verifier"), verdict, positive_control_passed
        )

    @staticmethod
    def _snapshot_audit_records(
        records: Sequence[AuditRecord],
    ) -> tuple[AuditRecord, ...]:
        try:
            source = tuple(records)
        except Exception as exc:
            raise TypeError("audit records must be a finite iterable") from exc
        snapshots: list[AuditRecord] = []
        for record in source:
            if type(record) is not AuditRecord:
                raise TypeError("every audit record must be an exact AuditRecord")
            snapshots.append(
                AuditRecord(
                    sequence=record.sequence,
                    previous_hash=record.previous_hash,
                    event_json=record.event_json,
                    record_hash=record.record_hash,
                    mac=record.mac,
                    namespace=record.namespace,
                    receipt_kind=record.receipt_kind,
                    max_hops=record.max_hops,
                    recorded_at=record.recorded_at,
                    engine_id=record.engine_id,
                    code_version=record.code_version,
                    key_id=record.key_id,
                )
            )
        return tuple(snapshots)

    @staticmethod
    def _secret_bytes(value: bytes, name: str) -> bytes:
        if type(value) is not bytes or len(value) < 32:
            raise ValueError(f"{name} must contain at least 32 bytes")
        return bytes(value)

    @staticmethod
    def _payload(value: bytes) -> bytes:
        if type(value) is not bytes:
            raise TypeError("payload must be bytes")
        return bytes(value)

    @classmethod
    def _target(cls, value: AliasTarget) -> AliasTarget:
        return cls._snapshot_alias_target(value)


__all__ = [
    "AliasHead", "AliasPathNode", "AliasTarget", "ArtifactLineage", "AuditCheckpoint",
    "AuditIntegrityError", "AuditRecord", "AuthorityStatus", "CapabilityError",
    "ConsumeCapability", "DerivedArtifact", "GenerationConflict", "IdempotencyConflict",
    "LifecycleError", "LifecycleKernel", "MissingAliasDependency", "OutOfScopeDependency",
    "PathWitness", "PolicyHead", "PreparedPublication", "PublicationError",
    "PublicationSink", "ReceiptKind", "RouteDecision", "RouteKind", "TargetKind",
    "UnresolvedDependency", "VerifierResult", "VerifierVerdict",
]
