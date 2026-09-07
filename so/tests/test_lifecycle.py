from __future__ import annotations

from dataclasses import replace
from threading import Barrier, Event, RLock, Thread, get_ident
import hashlib
import hmac
import json
import math

import pytest

from so.lifecycle import (
    AliasTarget,
    AliasPathNode,
    AuditIntegrityError,
    AuthorityStatus,
    CapabilityError,
    GenerationConflict,
    IdempotencyConflict,
    LifecycleKernel,
    MissingAliasDependency,
    OutOfScopeDependency,
    PathWitness,
    PublicationError,
    ReceiptKind,
    RouteKind,
    TargetKind,
    VerifierResult,
    VerifierVerdict,
)


AUDIT_SECRET = b"a" * 32
CAPABILITY_SECRET = b"c" * 32
WRONG_SECRET = b"w" * 32


def _kernel(*, max_hops: int = 8, clock=None) -> LifecycleKernel:
    kwargs = {"clock": clock} if clock is not None else {}
    return LifecycleKernel(
        namespace="tenant/acme",
        max_hops=max_hops,
        capability_secret=CAPABILITY_SECRET,
        audit_secret=AUDIT_SECRET,
        allow_time_override=True,
        **kwargs,
    )


def _resign(records):
    """Model an operator with the audit key creating a semantically invalid stream."""

    output = []
    previous_hash = "0" * 64
    for original in records:
        material = json.dumps(
            {
                "sequence": len(output) + 1,
                "previous_hash": previous_hash,
                "event_json": original.event_json,
                "namespace": original.namespace,
                "receipt_kind": original.receipt_kind.value,
                "max_hops": original.max_hops,
                "recorded_at": original.recorded_at,
                "engine_id": original.engine_id,
                "code_version": original.code_version,
                "key_id": original.key_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        record_hash = hashlib.sha256(
            b"FINX-LIFECYCLE-AUDIT-HASH-V1\x00" + material
        ).hexdigest()
        mac = hmac.new(
            AUDIT_SECRET,
            b"FINX-LIFECYCLE-AUDIT-MAC-V1\x00" + record_hash.encode(),
            hashlib.sha256,
        ).hexdigest()
        output.append(
            replace(
                original,
                sequence=len(output) + 1,
                previous_hash=previous_hash,
                record_hash=record_hash,
                mac=mac,
            )
        )
        previous_hash = record_hash
    return tuple(output)


def test_payload_revision_and_authority_generation_are_distinct() -> None:
    kernel = _kernel()
    created = kernel.create_pod("pod", b"v1", idempotency_key="create")
    edited = kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    revoked = kernel.revoke_pod("pod", expected_generation=2, idempotency_key="revoke")
    restored = kernel.restore_pod("pod", expected_generation=3, idempotency_key="restore")

    assert (created.authority_generation, created.payload_revision) == (1, 1)
    assert (edited.authority_generation, edited.payload_revision) == (2, 2)
    assert (revoked.authority_generation, revoked.payload_revision) == (3, 2)
    assert (restored.authority_generation, restored.payload_revision) == (4, 2)


def test_alias_relink_invalidates_complete_path_while_old_pod_remains_live() -> None:
    kernel = _kernel()
    kernel.create_pod("old", b"old", idempotency_key="p-old")
    kernel.create_pod("new", b"new", idempotency_key="p-new")
    kernel.create_alias("answer", AliasTarget.pod("old"), idempotency_key="a")
    before = kernel.route("answer", in_scope=True)

    kernel.relink_alias(
        "answer",
        AliasTarget.pod("new"),
        expected_generation=1,
        idempotency_key="relink",
    )

    assert before.kind is RouteKind.RESOLVE
    assert kernel.read_pod("old") == b"old"
    assert not kernel.validate_witness(before.witness)
    assert kernel.route("answer", in_scope=True).witness.pod_id == "new"


def test_revoke_and_same_byte_restore_never_revives_old_witness() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"same", idempotency_key="create")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    old = kernel.route("alias", in_scope=True).witness

    kernel.revoke_pod("pod", expected_generation=1, idempotency_key="revoke")
    restored = kernel.restore_pod("pod", expected_generation=2, idempotency_key="restore")

    assert restored.payload_revision == old.payload_revision
    assert restored.authority_generation > old.pod_generation
    assert not kernel.validate_witness(old)


def test_rollback_selects_immutable_bytes_under_a_new_generation() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")

    rolled_back = kernel.rollback_pod(
        "pod", to_payload_revision=1, expected_generation=2, idempotency_key="rollback"
    )

    assert (rolled_back.authority_generation, rolled_back.payload_revision) == (3, 1)
    assert kernel.read_pod("pod") == b"v1"


def test_deleted_pod_id_is_terminal() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    deleted = kernel.delete_pod("pod", expected_generation=1, idempotency_key="delete")
    assert deleted.status is AuthorityStatus.DELETED

    with pytest.raises(PermissionError):
        kernel.restore_pod("pod", expected_generation=2, idempotency_key="restore")
    with pytest.raises(ValueError):
        kernel.create_pod("pod", b"replacement", idempotency_key="recreate")


def test_intermediate_alias_mutation_invalidates_entire_path() -> None:
    kernel = _kernel()
    kernel.create_pod("one", b"1", idempotency_key="p1")
    kernel.create_pod("two", b"2", idempotency_key="p2")
    kernel.create_alias("leaf", AliasTarget.pod("one"), idempotency_key="leaf")
    kernel.create_alias("root", AliasTarget.alias("leaf"), idempotency_key="root")
    old = kernel.route("root", in_scope=True).witness

    kernel.relink_alias(
        "leaf", AliasTarget.pod("two"), expected_generation=1, idempotency_key="move-leaf"
    )

    assert [node.alias_id for node in old.alias_path] == ["root", "leaf"]
    assert not kernel.validate_witness(old)


@pytest.mark.parametrize("failure", ["missing", "dead"])
def test_in_scope_resolution_failures_are_unknown(failure: str) -> None:
    kernel = _kernel(max_hops=2)
    if failure == "missing":
        kernel.set_policy_scope(
            ["missing"], expected_generation=1, idempotency_key="scope-missing"
        )
        alias_id = "missing"
    else:
        kernel.create_pod("pod", b"v", idempotency_key="p")
        kernel.create_alias("dead", AliasTarget.pod("pod"), idempotency_key="a")
        kernel.revoke_alias("dead", expected_generation=1, idempotency_key="revoke")
        alias_id = "dead"

    assert kernel.route(alias_id, in_scope=True).kind is RouteKind.UNKNOWN


def test_route_is_exact_and_tri_state() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("exact", AliasTarget.pod("pod"), idempotency_key="a")

    assert kernel.route("anything", in_scope=True).kind is RouteKind.BYPASS
    assert kernel.route("exact", in_scope=False).kind is RouteKind.RESOLVE
    assert kernel.route("EXACT", in_scope=True).kind is RouteKind.BYPASS


def test_authority_errors_fail_closed_but_out_of_scope_still_bypasses() -> None:
    class BrokenAuthority(LifecycleKernel):
        def _resolve_locked(self, alias_id: str):
            raise RuntimeError("authority unavailable")

    kernel = BrokenAuthority(
        namespace="tenant/acme",
        capability_secret=CAPABILITY_SECRET,
        audit_secret=AUDIT_SECRET,
    )
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    assert kernel.route("alias", in_scope=False).kind is RouteKind.UNKNOWN
    assert kernel.route("outside", in_scope=True).kind is RouteKind.BYPASS


def test_idempotency_replay_is_noop_and_collision_is_rejected() -> None:
    kernel = _kernel()
    first = kernel.create_pod("pod", b"v1", idempotency_key="same")
    audit_length = len(kernel.audit_records)
    replay = kernel.create_pod("pod", b"v1", idempotency_key="same")

    assert replay == first
    assert len(kernel.audit_records) == audit_length
    with pytest.raises(IdempotencyConflict):
        kernel.create_pod("other", b"v1", idempotency_key="same")


def test_expected_generation_cas_rejects_stale_mutation() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod("pod", b"winner", expected_generation=1, idempotency_key="winner")

    with pytest.raises(GenerationConflict):
        kernel.edit_pod("pod", b"loser", expected_generation=1, idempotency_key="loser")


def test_two_same_generation_mutations_have_exactly_one_winner() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    barrier = Barrier(3)
    outcomes: list[str] = []

    def edit(value: bytes, key: str) -> None:
        barrier.wait()
        try:
            kernel.edit_pod("pod", value, expected_generation=1, idempotency_key=key)
        except GenerationConflict:
            outcomes.append("conflict")
        else:
            outcomes.append("committed")

    threads = [Thread(target=edit, args=(b"a", "a")), Thread(target=edit, args=(b"b", "b"))]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["committed", "conflict"]
    assert kernel.pod_head("pod").authority_generation == 2


def test_audit_chain_verifies_replays_and_detects_modification_reorder_and_gap() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    records = kernel.audit_records

    checkpoint = kernel.audit_checkpoint()
    assert LifecycleKernel.verify_audit(
        records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    replayed = LifecycleKernel.replay(
        records,
        namespace="tenant/acme",
        max_hops=8,
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        allow_time_override=True,
    )
    assert replayed.pod_head("pod") == kernel.pod_head("pod")
    assert replayed.alias_head("alias") == kernel.alias_head("alias")

    modified = list(records)
    modified[0] = replace(modified[0], event_json='{"operation":"tampered"}')
    verify = lambda candidate: LifecycleKernel.verify_audit(
        candidate,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert not verify(modified)
    assert not verify(list(reversed(records)))
    assert not verify(records[1:])
    assert not verify(records[:-1])
    with pytest.raises(AuditIntegrityError):
        LifecycleKernel.replay(
            modified,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=checkpoint.terminal_hash,
            expected_length=checkpoint.length,
        )
    with pytest.raises(AuditIntegrityError):
        LifecycleKernel.replay(
            records,
            namespace="tenant/other",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=checkpoint.terminal_hash,
            expected_length=checkpoint.length,
        )


def test_lineage_is_transitive_and_selectively_stale() -> None:
    kernel = _kernel()
    for pod_id in ("one", "two"):
        kernel.create_pod(pod_id, pod_id.encode(), idempotency_key=f"p-{pod_id}")
        kernel.create_alias(
            f"{pod_id}-alias",
            AliasTarget.pod(pod_id),
            idempotency_key=f"a-{pod_id}",
        )
    one = kernel.derive_artifact(
        "one", b"cached-one", decisions=[kernel.route("one-alias", in_scope=True)]
    )
    two = kernel.derive_artifact(
        "two", b"cached-two", decisions=[kernel.route("two-alias", in_scope=True)]
    )
    combined = kernel.derive_artifact("combined", b"both", parents=["one", "two"])

    kernel.edit_pod("one", b"one-v2", expected_generation=1, idempotency_key="edit-one")

    assert not kernel.artifact_current(one.artifact_id)
    assert kernel.artifact_current(two.artifact_id)
    assert not kernel.artifact_current(combined.artifact_id)
    assert len(combined.lineage.dependencies) == 2


def test_cached_miss_records_negative_dependency() -> None:
    kernel = _kernel()
    kernel.set_policy_scope(["future"], expected_generation=1, idempotency_key="scope")
    miss = kernel.route("future", in_scope=True)
    artifact = kernel.derive_artifact("miss", b"negative", decisions=[miss])
    assert isinstance(artifact.lineage.dependencies[0], MissingAliasDependency)
    assert kernel.artifact_current("miss")

    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("future", AliasTarget.pod("pod"), idempotency_key="a")
    assert not kernel.artifact_current("miss")


def test_one_use_capability_binds_path_principal_namespace_nonce_and_expiry() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"secret", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    witness = kernel.route("alias", in_scope=True).witness
    capability = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=100.0
    )

    assert kernel.consume(capability, principal="alice", namespace="tenant/acme", now=105.0) == b"secret"
    with pytest.raises(CapabilityError):
        kernel.consume(capability, principal="alice", namespace="tenant/acme", now=105.0)

    fresh = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=106.0
    )
    with pytest.raises(CapabilityError):
        kernel.consume(fresh, principal="mallory", namespace="tenant/acme", now=107.0)
    assert kernel.consume(fresh, principal="alice", namespace="tenant/acme", now=107.0) == b"secret"

    tamper_target = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=108.0
    )
    with pytest.raises(CapabilityError):
        kernel.consume(
            replace(tamper_target, nonce="different"),
            principal="alice",
            namespace="tenant/acme",
            now=109.0,
        )
    with pytest.raises(CapabilityError):
        kernel.consume(
            tamper_target, principal="alice", namespace="tenant/other", now=109.0
        )

    expiring = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=1, now=110.0
    )
    with pytest.raises(CapabilityError):
        kernel.consume(expiring, principal="alice", namespace="tenant/acme", now=111.0)


def test_capability_is_invalidated_by_any_path_mutation() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"secret", idempotency_key="p")
    kernel.create_alias("leaf", AliasTarget.pod("pod"), idempotency_key="leaf")
    kernel.create_alias("root", AliasTarget.alias("leaf"), idempotency_key="root")
    witness = kernel.route("root", in_scope=True).witness
    capability = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=100.0
    )

    kernel.revoke_alias("leaf", expected_generation=1, idempotency_key="revoke")

    with pytest.raises(CapabilityError):
        kernel.consume(capability, principal="alice", namespace="tenant/acme", now=101.0)


def test_alias_heads_have_independent_generations_and_typed_targets() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    root = kernel.create_alias("root", AliasTarget.pod("pod"), idempotency_key="root")
    leaf = kernel.create_alias("leaf", AliasTarget.alias("root"), idempotency_key="leaf")
    assert root.target.kind is TargetKind.POD
    assert leaf.target.kind is TargetKind.ALIAS

    revoked = kernel.revoke_alias("root", expected_generation=1, idempotency_key="revoke")
    restored = kernel.restore_alias("root", expected_generation=2, idempotency_key="restore")
    assert revoked.authority_generation == 2
    assert restored.authority_generation == 3


def test_alias_create_and_relink_validate_the_complete_hypothetical_path() -> None:
    kernel = _kernel(max_hops=2)
    kernel.create_pod("live", b"v", idempotency_key="live")
    kernel.create_pod("dead", b"v", idempotency_key="dead")
    kernel.revoke_pod("dead", expected_generation=1, idempotency_key="kill")
    kernel.create_alias("leaf", AliasTarget.pod("live"), idempotency_key="leaf")
    kernel.create_alias("root", AliasTarget.alias("leaf"), idempotency_key="root")
    kernel.create_alias("other", AliasTarget.pod("live"), idempotency_key="other")

    with pytest.raises((KeyError, ValueError)):
        kernel.create_alias("missing", AliasTarget.pod("absent"), idempotency_key="missing")
    with pytest.raises(PermissionError):
        kernel.create_alias("dead-target", AliasTarget.pod("dead"), idempotency_key="dead-a")
    with pytest.raises(ValueError):
        kernel.create_alias("self", AliasTarget.alias("self"), idempotency_key="self")
    with pytest.raises(ValueError):
        kernel.relink_alias(
            "leaf", AliasTarget.alias("root"), expected_generation=1, idempotency_key="cycle"
        )
    with pytest.raises(ValueError):
        kernel.relink_alias(
            "root", AliasTarget.alias("root"), expected_generation=1, idempotency_key="self-r"
        )
    with pytest.raises(KeyError):
        kernel.relink_alias(
            "root",
            AliasTarget.alias("absent"),
            expected_generation=1,
            idempotency_key="missing-r",
        )
    with pytest.raises(ValueError):
        kernel.create_alias("too-deep", AliasTarget.alias("root"), idempotency_key="deep")
    with pytest.raises(PermissionError):
        kernel.relink_alias(
            "root", AliasTarget.pod("dead"), expected_generation=1, idempotency_key="dead-r"
        )
    with pytest.raises(ValueError):
        kernel.relink_alias(
            "other", AliasTarget.alias("root"), expected_generation=1, idempotency_key="deep-r"
        )

    assert kernel.alias_head("leaf").target == AliasTarget.pod("live")
    with pytest.raises(KeyError):
        kernel.alias_head("self")


def test_forged_cycle_and_over_depth_witnesses_are_rejected() -> None:
    kernel = _kernel(max_hops=2)
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("leaf", AliasTarget.pod("pod"), idempotency_key="leaf")
    kernel.create_alias("root", AliasTarget.alias("leaf"), idempotency_key="root")
    witness = kernel.route("root").witness
    assert witness is not None

    duplicate = replace(witness, alias_path=witness.alias_path + (witness.alias_path[0],))
    invented = replace(
        witness,
        alias_path=witness.alias_path
        + (AliasPathNode("invented", 1, AliasTarget.pod("pod")),),
    )
    wrong_root = replace(witness, alias_path=witness.alias_path[1:])
    assert not kernel.validate_witness(duplicate)
    assert not kernel.validate_witness(invented)
    assert not kernel.validate_witness(wrong_root)


def test_policy_scope_is_authoritative_and_generation_invalidates_dependents() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    decision = kernel.route("alias", in_scope=False)
    artifact = kernel.derive_artifact("cache", b"cached", decisions=[decision])
    capability = kernel.issue_capability(
        decision.witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    before = kernel.pod_head("pod")

    policy = kernel.set_policy_scope(
        ["future"], expected_generation=1, idempotency_key="policy"
    )

    assert policy.authority_generation == 2
    assert kernel.pod_head("pod").policy_generation == 2
    assert kernel.pod_head("pod").authority_generation == before.authority_generation + 1
    assert not kernel.validate_witness(decision.witness)
    assert not kernel.artifact_current("cache")
    with pytest.raises(CapabilityError):
        kernel.consume(
            capability, principal="alice", namespace="tenant/acme", now=101.0
        )
    assert kernel.route("future").kind is RouteKind.UNKNOWN
    assert kernel.route("outside", in_scope=True).kind is RouteKind.BYPASS


def test_cross_kind_identity_create_generation_and_terminal_alias_delete() -> None:
    kernel = _kernel()
    with pytest.raises(GenerationConflict):
        kernel.create_pod("bad", b"v", expected_generation=1, idempotency_key="bad")
    kernel.create_pod("shared", b"v", expected_generation=0, idempotency_key="pod")
    with pytest.raises(ValueError):
        kernel.create_alias("shared", AliasTarget.pod("shared"), idempotency_key="alias")

    kernel.create_alias("alias", AliasTarget.pod("shared"), idempotency_key="create-a")
    with pytest.raises(ValueError):
        kernel.create_pod("alias", b"v", idempotency_key="cross-kind")
    deleted = kernel.delete_alias("alias", expected_generation=1, idempotency_key="delete-a")
    assert deleted.status is AuthorityStatus.DELETED
    with pytest.raises(PermissionError):
        kernel.restore_alias("alias", expected_generation=2, idempotency_key="restore-a")
    with pytest.raises(ValueError):
        kernel.create_alias("alias", AliasTarget.pod("shared"), idempotency_key="recreate-a")


def test_restore_pod_can_select_an_explicit_immutable_revision() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    kernel.revoke_pod("pod", expected_generation=2, idempotency_key="revoke")

    restored = kernel.restore_pod(
        "pod",
        payload_revision=1,
        expected_generation=3,
        idempotency_key="restore-v1",
    )
    assert restored.payload_revision == 1
    assert restored.authority_generation == 4
    assert kernel.read_pod("pod") == b"v1"


def test_audit_append_failure_rolls_back_the_authoritative_mutation(monkeypatch) -> None:
    kernel = _kernel()

    def fail(_event, _kind=ReceiptKind.MUTATION):
        raise OSError("audit unavailable")

    monkeypatch.setattr(kernel, "_append_audit", fail)
    with pytest.raises(OSError):
        kernel.create_pod("pod", b"v", idempotency_key="create")
    with pytest.raises(KeyError):
        kernel.pod_head("pod")
    assert kernel.audit_records == ()


def test_capability_rejects_non_finite_backward_and_boundary_times() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    witness = kernel.route("alias").witness

    for ttl in (math.nan, math.inf, -math.inf):
        with pytest.raises(CapabilityError):
            kernel.issue_capability(
                witness,
                principal="alice",
                namespace="tenant/acme",
                ttl_seconds=ttl,
                now=100.0,
            )
    with pytest.raises(CapabilityError):
        kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=1,
            now=math.nan,
        )

    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    with pytest.raises(CapabilityError):
        kernel.consume(
            capability, principal="alice", namespace="tenant/acme", now=99.0
        )
    with pytest.raises(CapabilityError):
        kernel.consume(
            capability, principal="alice", namespace="tenant/acme", now=110.0
        )

    strict = LifecycleKernel(namespace="tenant/strict", secret=b"s" * 32)
    strict.create_pod("pod", b"v", idempotency_key="p")
    strict.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    with pytest.raises(CapabilityError, match="inject a trusted clock"):
        strict.issue_capability(
            strict.route("alias").witness,
            principal="alice",
            namespace="tenant/strict",
            ttl_seconds=1,
            now=100.0,
        )


def test_consumed_lineage_publication_is_fenced_against_concurrent_mutation() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    witness = kernel.route("alias").witness
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=20,
        now=100.0,
    )
    kernel.consume_to_artifact(
        capability,
        artifact_id="kv",
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    )
    sink_entered = Event()
    release_sink = Event()
    ordering: list[str] = []

    class Sink:
        def commit(self, publication_id: str, payload: bytes) -> bool:
            assert publication_id
            sink_entered.set()
            assert release_sink.wait(timeout=2)
            assert payload == b"v1"
            ordering.append("published")
            return True

    def publish() -> None:
        prepared = kernel.prepare_publication("kv")
        kernel.commit_publication(prepared, Sink())

    def mutate() -> None:
        kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
        ordering.append("mutated")

    publisher = Thread(target=publish)
    publisher.start()
    assert sink_entered.wait(timeout=2)
    mutator = Thread(target=mutate)
    mutator.start()
    assert ordering == []
    release_sink.set()
    publisher.join(timeout=2)
    mutator.join(timeout=2)

    assert ordering == ["published", "mutated"]
    receipts = [record.receipt_kind for record in kernel.audit_records]
    assert ReceiptKind.CONSUMPTION in receipts
    assert ReceiptKind.PUBLICATION in receipts


def test_stale_artifact_or_kv_is_never_sent_to_publication_sink() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    decision = kernel.route("alias")
    kernel.derive_artifact("kv", b"derived", decisions=[decision])
    prepared = kernel.prepare_publication("kv")
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")

    class Sink:
        def __init__(self) -> None:
            self.output: list[bytes] = []

        def commit(self, _publication_id: str, payload: bytes) -> bool:
            self.output.append(payload)
            return True

    sink = Sink()

    with pytest.raises(PublicationError):
        kernel.commit_publication(prepared, sink)
    assert sink.output == []


def test_authenticated_audit_covers_all_receipt_kinds_and_rejects_rechaining() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    decision = kernel.route("alias")
    capability = kernel.issue_capability(
        decision.witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    kernel.consume_to_artifact(
        capability,
        artifact_id="artifact",
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    )
    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(kernel.prepare_publication("artifact"), Sink())
    kernel.record_verifier_run(
        VerifierResult("lifecycle-tests", VerifierVerdict.PASS, positive_control_passed=True)
    )
    records = kernel.audit_records
    checkpoint = kernel.audit_checkpoint()

    assert {
        ReceiptKind.MUTATION,
        ReceiptKind.RESOLUTION,
        ReceiptKind.CONSUMPTION,
        ReceiptKind.PUBLICATION,
        ReceiptKind.VERIFIER_RUN,
    } <= {record.receipt_kind for record in records}
    assert all(
        record.namespace == "tenant/acme"
        and record.engine_id
        and record.code_version
        and record.key_id == "local-v1"
        and math.isfinite(record.recorded_at)
        for record in records
    )

    forged = []
    previous_hash = "0" * 64
    for index, original in enumerate(records):
        event_json = (
            '{"kind":"pod","operation":"forged"}' if index == 0 else original.event_json
        )
        material = json.dumps(
            {
                "sequence": original.sequence,
                "previous_hash": previous_hash,
                "event_json": event_json,
                "namespace": original.namespace,
                "receipt_kind": original.receipt_kind.value,
                "max_hops": original.max_hops,
                "recorded_at": original.recorded_at,
                "engine_id": original.engine_id,
                "code_version": original.code_version,
                "key_id": original.key_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        attacker_hash = hashlib.sha256(
            b"FINX-LIFECYCLE-AUDIT-HASH-V1\x00" + material
        ).hexdigest()
        forged.append(
            replace(
                original,
                previous_hash=previous_hash,
                event_json=event_json,
                record_hash=attacker_hash,
            )
        )
        previous_hash = attacker_hash
    assert not LifecycleKernel.verify_audit(
        forged,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=forged[-1].record_hash,
        expected_length=len(forged),
    )
    assert not LifecycleKernel.verify_audit(
        records,
        audit_secret=WRONG_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )


def test_audit_replay_restores_spent_nonce_and_policy_state() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="p")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="a")
    witness = kernel.route("alias").witness
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=100,
        now=100.0,
    )
    assert kernel.consume(
        capability, principal="alice", namespace="tenant/acme", now=101.0
    ) == b"v"
    kernel.set_policy_scope(["future"], expected_generation=1, idempotency_key="policy")
    checkpoint = kernel.audit_checkpoint()

    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        allow_time_override=True,
    )

    assert replayed.policy_head() == kernel.policy_head()
    assert replayed.pod_head("pod") == kernel.pod_head("pod")
    with pytest.raises(CapabilityError, match="already consumed"):
        replayed.consume(
            capability, principal="alice", namespace="tenant/acme", now=102.0
        )


def test_bypass_has_negative_lineage_and_empty_lineage_is_forbidden() -> None:
    kernel = _kernel()
    bypass = kernel.route("future")
    assert bypass.kind is RouteKind.BYPASS
    assert isinstance(bypass.dependency, OutOfScopeDependency)
    artifact = kernel.derive_artifact("negative", b"cached", decisions=[bypass])
    assert kernel.artifact_current(artifact.artifact_id)

    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("future", AliasTarget.pod("pod"), idempotency_key="alias")
    assert not kernel.artifact_current(artifact.artifact_id)

    policy_kernel = _kernel()
    policy_bypass = policy_kernel.route("future")
    policy_kernel.derive_artifact("negative", b"cached", decisions=[policy_bypass])
    policy_kernel.set_policy_scope(
        ["future"], expected_generation=1, idempotency_key="scope"
    )
    assert not policy_kernel.artifact_current("negative")
    with pytest.raises(ValueError, match="lineage"):
        kernel.derive_artifact("empty", b"unsafe")


def test_consume_to_artifact_rolls_back_on_audit_commit_failure(monkeypatch) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    before = kernel.audit_checkpoint()

    def fail(_record) -> None:
        raise OSError("durable audit unavailable")

    monkeypatch.setattr(kernel, "_commit_audit", fail)
    with pytest.raises(OSError):
        kernel.consume_to_artifact(
            capability,
            artifact_id="kv",
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )
    assert kernel.audit_checkpoint() == before
    assert not kernel.artifact_current("kv")

    monkeypatch.undo()
    assert kernel.consume_to_artifact(
        capability,
        artifact_id="kv",
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ).payload == b"v"


def test_audit_and_capability_use_distinct_key_material() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    checkpoint = kernel.audit_checkpoint()

    assert LifecycleKernel.verify_audit(
        kernel.audit_records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert not LifecycleKernel.verify_audit(
        kernel.audit_records,
        audit_secret=CAPABILITY_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )


def test_serialized_capability_expiry_must_be_finite() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    with pytest.raises(CapabilityError, match="finite"):
        kernel.consume(
            replace(capability, expires_at=math.nan),
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )


def test_namespace_and_hop_limit_are_immutable_and_idempotency_key_is_typed() -> None:
    kernel = _kernel()
    with pytest.raises(AttributeError):
        kernel.namespace = "tenant/other"
    with pytest.raises(AttributeError):
        kernel.max_hops = 99
    with pytest.raises((TypeError, ValueError)):
        kernel.create_pod("pod", b"v", idempotency_key=123)


@pytest.mark.parametrize("max_hops", [True, 1.5, math.nan, math.inf, -math.inf, 0, -1])
def test_max_hops_requires_a_positive_non_boolean_integer(max_hops) -> None:
    with pytest.raises((TypeError, ValueError)):
        _kernel(max_hops=max_hops)


@pytest.mark.parametrize(
    "secrets_kwargs",
    [
        {"capability_secret": b"", "audit_secret": b"audit"},
        {"capability_secret": b"capability", "audit_secret": b""},
    ],
)
def test_explicit_empty_secrets_are_rejected(secrets_kwargs) -> None:
    with pytest.raises(ValueError, match="secret"):
        LifecycleKernel(namespace="tenant/acme", **secrets_kwargs)


@pytest.mark.parametrize("to_artifact", [False, True])
def test_consume_audit_failure_rolls_back_security_time(monkeypatch, to_artifact) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=20,
        now=100.0,
    )

    def fail(_record) -> None:
        raise OSError("durable audit unavailable")

    monkeypatch.setattr(kernel, "_commit_audit", fail)
    with pytest.raises(OSError):
        if to_artifact:
            kernel.consume_to_artifact(
                capability,
                artifact_id="failed",
                principal="alice",
                namespace="tenant/acme",
                now=105.0,
            )
        else:
            kernel.consume(
                capability,
                principal="alice",
                namespace="tenant/acme",
                now=105.0,
            )
    monkeypatch.undo()

    if to_artifact:
        assert kernel.consume_to_artifact(
            capability,
            artifact_id="committed",
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        ).payload == b"v"
    else:
        assert kernel.consume(
            capability,
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        ) == b"v"


def test_verifier_combination_requires_positive_controls_and_unanimous_pass() -> None:
    control = VerifierResult("control", VerifierVerdict.PASS, True)
    workspace = VerifierResult("workspace", VerifierVerdict.PASS, True)
    no_positive_control = VerifierResult("control", VerifierVerdict.PASS, False)
    failed_control = VerifierResult("control", VerifierVerdict.FAIL, True)
    inconclusive = VerifierResult(
        "workspace", VerifierVerdict.INCONCLUSIVE, True
    )
    boundary_violation = VerifierResult("workspace", VerifierVerdict.FAIL, True)

    assert LifecycleKernel.combine_verifier_results([]) is VerifierVerdict.INCONCLUSIVE
    assert LifecycleKernel.combine_verifier_results([control]) is VerifierVerdict.INCONCLUSIVE
    assert LifecycleKernel.combine_verifier_results([workspace]) is VerifierVerdict.INCONCLUSIVE
    assert LifecycleKernel.combine_verifier_results(
        [control, workspace]
    ) is VerifierVerdict.PASS
    assert LifecycleKernel.combine_verifier_results(
        [control, inconclusive]
    ) is VerifierVerdict.INCONCLUSIVE
    assert LifecycleKernel.combine_verifier_results(
        [control, boundary_violation]
    ) is VerifierVerdict.FAIL
    assert LifecycleKernel.combine_verifier_results(
        [no_positive_control, workspace]
    ) is VerifierVerdict.INCONCLUSIVE
    assert LifecycleKernel.combine_verifier_results(
        [failed_control, workspace]
    ) is VerifierVerdict.INCONCLUSIVE


def test_verifier_run_records_and_replays_control_workspace_combination() -> None:
    kernel = _kernel()
    record = kernel.record_verifier_run(
        (
            VerifierResult("control", VerifierVerdict.PASS, True),
            VerifierResult("workspace", VerifierVerdict.PASS, True),
        )
    )
    event = json.loads(record.event_json)
    assert event["verdict"] == VerifierVerdict.PASS.value
    checkpoint = kernel.audit_checkpoint()

    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert replayed.audit_checkpoint() == checkpoint


def test_idempotent_publication_retry_after_ambiguous_sink_failure_has_one_effect() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    artifact = kernel.derive_artifact(
        "artifact", b"derived", decisions=[kernel.route("alias")]
    )
    prepared = kernel.prepare_publication(artifact.artifact_id)

    class AmbiguousSink:
        def __init__(self) -> None:
            self.effects: dict[str, bytes] = {}
            self.first = True

        def commit(self, publication_id: str, payload: bytes) -> bool:
            self.effects.setdefault(publication_id, payload)
            if self.first:
                self.first = False
                raise OSError("reply lost after durable commit")
            return self.effects[publication_id] == payload

    sink = AmbiguousSink()
    before_receipts = sum(
        record.receipt_kind is ReceiptKind.PUBLICATION for record in kernel.audit_records
    )
    with pytest.raises(OSError):
        kernel.commit_publication(prepared, sink)
    assert sum(
        record.receipt_kind is ReceiptKind.PUBLICATION for record in kernel.audit_records
    ) == before_receipts

    kernel.commit_publication(prepared, sink)
    assert list(sink.effects.values()) == [b"derived"]
    assert sum(
        record.receipt_kind is ReceiptKind.PUBLICATION for record in kernel.audit_records
    ) == before_receipts + 1


def test_replay_rejects_duplicate_consumption_nonce_even_when_stream_is_signed() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    kernel.consume(capability, principal="alice", namespace="tenant/acme", now=101.0)
    consumption = next(
        record
        for record in kernel.audit_records
        if record.receipt_kind is ReceiptKind.CONSUMPTION
    )
    signed = _resign(kernel.audit_records + (consumption,))

    with pytest.raises(AuditIntegrityError, match="duplicate consumption nonce"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


@pytest.mark.parametrize("corruption", ["revision-overwrite", "illegal-status"])
def test_replay_rejects_signed_illegal_mutation_transitions(corruption: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    if corruption == "revision-overwrite":
        kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    else:
        kernel.revoke_pod("pod", expected_generation=1, idempotency_key="revoke")
    records = list(kernel.audit_records)
    event = json.loads(records[-1].event_json)
    if corruption == "revision-overwrite":
        event["head"]["payload_revision"] = 1
        event["new_revision"]["payload_revision"] = 1
    else:
        event["operation"] = "DELETE_POD"
    records[-1] = replace(
        records[-1],
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="illegal|overwritten"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_replay_restores_security_time_floor() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    kernel.consume(capability, principal="alice", namespace="tenant/acme", now=105.0)
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        allow_time_override=True,
    )

    with pytest.raises(CapabilityError, match="backwards"):
        replayed.issue_capability(
            replayed.route("alias").witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=104.0,
        )


def test_reentrant_clock_cannot_mutate_or_duplicate_audit_sequence() -> None:
    class ReentrantClock:
        def __init__(self) -> None:
            self.kernel = None
            self.armed = False
            self.error = None

        def __call__(self) -> float:
            if self.armed:
                self.armed = False
                try:
                    self.kernel.create_pod("inner", b"bad", idempotency_key="inner")
                except Exception as exc:
                    self.error = exc
            return 100.0

    clock = ReentrantClock()
    kernel = _kernel(clock=clock)
    clock.kernel = kernel
    clock.armed = True
    kernel.create_pod("outer", b"good", idempotency_key="outer")

    assert isinstance(clock.error, AuditIntegrityError)
    with pytest.raises(KeyError):
        kernel.pod_head("inner")
    assert [record.sequence for record in kernel.audit_records] == [1]

    kernel.create_alias("alias", AliasTarget.pod("outer"), idempotency_key="alias")
    clock.error = None
    clock.armed = True
    kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
    )
    assert isinstance(clock.error, AuditIntegrityError)
    with pytest.raises(KeyError):
        kernel.pod_head("inner")
    assert [record.sequence for record in kernel.audit_records] == list(
        range(1, len(kernel.audit_records) + 1)
    )


def test_receipt_kind_and_event_schema_are_strictly_coupled_cross_product() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    kernel.consume_to_artifact(
        capability,
        artifact_id="artifact",
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    )

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(kernel.prepare_publication("artifact"), Sink())
    kernel.record_verifier_run(VerifierResult("audit", VerifierVerdict.PASS, True))
    records = kernel.audit_records
    kinds = {record.receipt_kind for record in records}
    assert ReceiptKind.CAPABILITY_ISSUANCE in kinds

    for index, original in enumerate(records):
        for wrong_kind in ReceiptKind:
            if wrong_kind is original.receipt_kind:
                continue
            forged = list(records)
            forged[index] = replace(original, receipt_kind=wrong_kind)
            signed = _resign(forged)
            assert not LifecycleKernel.verify_audit(
                signed,
                audit_secret=AUDIT_SECRET,
                namespace="tenant/acme",
                expected_terminal_hash=signed[-1].record_hash,
                expected_length=len(signed),
            ), (original.receipt_kind, wrong_kind)


def test_issue_only_security_time_floor_survives_authenticated_replay() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=105.0,
    )
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        allow_time_override=True,
    )

    with pytest.raises(CapabilityError, match="backwards"):
        replayed.issue_capability(
            replayed.route("alias").witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=104.0,
        )


@pytest.mark.parametrize("tamper", ["fingerprint", "command-body"])
def test_replay_recomputes_command_fingerprint_from_audited_body(tamper: str) -> None:
    kernel = _kernel()
    result = kernel.create_pod("pod", b"v", idempotency_key="create")
    records = list(kernel.audit_records)
    event = json.loads(records[0].event_json)
    if tamper == "fingerprint":
        event["command_fingerprint"] = "0" * 64
    else:
        event["command_body"]["payload_hex"] = b"other".hex()
    records[0] = replace(
        records[0], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)
    with pytest.raises(AuditIntegrityError, match="fingerprint"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )

    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    before = replayed.audit_checkpoint()
    assert replayed.create_pod("pod", b"v", idempotency_key="create") == result
    assert replayed.audit_checkpoint() == before


def test_replay_rejects_non_string_object_identifier() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    records = list(kernel.audit_records)
    event = json.loads(records[0].event_json)
    event["head"]["pod_id"] = 7
    event["command_body"]["pod_id"] = 7
    records[0] = replace(
        records[0], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="pod head|identifier"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


@pytest.mark.parametrize("corruption", ["non-dict", "duplicate", "wrong-length"])
def test_replay_requires_exact_unique_dict_policy_pod_heads(corruption: str) -> None:
    kernel = _kernel()
    kernel.create_pod("one", b"1", idempotency_key="one")
    kernel.create_pod("two", b"2", idempotency_key="two")
    kernel.set_policy_scope([], expected_generation=1, idempotency_key="policy")
    records = list(kernel.audit_records)
    event = json.loads(records[-1].event_json)
    if corruption == "non-dict":
        event["pod_heads"][0] = "not-a-head"
    elif corruption == "duplicate":
        event["pod_heads"][1] = event["pod_heads"][0]
    else:
        event["pod_heads"].append(event["pod_heads"][0])
    records[-1] = replace(
        records[-1], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="policy event|pod"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


_INTEGER_API_CASES = (
    "create-pod", "edit-pod", "revoke-pod", "restore-pod-generation",
    "restore-pod-revision", "rollback-pod-generation", "rollback-pod-revision",
    "delete-pod", "create-alias", "relink-alias", "revoke-alias",
    "restore-alias", "delete-alias", "policy",
)


@pytest.mark.parametrize("invalid", [True, 1.0, "1"])
@pytest.mark.parametrize("api_case", _INTEGER_API_CASES)
def test_all_live_generation_and_revision_inputs_are_exact_integers(
    api_case: str, invalid
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="setup-pod")
    if "alias" in api_case:
        kernel.create_alias(
            "alias", AliasTarget.pod("pod"), idempotency_key="setup-alias"
        )
    if api_case == "restore-pod-generation" or api_case == "restore-pod-revision":
        kernel.revoke_pod("pod", expected_generation=1, idempotency_key="setup-revoke")
    if api_case == "restore-alias":
        kernel.revoke_alias("alias", expected_generation=1, idempotency_key="setup-revoke")
    before = kernel.audit_checkpoint()

    calls = {
        "create-pod": lambda: kernel.create_pod(
            "new-pod", b"v", expected_generation=invalid, idempotency_key="bad"
        ),
        "edit-pod": lambda: kernel.edit_pod(
            "pod", b"v2", expected_generation=invalid, idempotency_key="bad"
        ),
        "revoke-pod": lambda: kernel.revoke_pod(
            "pod", expected_generation=invalid, idempotency_key="bad"
        ),
        "restore-pod-generation": lambda: kernel.restore_pod(
            "pod", expected_generation=invalid, idempotency_key="bad"
        ),
        "restore-pod-revision": lambda: kernel.restore_pod(
            "pod", expected_generation=2, payload_revision=invalid,
            idempotency_key="bad",
        ),
        "rollback-pod-generation": lambda: kernel.rollback_pod(
            "pod", to_payload_revision=1, expected_generation=invalid,
            idempotency_key="bad",
        ),
        "rollback-pod-revision": lambda: kernel.rollback_pod(
            "pod", to_payload_revision=invalid, expected_generation=1,
            idempotency_key="bad",
        ),
        "delete-pod": lambda: kernel.delete_pod(
            "pod", expected_generation=invalid, idempotency_key="bad"
        ),
        "create-alias": lambda: kernel.create_alias(
            "new-alias", AliasTarget.pod("pod"), expected_generation=invalid,
            idempotency_key="bad",
        ),
        "relink-alias": lambda: kernel.relink_alias(
            "alias", AliasTarget.pod("pod"), expected_generation=invalid,
            idempotency_key="bad",
        ),
        "revoke-alias": lambda: kernel.revoke_alias(
            "alias", expected_generation=invalid, idempotency_key="bad"
        ),
        "restore-alias": lambda: kernel.restore_alias(
            "alias", expected_generation=invalid, idempotency_key="bad"
        ),
        "delete-alias": lambda: kernel.delete_alias(
            "alias", expected_generation=invalid, idempotency_key="bad"
        ),
        "policy": lambda: kernel.set_policy_scope(
            [], expected_generation=invalid, idempotency_key="bad"
        ),
    }
    with pytest.raises((TypeError, ValueError), match="integer|generation|revision"):
        calls[api_case]()
    assert kernel.audit_checkpoint() == before

    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=before.terminal_hash,
        expected_length=before.length,
        allow_time_override=True,
    )
    assert replayed.audit_checkpoint() == before


@pytest.mark.parametrize(
    ("api_case", "invalid"),
    [
        ("create-pod", -1),
        ("create-alias", 1),
        ("edit-pod", 0),
        ("restore-pod-revision", 0),
        ("rollback-pod-revision", -1),
        ("policy", 0),
    ],
)
def test_live_integer_domains_enforce_create_zero_and_mutation_positive(
    api_case: str, invalid: int
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="setup-pod")
    if api_case == "create-alias":
        call = lambda: kernel.create_alias(
            "alias", AliasTarget.pod("pod"), expected_generation=invalid,
            idempotency_key="bad",
        )
    elif api_case == "create-pod":
        call = lambda: kernel.create_pod(
            "new", b"v", expected_generation=invalid, idempotency_key="bad"
        )
    elif api_case == "edit-pod":
        call = lambda: kernel.edit_pod(
            "pod", b"v2", expected_generation=invalid, idempotency_key="bad"
        )
    elif api_case == "restore-pod-revision":
        kernel.revoke_pod("pod", expected_generation=1, idempotency_key="revoke")
        call = lambda: kernel.restore_pod(
            "pod", expected_generation=2, payload_revision=invalid,
            idempotency_key="bad",
        )
    elif api_case == "rollback-pod-revision":
        call = lambda: kernel.rollback_pod(
            "pod", to_payload_revision=invalid, expected_generation=1,
            idempotency_key="bad",
        )
    else:
        call = lambda: kernel.set_policy_scope(
            [], expected_generation=invalid, idempotency_key="bad"
        )
    with pytest.raises((GenerationConflict, ValueError), match="generation|revision|create"):
        call()


def test_external_integer_fields_reject_boolean_equivalence() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness

    assert not kernel.validate_witness(replace(witness, pod_generation=True))
    assert not kernel.validate_witness(replace(witness, payload_revision=True))
    assert not kernel.validate_witness(replace(witness, policy_generation=True))
    assert not kernel.validate_witness(
        replace(
            witness,
            alias_path=(replace(witness.alias_path[0], authority_generation=True),),
        )
    )
    one_record = _kernel()
    one_record.create_pod("only", b"v", idempotency_key="only")
    checkpoint = one_record.audit_checkpoint()
    assert not LifecycleKernel.verify_audit(
        one_record.audit_records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=True,
    )


@pytest.mark.parametrize(
    "operation",
    [
        "create-pod", "edit-pod", "revoke-pod", "restore-pod", "rollback-pod",
        "delete-pod", "create-alias", "relink-alias", "revoke-alias",
        "restore-alias", "delete-alias", "policy",
    ],
)
def test_every_authentic_live_mutation_stream_is_replayable(operation: str) -> None:
    kernel = _kernel()
    if operation == "create-pod":
        kernel.create_pod("pod", b"v", idempotency_key="operation")
    else:
        kernel.create_pod("pod", b"v1", idempotency_key="setup-pod")
        if operation == "edit-pod":
            kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="operation")
        elif operation == "revoke-pod":
            kernel.revoke_pod("pod", expected_generation=1, idempotency_key="operation")
        elif operation == "restore-pod":
            kernel.revoke_pod("pod", expected_generation=1, idempotency_key="setup-revoke")
            kernel.restore_pod("pod", expected_generation=2, idempotency_key="operation")
        elif operation == "rollback-pod":
            kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="setup-edit")
            kernel.rollback_pod(
                "pod", to_payload_revision=1, expected_generation=2,
                idempotency_key="operation",
            )
        elif operation == "delete-pod":
            kernel.delete_pod("pod", expected_generation=1, idempotency_key="operation")
        elif operation == "policy":
            kernel.set_policy_scope(
                ["future"], expected_generation=1, idempotency_key="operation"
            )
        else:
            kernel.create_pod("pod-2", b"v2", idempotency_key="setup-pod-2")
            kernel.create_alias(
                "alias", AliasTarget.pod("pod"), idempotency_key="setup-alias"
            )
            if operation == "create-alias":
                kernel.create_alias(
                    "new-alias", AliasTarget.pod("pod"), idempotency_key="operation"
                )
            elif operation == "relink-alias":
                kernel.relink_alias(
                    "alias", AliasTarget.pod("pod-2"), expected_generation=1,
                    idempotency_key="operation",
                )
            elif operation == "revoke-alias":
                kernel.revoke_alias(
                    "alias", expected_generation=1, idempotency_key="operation"
                )
            elif operation == "restore-alias":
                kernel.revoke_alias(
                    "alias", expected_generation=1, idempotency_key="setup-revoke"
                )
                kernel.restore_alias(
                    "alias", expected_generation=2, idempotency_key="operation"
                )
            else:
                kernel.delete_alias(
                    "alias", expected_generation=1, idempotency_key="operation"
                )
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert replayed.audit_checkpoint() == checkpoint


def test_replay_default_restore_cannot_select_a_different_revision() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    kernel.revoke_pod("pod", expected_generation=2, idempotency_key="revoke")
    kernel.restore_pod("pod", expected_generation=3, idempotency_key="restore")
    records = list(kernel.audit_records)
    event = json.loads(records[-1].event_json)
    first = json.loads(records[0].event_json)["head"]
    event["head"]["payload_revision"] = 1
    event["head"]["payload_hash"] = first["payload_hash"]
    records[-1] = replace(
        records[-1], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="restore|revision"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_replay_rejects_backward_issuance_security_time() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=100.0
    )
    kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=110.0
    )
    records = list(kernel.audit_records)
    issuance_indexes = [
        index
        for index, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.CAPABILITY_ISSUANCE
    ]
    event = json.loads(records[issuance_indexes[-1]].event_json)
    event["security_time"] = 50.0
    records[issuance_indexes[-1]] = replace(
        records[issuance_indexes[-1]],
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="backward|monotone|security time"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_capability_mac_and_replayed_issuance_bind_issued_at() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    records = list(kernel.audit_records)
    index = next(
        i for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.CAPABILITY_ISSUANCE
    )
    event = json.loads(records[index].event_json)
    event["security_time"] = 50.0
    records[index] = replace(
        records[index], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)
    with pytest.raises(AuditIntegrityError, match="issuance|MAC|binding"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
            allow_time_override=True,
        )


def test_replay_reconstructs_derived_and_consumed_artifacts() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    kernel.derive_artifact("derived", b"cache", decisions=[decision])
    capability = kernel.issue_capability(
        decision.witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    kernel.consume_to_artifact(
        capability,
        artifact_id="consumed",
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    )
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )

    assert replayed.artifact_current("derived")
    assert replayed.artifact_current("consumed")
    assert replayed.prepare_publication("derived").artifact_id == "derived"


def test_replay_rejects_duplicate_consumed_artifact_id() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    for index in range(2):
        issued_at = 100.0 + index * 3
        capability = kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=20,
            now=issued_at,
        )
        kernel.consume_to_artifact(
            capability,
            artifact_id=f"artifact-{index}",
            principal="alice",
            namespace="tenant/acme",
            now=issued_at + 1,
        )
    records = list(kernel.audit_records)
    consumptions = [
        index
        for index, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.CONSUMPTION
    ]
    event = json.loads(records[consumptions[-1]].event_json)
    event["artifact_id"] = "artifact-0"
    records[consumptions[-1]] = replace(
        records[consumptions[-1]],
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="duplicate|artifact"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


@pytest.mark.parametrize("corruption", ["missing", "already-published", "stale"])
def test_replay_publish_requires_current_unpublished_artifact(corruption: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.derive_artifact("artifact", b"cache", decisions=[kernel.route("alias")])

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(kernel.prepare_publication("artifact"), Sink())
    records = list(kernel.audit_records)
    publish_index = next(
        i for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.PUBLICATION
    )
    if corruption == "missing":
        event = json.loads(records[publish_index].event_json)
        event["artifact_id"] = "missing"
        records[publish_index] = replace(
            records[publish_index],
            event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
        )
    elif corruption == "already-published":
        event = json.loads(records[publish_index].event_json)
        event["publication_id"] = "b" * 32
        duplicate = replace(
            records[publish_index],
            event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
        )
        records.append(duplicate)
    else:
        kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
        records = list(kernel.audit_records)
        publication = records.pop(publish_index)
        records.append(replace(publication, recorded_at=records[-1].recorded_at))
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="artifact|publication|stale"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_live_capability_nonce_collision_cannot_create_unreplayable_audit(
    monkeypatch,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    first = kernel.issue_capability(
        witness, principal="alice", namespace="tenant/acme", ttl_seconds=10, now=100.0
    )
    before = kernel.audit_checkpoint()
    monkeypatch.setattr("so.lifecycle.secrets.token_hex", lambda _size: first.nonce)

    with pytest.raises(CapabilityError, match="nonce|collision"):
        kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=101.0,
        )
    assert kernel.audit_checkpoint() == before
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=before.terminal_hash,
        expected_length=before.length,
    )
    assert replayed.audit_checkpoint() == before


def test_publication_id_collision_is_bound_to_one_artifact(monkeypatch) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    kernel.derive_artifact("one", b"one", decisions=[decision])
    kernel.derive_artifact("two", b"two", decisions=[decision])
    monkeypatch.setattr("so.lifecycle.secrets.token_hex", lambda _size: "a" * 32)
    prepared = kernel.prepare_publication("one")

    with pytest.raises(PublicationError, match="publication|collision"):
        kernel.prepare_publication("two")

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(prepared, Sink())
    with pytest.raises(PublicationError, match="authentication|artifact|bound"):
        kernel.commit_publication(replace(prepared, artifact_id="two"), Sink())


def test_replay_rejects_publication_id_rebound_to_another_artifact() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    kernel.derive_artifact("one", b"one", decisions=[decision])
    kernel.derive_artifact("two", b"two", decisions=[decision])

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(kernel.prepare_publication("one"), Sink())
    records = list(kernel.audit_records)
    original = next(
        record for record in records if record.receipt_kind is ReceiptKind.PUBLICATION
    )
    event = json.loads(original.event_json)
    event["artifact_id"] = "two"
    event["payload_hash"] = hashlib.sha256(b"two").hexdigest()
    rebound = replace(
        original,
        recorded_at=records[-1].recorded_at,
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records + [rebound])

    with pytest.raises(AuditIntegrityError, match="publication|duplicate|bound"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


@pytest.mark.parametrize("bad_time", [True, "100"])
@pytest.mark.parametrize("field", ["ttl", "issue-now", "consume-now"])
def test_live_security_times_reject_boolean_and_string(field: str, bad_time) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    if field == "ttl":
        call = lambda: kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=bad_time,
            now=100.0,
        )
    elif field == "issue-now":
        call = lambda: kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=bad_time,
        )
    else:
        capability = kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=100.0,
        )
        call = lambda: kernel.consume(
            capability,
            principal="alice",
            namespace="tenant/acme",
            now=bad_time,
        )
    with pytest.raises(CapabilityError, match="time|ttl|numeric"):
        call()


@pytest.mark.parametrize("bad_time", [True, "100"])
def test_trusted_clock_rejects_boolean_and_string_recorded_at(bad_time) -> None:
    kernel = _kernel(clock=lambda: bad_time)
    with pytest.raises(AuditIntegrityError, match="clock|numeric"):
        kernel.create_pod("pod", b"v", idempotency_key="pod")
    assert kernel.audit_records == ()


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
@pytest.mark.parametrize("bad_time", [True, "100"])
def test_serialized_capability_times_reject_boolean_and_string(
    field: str, bad_time
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    with pytest.raises(CapabilityError, match="time"):
        kernel.consume(
            replace(capability, **{field: bad_time}),
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )


@pytest.mark.parametrize("bad_time", [True, "100"])
def test_replay_rejects_non_numeric_recorded_at(bad_time) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    record = replace(kernel.audit_records[0], recorded_at=bad_time)
    signed = _resign([record])
    assert not LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=1,
    )


def test_explicit_equal_audit_and_capability_secrets_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct|separate"):
        LifecycleKernel(
            namespace="tenant/acme",
            capability_secret=b"z" * 32,
            audit_secret=b"z" * 32,
        )


def test_artifact_identifier_collision_does_not_spend_capability_or_append() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    kernel.derive_artifact("artifact", b"derived", decisions=[decision])
    capability = kernel.issue_capability(
        decision.witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    before = kernel.audit_checkpoint()
    with pytest.raises(ValueError, match="already exists"):
        kernel.consume_to_artifact(
            capability,
            artifact_id="artifact",
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )
    assert kernel.audit_checkpoint() == before
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"


def test_replay_plain_consume_revalidates_witness_at_event_position() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=20,
        now=100.0,
    )
    kernel.consume(
        capability, principal="alice", namespace="tenant/acme", now=101.0
    )
    kernel.edit_pod("pod", b"v2", expected_generation=1, idempotency_key="edit")
    records = list(kernel.audit_records)
    consume_index = next(
        i for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.CONSUMPTION
    )
    consumption = records.pop(consume_index)
    records.append(replace(consumption, recorded_at=records[-1].recorded_at))
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="witness|stale|consumption"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_audit_binds_max_hops_and_replay_rejects_different_configuration() -> None:
    kernel = _kernel(max_hops=2)
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("leaf", AliasTarget.pod("pod"), idempotency_key="leaf")
    kernel.create_alias("root", AliasTarget.alias("leaf"), idempotency_key="root")
    kernel.route("root")
    checkpoint = kernel.audit_checkpoint()

    with pytest.raises(AuditIntegrityError, match="configuration|max_hops"):
        LifecycleKernel.replay(
            kernel.audit_records,
            namespace="tenant/acme",
            max_hops=3,
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=checkpoint.terminal_hash,
            expected_length=checkpoint.length,
        )


_BAD_RANDOM_IDS = ("A" * 32, "a" * 30 + "  ", "a" * 31, "a" * 33, "g" * 32)


@pytest.mark.parametrize("bad_id", _BAD_RANDOM_IDS)
@pytest.mark.parametrize("kind", ["nonce", "publication"])
def test_replay_rejects_noncanonical_random_identifiers(kind: str, bad_id: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    if kind == "nonce":
        kernel.issue_capability(
            decision.witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=100.0,
        )
        receipt_kind = ReceiptKind.CAPABILITY_ISSUANCE
        field = "nonce"
    else:
        kernel.derive_artifact("artifact", b"v", decisions=[decision])

        class Sink:
            def commit(self, _publication_id: str, _payload: bytes) -> bool:
                return True

        kernel.commit_publication(kernel.prepare_publication("artifact"), Sink())
        receipt_kind = ReceiptKind.PUBLICATION
        field = "publication_id"
    records = list(kernel.audit_records)
    index = next(
        i for i, record in enumerate(records) if record.receipt_kind is receipt_kind
    )
    event = json.loads(records[index].event_json)
    event[field] = bad_id
    records[index] = replace(
        records[index], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


@pytest.mark.parametrize("bad_id", _BAD_RANDOM_IDS)
@pytest.mark.parametrize("operation", ["issue", "prepare"])
def test_live_random_identifier_generator_must_return_canonical_128_bits(
    monkeypatch, operation: str, bad_id: str
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    monkeypatch.setattr("so.lifecycle.secrets.token_hex", lambda _size: bad_id)
    if operation == "issue":
        with pytest.raises(CapabilityError, match="nonce|random"):
            kernel.issue_capability(
                decision.witness,
                principal="alice",
                namespace="tenant/acme",
                ttl_seconds=10,
                now=100.0,
            )
    else:
        kernel.derive_artifact("artifact", b"v", decisions=[decision])
        with pytest.raises(PublicationError, match="publication|random"):
            kernel.prepare_publication("artifact")


@pytest.mark.parametrize(
    "bad_digest", ["A" * 64, "a" * 62 + "  ", "a" * 63, "a" * 65, "g" * 64]
)
def test_audit_digest_requires_exact_lowercase_hex(bad_digest: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    forged = [replace(kernel.audit_records[0], record_hash=bad_digest)]
    assert not LifecycleKernel.verify_audit(
        forged,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=bad_digest,
        expected_length=1,
    )


@pytest.mark.parametrize("corruption", ["decision", "reason", "policy", "witness"])
def test_replay_semantically_validates_resolution_receipt(corruption: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.route("alias")
    records = list(kernel.audit_records)
    index = next(
        i for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.RESOLUTION
    )
    event = json.loads(records[index].event_json)
    if corruption == "decision":
        event["decision"] = RouteKind.BYPASS.value
    elif corruption == "reason":
        event["reason"] = "FORGED"
    elif corruption == "policy":
        event["policy_generation"] = 2
    else:
        event["witness"]["pod_generation"] = 2
    records[index] = replace(
        records[index], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="resolution|route|witness"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_replay_recomputes_verifier_verdict_from_audited_inputs() -> None:
    kernel = _kernel()
    kernel.record_verifier_run(
        (
            VerifierResult("control", VerifierVerdict.PASS, True),
            VerifierResult("workspace", VerifierVerdict.PASS, True),
        )
    )
    records = list(kernel.audit_records)
    event = json.loads(records[0].event_json)
    event["inputs"][1]["verdict"] = VerifierVerdict.FAIL.value
    records[0] = replace(
        records[0], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="verifier|verdict"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_audit_never_materializes_bearer_capability_mac_and_replay_still_consumes() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"secret", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    checkpoint = kernel.audit_checkpoint()
    for record in kernel.audit_records:
        event = json.loads(record.event_json)
        assert "capability_mac" not in event
        assert capability.mac not in record.event_json
    issuance = next(
        json.loads(record.event_json)
        for record in kernel.audit_records
        if record.receipt_kind is ReceiptKind.CAPABILITY_ISSUANCE
    )
    assert issuance["capability_commitment"] != capability.mac
    with pytest.raises(CapabilityError, match="authentication"):
        kernel.consume(
            replace(capability, mac=issuance["capability_commitment"]),
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )

    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        allow_time_override=True,
    )
    assert replayed.consume(
        capability, principal="alice", namespace="tenant/acme", now=101.0
    ) == b"secret"
    consumption = replayed.audit_records[-1]
    assert consumption.receipt_kind is ReceiptKind.CONSUMPTION
    assert "capability_mac" not in json.loads(consumption.event_json)
    assert capability.mac not in consumption.event_json


def test_replay_accepts_trusted_clock_for_post_recovery_append() -> None:
    future = 4_000_000_000.0
    kernel = _kernel(clock=lambda: future)
    kernel.create_pod("pod", b"v1", idempotency_key="pod")
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        clock=lambda: future + 1,
    )
    replayed.edit_pod(
        "pod", b"v2", expected_generation=1, idempotency_key="post-recovery"
    )
    new_checkpoint = replayed.audit_checkpoint()
    assert LifecycleKernel.verify_audit(
        replayed.audit_records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=new_checkpoint.terminal_hash,
        expected_length=new_checkpoint.length,
    )


def test_route_rejects_invalid_alias_before_creating_receipt() -> None:
    kernel = _kernel()
    before = kernel.audit_checkpoint()
    with pytest.raises(ValueError, match="alias_id"):
        kernel.route(None)
    assert kernel.audit_checkpoint() == before


@pytest.mark.parametrize("error_type", [OSError, ValueError, TypeError, KeyError])
def test_transient_authority_error_resolution_is_fail_closed_and_replayable(
    error_type: type[Exception],
) -> None:
    class OneShotClock:
        def __init__(self) -> None:
            self.fail = False

        def __call__(self) -> float:
            if self.fail:
                self.fail = False
                raise error_type("transient clock failure")
            return 100.0

    clock = OneShotClock()
    kernel = _kernel(clock=clock)
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    clock.fail = True
    decision = kernel.route("alias")
    assert decision.kind is RouteKind.UNKNOWN
    assert decision.reason == "AUTHORITY_ERROR"
    checkpoint = kernel.audit_checkpoint()

    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        clock=lambda: 101.0,
    )
    assert replayed.audit_checkpoint() == checkpoint
    assert replayed.route("alias").kind is RouteKind.RESOLVE


def test_authority_error_receipt_uses_one_locked_policy_epoch() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    main_thread = get_ident()
    mutate = Event()
    mutated = Event()

    class InterposingLock:
        def __init__(self) -> None:
            self.inner = RLock()
            self.failed_exit = False
            self.interposed = False

        def __enter__(self):
            if (
                self.failed_exit
                and not self.interposed
                and get_ident() == main_thread
            ):
                self.interposed = True
                mutate.set()
                assert mutated.wait(2)
            self.inner.acquire()
            return self

        def __exit__(self, exc_type, _exc, _traceback) -> None:
            self.inner.release()
            if exc_type is not None:
                self.failed_exit = True

    base_time = kernel.audit_records[-1].recorded_at

    class OneShotClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> float:
            self.calls += 1
            if self.calls == 1:
                raise OSError("injected route audit failure")
            return base_time + self.calls

    kernel._lock = InterposingLock()
    kernel._clock = OneShotClock()

    def mutate_policy() -> None:
        assert mutate.wait(2)
        kernel.set_policy_scope(
            ["alias"], expected_generation=1, idempotency_key="policy-race"
        )
        mutated.set()

    worker = Thread(target=mutate_policy)
    worker.start()
    decision = kernel.route("alias")
    worker.join(2)
    assert not worker.is_alive()
    assert decision.kind is RouteKind.UNKNOWN
    assert decision.dependency.policy_generation == 2
    event = json.loads(kernel.audit_records[-1].event_json)
    assert event["policy_generation"] == 2
    assert event["dependency"]["value"]["policy_generation"] == 2

    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
        clock=lambda: base_time + 10,
    )
    assert replayed.audit_checkpoint() == checkpoint


@pytest.mark.parametrize("field", ["engine_id", "code_version", "key_id"])
@pytest.mark.parametrize("bad_value", [1, True, ""])
def test_constructor_audit_metadata_requires_nonempty_strings(
    field: str, bad_value
) -> None:
    kwargs = {
        "namespace": "tenant/acme",
        "capability_secret": CAPABILITY_SECRET,
        "audit_secret": AUDIT_SECRET,
        field: bad_value,
    }
    with pytest.raises(ValueError, match="engine_id|code_version|key_id|metadata"):
        LifecycleKernel(**kwargs)


@pytest.mark.parametrize("bad_value", [0, 1, "true", None])
def test_constructor_requires_exact_bool_time_override(bad_value) -> None:
    with pytest.raises((TypeError, ValueError), match="allow_time_override|bool"):
        LifecycleKernel(
            namespace="tenant/acme",
            capability_secret=CAPABILITY_SECRET,
            audit_secret=AUDIT_SECRET,
            allow_time_override=bad_value,
        )


@pytest.mark.parametrize("bad_clock", [0, 1, True, "clock"])
def test_constructor_requires_callable_clock(bad_clock) -> None:
    with pytest.raises((TypeError, ValueError), match="clock|callable"):
        LifecycleKernel(
            namespace="tenant/acme",
            capability_secret=CAPABILITY_SECRET,
            audit_secret=AUDIT_SECRET,
            clock=bad_clock,
        )


def test_custom_string_audit_metadata_produces_replayable_stream() -> None:
    kernel = LifecycleKernel(
        namespace="tenant/acme",
        capability_secret=CAPABILITY_SECRET,
        audit_secret=AUDIT_SECRET,
        engine_id="engine/custom",
        code_version="reference/custom",
        key_id="key/custom",
    )
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        capability_secret=CAPABILITY_SECRET,
        audit_secret=AUDIT_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert replayed.pod_head("pod") == kernel.pod_head("pod")


@pytest.mark.parametrize("field", ["engine_id", "code_version"])
def test_audit_rejects_resigned_midstream_metadata_change(field: str) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    kernel.edit_pod(
        "pod", b"v2", expected_generation=1, idempotency_key="edit"
    )
    records = list(kernel.audit_records)
    records[1] = replace(records[1], **{field: "attacker-v2"})
    signed = _resign(records)

    assert not LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=len(signed),
    )
    with pytest.raises(AuditIntegrityError, match="checkpoint verification"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_replay_and_verify_snapshot_external_record_sequence_once() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    good = kernel.audit_records[0]
    event = json.loads(good.event_json)
    event["command_body"]["payload_hex"] = b"evil".hex()
    event["new_revision"]["payload_hex"] = b"evil".hex()
    event["head"]["payload_hash"] = hashlib.sha256(b"evil").hexdigest()
    event["command_fingerprint"] = hashlib.sha256(
        json.dumps(
            {"operation": "CREATE_POD", "body": event["command_body"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    unsigned_evil = replace(
        good, event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )

    class SwitchingRecords:
        def __init__(self) -> None:
            self.iterations = 0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            if index in (0, -1):
                return good
            raise IndexError(index)

        def __iter__(self):
            self.iterations += 1
            return iter((good,) if self.iterations != 2 else (unsigned_evil,))

    checkpoint = kernel.audit_checkpoint()
    verify_records = SwitchingRecords()
    assert LifecycleKernel.verify_audit(
        verify_records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert verify_records.iterations == 1

    replay_records = SwitchingRecords()
    replayed = LifecycleKernel.replay(
        replay_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert replay_records.iterations == 1
    assert replayed._payloads["pod"][1] == b"v1"
    assert replayed.audit_records == (good,)


def test_replay_rejects_stateful_non_record_element_before_field_reads() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v1", idempotency_key="create")
    good = kernel.audit_records[0]

    class ShapeShifter:
        def __init__(self) -> None:
            self.event_reads = 0

        def __getattr__(self, name: str):
            if name == "event_json":
                self.event_reads += 1
            return getattr(good, name)

    shifter = ShapeShifter()
    assert not LifecycleKernel.verify_audit(
        [shifter],
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=good.record_hash,
        expected_length=1,
    )
    assert shifter.event_reads == 0
    with pytest.raises(AuditIntegrityError, match="snapshot"):
        LifecycleKernel.replay(
            [shifter],
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=good.record_hash,
            expected_length=1,
        )
    assert shifter.event_reads == 0


@pytest.mark.parametrize("bad_principal", [1, True, b"alice", ""])
def test_issue_rejects_nonstring_principal_before_security_time(
    bad_principal: object,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness

    with pytest.raises(CapabilityError, match="principal"):
        kernel.issue_capability(
            witness,
            principal=bad_principal,
            namespace="tenant/acme",
            ttl_seconds=10,
            now=200.0,
        )
    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    assert capability.principal == "alice"


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("principal", 1),
        ("namespace", True),
        ("nonce", "A" * 32),
        ("mac", "A" * 64),
    ],
)
def test_consume_rejects_malformed_public_fields_before_security_time(
    field: str, bad: object
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    forged = replace(capability, **{field: bad})

    with pytest.raises(CapabilityError, match="capability|principal|namespace|nonce|MAC"):
        kernel.consume(
            forged,
            principal="alice",
            namespace="tenant/acme",
            now=200.0,
        )
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"


@pytest.mark.parametrize("secret_kind", ["secret", "audit_secret", "capability_secret"])
def test_constructor_rejects_secrets_shorter_than_32_bytes(secret_kind: str) -> None:
    kwargs = {"namespace": "tenant/acme"}
    if secret_kind == "secret":
        kwargs["secret"] = b"short"
    else:
        kwargs["audit_secret"] = AUDIT_SECRET
        kwargs["capability_secret"] = CAPABILITY_SECRET
        kwargs[secret_kind] = b"short"
    with pytest.raises(ValueError, match="32 bytes"):
        LifecycleKernel(**kwargs)


def test_huge_numeric_inputs_fail_closed_without_overflow() -> None:
    huge = 10**1000
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    with pytest.raises(CapabilityError, match="ttl_seconds"):
        kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=huge,
            now=100.0,
        )
    with pytest.raises(CapabilityError, match="security time"):
        kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=huge,
        )

    bad_clock = _kernel(clock=lambda: huge)
    with pytest.raises(AuditIntegrityError, match="clock"):
        bad_clock.create_pod("pod", b"v", idempotency_key="pod")

    record = replace(kernel.audit_records[0], recorded_at=huge)
    assert not LifecycleKernel.verify_audit(
        [record],
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=record.record_hash,
        expected_length=1,
    )


@pytest.mark.parametrize("field", ["issued_at", "expires_at"])
def test_huge_serialized_capability_time_is_rejected_without_advancing_floor(
    field: str,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    with pytest.raises(CapabilityError, match="capability"):
        kernel.consume(
            replace(capability, **{field: 10**1000}),
            principal="alice",
            namespace="tenant/acme",
            now=200.0,
        )
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"


def test_public_capability_and_witness_subclasses_are_rejected() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness

    class ShiftingWitness(PathWitness):
        pass

    forged_witness = ShiftingWitness(
        witness.namespace,
        witness.root_alias_id,
        witness.alias_path,
        witness.pod_id,
        witness.pod_generation,
        witness.payload_revision,
        witness.payload_hash,
        witness.policy_generation,
    )
    with pytest.raises(CapabilityError, match="witness"):
        kernel.issue_capability(
            forged_witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=100.0,
        )

    capability = kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )

    class ShiftingCapability(type(capability)):
        pass

    forged_capability = ShiftingCapability(
        capability.witness,
        capability.principal,
        capability.namespace,
        capability.nonce,
        capability.issued_at,
        capability.expires_at,
        capability.mac,
    )
    with pytest.raises(CapabilityError, match="capability"):
        kernel.consume(
            forged_capability,
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )


def test_public_decision_dependency_and_target_subclasses_are_rejected() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")

    class ShiftingDecision(type(decision)):
        pass

    with pytest.raises(TypeError, match="decision"):
        kernel.derive_artifact(
            "decision",
            b"unsafe",
            decisions=[
                ShiftingDecision(
                    decision.kind,
                    decision.witness,
                    decision.dependency,
                    decision.reason,
                )
            ],
        )

    outside = kernel.route("outside")

    class ShiftingDependency(OutOfScopeDependency):
        pass

    dependency = outside.dependency
    forged_dependency = ShiftingDependency(
        dependency.namespace, dependency.alias_id, dependency.policy_generation
    )
    with pytest.raises(TypeError, match="decision"):
        kernel.derive_artifact(
            "dependency",
            b"unsafe",
            decisions=[replace(outside, dependency=forged_dependency)],
        )

    class ShiftingTarget(AliasTarget):
        pass

    with pytest.raises(TypeError, match="target"):
        kernel.create_alias(
            "forged-target",
            ShiftingTarget(TargetKind.POD, "pod"),
            idempotency_key="forged-target",
        )


def test_public_prepared_publication_and_verifier_subclasses_are_rejected() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.derive_artifact("artifact", b"v", decisions=[kernel.route("alias")])
    prepared = kernel.prepare_publication("artifact")

    class ShiftingPublication(type(prepared)):
        pass

    forged = ShiftingPublication(
        prepared.publication_id,
        prepared.artifact_id,
        prepared.namespace,
        prepared.payload_hash,
        prepared.lineage_hash,
        prepared.mac,
    )
    sink_calls = 0

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            nonlocal sink_calls
            sink_calls += 1
            return True

    with pytest.raises(PublicationError, match="canonical"):
        kernel.commit_publication(forged, Sink())
    assert sink_calls == 0

    class ShiftingVerifier(VerifierResult):
        pass

    with pytest.raises(TypeError, match="VerifierResult|verifier result"):
        kernel.record_verifier_run(
            ShiftingVerifier("control", VerifierVerdict.PASS, True)
        )


def test_canonical_but_unauthenticated_capability_does_not_poison_time_floor() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    with pytest.raises(CapabilityError, match="authentication"):
        kernel.consume(
            replace(capability, mac="0" * 64),
            principal="alice",
            namespace="tenant/acme",
            now=1e300,
        )
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"


def test_ttl_must_advance_representable_expiry_without_poisoning_time_floor() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    with pytest.raises(CapabilityError, match="ttl_seconds|later expiry"):
        kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=1e-10,
            now=1e9,
        )
    assert kernel.issue_capability(
        witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    ).issued_at == 100.0


def test_constructor_and_object_ids_reject_scalar_subclasses() -> None:
    class IntSubclass(int):
        pass

    class StrSubclass(str):
        pass

    with pytest.raises(ValueError, match="max_hops"):
        LifecycleKernel(
            namespace="tenant/acme",
            max_hops=IntSubclass(8),
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
        )
    with pytest.raises(ValueError, match="namespace"):
        LifecycleKernel(
            namespace=StrSubclass("tenant/acme"),
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
        )
    kernel = _kernel()
    with pytest.raises(ValueError, match="pod_id"):
        kernel.create_pod(
            StrSubclass("pod"), b"v", idempotency_key="subclass-id"
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "mutation-operation",
        "mutation-head-status",
        "alias-target-kind",
        "resolution-decision",
        "verifier-verdict",
        "nested-verifier-verdict",
    ],
)
def test_audit_schema_rejects_unhashable_enum_and_nested_fields(
    corruption: str,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.route("alias")
    kernel.record_verifier_run(
        (
            VerifierResult("control", VerifierVerdict.PASS, True),
            VerifierResult("workspace", VerifierVerdict.PASS, True),
        )
    )
    records = list(kernel.audit_records)
    if corruption.startswith("mutation"):
        index = 0
    elif corruption == "alias-target-kind":
        index = 1
    elif corruption == "resolution-decision":
        index = 2
    else:
        index = 3
    event = json.loads(records[index].event_json)
    if corruption == "mutation-operation":
        event["operation"] = []
    elif corruption == "mutation-head-status":
        event["head"]["status"] = []
    elif corruption == "alias-target-kind":
        event["head"]["target"]["kind"] = []
    elif corruption == "resolution-decision":
        event["decision"] = []
    elif corruption == "verifier-verdict":
        event["verdict"] = []
    else:
        event["inputs"][0]["verdict"] = []
    records[index] = replace(
        records[index],
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records)

    assert not LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=len(signed),
    )
    with pytest.raises(AuditIntegrityError, match="verification"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_audit_parser_rejects_oversized_integer_without_leaking_value_error() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    record = replace(
        kernel.audit_records[0], event_json='{"operation":' + "9" * 5000 + "}"
    )
    signed = _resign([record])

    assert not LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=1,
    )
    with pytest.raises(AuditIntegrityError, match="verification"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=1,
        )


def test_replay_wraps_schema_valid_but_impossible_alias_transition() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    alias_record = next(
        record
        for record in kernel.audit_records
        if json.loads(record.event_json)["operation"] == "CREATE_ALIAS"
    )
    signed = _resign([alias_record])
    assert LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=1,
    )

    with pytest.raises(AuditIntegrityError, match="semantic validation"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=1,
        )


def test_consume_to_artifact_collision_preserves_existing_artifact() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    existing = kernel.derive_artifact("artifact", b"derived", decisions=[decision])
    capability = kernel.issue_capability(
        decision.witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )

    with pytest.raises(ValueError, match="already exists"):
        kernel.consume_to_artifact(
            capability,
            artifact_id="artifact",
            principal="alice",
            namespace="tenant/acme",
            now=101.0,
        )
    assert kernel._artifacts["artifact"] == existing
    assert kernel.artifact_current("artifact")
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"


def test_failed_reconsume_to_artifact_does_not_unspend_nonce() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    capability = kernel.issue_capability(
        kernel.route("alias").witness,
        principal="alice",
        namespace="tenant/acme",
        ttl_seconds=10,
        now=100.0,
    )
    assert kernel.consume(
        capability,
        principal="alice",
        namespace="tenant/acme",
        now=101.0,
    ) == b"v"

    with pytest.raises(CapabilityError, match="already consumed"):
        kernel.consume_to_artifact(
            capability,
            artifact_id="artifact",
            principal="alice",
            namespace="tenant/acme",
            now=102.0,
        )
    with pytest.raises(CapabilityError, match="already consumed"):
        kernel.consume(
            capability,
            principal="alice",
            namespace="tenant/acme",
            now=102.0,
        )


def test_huge_audit_envelope_max_hops_is_total_and_fail_closed() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    record = replace(kernel.audit_records[0], max_hops=10**5000)
    assert not LifecycleKernel.verify_audit(
        [record],
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=record.record_hash,
        expected_length=1,
    )
    with pytest.raises(AuditIntegrityError, match="verification"):
        LifecycleKernel.replay(
            [record],
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=record.record_hash,
            expected_length=1,
        )


def test_audit_clock_cannot_observe_provisional_mutation_through_public_reads() -> None:
    observations: list[object] = []
    blocked: list[str] = []
    kernel: LifecycleKernel

    def hostile_clock() -> float:
        readers = {
            "pod_head": lambda: kernel.pod_head("pod"),
            "read_pod": lambda: kernel.read_pod("pod"),
            "audit_checkpoint": kernel.audit_checkpoint,
            "audit_records": lambda: kernel.audit_records,
            "policy_head": kernel.policy_head,
        }
        for name, reader in readers.items():
            try:
                observations.append(reader())
            except AuditIntegrityError:
                blocked.append(name)
        raise OSError("abort audit after attempted re-entry")

    kernel = LifecycleKernel(
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        clock=hostile_clock,
    )
    with pytest.raises(OSError, match="abort audit"):
        kernel.create_pod("pod", b"secret", idempotency_key="pod")
    assert observations == []
    assert set(blocked) == {
        "pod_head", "read_pod", "audit_checkpoint", "audit_records", "policy_head"
    }
    assert kernel.audit_records == ()
    with pytest.raises(KeyError):
        kernel.pod_head("pod")


@pytest.mark.parametrize(
    ("source", "path", "bad"),
    [
        ("resolve", ("witness", "alias_path", 0, "authority_generation"), True),
        ("resolve", ("witness", "alias_path", 0, "authority_generation"), 1.0),
        ("resolve", ("witness", "pod_generation"), True),
        ("resolve", ("witness", "pod_generation"), 1.0),
        ("resolve", ("witness", "payload_revision"), True),
        ("resolve", ("witness", "policy_generation"), 1.0),
        ("resolve", ("witness", "root_alias_id"), True),
        ("resolve", ("witness", "alias_path", 0, "alias_id"), True),
        ("resolve", ("witness", "alias_path", 0, "target", "kind"), "BOGUS"),
        ("resolve", ("witness", "alias_path", 0, "target", "object_id"), True),
        ("resolve", ("witness", "pod_id"), True),
        ("resolve", ("witness", "payload_hash"), "A" * 64),
        ("resolve", ("dependency", "value", "pod_generation"), True),
        ("resolve", ("dependency", "value", "payload_revision"), 1.0),
        ("resolve", ("dependency", "kind"), "BOGUS"),
        ("bypass", ("dependency", "value", "policy_generation"), True),
        ("bypass", ("dependency", "value", "policy_generation"), 1.0),
        ("bypass", ("dependency", "value", "alias_id"), True),
        ("unknown", ("dependency", "value", "policy_generation"), True),
        ("unknown", ("dependency", "value", "namespace"), True),
    ],
)
def test_resolution_replay_rejects_nested_noncanonical_scalar_tampering(
    source: str, path: tuple[object, ...], bad: object
) -> None:
    kernel = _kernel()
    if source == "resolve":
        kernel.create_pod("pod", b"v", idempotency_key="pod")
        kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
        kernel.route("alias")
        alias_id = "alias"
    elif source == "bypass":
        kernel.route("outside")
        alias_id = "outside"
    else:
        kernel.set_policy_scope(
            ["missing"], expected_generation=1, idempotency_key="policy"
        )
        kernel.route("missing")
        alias_id = "missing"
    records = list(kernel.audit_records)
    index = next(
        i
        for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.RESOLUTION
        and json.loads(record.event_json)["alias_id"] == alias_id
    )
    event = json.loads(records[index].event_json)
    cursor = event
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = bad
    records[index] = replace(
        records[index],
        event_json=json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    signed = _resign(records)
    assert LifecycleKernel.verify_audit(
        signed,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=signed[-1].record_hash,
        expected_length=len(signed),
    )

    with pytest.raises(AuditIntegrityError, match="resolution|lineage|witness"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )


def test_route_never_returns_an_unaudited_decision_when_audit_stays_down() -> None:
    class FailingClock:
        fail = False

        def __call__(self) -> float:
            if self.fail:
                raise OSError("persistent audit clock failure")
            return 100.0

    clock = FailingClock()
    kernel = _kernel(clock=clock)
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    before = kernel.audit_checkpoint()
    clock.fail = True

    with pytest.raises(AuditIntegrityError, match="audit|route"):
        kernel.route("alias")
    assert kernel.audit_checkpoint() == before
    assert LifecycleKernel.verify_audit(
        kernel.audit_records,
        audit_secret=AUDIT_SECRET,
        namespace="tenant/acme",
        expected_terminal_hash=before.terminal_hash,
        expected_length=before.length,
    )


def test_keyboard_interrupt_rolls_back_entire_mutation() -> None:
    class InterruptingClock:
        interrupt = True

        def __call__(self) -> float:
            if self.interrupt:
                raise KeyboardInterrupt
            return 100.0

    clock = InterruptingClock()
    kernel = LifecycleKernel(
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        clock=clock,
    )
    with pytest.raises(KeyboardInterrupt):
        kernel.create_pod("pod", b"secret", idempotency_key="pod")
    assert kernel.audit_records == ()
    with pytest.raises(KeyError):
        kernel.pod_head("pod")

    clock.interrupt = False
    assert kernel.create_pod(
        "pod", b"secret", idempotency_key="pod"
    ).authority_generation == 1


def test_system_exit_after_audit_append_rolls_back_derived_artifact(
    monkeypatch,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    before = kernel.audit_checkpoint()
    original_commit = kernel._commit_audit

    def append_then_interrupt(record) -> None:
        original_commit(record)
        raise SystemExit("interrupt after append")

    monkeypatch.setattr(kernel, "_commit_audit", append_then_interrupt)
    with pytest.raises(SystemExit, match="after append"):
        kernel.derive_artifact("artifact", b"derived", decisions=[decision])
    assert kernel.audit_checkpoint() == before
    assert not kernel.artifact_current("artifact")

    monkeypatch.undo()
    kernel.derive_artifact("artifact", b"derived", decisions=[decision])

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(kernel.prepare_publication("artifact"), Sink())
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    assert replayed.audit_checkpoint() == checkpoint


@pytest.mark.parametrize("operation", ["issue", "consume", "consume_to_artifact"])
def test_base_exception_after_audit_append_rolls_back_capability_unit(
    operation: str, monkeypatch
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    witness = kernel.route("alias").witness
    capability = None
    if operation != "issue":
        capability = kernel.issue_capability(
            witness,
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=100.0,
        )
    before = kernel.audit_checkpoint()
    original_commit = kernel._commit_audit

    def append_then_interrupt(record) -> None:
        original_commit(record)
        raise KeyboardInterrupt

    monkeypatch.setattr(kernel, "_commit_audit", append_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        if operation == "issue":
            kernel.issue_capability(
                witness,
                principal="alice",
                namespace="tenant/acme",
                ttl_seconds=10,
                now=100.0,
            )
        elif operation == "consume":
            kernel.consume(
                capability,
                principal="alice",
                namespace="tenant/acme",
                now=101.0,
            )
        else:
            kernel.consume_to_artifact(
                capability,
                artifact_id="artifact",
                principal="alice",
                namespace="tenant/acme",
                now=101.0,
            )
    assert kernel.audit_checkpoint() == before
    if operation == "issue":
        assert kernel._issued_capabilities == {}
    else:
        assert capability.nonce not in kernel._spent_nonces
        assert "artifact" not in kernel._artifacts


def test_publication_interrupt_restores_receipt_and_internal_markers(
    monkeypatch,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.derive_artifact("artifact", b"derived", decisions=[kernel.route("alias")])
    prepared = kernel.prepare_publication("artifact")
    before = kernel.audit_checkpoint()
    effects: dict[str, bytes] = {}

    class Sink:
        def commit(self, publication_id: str, payload: bytes) -> bool:
            effects.setdefault(publication_id, payload)
            return True

    original_commit = kernel._commit_audit

    def append_then_interrupt(record) -> None:
        original_commit(record)
        raise SystemExit("interrupt publication")

    monkeypatch.setattr(kernel, "_commit_audit", append_then_interrupt)
    with pytest.raises(SystemExit, match="publication"):
        kernel.commit_publication(prepared, Sink())
    assert kernel.audit_checkpoint() == before
    assert prepared.publication_id not in kernel._published
    assert prepared.artifact_id not in kernel._published_artifacts
    assert prepared.publication_id not in kernel._publication_bindings
    assert effects == {prepared.publication_id: b"derived"}

    monkeypatch.undo()
    kernel.commit_publication(prepared, Sink())
    assert effects == {prepared.publication_id: b"derived"}


@pytest.mark.parametrize("corruption", ["alias-id", "policy-generation", "path-generation"])
def test_public_forged_dependency_objects_are_rejected_before_audit(
    corruption: str,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    if corruption == "alias-id":
        dependency = OutOfScopeDependency("tenant/acme", True, 1)
        forged = replace(decision, kind=RouteKind.BYPASS, witness=None, dependency=dependency)
    elif corruption == "policy-generation":
        dependency = OutOfScopeDependency("tenant/acme", "future", True)
        forged = replace(decision, kind=RouteKind.BYPASS, witness=None, dependency=dependency)
    else:
        witness = replace(
            decision.witness,
            alias_path=(
                replace(decision.witness.alias_path[0], authority_generation=True),
            ),
        )
        forged = replace(decision, witness=witness, dependency=witness)
    before = kernel.audit_checkpoint()

    with pytest.raises((TypeError, ValueError), match="decision|dependency|lineage"):
        kernel.derive_artifact("forged", b"unsafe", decisions=[forged])
    assert kernel.audit_checkpoint() == before
    assert not kernel.artifact_current("forged")


def test_public_forged_witness_and_prepared_publication_are_rejected() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    decision = kernel.route("alias")
    with pytest.raises(CapabilityError, match="witness"):
        kernel.issue_capability(
            replace(decision.witness, payload_revision=True),
            principal="alice",
            namespace="tenant/acme",
            ttl_seconds=10,
            now=100.0,
        )
    kernel.derive_artifact("artifact", b"v", decisions=[decision])
    prepared = kernel.prepare_publication("artifact")

    class Sink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    with pytest.raises(PublicationError, match="prepared|artifact|invalid"):
        kernel.commit_publication(replace(prepared, artifact_id=True), Sink())


def test_replay_restores_publication_binding_for_idempotent_retry() -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    kernel.derive_artifact("artifact", b"v", decisions=[kernel.route("alias")])
    prepared = kernel.prepare_publication("artifact")

    class InitialSink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            return True

    kernel.commit_publication(prepared, InitialSink())
    checkpoint = kernel.audit_checkpoint()
    replayed = LifecycleKernel.replay(
        kernel.audit_records,
        namespace="tenant/acme",
        audit_secret=AUDIT_SECRET,
        capability_secret=CAPABILITY_SECRET,
        expected_terminal_hash=checkpoint.terminal_hash,
        expected_length=checkpoint.length,
    )
    calls = 0

    class RetrySink:
        def commit(self, _publication_id: str, _payload: bytes) -> bool:
            nonlocal calls
            calls += 1
            return True

    replayed.commit_publication(prepared, RetrySink())
    assert calls == 0
    assert replayed.audit_checkpoint() == checkpoint


@pytest.mark.parametrize("future_kind", ["policy", "alias", "pod"])
def test_derive_rejects_well_typed_future_or_noncurrent_dependency(
    future_kind: str,
) -> None:
    kernel = _kernel()
    kernel.create_pod("pod", b"v", idempotency_key="pod")
    kernel.create_alias("alias", AliasTarget.pod("pod"), idempotency_key="alias")
    resolved = kernel.route("alias")
    if future_kind == "policy":
        dependency = OutOfScopeDependency("tenant/acme", "future", 2)
        decision = replace(
            resolved, kind=RouteKind.BYPASS, witness=None, dependency=dependency
        )
    elif future_kind == "alias":
        dependency = MissingAliasDependency("tenant/acme", "alias", 1)
        decision = replace(
            resolved, kind=RouteKind.UNKNOWN, witness=None, dependency=dependency
        )
    else:
        witness = replace(resolved.witness, pod_generation=2)
        decision = replace(resolved, witness=witness, dependency=witness)

    with pytest.raises(ValueError, match="current|dependency|lineage"):
        kernel.derive_artifact("future", b"unsafe", decisions=[decision])
    if future_kind == "policy":
        kernel.set_policy_scope([], expected_generation=1, idempotency_key="policy")
    assert not kernel.artifact_current("future")


def test_derive_accepts_only_current_negative_dependencies() -> None:
    kernel = _kernel()
    out_of_scope = kernel.route("outside")
    assert kernel.derive_artifact(
        "outside", b"negative", decisions=[out_of_scope]
    ).lineage.dependencies == (out_of_scope.dependency,)
    kernel.set_policy_scope(["missing"], expected_generation=1, idempotency_key="policy")
    missing = kernel.route("missing")
    assert kernel.derive_artifact(
        "missing", b"negative", decisions=[missing]
    ).lineage.dependencies == (missing.dependency,)


def test_replay_derivation_rejects_dependency_that_is_future_at_event_position() -> None:
    kernel = _kernel()
    decision = kernel.route("outside")
    kernel.derive_artifact("artifact", b"negative", decisions=[decision])
    records = list(kernel.audit_records)
    index = next(
        i for i, record in enumerate(records)
        if record.receipt_kind is ReceiptKind.DERIVATION
    )
    event = json.loads(records[index].event_json)
    event["lineage"][0]["value"]["policy_generation"] = 2
    records[index] = replace(
        records[index], event_json=json.dumps(event, sort_keys=True, separators=(",", ":"))
    )
    signed = _resign(records)

    with pytest.raises(AuditIntegrityError, match="current|lineage|artifact"):
        LifecycleKernel.replay(
            signed,
            namespace="tenant/acme",
            audit_secret=AUDIT_SECRET,
            capability_secret=CAPABILITY_SECRET,
            expected_terminal_hash=signed[-1].record_hash,
            expected_length=len(signed),
        )
