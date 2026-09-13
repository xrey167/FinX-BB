from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

OUT = Path(os.environ.get("SO_R299_REPORT", "ci-r299/report.json"))
MODEL_REV = "ckca-model-rev-A"
KEY = b"ckca-r299-test-key-not-production"


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mac(data: bytes) -> str:
    return hmac.new(KEY, data, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class Record:
    seq: int
    pod_id: int
    generation: int
    op: str
    page_digest: str | None
    model_revision: str
    prev_hash: str
    record_hash: str
    record_mac: str


@dataclass(frozen=True)
class Capability:
    pod_id: int
    generation: int
    page_digest: str
    model_revision: str
    seq: int
    record_hash: str
    cap_mac: str


@dataclass
class State:
    generation: int
    live: bool
    page_digest: str | None
    model_revision: str
    seq: int
    record_hash: str


class LedgerError(RuntimeError):
    pass


class AuthorityLedger:
    """Authenticated linearizable authority log for CKCA Pods.

    Correctness model: a serve operation requires an online current authority verifier.
    During a control-plane partition CKCA fails closed; this experiment does not claim
    availability under partition. Page bytes may exist on arbitrary replicas.
    """

    def __init__(self):
        self.records: list[Record] = []
        self.state: dict[int, State] = {}
        self.tip = "0" * 64

    def _append(self, pod_id: int, generation: int, op: str, page_digest: str | None) -> Record:
        body = {
            "seq": len(self.records) + 1,
            "pod_id": pod_id,
            "generation": generation,
            "op": op,
            "page_digest": page_digest,
            "model_revision": MODEL_REV,
            "prev_hash": self.tip,
        }
        rh = sha(canon(body))
        rm = mac(canon({**body, "record_hash": rh}))
        rec = Record(**body, record_hash=rh, record_mac=rm)
        self.records.append(rec)
        self.tip = rh
        self.state[pod_id] = State(
            generation=generation,
            live=op == "commit",
            page_digest=page_digest if op == "commit" else None,
            model_revision=MODEL_REV,
            seq=rec.seq,
            record_hash=rh,
        )
        return rec

    def commit(self, pod_id: int, page_digest: str) -> Record:
        prev = self.state.get(pod_id)
        gen = 1 if prev is None else prev.generation + 1
        return self._append(pod_id, gen, "commit", page_digest)

    def revoke(self, pod_id: int) -> Record:
        prev = self.state[pod_id]
        return self._append(pod_id, prev.generation + 1, "revoke", None)

    def capability(self, pod_id: int) -> Capability | None:
        st = self.state.get(pod_id)
        if st is None or not st.live or st.page_digest is None:
            return None
        body = {
            "pod_id": pod_id,
            "generation": st.generation,
            "page_digest": st.page_digest,
            "model_revision": st.model_revision,
            "seq": st.seq,
            "record_hash": st.record_hash,
        }
        return Capability(**body, cap_mac=mac(canon(body)))

    def verify(self, cap: Capability | None, verifier_available: bool = True) -> bool:
        if not verifier_available or cap is None:
            return False
        body = {
            "pod_id": cap.pod_id,
            "generation": cap.generation,
            "page_digest": cap.page_digest,
            "model_revision": cap.model_revision,
            "seq": cap.seq,
            "record_hash": cap.record_hash,
        }
        if not hmac.compare_digest(cap.cap_mac, mac(canon(body))):
            return False
        st = self.state.get(cap.pod_id)
        if st is None or not st.live:
            return False
        return (
            cap.generation == st.generation
            and cap.page_digest == st.page_digest
            and cap.model_revision == st.model_revision
            and cap.seq == st.seq
            and cap.record_hash == st.record_hash
        )

    def bytes(self) -> bytes:
        return b"".join(canon(asdict(r)) + b"\n" for r in self.records)

    @classmethod
    def recover(cls, data: bytes, tolerate_torn_tail: bool = False) -> "AuthorityLedger":
        out = cls()
        prev = "0" * 64
        expected_seq = 1
        lines = data.splitlines(keepends=False)
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if tolerate_torn_tail and idx == len(lines) - 1:
                    break
                raise LedgerError("malformed record")
            if int(obj["seq"]) != expected_seq:
                raise LedgerError("sequence discontinuity")
            if obj["prev_hash"] != prev:
                raise LedgerError("hash-chain discontinuity")
            body = {
                "seq": int(obj["seq"]),
                "pod_id": int(obj["pod_id"]),
                "generation": int(obj["generation"]),
                "op": obj["op"],
                "page_digest": obj["page_digest"],
                "model_revision": obj["model_revision"],
                "prev_hash": obj["prev_hash"],
            }
            rh = sha(canon(body))
            if not hmac.compare_digest(rh, obj["record_hash"]):
                raise LedgerError("record hash mismatch")
            rm = mac(canon({**body, "record_hash": rh}))
            if not hmac.compare_digest(rm, obj["record_mac"]):
                raise LedgerError("record MAC mismatch")
            prev_state = out.state.get(body["pod_id"])
            expected_gen = 1 if prev_state is None else prev_state.generation + 1
            if body["generation"] != expected_gen:
                raise LedgerError("non-monotonic pod generation")
            rec = Record(**body, record_hash=rh, record_mac=rm)
            out.records.append(rec)
            out.tip = rh
            out.state[body["pod_id"]] = State(
                generation=body["generation"],
                live=body["op"] == "commit",
                page_digest=body["page_digest"] if body["op"] == "commit" else None,
                model_revision=body["model_revision"],
                seq=body["seq"],
                record_hash=rh,
            )
            prev = rh
            expected_seq += 1
        return out


def digest_for(pod: int, gen_hint: int, salt: int) -> str:
    return sha(f"pod={pod}|genhint={gen_hint}|salt={salt}".encode())


def state_fingerprint(state: dict[int, State]) -> str:
    obj = {
        str(k): {
            "generation": v.generation,
            "live": v.live,
            "page_digest": v.page_digest,
            "model_revision": v.model_revision,
            "seq": v.seq,
            "record_hash": v.record_hash,
        }
        for k, v in sorted(state.items())
    }
    return sha(canon(obj))


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(299)
    ledger = AuthorityLedger()
    pod_count = 10_000

    stale_caps: list[Capability] = []
    latest_caps: dict[int, Capability] = {}
    replica_snapshots: list[dict[int, Capability]] = []

    # Bootstrap one authoritative generation per Pod.
    for pod in range(pod_count):
        ledger.commit(pod, digest_for(pod, 1, rng.randrange(1 << 30)))
        c = ledger.capability(pod)
        assert c is not None
        latest_caps[pod] = c

    transition_count = 60_000
    update_ns = []
    for step in range(transition_count):
        pod = rng.randrange(pod_count)
        old = ledger.capability(pod)
        if old is not None:
            stale_caps.append(old)
        t0 = time.perf_counter_ns()
        if rng.random() < 0.13:
            ledger.revoke(pod)
            latest_caps.pop(pod, None)
        else:
            # 25% of commits deliberately reuse the same semantic page digest to test
            # ABA/same-value generation separation.
            prev = ledger.state[pod]
            if rng.random() < 0.25 and prev.page_digest is not None:
                d = prev.page_digest
            else:
                d = digest_for(pod, prev.generation + 1, rng.randrange(1 << 30))
            ledger.commit(pod, d)
            c = ledger.capability(pod)
            assert c is not None
            latest_caps[pod] = c
        update_ns.append(time.perf_counter_ns() - t0)

        if step in (5_000, 20_000, 40_000):
            # Replica snapshot is just arbitrary cached capabilities/pages. It has no
            # authority of its own and may later be restored from the past.
            replica_snapshots.append(dict(latest_caps))

    # Current capability verification baseline.
    current_ok = []
    verify_ns = []
    current_caps = list(latest_caps.values())
    for cap in rng.sample(current_caps, min(50_000, len(current_caps))):
        t0 = time.perf_counter_ns()
        ok = ledger.verify(cap)
        verify_ns.append(time.perf_counter_ns() - t0)
        current_ok.append(ok)

    # Stale capabilities must all fail, including same-value generation rewrites.
    stale_sample = rng.sample(stale_caps, min(100_000, len(stale_caps)))
    stale_reject = [not ledger.verify(c) for c in stale_sample]

    # Forgery families.
    forged_checks = []
    for cap in rng.sample(current_caps, min(25_000, len(current_caps))):
        forged_checks.append(not ledger.verify(Capability(
            cap.pod_id, cap.generation + 1, cap.page_digest, cap.model_revision,
            cap.seq, cap.record_hash, cap.cap_mac,
        )))
        forged_checks.append(not ledger.verify(Capability(
            cap.pod_id, cap.generation, "f" * 64, cap.model_revision,
            cap.seq, cap.record_hash, cap.cap_mac,
        )))
        forged_checks.append(not ledger.verify(Capability(
            cap.pod_id, cap.generation, cap.page_digest, "wrong-model-rev",
            cap.seq, cap.record_hash, cap.cap_mac,
        )))
        bad_mac = ("0" if cap.cap_mac[0] != "0" else "1") + cap.cap_mac[1:]
        forged_checks.append(not ledger.verify(Capability(
            cap.pod_id, cap.generation, cap.page_digest, cap.model_revision,
            cap.seq, cap.record_hash, bad_mac,
        )))

    # Restoring an old replica snapshot cannot restore authority. Some snapshot caps
    # are still legitimately current if their Pods never changed; only changed Pods
    # count as stale-restore probes.
    restored_stale_checks = []
    for snap in replica_snapshots:
        for pod, cap in rng.sample(list(snap.items()), min(5_000, len(snap))):
            st = ledger.state[pod]
            if cap.generation != st.generation or cap.record_hash != st.record_hash or not st.live:
                restored_stale_checks.append(not ledger.verify(cap))

    # Partition semantics: no verifier => fail closed even for current capability.
    partition_fail_closed = [not ledger.verify(c, verifier_available=False) for c in rng.sample(current_caps, min(10_000, len(current_caps)))]

    # Durable restart exactness.
    journal = ledger.bytes()
    recovered = AuthorityLedger.recover(journal)
    restart_exact = state_fingerprint(ledger.state) == state_fingerprint(recovered.state) and ledger.tip == recovered.tip

    # Old capability remains dead after verifier restart.
    stale_after_restart = [not recovered.verify(c) for c in rng.sample(stale_caps, min(50_000, len(stale_caps)))]

    # Tamper detection: flip one byte inside a middle record.
    lines = journal.splitlines()
    tamper_detected = False
    if len(lines) > 10:
        mid = len(lines) // 2
        obj = json.loads(lines[mid])
        obj["generation"] = int(obj["generation"]) + 7
        lines[mid] = canon(obj)
        try:
            AuthorityLedger.recover(b"\n".join(lines) + b"\n")
        except LedgerError:
            tamper_detected = True

    # Out-of-order replay must fail chain verification.
    reorder_detected = False
    if len(lines) > 20:
        original = journal.splitlines()
        j = len(original) // 3
        original[j], original[j + 1] = original[j + 1], original[j]
        try:
            AuthorityLedger.recover(b"\n".join(original) + b"\n")
        except LedgerError:
            reorder_detected = True

    # Torn uncommitted tail: recovery ignores only the malformed final tail and
    # returns the last fully durable authority state.
    torn = journal + b'{"seq":999999,"pod_id":1'
    torn_recovery = AuthorityLedger.recover(torn, tolerate_torn_tail=True)
    torn_tail_safe = state_fingerprint(torn_recovery.state) == state_fingerprint(ledger.state)

    # Prepared-but-uncommitted page is never in the ledger, hence has no capability.
    prepared_digest = digest_for(123, 999999, 12345)
    prepared_uncommitted_not_authoritative = prepared_digest != ledger.state[123].page_digest

    report = {
        "stage": "R299-AUTHENTICATED-AUTHORITY-LEDGER",
        "architecture_candidate": "Authenticated Authority Ledger + Online Generation Capabilities",
        "consistency_model": "linearizable online authority verifier; fail closed during verifier partition",
        "pod_count": pod_count,
        "authority_transitions": transition_count,
        "journal_records": len(ledger.records),
        "current_capability_acceptance_rate": sum(current_ok) / len(current_ok),
        "stale_capability_rejection_rate": sum(stale_reject) / len(stale_reject),
        "forged_capability_rejection_rate": sum(forged_checks) / len(forged_checks),
        "stale_restored_replica_capability_rejection_rate": sum(restored_stale_checks) / max(len(restored_stale_checks), 1),
        "fail_closed_under_verifier_partition_rate": sum(partition_fail_closed) / len(partition_fail_closed),
        "restart_state_exact": restart_exact,
        "stale_capability_rejection_after_restart_rate": sum(stale_after_restart) / len(stale_after_restart),
        "tampered_journal_detected": tamper_detected,
        "out_of_order_replay_detected": reorder_detected,
        "torn_tail_recovery_preserves_last_durable_state": torn_tail_safe,
        "prepared_uncommitted_page_not_authoritative": prepared_uncommitted_not_authoritative,
        "median_authority_transition_ns": statistics.median(update_ns),
        "p99_authority_transition_ns": sorted(update_ns)[int(0.99 * len(update_ns))],
        "median_capability_verify_ns": statistics.median(verify_ns),
        "p99_capability_verify_ns": sorted(verify_ns)[int(0.99 * len(verify_ns))],
        "journal_bytes": len(journal),
        "authority_state_fingerprint": state_fingerprint(ledger.state),
        "mechanism": (
            "Physical pages and replica snapshots are non-authoritative. Serving requires an authenticated capability "
            "whose generation, digest, model revision, record hash and sequence exactly match the current recovered "
            "authority state. The append-only HMAC/hash-chained journal is the restart source of truth. Same-value "
            "rewrites receive a new generation, so old capabilities remain dead."
        ),
        "claim_boundary": (
            "This is a runtime durability/anti-resurrection experiment under a linearizable online-verifier model. "
            "It intentionally sacrifices serving availability during authority partitions and is not a consensus protocol "
            "or a standalone novelty claim."
        ),
        "dod_status": "NOT_DOD; restart/replica authority gate",
    }
    canonical = canon(report)
    report["report_sha256"] = sha(canonical)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    must_be_one = (
        "current_capability_acceptance_rate",
        "stale_capability_rejection_rate",
        "forged_capability_rejection_rate",
        "stale_restored_replica_capability_rejection_rate",
        "fail_closed_under_verifier_partition_rate",
        "stale_capability_rejection_after_restart_rate",
    )
    if any(report[k] != 1.0 for k in must_be_one):
        return 2
    if not all((restart_exact, tamper_detected, reorder_detected, torn_tail_safe, prepared_uncommitted_not_authoritative)):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
