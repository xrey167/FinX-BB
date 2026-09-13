from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

from research.r299_authenticated_authority_ledger import AuthorityLedger, Capability, LedgerError

OUT = Path(os.environ.get("SO_R306_REPORT", "ci-r306/report.json"))


@dataclass
class DState:
    state_id: int
    deps: tuple[tuple[int, int], ...]
    valid: bool = True


class PersistentLifetimeIndex:
    """Factorized lifetime index with snapshot/replay recovery.

    Each derived state has one O(1) validity bit; reverse edges are rebuilt/persisted
    as an optimization. Correctness after restore is established by replaying authority
    transitions newer than the snapshot watermark.
    """

    def __init__(self):
        self.states: dict[int, DState] = {}
        self.reverse: dict[tuple[int, int], set[int]] = defaultdict(set)
        self.next_id = 1
        self.ledger_seq = 0

    def add(self, deps) -> int:
        deps = tuple(sorted(set(deps)))
        sid = self.next_id; self.next_id += 1
        s = DState(sid, deps, True)
        self.states[sid] = s
        for dep in deps:
            self.reverse[dep].add(sid)
        return sid

    def invalidate_generation(self, pod: int, generation: int) -> int:
        ids = list(self.reverse.get((pod, generation), ()))
        touched = 0
        for sid in ids:
            s = self.states.get(sid)
            if s is not None and s.valid:
                s.valid = False; touched += 1
        return touched

    def is_valid(self, sid: int) -> bool:
        s = self.states.get(sid)
        return bool(s and s.valid)

    def snapshot(self) -> bytes:
        obj = {
            "ledger_seq": self.ledger_seq,
            "next_id": self.next_id,
            "states": [asdict(s) for s in self.states.values()],
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def recover(cls, data: bytes) -> "PersistentLifetimeIndex":
        obj = json.loads(data)
        out = cls()
        out.ledger_seq = int(obj["ledger_seq"])
        out.next_id = int(obj["next_id"])
        for raw in obj["states"]:
            deps = tuple((int(p), int(g)) for p, g in raw["deps"])
            s = DState(int(raw["state_id"]), deps, bool(raw["valid"]))
            out.states[s.state_id] = s
            for dep in deps:
                out.reverse[dep].add(s.state_id)
        return out

    def replay_authority(self, ledger: AuthorityLedger) -> int:
        touched = 0
        # Every record after the factor snapshot moves one Pod to a fresh generation.
        # The previous generation therefore expires exactly once.
        for rec in ledger.records:
            if rec.seq <= self.ledger_seq:
                continue
            old_gen = rec.generation - 1
            if old_gen >= 1:
                touched += self.invalidate_generation(rec.pod_id, old_gen)
        self.ledger_seq = len(ledger.records)
        return touched


def current_dep_valid(ledger: AuthorityLedger, dep: tuple[int, int]) -> bool:
    pod, gen = dep
    st = ledger.state.get(pod)
    return bool(st is not None and st.live and st.generation == gen)


def state_truth(ledger: AuthorityLedger, s: DState) -> bool:
    return all(current_dep_valid(ledger, dep) for dep in s.deps)


def digest(pod: int, value: int, nonce: int) -> str:
    return hashlib.sha256(f"p={pod}|v={value}|n={nonce}".encode()).hexdigest()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(306)
    pod_count = 8_000
    ledger = AuthorityLedger()
    index = PersistentLifetimeIndex()

    # Bootstrap authoritative pages.
    for pod in range(pod_count):
        ledger.commit(pod, digest(pod, rng.randrange(64), rng.randrange(1 << 30)))
    index.ledger_seq = len(ledger.records)

    # Create initial derived neural artifacts over 1-6 current generations.
    for _ in range(80_000):
        deps = []
        for pod in rng.sample(range(pod_count), rng.randint(1, 6)):
            st = ledger.state[pod]
            if st.live:
                deps.append((pod, st.generation))
        index.add(deps)

    snapshots: list[tuple[bytes, bytes]] = []
    snapshots.append((ledger.bytes(), index.snapshot()))
    invalidated_per_update = []
    transitions = 25_000

    for step in range(transitions):
        pod = rng.randrange(pod_count)
        old = ledger.state[pod]
        old_gen = old.generation
        if rng.random() < 0.10:
            ledger.revoke(pod)
        else:
            # Include semantic same-value-like rewrites by reusing digest 20% of the time.
            if old.page_digest is not None and rng.random() < 0.20:
                d = old.page_digest
            else:
                d = digest(pod, rng.randrange(64), rng.randrange(1 << 30))
            ledger.commit(pod, d)
        invalidated_per_update.append(index.invalidate_generation(pod, old_gen))
        index.ledger_seq = len(ledger.records)

        # Keep creating fresh derived artifacts after updates.
        if step % 5 == 0:
            for _ in range(4):
                deps=[]
                for p in rng.sample(range(pod_count), rng.randint(1, 5)):
                    st=ledger.state[p]
                    if st.live: deps.append((p,st.generation))
                index.add(deps)

        # Snapshots are intentionally sparse so recovery often replays many records.
        if step in (2_000, 7_000, 13_000, 19_000):
            snapshots.append((ledger.bytes(), index.snapshot()))

    final_ledger_bytes = ledger.bytes()
    final_state_count = len(index.states)

    # Current exactness audit before crash.
    pre_mismatch = 0
    for sid in rng.sample(list(index.states), min(50_000, len(index.states))):
        s = index.states[sid]
        pre_mismatch += int(index.is_valid(sid) != state_truth(ledger, s))

    # Crash/recovery trials: restore a stale lifetime-index snapshot but replay the full
    # authenticated authority journal to current tip. This models stale replica/snapshot state.
    crash_trials = 100
    recovery_mismatches = 0
    stale_valid_before_replay = 0
    stale_valid_after_replay = 0
    replay_touched = []
    replay_ns = []
    restored_state_counts = []

    for _ in range(crash_trials):
        _snap_ledger_bytes, snap_index_bytes = snapshots[rng.randrange(len(snapshots))]
        recovered_ledger = AuthorityLedger.recover(final_ledger_bytes)
        recovered_index = PersistentLifetimeIndex.recover(snap_index_bytes)
        restored_state_counts.append(len(recovered_index.states))

        sample_ids = rng.sample(list(recovered_index.states), min(5_000, len(recovered_index.states)))
        for sid in sample_ids:
            s = recovered_index.states[sid]
            if recovered_index.is_valid(sid) and not state_truth(recovered_ledger, s):
                stale_valid_before_replay += 1

        t0 = time.perf_counter_ns()
        replay_touched.append(recovered_index.replay_authority(recovered_ledger))
        replay_ns.append(time.perf_counter_ns() - t0)

        for sid in sample_ids:
            s = recovered_index.states[sid]
            got = recovered_index.is_valid(sid)
            truth = state_truth(recovered_ledger, s)
            recovery_mismatches += int(got != truth)
            stale_valid_after_replay += int(got and not truth)

    # Replica possession test: old capabilities/pages from old snapshots cannot be admitted by
    # current authority verifier after recovery.
    recovered_final = AuthorityLedger.recover(final_ledger_bytes)
    stale_cap_checks = []
    for snap_ledger_bytes, _idx in snapshots[:-1]:
        old_ledger = AuthorityLedger.recover(snap_ledger_bytes)
        caps=[]
        for pod in rng.sample(list(old_ledger.state), min(1_000, len(old_ledger.state))):
            c = old_ledger.capability(pod)
            if c is not None:
                cur = recovered_final.state.get(pod)
                if cur is None or cur.generation != c.generation or cur.record_hash != c.record_hash:
                    caps.append(c)
        stale_cap_checks.extend(not recovered_final.verify(c) for c in caps)

    # Same-value generation ABA witness.
    aba_pod = 3
    old_cap = recovered_final.capability(aba_pod)
    if old_cap is None:
        recovered_final.commit(aba_pod, digest(aba_pod, 7, 1)); old_cap = recovered_final.capability(aba_pod)
    assert old_cap is not None
    same_digest = old_cap.page_digest
    recovered_final.commit(aba_pod, same_digest)
    aba_old_cap_dead = not recovered_final.verify(old_cap)

    report = {
        "stage": "R306-PERSISTENT-COHERENCE-RECOVERY",
        "architecture_candidate": "Persistent CKCA Authority + Lifetime Recovery",
        "pod_count": pod_count,
        "authority_transitions": transitions,
        "final_derived_state_count": final_state_count,
        "pre_crash_exactness_audit_mismatches": pre_mismatch,
        "crash_recovery_trials": crash_trials,
        "stale_valid_states_detected_before_replay": stale_valid_before_replay,
        "stale_valid_states_after_authority_replay": stale_valid_after_replay,
        "post_replay_factor_vs_authority_mismatches": recovery_mismatches,
        "stale_replica_capability_rejection_rate": sum(stale_cap_checks) / max(len(stale_cap_checks), 1),
        "same_digest_new_generation_old_capability_dead": aba_old_cap_dead,
        "median_invalidated_states_per_online_update": statistics.median(invalidated_per_update),
        "p99_invalidated_states_per_online_update": sorted(invalidated_per_update)[int(0.99 * len(invalidated_per_update))],
        "median_recovery_replay_ms": statistics.median(replay_ns) / 1e6,
        "p99_recovery_replay_ms": sorted(replay_ns)[int(0.99 * len(replay_ns))] / 1e6,
        "median_factor_nodes_touched_during_recovery": statistics.median(replay_touched),
        "median_restored_state_count": statistics.median(restored_state_counts),
        "recovery_rule": (
            "restore persisted J/lifetime snapshot at watermark S; recover authenticated authority ledger to current tip T; "
            "replay authority generations S<T to invalidate old lifetime leaves before serving any restored derived state"
        ),
        "mechanism": (
            "Authority and derived-neural lifetime state are independently durable. A stale factor snapshot is never trusted "
            "as current by itself. Recovery replays the authenticated authority history newer than the factor watermark, "
            "invalidating source generations before J/cache admission is enabled."
        ),
        "claim_boundary": (
            "Pure runtime recovery experiment, not a consensus system and not a neural-quality test. It demonstrates a "
            "practical restart rule for the CKCA coherence contract under the same linearizable-verifier safety model as R299."
        ),
        "dod_status": "NOT_DOD; persistent coherence integration gate",
    }
    canonical=json.dumps(report,sort_keys=True,separators=(",",":")).encode(); report["report_sha256"]=hashlib.sha256(canonical).hexdigest(); OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

    if pre_mismatch != 0:return 2
    if recovery_mismatches != 0 or stale_valid_after_replay != 0:return 3
    if report["stale_replica_capability_rejection_rate"] != 1.0:return 4
    if not aba_old_cap_dead:return 5
    if stale_valid_before_replay == 0:return 6  # prove replay actually mattered
    return 0

if __name__=="__main__":raise SystemExit(main())
