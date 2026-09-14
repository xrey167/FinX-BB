from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

STAGE = "R324-CAUSAL-SEMANTIC-OBJECT-GRAPH"
PODS = int(os.environ.get("SO_R324_PODS", "10000"))
MESSAGES = int(os.environ.get("SO_R324_MESSAGES", "50000"))
SLOTS_PER_MESSAGE = int(os.environ.get("SO_R324_SLOTS", "4"))
UPDATES = int(os.environ.get("SO_R324_UPDATES", "12000"))
AUDIT_EVERY = int(os.environ.get("SO_R324_AUDIT_EVERY", "200"))
REPORT_PATH = Path(os.environ.get("SO_R324_REPORT", "r324_report.json"))
SEED = 3240914


@dataclass
class Version:
    generation: int
    value: str | None
    live: bool


class World:
    def __init__(self, rng: random.Random):
        self.current: dict[int, Version] = {
            pid: Version(1, f"V{rng.randrange(1_000_000):06d}", True)
            for pid in range(PODS)
        }
        self.history: dict[tuple[int, int], Version] = {
            (pid, 1): Version(v.generation, v.value, v.live)
            for pid, v in self.current.items()
        }

    def update(self, pid: int, value: str | None, live: bool = True):
        old = self.current[pid]
        nxt = Version(old.generation + 1, value, live)
        self.current[pid] = nxt
        self.history[(pid, nxt.generation)] = Version(nxt.generation, nxt.value, nxt.live)
        return (pid, old.generation), (pid, nxt.generation)

    def render_current(self, pid: int) -> str:
        v = self.current[pid]
        return v.value if v.live and v.value is not None else "UNKNOWN"


@dataclass(frozen=True)
class Text:
    value: str


@dataclass(frozen=True)
class CurrentSlot:
    pod_id: int


@dataclass(frozen=True)
class HistoricalRef:
    pod_id: int
    generation: int


Segment = Text | CurrentSlot | HistoricalRef


@dataclass
class SemanticMessage:
    segments: tuple[Segment, ...]
    # User-visible audit snapshot can preserve what was rendered then, but it is
    # deliberately not used to build future model context.
    archival_render: str


class SemanticTranscript:
    def __init__(self, world: World):
        self.world = world
        self.messages: list[SemanticMessage] = []
        self.current_reverse: dict[int, set[int]] = defaultdict(set)
        self.active_cache: list[str] = []

    def _render_active(self, msg: SemanticMessage) -> str:
        parts = []
        for seg in msg.segments:
            if isinstance(seg, Text):
                parts.append(seg.value)
            elif isinstance(seg, CurrentSlot):
                parts.append(self.world.render_current(seg.pod_id))
            elif isinstance(seg, HistoricalRef):
                # Historical values are addressable for audit, but the active model
                # transcript receives an opaque temporal reference rather than the
                # historical literal. This prevents a stale transcript from becoming
                # an uncontrolled factual address.
                parts.append(f"<HISTREF P{seg.pod_id}:g{seg.generation}>")
            else:
                raise TypeError(seg)
        return "".join(parts)

    def append(self, segments: tuple[Segment, ...]) -> int:
        # UI snapshot is allowed to include current literals at creation time.
        provisional = SemanticMessage(segments, "")
        archival = []
        for seg in segments:
            if isinstance(seg, Text): archival.append(seg.value)
            elif isinstance(seg, CurrentSlot): archival.append(self.world.render_current(seg.pod_id))
            elif isinstance(seg, HistoricalRef):
                old = self.world.history[(seg.pod_id, seg.generation)]
                archival.append(old.value if old.value is not None else "UNKNOWN")
        msg = SemanticMessage(segments, "".join(archival))
        idx = len(self.messages)
        self.messages.append(msg)
        self.active_cache.append(self._render_active(msg))
        for seg in segments:
            if isinstance(seg, CurrentSlot):
                self.current_reverse[seg.pod_id].add(idx)
        return idx

    def patch_current(self, pid: int) -> int:
        touched = 0
        for idx in self.current_reverse.get(pid, ()):
            self.active_cache[idx] = self._render_active(self.messages[idx])
            touched += 1
        return touched

    def full_active(self) -> list[str]:
        return [self._render_active(m) for m in self.messages]

    def active_model_context_digest(self) -> str:
        h = hashlib.sha256()
        for x in self.active_cache:
            h.update(x.encode())
            h.update(b"\0")
        return h.hexdigest()

    def archive_digest(self) -> str:
        h = hashlib.sha256()
        for m in self.messages:
            h.update(m.archival_render.encode())
            h.update(b"\0")
        return h.hexdigest()


def build_transcript(world: World, rng: random.Random) -> SemanticTranscript:
    t = SemanticTranscript(world)
    for i in range(MESSAGES):
        pids = rng.sample(range(PODS), SLOTS_PER_MESSAGE)
        segs: list[Segment] = [Text(f"M{i}: ")]
        for j, pid in enumerate(pids):
            segs.extend([Text(f"F{j}="), CurrentSlot(pid), Text("; ")])
        # Roughly 1/8 messages retain an explicit historical *reference* for audit.
        if (i & 7) == 0:
            pid = pids[0]
            g = world.current[pid].generation
            segs.extend([Text("audit="), HistoricalRef(pid, g)])
        t.append(tuple(segs))
    return t


def main() -> None:
    rng = random.Random(SEED)
    world = World(rng)
    t0 = time.perf_counter()
    transcript = build_transcript(world, rng)
    build_s = time.perf_counter() - t0

    archive_before = transcript.archive_digest()
    active_before = transcript.active_model_context_digest()
    touched_counts = []
    patch_ns = []
    same_value_updates = 0
    revocations = 0
    audit_mismatches = 0
    archive_changed_after_world_updates = 0
    active_changed_when_unreferenced = 0

    # Keep a copy of archival strings to prove they are not rewritten on current updates.
    original_archives = [m.archival_render for m in transcript.messages]

    for u in range(UPDATES):
        pid = rng.randrange(PODS)
        old = world.current[pid]
        referenced = bool(transcript.current_reverse.get(pid))
        prior_active_digest = transcript.active_model_context_digest() if (u % AUDIT_EVERY == 0 and not referenced) else None

        r = rng.random()
        if r < 0.12:
            value = None
            live = False
            revocations += 1
        elif r < 0.48:
            value = old.value
            live = True
            same_value_updates += 1
        else:
            value = f"V{rng.randrange(1_000_000):06d}"
            live = True
        world.update(pid, value, live)

        ts = time.perf_counter_ns()
        touched = transcript.patch_current(pid)
        patch_ns.append(time.perf_counter_ns() - ts)
        touched_counts.append(touched)

        if prior_active_digest is not None:
            active_changed_when_unreferenced += int(prior_active_digest != transcript.active_model_context_digest())

        if u % AUDIT_EVERY == 0:
            fresh = transcript.full_active()
            audit_mismatches += sum(a != b for a, b in zip(fresh, transcript.active_cache))
            archive_changed_after_world_updates += sum(
                old_text != msg.archival_render
                for old_text, msg in zip(original_archives, transcript.messages)
            )

    # Mutate the physically retained archival strings after all updates. This is an
    # explicit stale-byte attack: active context must be independent of archive bytes.
    active_pre_attack = transcript.active_model_context_digest()
    archive_pre_attack = transcript.archive_digest()
    attacked_messages = 0
    for i in range(0, len(transcript.messages), 97):
        m = transcript.messages[i]
        m.archival_render = "CORRUPTED_STALE_ARCHIVE::" + m.archival_render[::-1]
        attacked_messages += 1
    archive_post_attack = transcript.archive_digest()
    active_post_attack = transcript.active_model_context_digest()

    total_slot_refs = MESSAGES * SLOTS_PER_MESSAGE
    mean_touched = statistics.mean(touched_counts)
    p99_touched = sorted(touched_counts)[int(0.99 * (len(touched_counts) - 1))]
    report = {
        "stage": STAGE,
        "architecture_candidate": "Causal Semantic Object Graph (CSOG) / Virtualized Transcript",
        "pods": PODS,
        "messages": MESSAGES,
        "current_slots_per_message": SLOTS_PER_MESSAGE,
        "total_current_slot_refs": total_slot_refs,
        "updates": UPDATES,
        "same_value_generation_updates": same_value_updates,
        "revocations": revocations,
        "build_seconds": build_s,
        "mean_messages_patched_per_update": mean_touched,
        "p99_messages_patched_per_update": p99_touched,
        "mean_message_patch_fraction": mean_touched / MESSAGES,
        "median_patch_ns_python": statistics.median(patch_ns),
        "active_vs_full_rerender_audit_mismatches": audit_mismatches,
        "archival_snapshot_mutations_caused_by_world_updates": archive_changed_after_world_updates,
        "unreferenced_world_updates_changed_active_context": active_changed_when_unreferenced,
        "archive_digest_before_world_updates": archive_before,
        "active_digest_before_world_updates": active_before,
        "stale_archive_attack_messages": attacked_messages,
        "stale_archive_digest_changed_by_attack": archive_pre_attack != archive_post_attack,
        "active_model_context_changed_by_stale_archive_attack": active_pre_attack != active_post_attack,
        "mechanism": (
            "Persisted assistant/user transcripts are semantic objects rather than flat authoritative text. Governed current claims are "
            "stored as CurrentSlot(PodRef), while historical snapshots remain in an archival plane and active model context receives only "
            "opaque HISTREF handles for historical data. World updates patch only reverse-indexed current slots; old archival bytes are "
            "never re-admitted into the active factual context."
        ),
        "architectural_hypothesis": (
            "Conversation history itself is a neural resurrection vector. A lifecycle-safe system must virtualize governed factual spans "
            "inside persisted transcripts so stale assistant text cannot become a second factual address on a later turn."
        ),
        "dod_status": "NOT_DOD; transcript-lifecycle mechanism gate",
        "claim_boundary": (
            "Structured documents, reactive UIs, materialized views, provenance annotations and dynamic templates are established. R324 "
            "tests their use as a generation-scoped semantic transcript substrate for CKCA; standalone novelty is not claimed."
        ),
    }
    report["contract_pass"] = (
        audit_mismatches == 0
        and archive_changed_after_world_updates == 0
        and active_changed_when_unreferenced == 0
        and archive_pre_attack != archive_post_attack
        and active_pre_attack == active_post_attack
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["contract_pass"]:
        raise SystemExit("CSOG transcript virtualization contract failed")


if __name__ == "__main__":
    main()
