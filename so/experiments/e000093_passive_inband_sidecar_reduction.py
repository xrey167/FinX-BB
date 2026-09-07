"""E-000093 -- scoped reduction of passive in-band freshness to a co-located sidecar.

This is a kill-screen, not a novelty claim.  It enumerates finite deterministic
freshness schemes whose reuse decision depends on current authority A and a
statistic D(T) recoverable from a cached artifact T.  A sidecar materialized as
S=D(T) must reproduce the same decision exactly.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Artifact:
    object_id: int
    generation: int
    payload: int
    marker: int

    def serialize(self) -> bytes:
        return bytes((self.object_id, self.generation, self.payload, self.marker))


@dataclass(frozen=True)
class Authority:
    object_id: int
    generation: int
    live: bool


Decoder = Callable[[Artifact], Tuple[int, ...]]
Gate = Callable[[Authority, Tuple[int, ...]], bool]


def dec_generation(t: Artifact) -> Tuple[int, ...]:
    return (t.object_id, t.generation)


def dec_generation_marker(t: Artifact) -> Tuple[int, ...]:
    return (t.object_id, t.generation, t.marker)


def dec_content_bound(t: Artifact) -> Tuple[int, ...]:
    h = hashlib.sha256(t.serialize()).digest()
    return (t.object_id, t.generation, int.from_bytes(h[:4], "big"))


def gate_exact(a: Authority, z: Tuple[int, ...]) -> bool:
    return bool(a.live and len(z) >= 2 and a.object_id == z[0] and a.generation == z[1])


def gate_exact_marker(a: Authority, z: Tuple[int, ...]) -> bool:
    return bool(gate_exact(a, z) and len(z) >= 3 and (z[2] % 2 == 1))


def artifact_domain() -> Iterable[Artifact]:
    for oid, gen, payload, marker in itertools.product(range(3), range(4), range(3), range(2)):
        yield Artifact(oid, gen, payload, marker)


def authority_domain() -> Iterable[Authority]:
    for oid, gen, live in itertools.product(range(3), range(4), (False, True)):
        yield Authority(oid, gen, live)


def registered_faults(t: Artifact, a: Authority) -> List[Tuple[str, Artifact, Authority]]:
    # External metadata deletion/swap is intentionally absent from Artifact: a
    # correctly co-located baseline transports its sidecar with this artifact.
    other_oid = (t.object_id + 1) % 3
    next_gen = (t.generation + 1) % 4
    prev_gen = (t.generation - 1) % 4
    return [
        ("no_mutation", t, a),
        ("current_update", t, Authority(a.object_id, (a.generation + 1) % 4, a.live)),
        ("unrelated_mutation", t, Authority(other_oid, (a.generation + 1) % 4, a.live)),
        ("metadata_deleted", t, a),
        ("metadata_swapped", t, a),
        ("serialized_relocated", Artifact(*t.serialize()), a),
        ("stale_replay", Artifact(t.object_id, prev_gen, t.payload, t.marker), a),
        ("aba_authority_return", t, Authority(a.object_id, t.generation, a.live)),
        ("rollback", t, Authority(a.object_id, prev_gen, a.live)),
        ("same_content_new_generation", Artifact(t.object_id, next_gen, t.payload, t.marker), a),
    ]


def check_scheme(name: str, decoder: Decoder, gate: Gate) -> Dict[str, object]:
    comparisons = 0
    mismatches: List[Dict[str, object]] = []
    fault_counts: Dict[str, int] = {}
    for t0 in artifact_domain():
        for a0 in authority_domain():
            for fault, t, a in registered_faults(t0, a0):
                z = decoder(t)
                # In-band candidate decodes Z from T at decision time.
                candidate = gate(a, decoder(t))
                # Guarantee-matched sidecar stores exactly that sufficient statistic.
                sidecar = gate(a, z)
                comparisons += 1
                fault_counts[fault] = fault_counts.get(fault, 0) + 1
                if candidate != sidecar:
                    mismatches.append({
                        "fault": fault,
                        "artifact": t.__dict__,
                        "authority": a.__dict__,
                        "z": list(z),
                        "candidate": candidate,
                        "sidecar": sidecar,
                    })
    return {
        "scheme": name,
        "comparisons": comparisons,
        "fault_counts": fault_counts,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
        "exact_equivalence": len(mismatches) == 0,
    }


def main() -> None:
    schemes = [
        ("generation", dec_generation, gate_exact),
        ("generation_marker", dec_generation_marker, gate_exact_marker),
        ("content_bound_generation", dec_content_bound, gate_exact),
    ]
    rows = [check_scheme(*s) for s in schemes]
    rec = {
        "experiment": "E-000093",
        "scope": "passive in-band freshness schemes factorable through a deterministic tensor statistic D(T)",
        "rows": rows,
        "all_exact_equivalence": all(bool(r["exact_equivalence"]) for r in rows),
        "kill_if": "all registered passive schemes are decision-equivalent to the co-located sidecar shadow",
        "escape": "active lifecycle-conditioned computation or measured systems advantage beyond guarantee-matched sidecar",
    }
    out = Path("so/results/e000093_passive_inband_sidecar_reduction.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))
    if not rec["all_exact_equivalence"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
