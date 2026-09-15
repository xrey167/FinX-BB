"""E-000106 exact region-transition witness reduction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    left: Fraction
    right: Fraction
    slopes: tuple[Fraction, ...]
    intercepts: tuple[Fraction, ...]


class PodTransitionWitness:
    """Candidate: Pod-named exact region transition tape."""

    def __init__(self, segments: list[Segment]):
        self.segments = tuple(segments)
        _validate(self.segments)

    def evaluate(self, p: Fraction) -> tuple[Fraction, ...]:
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            seg = self.segments[mid]
            if p < seg.left:
                hi = mid - 1
            elif p > seg.right or (p == seg.right and mid < len(self.segments) - 1):
                lo = mid + 1
            else:
                return tuple(m * p + b for m, b in zip(seg.slopes, seg.intercepts))
        raise ValueError(f"outside registered domain: {p}")

    @property
    def rational_slots(self) -> int:
        return (len(self.segments) - 1) + sum(2 * len(s.slopes) for s in self.segments)


class GenericPiecewiseAffine:
    """Independent generic baseline using the identical sufficient representation."""

    def __init__(self, serialized: list[dict[str, object]]):
        self.bounds: list[tuple[Fraction, Fraction]] = []
        self.maps: list[tuple[tuple[Fraction, ...], tuple[Fraction, ...]]] = []
        for row in serialized:
            left = _decode(row["left"])
            right = _decode(row["right"])
            slopes = tuple(_decode(x) for x in row["slopes"])
            intercepts = tuple(_decode(x) for x in row["intercepts"])
            self.bounds.append((left, right))
            self.maps.append((slopes, intercepts))

    def evaluate(self, p: Fraction) -> tuple[Fraction, ...]:
        first, last = 0, len(self.bounds)
        while first < last:
            mid = (first + last) // 2
            left, right = self.bounds[mid]
            if p < left:
                last = mid
            elif p > right or (p == right and mid + 1 < len(self.bounds)):
                first = mid + 1
            else:
                slopes, intercepts = self.maps[mid]
                return tuple(slopes[j] * p + intercepts[j] for j in range(len(slopes)))
        raise ValueError(f"outside registered domain: {p}")

    @property
    def rational_slots(self) -> int:
        dims = len(self.maps[0][0])
        return (len(self.maps) - 1) + len(self.maps) * 2 * dims


def _encode(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _decode(x: object) -> Fraction:
    num, den = str(x).split("/")
    return Fraction(int(num), int(den))


def _serialize(segments: tuple[Segment, ...] | list[Segment]) -> list[dict[str, object]]:
    return [
        {
            "left": _encode(s.left),
            "right": _encode(s.right),
            "slopes": [_encode(x) for x in s.slopes],
            "intercepts": [_encode(x) for x in s.intercepts],
        }
        for s in segments
    ]


def _validate(segments: tuple[Segment, ...] | list[Segment]) -> None:
    assert segments
    assert segments[0].left == 0
    assert segments[-1].right == 1
    dims = len(segments[0].slopes)
    for i, seg in enumerate(segments):
        assert len(seg.slopes) == dims == len(seg.intercepts)
        assert seg.left < seg.right
        if i:
            prev = segments[i - 1]
            assert prev.right == seg.left
            t = seg.left
            for j in range(dims):
                assert prev.slopes[j] * t + prev.intercepts[j] == seg.slopes[j] * t + seg.intercepts[j]


def make_session(seed: int, dims: int, region_count: int) -> PodTransitionWitness:
    rng = random.Random(seed)
    pool = [x for x in range(1, 1024) if Fraction(x, 1024) not in {Fraction(1, 8), Fraction(7, 8)}]
    cuts = sorted(rng.sample(pool, region_count - 1))
    bounds = [Fraction(0)] + [Fraction(x, 1024) for x in cuts] + [Fraction(1)]

    slopes = [Fraction(rng.choice([x for x in range(-9, 10) if x]), rng.randint(1, 7)) for _ in range(dims)]
    intercepts = [Fraction(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(dims)]
    segments: list[Segment] = []

    for i in range(region_count):
        left, right = bounds[i], bounds[i + 1]
        segments.append(Segment(left, right, tuple(slopes), tuple(intercepts)))
        if i + 1 < region_count:
            t = right
            y = [m * t + b for m, b in zip(slopes, intercepts)]
            next_slopes = [
                Fraction(rng.choice([x for x in range(-9, 10) if x]), rng.randint(1, 7))
                for _ in range(dims)
            ]
            intercepts = [yy - mm * t for yy, mm in zip(y, next_slopes)]
            slopes = next_slopes

    return PodTransitionWitness(segments)


def tent_tape(depth: int) -> PodTransitionWitness:
    n = 2**depth
    segments: list[Segment] = []
    for k in range(n):
        left, right = Fraction(k, n), Fraction(k + 1, n)
        if k % 2 == 0:
            slope, intercept = Fraction(n), Fraction(-k)
        else:
            slope, intercept = Fraction(-n), Fraction(k + 1)
        segments.append(Segment(left, right, (slope,), (intercept,)))
    return PodTransitionWitness(segments)


def run(seed_count: int, sessions_per_seed: int, dims: int, region_count: int) -> dict[str, object]:
    endpoints = {
        "OLD": Fraction(1, 8),
        "UPDATE": Fraction(7, 8),
        "DELETE": Fraction(0),
        "RESTORE": Fraction(7, 8),
        "ABA": Fraction(1, 8),
    }

    total_sessions = seed_count * sessions_per_seed
    candidate_generic_cases = 0
    candidate_generic_mismatches = 0
    material_updates = 0
    endpoint_materialization_cases = 0
    endpoint_materialization_mismatches = 0
    candidate_slots: list[int] = []
    endpoint_slots: list[int] = []

    for seed in range(seed_count):
        for session in range(sessions_per_seed):
            witness = make_session(seed * 100_000 + session, dims, region_count)
            generic = GenericPiecewiseAffine(_serialize(witness.segments))
            assert witness.rational_slots == generic.rational_slots

            old_state = witness.evaluate(endpoints["OLD"])
            current_state = witness.evaluate(endpoints["UPDATE"])
            if old_state != current_state:
                material_updates += 1

            unique_endpoints = sorted(set(endpoints.values()))
            target_states = {p: witness.evaluate(p) for p in unique_endpoints}

            for p in endpoints.values():
                candidate = witness.evaluate(p)
                generic_state = generic.evaluate(p)
                candidate_generic_cases += 1
                if candidate != generic_state:
                    candidate_generic_mismatches += 1

                endpoint_materialization_cases += 1
                if candidate != target_states[p]:
                    endpoint_materialization_mismatches += 1

            candidate_slots.append(witness.rational_slots)
            endpoint_slots.append(len(unique_endpoints) * dims)

    folding = {}
    folding_failures = 0
    for depth in range(1, 13):
        witness = tent_tape(depth)
        expected = 2**depth
        if len(witness.segments) != expected:
            folding_failures += 1
        generic = GenericPiecewiseAffine(_serialize(witness.segments))
        for j in range(17):
            p = Fraction(j, 16)
            if witness.evaluate(p) != generic.evaluate(p):
                folding_failures += 1
            y = witness.evaluate(p)[0]
            assert 0 <= y <= 1
        folding[str(depth)] = {
            "regions": len(witness.segments),
            "expected_regions": expected,
            "rational_slots": witness.rational_slots,
        }

    kill = (
        candidate_generic_mismatches == 0
        and endpoint_materialization_mismatches == 0
        and material_updates == total_sessions
        and folding_failures == 0
        and folding["12"]["regions"] == 4096
    )

    return {
        "experiment": "E-000106",
        "scope": "canonical-edit exact region-transition witness reduction",
        "seed_count": seed_count,
        "sessions_per_seed": sessions_per_seed,
        "total_sessions": total_sessions,
        "dimensions": dims,
        "regions_per_primary_session": region_count,
        "candidate_generic_cases": candidate_generic_cases,
        "candidate_generic_mismatches": candidate_generic_mismatches,
        "material_updates": material_updates,
        "endpoint_materialization_cases": endpoint_materialization_cases,
        "endpoint_materialization_mismatches": endpoint_materialization_mismatches,
        "candidate_generic_same_representation_slots": len(set(candidate_slots)) == 1,
        "candidate_rational_slots_per_session": candidate_slots[0],
        "finite_endpoint_state_slots_per_session": endpoint_slots[0],
        "folding_control": folding,
        "folding_failures": folding_failures,
        "kill_screen_pass": kill,
        "decision": (
            "KILL_PRECOMPUTED_REGION_TRANSITION_WITNESS_AS_STANDALONE_NOVELTY_SEAM"
            if kill
            else "SCREEN_NOT_DECISIVE"
        ),
        "not_claimed": (
            "No impossibility theorem for mutation-time neural-specific nonlinear quotients, "
            "no trained-transformer result, and no systems speed claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--sessions-per-seed", type=int, default=64)
    parser.add_argument("--dims", type=int, default=8)
    parser.add_argument("--regions", type=int, default=12)
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/e000106"))
    args = parser.parse_args()

    result = run(args.seed_count, args.sessions_per_seed, args.dims, args.regions)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000106_region_transition_witness_reduction.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
