"""LCC-001. Exact structural countermodels, NOT a trained-reader experiment.

Standard library only. Run: python assay.py --out result.json
All decisions use Fraction; no numeric tolerances, real LLM, or J-lens execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import unittest
from fractions import Fraction as Q
from pathlib import Path

BASE = ((1, 1, 1), (2, -1), (0, -2, 0, 1), (0, 1, 0, 0, 1))
VECTOR = (1, -2, 3, -4)
SEEDS = (0, 1, 2)
COUNTS = (8, 32, 128)
PREREG_COMMIT = "b0d25b651ff02c5b847f30c30778e51febd8ce90"


def probes(seed: int, n: int) -> tuple[Q, ...]:
    if not 2 <= n <= 65537:
        raise ValueError("probe count must be between 2 and 65537")
    rng = random.Random(1000 * seed + n)
    return tuple(sorted([Q(0), Q(1)] + [Q(i, 65536) for i in rng.sample(range(1, 65536), n - 2)]))


def knots(xs: tuple[Q, ...]) -> tuple[Q, Q, Q]:
    if len(xs) < 2 or tuple(sorted(set(xs))) != xs or xs[0] != 0 or xs[-1] != 1:
        raise ValueError("need distinct ordered probes including 0 and 1")
    l, u = max(zip(xs, xs[1:]), key=lambda pair: pair[1] - pair[0])
    return l + (u - l) / 4, (l + u) / 2, l + 3 * (u - l) / 4


def bump(x: Q, abc: tuple[Q, Q, Q]) -> Q:
    a, b, c = abc
    if not a < b < c or a + c != 2 * b:
        raise ValueError("knots must be strictly increasing and equally spaced")
    return (max(Q(0), x - a) - 2 * max(Q(0), x - b) + max(Q(0), x - c)) / (b - a)


def bump_derivative(x: Q, order: int, abc: tuple[Q, Q, Q]) -> Q:
    if order < 0:
        raise ValueError("negative derivative order")
    a, b, c = abc
    if order == 0:
        return bump(x, abc)
    if x in abc:
        raise ValueError("derivative at ReLU knot is not defined")
    # Independent piecewise expression for the local linear polynomial.
    if order > 1 or x < a or x > c:
        return Q(0)
    return (Q(1) if x < b else Q(-1)) / (b - a)


def poly_derivative(coeff: tuple[int, ...], x: Q, order: int) -> Q:
    return sum((Q(c) * math.factorial(i) / math.factorial(i - order) * x ** (i - order)
                for i, c in enumerate(coeff) if i >= order), Q(0))


def state(x: Q, m: Q, abc: tuple[Q, Q, Q], dependent: bool) -> tuple[Q, ...]:
    return tuple(poly_derivative(p, x, 0) + int(dependent) * m * bump(x, abc) * v
                 for p, v in zip(BASE, VECTOR))


def jet(x: Q, m: Q, dx: int, dm: int, abc: tuple[Q, Q, Q], dependent: bool) -> tuple[Q, ...]:
    if dx < 0 or dm < 0:
        raise ValueError("negative derivative order")
    base = tuple(poly_derivative(p, x, dx) if dm == 0 else Q(0) for p in BASE)
    factor = m if dm == 0 else Q(1) if dm == 1 else Q(0)
    extra = int(dependent) * factor * bump_derivative(x, dx, abc)
    return tuple(b + extra * v for b, v in zip(base, VECTOR))


def trace(x: Q, m: Q, abc: tuple[Q, Q, Q], dependent: bool) -> tuple:
    # Both graphs compute the same auxiliary gate; only its final connection differs.
    return (x, m, tuple(x - k for k in abc), tuple(max(Q(0), x - k) for k in abc),
            bump(x, abc), m * bump(x, abc), state(x, m, abc, dependent))


def certificate(l: Q, u: Q, abc: tuple[Q, Q, Q]) -> bool:
    if not 0 <= l <= u <= 1:
        raise ValueError("invalid context interval")
    return u <= abc[0] or l >= abc[2]


def interval_reference(l: Q, u: Q, abc: tuple[Q, Q, Q]) -> bool:
    # Exact range check: a continuous PWL function reaches extrema at endpoints/knots.
    points = {l, u} | {x for x in abc if l <= x <= u}
    return all(bump(x, abc) == 0 for x in points)


def collision_screen() -> dict:
    cases = 0
    for seed in SEEDS:
        for payload in range(-8, 8):
            ids = (f"A:{seed}:{payload}:g4", f"B:{seed}:{payload}:g4")
            snapshots = tuple((payload, payload * payload, 3 * payload + 1) for _ in ids)
            live = {ids[0]: False, ids[1]: True}
            assert snapshots[0] == snapshots[1]
            assert live[ids[0]] != live[ids[1]]
            # A supplied provenance record resolves the collision without any lens.
            assert [live[source] for source in ids] == [False, True]
            cases += 1
    return {"indistinguishable_history_pairs": cases, "different_validity_pairs": cases,
            "explicit_provenance_control_decisions_correct": 2 * cases}


def scenario(seed: int, n: int) -> dict:
    xs = probes(seed, n)
    abc = knots(xs)
    output_checks = trace_checks = jet_checks = revocation_checks = 0
    for x in xs:
        assert x not in abc and bump(x, abc) == 0
        for m in (Q(0), Q(1)):
            assert state(x, m, abc, False) == state(x, m, abc, True)
            output_checks += 1
            assert trace(x, m, abc, False) == trace(x, m, abc, True)
            trace_checks += 1
            for dx in range(5):
                for dm in range(5 - dx):
                    assert jet(x, m, dx, dm, abc, False) == jet(x, m, dx, dm, abc, True)
                    jet_checks += 1
        assert state(x, Q(0), abc, True) == state(x, Q(1), abc, True)
        revocation_checks += 1
    witness = abc[1]
    delta = tuple(a - b for a, b in zip(state(witness, Q(1), abc, True), state(witness, Q(0), abc, True)))
    assert delta == VECTOR
    assert state(witness, Q(1), abc, False) == state(witness, Q(0), abc, False)
    grid = sorted({Q(i, 16) for i in range(17)} | set(abc))
    cert_checks = accepts = 0
    for i, l in enumerate(grid):
        for u in grid[i:]:
            got = certificate(l, u, abc)
            assert got == interval_reference(l, u, abc)
            cert_checks += 1
            accepts += int(got)
    assert certificate(Q(0), Q(1), abc) is False
    assert certificate(abc[0], abc[0], abc) is True
    assert certificate(abc[2], abc[2], abc) is True
    assert certificate(witness, witness, abc) is False
    return {"seed": seed, "probe_count": n, "probes": [str(x) for x in xs],
            "knots": [str(x) for x in abc], "witness": str(witness),
            "output_equalities": output_checks, "shared_trace_equalities": trace_checks,
            "mixed_derivative_equalities_through_order4": jet_checks,
            "audited_revocations_with_zero_effect": revocation_checks,
            "unprobed_revocation_hidden_delta": [str(x) for x in delta],
            "whole_domain_certificate_accepts": False,
            "interval_certificate_comparisons": cert_checks, "intervals_certified": accepts}


class Controls(unittest.TestCase):
    def test_probe_generation(self):
        for s in SEEDS:
            for n in COUNTS:
                xs = probes(s, n)
                self.assertEqual(len(xs), n)
                self.assertEqual(xs, tuple(sorted(set(xs))))

    def test_invalid_probes(self):
        for n in (0, 1, 65538):
            with self.assertRaises(ValueError):
                probes(0, n)
        with self.assertRaises(ValueError):
            knots((Q(0), Q(0), Q(1)))

    def test_triangle(self):
        abc = Q(1, 4), Q(1, 2), Q(3, 4)
        for x, y in ((0, 0), (Q(1, 4), 0), (Q(3, 8), Q(1, 2)), (Q(1, 2), 1), (Q(5, 8), Q(1, 2)), (Q(3, 4), 0), (1, 0)):
            self.assertEqual(bump(Q(x), abc), y)

    def test_bad_knots(self):
        with self.assertRaises(ValueError):
            bump(Q(0), (Q(0), Q(1, 4), Q(1)))

    def test_piecewise_derivatives(self):
        abc = Q(1, 4), Q(1, 2), Q(3, 4)
        self.assertEqual(bump_derivative(Q(3, 8), 1, abc), 4)
        self.assertEqual(bump_derivative(Q(5, 8), 1, abc), -4)
        self.assertEqual(bump_derivative(Q(1, 8), 4, abc), 0)
        with self.assertRaises(ValueError):
            bump_derivative(abc[0], 1, abc)

    def test_polynomial_derivative(self):
        self.assertEqual(poly_derivative((0, 0, 0, 1), Q(2), 2), 12)
        self.assertEqual(poly_derivative((1, 2), Q(2), 3), 0)

    def test_interval_input(self):
        with self.assertRaises(ValueError):
            certificate(Q(1), Q(0), (Q(1, 4), Q(1, 2), Q(3, 4)))

    def test_provenance(self):
        self.assertEqual(collision_screen()["different_validity_pairs"], 48)

    def test_all_preregistered_scenarios(self):
        for s in SEEDS:
            for n in COUNTS:
                with self.subTest(seed=s, n=n):
                    row = scenario(s, n)
                    self.assertFalse(row["whole_domain_certificate_accepts"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("lcc001-result.json"))
    args = ap.parse_args()
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    if not tests.wasSuccessful():
        raise SystemExit(1)
    rows = [scenario(s, n) for s in SEEDS for n in COUNTS]
    sums = {k: sum(r[k] for r in rows) for k in (
        "output_equalities", "shared_trace_equalities", "mixed_derivative_equalities_through_order4",
        "audited_revocations_with_zero_effect", "interval_certificate_comparisons", "intervals_certified")}
    record = {"experiment": "LCC-001", "preregistration_commit": PREREG_COMMIT,
              "classification": "two restricted-family structural falsifications; NOT a technical novelty",
              "breakthrough": False, "arithmetic": "exact fractions; no tolerances", "python": platform.python_version(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tests_run": tests.testsRun, "tests_successful": tests.wasSuccessful(), "collision": collision_screen(),
              "scenarios": rows, "totals": sums, "unprobed_counterexamples": len(rows),
              "scope": "state-only historical provenance and finite-oracle whole-domain support; NOT white-box or scoped proof impossibility",
              "real_reader_capability": "NOT MEASURED", "trained_reader_seeds": 0,
              "backbones_evaluated": 0, "actual_J_lens_executed": False,
              "full_system_lifecycle_and_utility_gates": "NOT MEASURED; unchanged"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "scenarios"}, indent=2))


if __name__ == "__main__":
    main()
