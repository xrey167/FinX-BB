from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

Monomial = Tuple[int, ...]
Poly = Dict[Monomial, Fraction]


def _const(c: Fraction | int, n: int) -> Poly:
    return {(0,) * n: Fraction(c)}


def _var(i: int, n: int) -> Poly:
    exp = [0] * n
    exp[i] = 1
    return {tuple(exp): Fraction(1)}


def _add(a: Poly, b: Poly) -> Poly:
    out: defaultdict[Monomial, Fraction] = defaultdict(Fraction)
    for m, c in a.items():
        out[m] += c
    for m, c in b.items():
        out[m] += c
    return {m: c for m, c in out.items() if c}


def _mul(a: Poly, b: Poly) -> Poly:
    out: defaultdict[Monomial, Fraction] = defaultdict(Fraction)
    for ma, ca in a.items():
        for mb, cb in b.items():
            out[tuple(x + y for x, y in zip(ma, mb))] += ca * cb
    return {m: c for m, c in out.items() if c}


def _scale(a: Poly, s: Fraction) -> Poly:
    s = Fraction(s)
    return {m: c * s for m, c in a.items() if c * s}


def _eval(poly: Poly, values: List[Fraction]) -> Fraction:
    total = Fraction(0)
    for monomial, coefficient in poly.items():
        term = coefficient
        for exponent, value in zip(monomial, values):
            if exponent:
                term *= value ** exponent
        total += term
    return total


def candidate_lift(family: str, coeffs: List[Fraction]) -> Poly:
    """Lifecycle-native symbolic lift via network-style add/multiply propagation."""
    n = len(coeffs)
    one = _const(1, n)
    if family == "additive":
        state = one
        for i, a in enumerate(coeffs):
            state = _add(state, _scale(_var(i, n), a))
        return state
    if family == "pairwise":
        state = one
        for i, a in enumerate(coeffs):
            state = _add(state, _scale(_var(i, n), a))
        for i in range(n):
            for j in range(i + 1, n):
                interaction = _mul(_var(i, n), _var(j, n))
                state = _add(state, _scale(interaction, coeffs[i] * coeffs[j] / 2))
        return state
    if family == "full_product":
        state = one
        for i, a in enumerate(coeffs):
            factor = _add(one, _scale(_var(i, n), a))
            state = _mul(state, factor)
        return state
    raise ValueError(f"unknown family: {family}")


def provenance_baseline(family: str, coeffs: List[Fraction]) -> Poly:
    """Independent direct construction for a generic provenance-polynomial engine."""
    n = len(coeffs)
    out: Poly = {(0,) * n: Fraction(1)}
    if family in {"additive", "pairwise"}:
        for i, a in enumerate(coeffs):
            exp = [0] * n
            exp[i] = 1
            out[tuple(exp)] = a
        if family == "pairwise":
            for i in range(n):
                for j in range(i + 1, n):
                    exp = [0] * n
                    exp[i] = exp[j] = 1
                    out[tuple(exp)] = coeffs[i] * coeffs[j] / 2
        return {m: c for m, c in out.items() if c}
    if family == "full_product":
        out = {}
        for mask in range(1 << n):
            exp = [0] * n
            coefficient = Fraction(1)
            for i, a in enumerate(coeffs):
                if (mask >> i) & 1:
                    exp[i] = 1
                    coefficient *= a
            out[tuple(exp)] = coefficient
        return {m: c for m, c in out.items() if c}
    raise ValueError(f"unknown family: {family}")


def fresh_numeric(family: str, coeffs: List[Fraction], values: List[Fraction]) -> Fraction:
    if family == "additive":
        return Fraction(1) + sum((a * x for a, x in zip(coeffs, values)), Fraction(0))
    if family == "pairwise":
        total = Fraction(1) + sum((a * x for a, x in zip(coeffs, values)), Fraction(0))
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total += (coeffs[i] * coeffs[j] / 2) * values[i] * values[j]
        return total
    if family == "full_product":
        total = Fraction(1)
        for a, x in zip(coeffs, values):
            total *= 1 + a * x
        return total
    raise ValueError(f"unknown family: {family}")


def exact_edit_delta(
    poly: Poly,
    old_values: List[Fraction],
    new_values: List[Fraction],
    edited: int,
) -> Tuple[Fraction, int]:
    """Evaluate only expanded symbolic terms that depend on the edited Pod."""
    delta = Fraction(0)
    touched = 0
    for monomial, coefficient in poly.items():
        if monomial[edited] == 0:
            continue
        touched += 1
        old_term = coefficient
        new_term = coefficient
        for exponent, old, new in zip(monomial, old_values, new_values):
            if exponent:
                old_term *= old ** exponent
                new_term *= new ** exponent
        delta += new_term - old_term
    return delta, touched


class GenericDependencyProduct:
    """Generic dependency-directed incremental baseline over a balanced product DAG."""

    def __init__(self, leaves: List[Fraction]):
        size = 1
        while size < len(leaves):
            size *= 2
        self.size = size
        self.values = [Fraction(1)] * (2 * size)
        for i, value in enumerate(leaves):
            self.values[size + i] = value
        for i in range(size - 1, 0, -1):
            self.values[i] = self.values[2 * i] * self.values[2 * i + 1]

    @property
    def output(self) -> Fraction:
        return self.values[1]

    def update(self, index: int, value: Fraction) -> int:
        pos = self.size + index
        self.values[pos] = value
        multiplications = 0
        pos //= 2
        while pos:
            self.values[pos] = self.values[2 * pos] * self.values[2 * pos + 1]
            multiplications += 1
            pos //= 2
        return multiplications


def specialized_lifecycle_product_update(
    leaves: List[Fraction], edited: int, new_value: Fraction
) -> Tuple[Fraction, int]:
    """Specialized candidate factor-tree update, intentionally independent of the baseline class."""
    level = list(leaves)
    size = 1
    while size < len(level):
        size *= 2
    level.extend([Fraction(1)] * (size - len(level)))
    tree: List[List[Fraction]] = [level]
    while len(tree[-1]) > 1:
        prior = tree[-1]
        tree.append([prior[i] * prior[i + 1] for i in range(0, len(prior), 2)])

    tree[0][edited] = new_value
    idx = edited
    multiplications = 0
    for depth in range(1, len(tree)):
        parent = idx // 2
        left = tree[depth - 1][2 * parent]
        right = tree[depth - 1][2 * parent + 1]
        tree[depth][parent] = left * right
        multiplications += 1
        idx = parent
    return tree[-1][0], multiplications


def _different_fraction(rng: random.Random, current: Fraction) -> Fraction:
    candidate = current
    while candidate == current:
        candidate = Fraction(rng.randint(-5, 6), rng.randint(2, 9))
    return candidate


def run(seed_count: int, sizes: List[int]) -> dict:
    families = ("additive", "pairwise", "full_product")
    exact_cases = 0
    exact_mismatches = 0
    representation_mismatches = 0
    compact_cases = 0
    compact_output_mismatches = 0
    compact_work_mismatches = 0
    records = []

    for seed in range(seed_count):
        rng = random.Random(10_000 + seed)
        for n in sizes:
            coeffs = [Fraction(rng.randint(1, 9), rng.randint(2, 11)) for _ in range(n)]
            old_values = [Fraction(rng.randint(-4, 5), rng.randint(2, 9)) for _ in range(n)]
            edited_indices = sorted({0, n // 2, n - 1})

            for family in families:
                candidate = candidate_lift(family, coeffs)
                baseline = provenance_baseline(family, coeffs)
                if candidate != baseline:
                    representation_mismatches += 1

                for edited in edited_indices:
                    new_values = list(old_values)
                    new_values[edited] = _different_fraction(rng, old_values[edited])

                    old_symbolic = _eval(candidate, old_values)
                    new_symbolic = _eval(candidate, new_values)
                    new_baseline = _eval(baseline, new_values)
                    old_fresh = fresh_numeric(family, coeffs, old_values)
                    new_fresh = fresh_numeric(family, coeffs, new_values)
                    delta, touched = exact_edit_delta(candidate, old_values, new_values, edited)

                    exact_cases += 1
                    if not (
                        old_symbolic == old_fresh
                        and new_symbolic == new_fresh
                        and new_baseline == new_fresh
                        and old_symbolic + delta == new_fresh
                    ):
                        exact_mismatches += 1

                    expected_terms = {
                        "additive": 1 + n,
                        "pairwise": 1 + n + n * (n - 1) // 2,
                        "full_product": 1 << n,
                    }[family]
                    expected_touched = {
                        "additive": 1,
                        "pairwise": n,
                        "full_product": 1 << (n - 1),
                    }[family]
                    if len(candidate) != expected_terms or touched != expected_touched:
                        representation_mismatches += 1

                    records.append(
                        {
                            "seed": seed,
                            "n_pods": n,
                            "family": family,
                            "edited": edited,
                            "monomials": len(candidate),
                            "edit_terms_touched": touched,
                        }
                    )

            # Compact factorized control for the highest-interaction family.
            factors = [Fraction(1) + a * x for a, x in zip(coeffs, old_values)]
            for edited in edited_indices:
                new_x = _different_fraction(rng, old_values[edited])
                new_factor = Fraction(1) + coeffs[edited] * new_x

                candidate_output, candidate_ops = specialized_lifecycle_product_update(
                    factors, edited, new_factor
                )
                generic = GenericDependencyProduct(factors)
                generic_ops = generic.update(edited, new_factor)
                fresh_factors = list(factors)
                fresh_factors[edited] = new_factor
                fresh_output = math.prod(fresh_factors)

                compact_cases += 1
                if candidate_output != fresh_output or generic.output != fresh_output:
                    compact_output_mismatches += 1
                if candidate_ops != generic_ops:
                    compact_work_mismatches += 1

    max_n = max(sizes)
    full_records = [
        r for r in records if r["family"] == "full_product" and r["n_pods"] == max_n
    ]
    pair_records = [
        r for r in records if r["family"] == "pairwise" and r["n_pods"] == max_n
    ]
    add_records = [
        r for r in records if r["family"] == "additive" and r["n_pods"] == max_n
    ]

    summary = {
        "experiment": "E-000102",
        "scope": "symbolic Pod algebra / provenance / generic incremental-computation reduction",
        "seed_count": seed_count,
        "sizes": sizes,
        "exact_cases": exact_cases,
        "exact_mismatches": exact_mismatches,
        "representation_mismatches": representation_mismatches,
        "compact_cases": compact_cases,
        "compact_output_mismatches": compact_output_mismatches,
        "compact_work_mismatches": compact_work_mismatches,
        "max_n": max_n,
        "max_n_additive_monomials": add_records[0]["monomials"],
        "max_n_additive_edit_terms": add_records[0]["edit_terms_touched"],
        "max_n_pairwise_monomials": pair_records[0]["monomials"],
        "max_n_pairwise_edit_terms": pair_records[0]["edit_terms_touched"],
        "max_n_full_monomials": full_records[0]["monomials"],
        "max_n_full_edit_terms": full_records[0]["edit_terms_touched"],
        "expanded_full_growth_verified": all(
            r["monomials"] == (1 << r["n_pods"])
            and r["edit_terms_touched"] == (1 << (r["n_pods"] - 1))
            for r in records
            if r["family"] == "full_product"
        ),
        "kill_screen_pass": (
            exact_mismatches == 0
            and representation_mismatches == 0
            and compact_output_mismatches == 0
            and compact_work_mismatches == 0
        ),
        "decision": "KILL_SYMBOLIC_POD_ALGEBRA_ALONE_AS_NOVELTY_SEAM",
        "not_claimed": (
            "No impossibility result for compressed exact neural-specific interaction representations; "
            "no trained-language-model capability or systems speed claim."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=16)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--results-dir", type=Path, default=Path("so/results/e000102"))
    args = parser.parse_args()

    summary = run(args.seed_count, args.sizes)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "e000102-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["kill_screen_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
