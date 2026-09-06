from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

Scalar = Fraction
Jet = tuple[Scalar, ...]


@dataclass(frozen=True)
class PolynomialLifecycleNet:
    """Scalar nonlinear lifecycle-native toy network.

    The mutable Pod is scalar p. Context x is session-specific. Every layer is
    quadratic, so the exact Pod degree doubles with depth when the leading
    coefficient is nonzero.
    """

    ax: Scalar
    bp: Scalar
    c: Scalar
    layers: tuple[tuple[Scalar, Scalar, Scalar, Scalar], ...]

    def fresh(self, x: Scalar, p: Scalar) -> Scalar:
        h = self.ax * x + self.bp * p + self.c
        for alpha, beta, gamma, eta in self.layers:
            h = alpha * h * h + beta * h + gamma + eta * x
        return h


def make_net(seed: int, depth: int) -> PolynomialLifecycleNet:
    rng = random.Random(111_000_111 + seed * 1009 + depth * 97)
    ax = Fraction(rng.randint(1, 3), 5)
    bp = Fraction(rng.randint(1, 3), 7)
    c = Fraction(rng.randint(-2, 2), 11)
    layers = []
    for _ in range(depth):
        layers.append(
            (
                Fraction(rng.randint(1, 3), 7),
                Fraction(rng.randint(1, 3), 5),
                Fraction(rng.randint(-2, 2), 13),
                Fraction(rng.randint(-2, 2), 17),
            )
        )
    return PolynomialLifecycleNet(ax=ax, bp=bp, c=c, layers=tuple(layers))


def _candidate_convolution(a: Jet, b: Jet, order: int) -> tuple[Jet, int]:
    out = [Fraction(0) for _ in range(order + 1)]
    multiplies = 0
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > order:
                break
            if bj == 0:
                continue
            out[i + j] += ai * bj
            multiplies += 1
    return tuple(out), multiplies


def candidate_neural_edit_jet(
    net: PolynomialLifecycleNet, x: Scalar, p0: Scalar, order: int
) -> tuple[Jet, int]:
    """Candidate: co-compute an exact Pod edit jet during the forward path.

    Coefficients are for h(p0 + d) = sum_k jet[k] * d**k.
    """
    h = [Fraction(0) for _ in range(order + 1)]
    h[0] = net.ax * x + net.bp * p0 + net.c
    if order >= 1:
        h[1] = net.bp
    multiplies = 0

    for alpha, beta, gamma, eta in net.layers:
        square, work = _candidate_convolution(tuple(h), tuple(h), order)
        multiplies += work
        nxt = [Fraction(0) for _ in range(order + 1)]
        nxt[0] = alpha * square[0] + beta * h[0] + gamma + eta * x
        multiplies += 2
        for k in range(1, order + 1):
            nxt[k] = alpha * square[k] + beta * h[k]
            multiplies += 2
        h = nxt
    return tuple(h), multiplies


def generic_taylor_mode_ad(
    net: PolynomialLifecycleNet, x: Scalar, p0: Scalar, order: int
) -> tuple[Jet, int]:
    """Independent generic truncated-power-series AD baseline.

    This path has no Pod/lifecycle semantics. It propagates ordinary Taylor
    coefficients through the same arithmetic program.
    """
    series = [Fraction(0) for _ in range(order + 1)]
    series[0] = net.ax * x + net.bp * p0 + net.c
    if order >= 1:
        series[1] = net.bp
    multiplies = 0

    for alpha, beta, gamma, eta in net.layers:
        product = [Fraction(0) for _ in range(order + 1)]
        for degree in range(order + 1):
            for left_degree in range(degree + 1):
                right_degree = degree - left_degree
                left = series[left_degree]
                right = series[right_degree]
                if left == 0 or right == 0:
                    continue
                product[degree] += left * right
                multiplies += 1

        nxt = [Fraction(0) for _ in range(order + 1)]
        for degree in range(order + 1):
            if degree == 0:
                nxt[degree] = (
                    alpha * product[degree]
                    + beta * series[degree]
                    + gamma
                    + eta * x
                )
            else:
                nxt[degree] = alpha * product[degree] + beta * series[degree]
            multiplies += 2
        series = nxt
    return tuple(series), multiplies


def evaluate_jet(jet: Jet, delta: Scalar) -> Scalar:
    value = Fraction(0)
    for coefficient in reversed(jet):
        value = value * delta + coefficient
    return value


def true_degree(jet: Jet) -> int:
    for degree in range(len(jet) - 1, -1, -1):
        if jet[degree] != 0:
            return degree
    return 0


def rational_nonpolynomial_fresh(a: Scalar, p: Scalar) -> Scalar:
    denominator = Fraction(1) - a * p
    if denominator == 0:
        raise ZeroDivisionError("registered rational witness hit a pole")
    return Fraction(1, 1) / denominator


def rational_nonpolynomial_jet(a: Scalar, order: int) -> Jet:
    # Around p0 = 0: 1/(1-a*d) = sum_{k>=0} a**k d**k.
    return tuple(a**k for k in range(order + 1))


def registered_edit_jet_cells(
    seeds: Iterable[int] = range(16),
    depths: Iterable[int] = range(1, 7),
    contexts_per_cell: int = 12,
    deltas: Iterable[Scalar] = (
        Fraction(1, 5),
        Fraction(-1, 7),
        Fraction(2, 11),
    ),
) -> dict:
    seeds = tuple(seeds)
    depths = tuple(depths)
    deltas = tuple(deltas)

    exact_transport_cases = 0
    candidate_fresh_mismatches = 0
    generic_fresh_mismatches = 0
    candidate_generic_jet_mismatches = 0
    candidate_generic_work_mismatches = 0
    truncated_transport_cases = 0
    truncated_transport_failures = 0
    degree_mismatches = 0
    candidate_total_slots = 0
    generic_total_slots = 0
    candidate_total_multiplies = 0
    generic_total_multiplies = 0
    max_required_order = 0
    cells = []

    for seed in seeds:
        for depth in depths:
            net = make_net(seed=seed, depth=depth)
            expected_degree = 2**depth
            cell_exact_cases = 0
            cell_truncated_failures = 0
            cell_jet_mismatches = 0

            for context_id in range(contexts_per_cell):
                x = Fraction(context_id - contexts_per_cell // 2, 19)
                p0 = Fraction(0)
                candidate, candidate_work = candidate_neural_edit_jet(
                    net, x, p0, expected_degree
                )
                generic, generic_work = generic_taylor_mode_ad(
                    net, x, p0, expected_degree
                )

                if candidate != generic:
                    candidate_generic_jet_mismatches += 1
                    cell_jet_mismatches += 1
                if candidate_work != generic_work:
                    candidate_generic_work_mismatches += 1

                degree = true_degree(candidate)
                if degree != expected_degree:
                    degree_mismatches += 1
                max_required_order = max(max_required_order, degree)

                candidate_total_slots += len(candidate)
                generic_total_slots += len(generic)
                candidate_total_multiplies += candidate_work
                generic_total_multiplies += generic_work

                truncated_order = expected_degree // 2
                truncated = candidate[: truncated_order + 1]

                for delta in deltas:
                    fresh = net.fresh(x, p0 + delta)
                    candidate_updated = evaluate_jet(candidate, delta)
                    generic_updated = evaluate_jet(generic, delta)
                    exact_transport_cases += 1
                    cell_exact_cases += 1

                    if candidate_updated != fresh:
                        candidate_fresh_mismatches += 1
                    if generic_updated != fresh:
                        generic_fresh_mismatches += 1

                    truncated_transport_cases += 1
                    if evaluate_jet(truncated, delta) != fresh:
                        truncated_transport_failures += 1
                        cell_truncated_failures += 1

            cells.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "required_exact_jet_order": expected_degree,
                    "contexts": contexts_per_cell,
                    "edits_per_context": len(deltas),
                    "exact_cases": cell_exact_cases,
                    "candidate_generic_jet_mismatches": cell_jet_mismatches,
                    "half_order_truncated_failures": cell_truncated_failures,
                }
            )

    return {
        "cells": cells,
        "totals": {
            "registered_cells": len(cells),
            "exact_transport_cases": exact_transport_cases,
            "candidate_fresh_mismatches": candidate_fresh_mismatches,
            "generic_fresh_mismatches": generic_fresh_mismatches,
            "candidate_generic_jet_mismatches": candidate_generic_jet_mismatches,
            "candidate_generic_work_mismatches": candidate_generic_work_mismatches,
            "truncated_transport_cases": truncated_transport_cases,
            "truncated_transport_failures": truncated_transport_failures,
            "truncated_transport_failure_fraction": (
                truncated_transport_failures / truncated_transport_cases
            ),
            "degree_mismatches": degree_mismatches,
            "max_required_exact_jet_order": max_required_order,
            "candidate_total_slots": candidate_total_slots,
            "generic_total_slots": generic_total_slots,
            "candidate_to_generic_slot_ratio": (
                candidate_total_slots / generic_total_slots
            ),
            "candidate_total_multiplies": candidate_total_multiplies,
            "generic_total_multiplies": generic_total_multiplies,
            "candidate_to_generic_multiply_ratio": (
                candidate_total_multiplies / generic_total_multiplies
            ),
        },
    }


def registered_nonpolynomial_witness(
    orders: Iterable[int] = (1, 2, 4, 8, 16, 32),
    a_values: Iterable[Scalar] = (
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(3, 4),
    ),
    deltas: Iterable[Scalar] = (Fraction(1, 5), Fraction(-1, 7)),
) -> dict:
    orders = tuple(orders)
    a_values = tuple(a_values)
    deltas = tuple(deltas)

    cases = 0
    exact_matches = 0
    failures = 0
    witnesses = []
    for order in orders:
        for a in a_values:
            jet = rational_nonpolynomial_jet(a, order)
            for delta in deltas:
                fresh = rational_nonpolynomial_fresh(a, delta)
                transported = evaluate_jet(jet, delta)
                cases += 1
                if transported == fresh:
                    exact_matches += 1
                else:
                    failures += 1
                    witnesses.append(
                        {
                            "order": order,
                            "a": str(a),
                            "delta": str(delta),
                            "fresh": str(fresh),
                            "truncated_jet": str(transported),
                        }
                    )
    return {
        "cases": cases,
        "exact_matches": exact_matches,
        "failures": failures,
        "failure_fraction": failures / cases,
        "witnesses": witnesses[:8],
    }


def run_registered() -> dict:
    polynomial = registered_edit_jet_cells()
    nonpolynomial = registered_nonpolynomial_witness()
    totals = polynomial["totals"]

    kill = (
        totals["registered_cells"] == 96
        and totals["exact_transport_cases"] == 3456
        and totals["candidate_fresh_mismatches"] == 0
        and totals["generic_fresh_mismatches"] == 0
        and totals["candidate_generic_jet_mismatches"] == 0
        and totals["candidate_generic_work_mismatches"] == 0
        and totals["degree_mismatches"] == 0
        and totals["max_required_exact_jet_order"] == 64
        and totals["truncated_transport_failure_fraction"] == 1.0
        and totals["candidate_to_generic_slot_ratio"] == 1.0
        and totals["candidate_to_generic_multiply_ratio"] == 1.0
        and nonpolynomial["cases"] == 36
        and nonpolynomial["failures"] == 36
        and nonpolynomial["exact_matches"] == 0
    )

    return {
        "experiment": "E-000111",
        "title": "Finite Edit-Jet and Co-computed Higher-Order Sensitivity Reduction",
        "kill_screen_pass": kill,
        "decision": (
            "KILL_FINITE_EDIT_JETS_AND_CO_COMPUTED_HIGHER_ORDER_SENSITIVITY_AS_STANDALONE_EXACT_TRANSPORT_NOVELTY"
            if kill
            else "DO_NOT_KILL"
        ),
        "major_invention": False,
        "polynomial_edit_jet": polynomial,
        "nonpolynomial_exactness_witness": nonpolynomial,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ci-e111"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result = run_registered()
    out = args.results_dir / "e000111-result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["polynomial_edit_jet"]["totals"], indent=2, sort_keys=True))
    print(json.dumps(result["nonpolynomial_exactness_witness"], indent=2, sort_keys=True))
    print(result["decision"])


if __name__ == "__main__":
    main()
