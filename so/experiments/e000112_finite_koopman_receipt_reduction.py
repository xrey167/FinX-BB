from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

Scalar = Fraction
Lift = tuple[Scalar, ...]

LIFT_DIM = 6
# Structural coefficient multiplies for one exact lifted transition. Identity
# copies for the constant/context coordinates are not counted.
LIFT_STEP_MULTIPLIES = 8
# Fresh original-coordinate replay per nonlinear layer:
# lam*x; mu*y; x*c then nu*; x*x then rho*; context_drive*c.
FRESH_STEP_MULTIPLIES = 7
# The shared B_d receipt has support only in (x*c, y), so one session needs
# exactly two scalar multiplies by its preserved context coordinate.
SESSION_RECEIPT_MULTIPLIES = 2


@dataclass(frozen=True)
class FiniteKoopmanLifecycleNet:
    """Nonlinear lifecycle-native toy dynamics with an exact finite lift.

    Original state is (x, c, y). x is initialized from the mutable Pod,
    c is a session-specific context coordinate, and y is a hidden coordinate.
    The update is nonlinear in original coordinates through x*c and x**2:

        x' = lam*x
        c' = c
        y' = mu*y + nu*x*c + rho*x**2 + bias + context_drive*c

    The observable vector (1, x, c, x*c, x**2, y) is exactly closed under a
    six-dimensional linear Koopman action.
    """

    lam: Scalar
    mu: Scalar
    nu: Scalar
    rho: Scalar
    bias: Scalar
    context_drive: Scalar

    def fresh(
        self, context: Scalar, pod: Scalar, y0: Scalar, depth: int
    ) -> tuple[Scalar, Scalar, Scalar]:
        x = pod
        c = context
        y = y0
        for _ in range(depth):
            y = (
                self.mu * y
                + self.nu * x * c
                + self.rho * x * x
                + self.bias
                + self.context_drive * c
            )
            x = self.lam * x
        return x, c, y


def make_net(seed: int) -> FiniteKoopmanLifecycleNet:
    rng = random.Random(112_000_112 + seed * 1009)
    # Strictly positive nonzero rationals avoid accidental materiality or
    # support cancellations in the registered domain.
    return FiniteKoopmanLifecycleNet(
        lam=Fraction(rng.randint(1, 3), 5),
        mu=Fraction(rng.randint(1, 3), 7),
        nu=Fraction(rng.randint(1, 3), 11),
        rho=Fraction(rng.randint(1, 3), 13),
        bias=Fraction(rng.randint(1, 3), 17),
        context_drive=Fraction(rng.randint(1, 3), 19),
    )


def lift_original(context: Scalar, pod: Scalar, y: Scalar) -> Lift:
    return (
        Fraction(1),
        pod,
        context,
        pod * context,
        pod * pod,
        y,
    )


def candidate_lift_step(net: FiniteKoopmanLifecycleNet, q: Lift) -> Lift:
    """Candidate's hand-derived exact neural observable transition."""
    one, x, c, xc, x2, y = q
    return (
        one,
        net.lam * x,
        c,
        net.lam * xc,
        net.lam * net.lam * x2,
        (
            net.bias * one
            + net.context_drive * c
            + net.nu * xc
            + net.rho * x2
            + net.mu * y
        ),
    )


def candidate_propagate(
    net: FiniteKoopmanLifecycleNet, q: Lift, depth: int
) -> Lift:
    for _ in range(depth):
        q = candidate_lift_step(net, q)
    return q


def generic_koopman_matrix(net: FiniteKoopmanLifecycleNet) -> tuple[tuple[Scalar, ...], ...]:
    """Independent generic linear-operator representation of the same lift."""
    z = Fraction(0)
    o = Fraction(1)
    lam2 = net.lam * net.lam
    return (
        (o, z, z, z, z, z),
        (z, net.lam, z, z, z, z),
        (z, z, o, z, z, z),
        (z, z, z, net.lam, z, z),
        (z, z, z, z, lam2, z),
        (net.bias, z, net.context_drive, net.nu, net.rho, net.mu),
    )


def _generic_matvec(matrix: tuple[tuple[Scalar, ...], ...], vector: Lift) -> Lift:
    out: list[Scalar] = []
    for row in matrix:
        total = Fraction(0)
        for coefficient, value in zip(row, vector):
            if coefficient != 0 and value != 0:
                total += coefficient * value
        out.append(total)
    return tuple(out)


def generic_propagate(
    net: FiniteKoopmanLifecycleNet, q: Lift, depth: int
) -> Lift:
    matrix = generic_koopman_matrix(net)
    for _ in range(depth):
        q = _generic_matvec(matrix, q)
    return q


def receipt_initial_basis(old_pod: Scalar, new_pod: Scalar) -> tuple[Lift, Lift]:
    """Factor the lifted edit into a shared constant term plus context term.

    phi(c, p_new, y)-phi(c, p_old, y) = A_0 + c*B_0.
    """
    delta = new_pod - old_pod
    delta_square = new_pod * new_pod - old_pod * old_pod
    a0: Lift = (
        Fraction(0),
        delta,
        Fraction(0),
        Fraction(0),
        delta_square,
        Fraction(0),
    )
    b0: Lift = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        delta,
        Fraction(0),
        Fraction(0),
    )
    return a0, b0


def candidate_compile_receipt(
    net: FiniteKoopmanLifecycleNet,
    old_pod: Scalar,
    new_pod: Scalar,
    depth: int,
) -> tuple[Lift, Lift, int]:
    """Compile one exact state-dependent receipt for the whole session fleet."""
    a0, b0 = receipt_initial_basis(old_pod, new_pod)
    a_depth = candidate_propagate(net, a0, depth)
    b_depth = candidate_propagate(net, b0, depth)
    work = 2 * depth * LIFT_STEP_MULTIPLIES
    return a_depth, b_depth, work


def generic_compile_receipt(
    net: FiniteKoopmanLifecycleNet,
    old_pod: Scalar,
    new_pod: Scalar,
    depth: int,
) -> tuple[Lift, Lift, int]:
    """Generic Koopman/sparse-linear baseline with no lifecycle semantics."""
    a0, b0 = receipt_initial_basis(old_pod, new_pod)
    a_depth = generic_propagate(net, a0, depth)
    b_depth = generic_propagate(net, b0, depth)
    work = 2 * depth * LIFT_STEP_MULTIPLIES
    return a_depth, b_depth, work


def apply_receipt(old_final: Lift, a_depth: Lift, b_depth: Lift) -> Lift:
    context = old_final[2]
    return tuple(
        old_value + a_value + context * b_value
        for old_value, a_value, b_value in zip(old_final, a_depth, b_depth)
    )


def original_fresh_lift(
    net: FiniteKoopmanLifecycleNet,
    context: Scalar,
    pod: Scalar,
    y0: Scalar,
    depth: int,
) -> Lift:
    x, c, y = net.fresh(context=context, pod=pod, y0=y0, depth=depth)
    return (Fraction(1), x, c, x * c, x * x, y)


def registered_cells(
    seeds: Iterable[int] = range(16),
    depths: Iterable[int] = (1, 2, 4, 8, 16, 32),
    sessions_per_cell: int = 64,
) -> dict:
    seeds = tuple(seeds)
    depths = tuple(depths)
    old_pod = Fraction(1, 5)
    revised_pod = Fraction(3, 7)
    deleted_pod = Fraction(0)

    # Sequential lifecycle includes an update, deletion, restoration and ABA
    # return to the original Pod value.
    lifecycle = (
        ("UPDATE", old_pod, revised_pod),
        ("DELETE", revised_pod, deleted_pod),
        ("RESTORE", deleted_pod, revised_pod),
        ("ABA", revised_pod, old_pod),
    )

    total_cases = 0
    candidate_fresh_mismatches = 0
    generic_fresh_mismatches = 0
    candidate_generic_receipt_mismatches = 0
    candidate_generic_state_mismatches = 0
    candidate_generic_work_mismatches = 0
    material_cases = 0
    state_dependent_edit_cells = 0
    candidate_work = 0
    generic_work = 0
    replay_work = 0
    candidate_slots = 0
    generic_slots = 0
    cells = []

    for seed in seeds:
        net = make_net(seed)
        for depth in depths:
            contexts = tuple(Fraction(i + 1, 67) for i in range(sessions_per_cell))
            y0_values = tuple(Fraction(i + 2, 71) for i in range(sessions_per_cell))
            current_states = [
                original_fresh_lift(net, c, old_pod, y0, depth)
                for c, y0 in zip(contexts, y0_values)
            ]

            cell_cases = 0
            cell_material = 0
            cell_state_dependent = 0
            cell_candidate_work = 0
            cell_replay_work = 0

            for op, from_pod, to_pod in lifecycle:
                cand_a, cand_b, cand_compile_work = candidate_compile_receipt(
                    net, from_pod, to_pod, depth
                )
                gen_a, gen_b, gen_compile_work = generic_compile_receipt(
                    net, from_pod, to_pod, depth
                )
                if cand_a != gen_a or cand_b != gen_b:
                    candidate_generic_receipt_mismatches += 1
                if cand_compile_work != gen_compile_work:
                    candidate_generic_work_mismatches += 1

                # B_y != 0 means one shared receipt produces context-dependent
                # hidden-state corrections across the fleet.
                if cand_b[5] != 0:
                    hidden_deltas = {
                        cand_a[5] + context * cand_b[5] for context in contexts
                    }
                    if len(hidden_deltas) == sessions_per_cell:
                        state_dependent_edit_cells += 1
                        cell_state_dependent += 1

                candidate_work += cand_compile_work
                generic_work += gen_compile_work
                cell_candidate_work += cand_compile_work

                next_states: list[Lift] = []
                for session_id, (context, y0, old_state) in enumerate(
                    zip(contexts, y0_values, current_states)
                ):
                    candidate_state = apply_receipt(old_state, cand_a, cand_b)
                    generic_state = apply_receipt(old_state, gen_a, gen_b)
                    fresh_state = original_fresh_lift(
                        net, context, to_pod, y0, depth
                    )
                    total_cases += 1
                    cell_cases += 1
                    if candidate_state != fresh_state:
                        candidate_fresh_mismatches += 1
                    if generic_state != fresh_state:
                        generic_fresh_mismatches += 1
                    if candidate_state != generic_state:
                        candidate_generic_state_mismatches += 1

                    if candidate_state[5] != old_state[5]:
                        material_cases += 1
                        cell_material += 1
                    next_states.append(candidate_state)

                # Apply shared B_d to one preserved context scalar per session.
                session_work = sessions_per_cell * SESSION_RECEIPT_MULTIPLIES
                candidate_work += session_work
                generic_work += session_work
                cell_candidate_work += session_work

                this_replay_work = (
                    sessions_per_cell * depth * FRESH_STEP_MULTIPLIES
                )
                replay_work += this_replay_work
                cell_replay_work += this_replay_work
                current_states = next_states

            candidate_slots += sessions_per_cell * LIFT_DIM
            generic_slots += sessions_per_cell * LIFT_DIM
            cells.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "sessions": sessions_per_cell,
                    "lifecycle_cases": cell_cases,
                    "material_cases": cell_material,
                    "state_dependent_edit_cells": cell_state_dependent,
                    "candidate_normalized_work": cell_candidate_work,
                    "fresh_replay_normalized_work": cell_replay_work,
                    "replay_to_candidate_advantage": (
                        cell_replay_work / cell_candidate_work
                    ),
                }
            )

    return {
        "cells": cells,
        "totals": {
            "registered_cells": len(cells),
            "lifecycle_cases": total_cases,
            "candidate_fresh_mismatches": candidate_fresh_mismatches,
            "generic_fresh_mismatches": generic_fresh_mismatches,
            "candidate_generic_receipt_mismatches": candidate_generic_receipt_mismatches,
            "candidate_generic_state_mismatches": candidate_generic_state_mismatches,
            "candidate_generic_work_mismatches": candidate_generic_work_mismatches,
            "material_cases": material_cases,
            "material_fraction": material_cases / total_cases,
            "state_dependent_edit_cells": state_dependent_edit_cells,
            "expected_state_dependent_edit_cells": len(cells) * len(lifecycle),
            "candidate_normalized_work": candidate_work,
            "generic_normalized_work": generic_work,
            "fresh_replay_normalized_work": replay_work,
            "candidate_to_generic_work_ratio": candidate_work / generic_work,
            "replay_to_candidate_advantage": replay_work / candidate_work,
            "candidate_slots": candidate_slots,
            "generic_slots": generic_slots,
            "candidate_to_generic_slot_ratio": candidate_slots / generic_slots,
            "min_cell_replay_advantage": min(
                cell["replay_to_candidate_advantage"] for cell in cells
            ),
            "max_cell_replay_advantage": max(
                cell["replay_to_candidate_advantage"] for cell in cells
            ),
        },
    }


def run_registered() -> dict:
    registered = registered_cells()
    totals = registered["totals"]
    kill = (
        totals["registered_cells"] == 96
        and totals["lifecycle_cases"] == 24576
        and totals["candidate_fresh_mismatches"] == 0
        and totals["generic_fresh_mismatches"] == 0
        and totals["candidate_generic_receipt_mismatches"] == 0
        and totals["candidate_generic_state_mismatches"] == 0
        and totals["candidate_generic_work_mismatches"] == 0
        and totals["material_fraction"] == 1.0
        and totals["state_dependent_edit_cells"]
        == totals["expected_state_dependent_edit_cells"]
        and totals["candidate_to_generic_work_ratio"] == 1.0
        and totals["candidate_to_generic_slot_ratio"] == 1.0
        and totals["replay_to_candidate_advantage"] > 8.0
    )
    return {
        "experiment": "E-000112",
        "title": "Finite Koopman Lift Reusable Revision Receipt Reduction",
        "kill_screen_pass": kill,
        "decision": (
            "KILL_FINITE_EXACT_KOOPMAN_CARLEMAN_LIFT_AS_STANDALONE_LIFECYCLE_TRANSPORT_NOVELTY"
            if kill
            else "DO_NOT_KILL"
        ),
        "major_invention": False,
        "scope": (
            "Exact finite observable lifting can create a useful one-edit reusable "
            "state-dependent fleet receipt, but when the identical lift/operator is "
            "given to a generic sparse-linear/Koopman engine it inherits identical "
            "state, memory, and mutation work. This is a standalone novelty kill, "
            "not a universal impossibility result for lifecycle-native architectures."
        ),
        "registered": registered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ci-e112"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result = run_registered()
    out = args.results_dir / "e000112-result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["registered"]["totals"], indent=2, sort_keys=True))
    print(result["decision"])


if __name__ == "__main__":
    main()
