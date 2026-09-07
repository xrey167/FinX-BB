from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

State = tuple[int, int]


@dataclass(frozen=True)
class EditAction:
    """Compact action g_(u,v,w)(a,b) = (a+u, b+v*a+w) over Z_p."""

    u: int
    v: int
    w: int

    def apply(self, state: State, p: int) -> State:
        a, b = state
        return ((a + self.u) % p, (b + self.v * a + self.w) % p)


@dataclass(frozen=True)
class TriangularNormalizerLayer:
    """Exact nonlinear bijection that normalizes the compact edit-action family.

    F(a,b) = (r*a+t, s*b + alpha*a^2 + beta*a + gamma) mod p
    with r,s != 0.
    """

    r: int
    s: int
    t: int
    alpha: int
    beta: int
    gamma: int

    def forward(self, state: State, p: int) -> State:
        a, b = state
        return (
            (self.r * a + self.t) % p,
            (
                self.s * b
                + self.alpha * a * a
                + self.beta * a
                + self.gamma
            )
            % p,
        )

    def inverse(self, state: State, p: int) -> State:
        A, B = state
        rinv = pow(self.r, -1, p)
        sinv = pow(self.s, -1, p)
        a = ((A - self.t) * rinv) % p
        q = (self.alpha * a * a + self.beta * a + self.gamma) % p
        b = ((B - q) * sinv) % p
        return a, b

    def candidate_push_action(self, action: EditAction, p: int) -> EditAction:
        """Analytic lifecycle transport through one nonlinear normalizer layer."""
        rinv = pow(self.r, -1, p)
        u_new = (self.r * action.u) % p
        v_new = (
            (self.s * action.v + 2 * self.alpha * action.u) * rinv
        ) % p
        w_new = (
            self.s * action.w
            + self.alpha * action.u * action.u
            + self.beta * action.u
            - v_new * self.t
        ) % p
        return EditAction(u_new, v_new, w_new)


@dataclass(frozen=True)
class NormalizerSuffix:
    p: int
    layers: tuple[TriangularNormalizerLayer, ...]

    def forward(self, state: State) -> State:
        for layer in self.layers:
            state = layer.forward(state, self.p)
        return state

    def inverse(self, state: State) -> State:
        for layer in reversed(self.layers):
            state = layer.inverse(state, self.p)
        return state

    def candidate_transport_action(
        self, action: EditAction
    ) -> tuple[EditAction, bool]:
        transported = action
        state_dependent_action_appeared = False
        for layer in self.layers:
            transported = layer.candidate_push_action(transported, self.p)
            if transported.v % self.p != 0:
                state_dependent_action_appeared = True
        return transported, state_dependent_action_appeared


def make_suffix(seed: int, p: int, depth: int) -> NormalizerSuffix:
    rng = random.Random((seed + 1) * 1_000_003 + p * 1009 + depth * 37 + 110)
    layers = []
    for _ in range(depth):
        layers.append(
            TriangularNormalizerLayer(
                r=rng.randrange(1, p),
                s=rng.randrange(1, p),
                t=rng.randrange(0, p),
                alpha=rng.randrange(1, p),
                beta=rng.randrange(0, p),
                gamma=rng.randrange(0, p),
            )
        )
    return NormalizerSuffix(p=p, layers=tuple(layers))


def generic_push_action_from_normalizer_metadata(
    layer: TriangularNormalizerLayer, action: EditAction, p: int
) -> EditAction:
    """Generic group-action/compiler baseline using the identical normalizer metadata.

    This function has no Pod or lifecycle semantics. It applies the induced
    automorphism of the compact group representation supplied by the layer.
    """
    inv_r = pow(layer.r, -1, p)
    next_u = (layer.r * action.u) % p
    next_v = ((layer.s * action.v + 2 * layer.alpha * action.u) * inv_r) % p
    next_w = (
        layer.s * action.w
        + layer.alpha * action.u * action.u
        + layer.beta * action.u
        - next_v * layer.t
    ) % p
    return EditAction(next_u, next_v, next_w)


def generic_transport_action(
    suffix: NormalizerSuffix, action: EditAction
) -> EditAction:
    transported = action
    for layer in suffix.layers:
        transported = generic_push_action_from_normalizer_metadata(
            layer, transported, suffix.p
        )
    return transported


def independently_fit_conjugated_action_by_probes(
    layer: TriangularNormalizerLayer, action: EditAction, p: int
) -> EditAction:
    """Independent exact audit.

    Fit the conjugated action F o g o F^-1 from black-box exact evaluations
    at output-space probes, then verify held-out probes. This does not call
    either analytic action-transport implementation.
    """

    def conjugated(y: State) -> State:
        x = layer.inverse(y, p)
        edited_x = action.apply(x, p)
        return layer.forward(edited_x, p)

    z0 = conjugated((0, 0))
    z1 = conjugated((1, 0))
    u = z0[0] % p
    if z1[0] % p != (1 + u) % p:
        raise AssertionError("conjugated action left registered group family")
    w = z0[1] % p
    v = (z1[1] - w) % p
    fitted = EditAction(u, v, w)

    for probe in ((0, 1), (2 % p, 3 % p), ((p - 1) % p, 4 % p)):
        if conjugated(probe) != fitted.apply(probe, p):
            raise AssertionError("probe-fitted conjugate failed held-out audit")
    return fitted


def independently_compile_transport_action(
    suffix: NormalizerSuffix, action: EditAction
) -> EditAction:
    transported = action
    for layer in suffix.layers:
        transported = independently_fit_conjugated_action_by_probes(
            layer, transported, suffix.p
        )
    return transported


def registered_normalizer_cells(
    seeds: Iterable[int] = range(16),
    primes: Iterable[int] = (11, 13, 17),
    depths: Iterable[int] = (2, 4, 8, 16),
    deltas: Iterable[int] = (1, 2, -1),
) -> dict:
    seeds = tuple(seeds)
    primes = tuple(primes)
    depths = tuple(depths)
    deltas = tuple(deltas)

    exact_state_cases = 0
    candidate_fresh_mismatches = 0
    generic_fresh_mismatches = 0
    candidate_generic_action_mismatches = 0
    independent_probe_action_mismatches = 0
    aba_failures = 0
    material_cases = 0
    state_dependent_action_edit_cells = 0
    edit_cells = 0

    candidate_group_transport_steps = 0
    generic_group_transport_steps = 0
    candidate_action_applications = 0
    generic_action_applications = 0
    suffix_recompute_layer_evals = 0

    min_cell_replay_to_candidate_normalized_work_ratio = float("inf")
    max_cell_replay_to_candidate_normalized_work_ratio = 0.0
    cells = []

    for seed in seeds:
        for p in primes:
            for depth in depths:
                suffix = make_suffix(seed, p, depth)
                inverse_failures = 0
                for a in range(p):
                    for b in range(p):
                        x = (a, b)
                        if suffix.inverse(suffix.forward(x)) != x:
                            inverse_failures += 1
                if inverse_failures:
                    raise AssertionError(
                        f"invertibility failure seed={seed} p={p} depth={depth}"
                    )

                cell_cases = 0
                cell_candidate_mismatches = 0
                cell_generic_mismatches = 0
                cell_state_dependent_edits = 0
                n_sessions = p * p

                for delta in deltas:
                    edit_cells += 1
                    read_action = EditAction(delta % p, 0, 0)

                    candidate_final_action, appeared = (
                        suffix.candidate_transport_action(read_action)
                    )
                    generic_final_action = generic_transport_action(
                        suffix, read_action
                    )
                    probe_final_action = independently_compile_transport_action(
                        suffix, read_action
                    )

                    inverse_read_action = EditAction((-delta) % p, 0, 0)
                    candidate_inverse_final_action, _ = (
                        suffix.candidate_transport_action(inverse_read_action)
                    )

                    if candidate_final_action != generic_final_action:
                        candidate_generic_action_mismatches += 1
                    if candidate_final_action != probe_final_action:
                        independent_probe_action_mismatches += 1
                    if appeared:
                        state_dependent_action_edit_cells += 1
                        cell_state_dependent_edits += 1

                    candidate_group_transport_steps += depth
                    generic_group_transport_steps += depth

                    replay_work = n_sessions * depth
                    candidate_normalized_work = depth + n_sessions
                    ratio = replay_work / candidate_normalized_work
                    min_cell_replay_to_candidate_normalized_work_ratio = min(
                        min_cell_replay_to_candidate_normalized_work_ratio, ratio
                    )
                    max_cell_replay_to_candidate_normalized_work_ratio = max(
                        max_cell_replay_to_candidate_normalized_work_ratio, ratio
                    )

                    for a in range(p):
                        for b in range(p):
                            read_state = (a, b)
                            old_final = suffix.forward(read_state)
                            fresh_final = suffix.forward(
                                read_action.apply(read_state, p)
                            )
                            candidate_final = candidate_final_action.apply(
                                old_final, p
                            )
                            generic_final = generic_final_action.apply(
                                old_final, p
                            )

                            exact_state_cases += 1
                            cell_cases += 1
                            candidate_action_applications += 1
                            generic_action_applications += 1
                            suffix_recompute_layer_evals += depth

                            if candidate_final != fresh_final:
                                candidate_fresh_mismatches += 1
                                cell_candidate_mismatches += 1
                            if generic_final != fresh_final:
                                generic_fresh_mismatches += 1
                                cell_generic_mismatches += 1
                            if fresh_final != old_final:
                                material_cases += 1

                            restored = candidate_inverse_final_action.apply(
                                candidate_final, p
                            )
                            if restored != old_final:
                                aba_failures += 1

                cells.append(
                    {
                        "seed": seed,
                        "p": p,
                        "depth": depth,
                        "sessions": n_sessions,
                        "edits": len(deltas),
                        "cases": cell_cases,
                        "candidate_fresh_mismatches": cell_candidate_mismatches,
                        "generic_fresh_mismatches": cell_generic_mismatches,
                        "state_dependent_action_edits": cell_state_dependent_edits,
                        "replay_layer_evals_per_edit": n_sessions * depth,
                        "candidate_normalized_work_per_edit": depth + n_sessions,
                        "generic_normalized_work_per_edit": depth + n_sessions,
                        "replay_to_candidate_normalized_work_ratio": (
                            n_sessions * depth / (depth + n_sessions)
                        ),
                    }
                )

    candidate_normalized_work = (
        candidate_group_transport_steps + candidate_action_applications
    )
    generic_normalized_work = (
        generic_group_transport_steps + generic_action_applications
    )
    replay_to_candidate_ratio = (
        suffix_recompute_layer_evals / candidate_normalized_work
    )

    return {
        "cells": cells,
        "totals": {
            "registered_cells": len(cells),
            "edit_cells": edit_cells,
            "exact_state_transport_cases": exact_state_cases,
            "candidate_fresh_mismatches": candidate_fresh_mismatches,
            "generic_fresh_mismatches": generic_fresh_mismatches,
            "candidate_generic_action_mismatches": (
                candidate_generic_action_mismatches
            ),
            "independent_probe_action_mismatches": (
                independent_probe_action_mismatches
            ),
            "aba_failures": aba_failures,
            "material_cases": material_cases,
            "material_fraction": material_cases / exact_state_cases,
            "state_dependent_action_edit_cells": (
                state_dependent_action_edit_cells
            ),
            "state_dependent_action_fraction": (
                state_dependent_action_edit_cells / edit_cells
            ),
            "candidate_group_transport_steps": candidate_group_transport_steps,
            "generic_group_transport_steps": generic_group_transport_steps,
            "candidate_action_applications": candidate_action_applications,
            "generic_action_applications": generic_action_applications,
            "candidate_normalized_work": candidate_normalized_work,
            "generic_normalized_work": generic_normalized_work,
            "candidate_to_generic_normalized_work_ratio": (
                candidate_normalized_work / generic_normalized_work
            ),
            "suffix_recompute_layer_evals": suffix_recompute_layer_evals,
            "replay_to_candidate_normalized_work_ratio": replay_to_candidate_ratio,
            "min_cell_replay_to_candidate_normalized_work_ratio": (
                min_cell_replay_to_candidate_normalized_work_ratio
            ),
            "max_cell_replay_to_candidate_normalized_work_ratio": (
                max_cell_replay_to_candidate_normalized_work_ratio
            ),
        },
    }


def run_registered() -> dict:
    normalizer = registered_normalizer_cells()
    totals = normalizer["totals"]
    kill = (
        totals["registered_cells"] == 192
        and totals["edit_cells"] == 576
        and totals["candidate_fresh_mismatches"] == 0
        and totals["generic_fresh_mismatches"] == 0
        and totals["candidate_generic_action_mismatches"] == 0
        and totals["independent_probe_action_mismatches"] == 0
        and totals["aba_failures"] == 0
        and totals["material_fraction"] == 1.0
        and totals["state_dependent_action_fraction"] == 1.0
        and totals["candidate_to_generic_normalized_work_ratio"] == 1.0
        and totals["min_cell_replay_to_candidate_normalized_work_ratio"] > 1.9
    )
    return {
        "experiment": "E-000110",
        "title": "Lifecycle Equivariance and Edit-Group Normalizer Reduction",
        "kill_screen_pass": kill,
        "decision": (
            "KILL_LIFECYCLE_EQUIVARIANCE_AND_GROUP_NORMALIZER_AS_STANDALONE_EXACT_TRANSPORT_ADVANTAGE"
            if kill
            else "DO_NOT_KILL"
        ),
        "major_invention": False,
        "normalizer_transport": normalizer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("ci-e110"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    result = run_registered()
    out = args.results_dir / "e000110-result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["normalizer_transport"]["totals"], indent=2, sort_keys=True))
    print(result["decision"])


if __name__ == "__main__":
    main()
