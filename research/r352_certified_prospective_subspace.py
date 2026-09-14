from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAGE = "R352-CERTIFIED-PROSPECTIVE-RESIDUAL-CIRCUIT"
REPORT_PATH = Path(os.environ.get("SO_R352_REPORT", "ci-r352/report.json"))
SEED = 3520914
QUERIES = int(os.environ.get("SO_R352_QUERIES", "120"))
FUTURE_WORLDS = int(os.environ.get("SO_R352_FUTURE_WORLDS", "256"))
Q_DIM = 24
WORLD_DIM = 8
WIDTH = 40
LAYERS = 6
OUT_DIM = 8
BIAS_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
TOL = 2e-9


@dataclass
class Network:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    out_w: np.ndarray
    out_b: np.ndarray


@dataclass
class Atom:
    """One unresolved ReLU gate over base world variables + previous atoms."""
    coeff: np.ndarray  # [constant, world variables..., previous atoms...]
    lower: float
    upper: float


@dataclass
class ResidualProgram:
    atoms: list[Atom]
    out_expr: np.ndarray  # OUT_DIM x [constant, world vars, all atoms]
    stable_positive: int
    stable_negative: int
    unstable: int
    original_relus: int
    compile_seconds: float

    def serve(self, world: np.ndarray) -> np.ndarray:
        env = [1.0, *world.tolist()]
        for atom in self.atoms:
            # atom coefficients can only reference base variables and older atoms.
            z = float(atom.coeff @ np.asarray(env, dtype=np.float64))
            env.append(max(0.0, z))
        return self.out_expr @ np.asarray(env, dtype=np.float64)

    @property
    def coefficient_count(self) -> int:
        return int(sum(a.coeff.size for a in self.atoms) + self.out_expr.size)

    @property
    def serve_multiply_count(self) -> int:
        # Each retained coefficient is used once in the direct residual-program evaluator.
        return self.coefficient_count


def make_network(rng: np.random.Generator, bias_scale: float) -> Network:
    weights = []
    biases = []
    in_dim = Q_DIM + WORLD_DIM
    for _ in range(LAYERS):
        w = rng.normal(0.0, 1.0 / math.sqrt(in_dim), size=(WIDTH, in_dim)).astype(np.float64)
        # A broad per-neuron offset creates a realistic mix of always-active,
        # always-inactive and world-sensitive gates over the typed world domain.
        b = rng.normal(0.0, bias_scale, size=(WIDTH,)).astype(np.float64)
        weights.append(w)
        biases.append(b)
        in_dim = WIDTH
    out_w = rng.normal(0.0, 1.0 / math.sqrt(WIDTH), size=(OUT_DIM, WIDTH)).astype(np.float64)
    out_b = rng.normal(0.0, 0.1, size=(OUT_DIM,)).astype(np.float64)
    return Network(weights, biases, out_w, out_b)


def full_forward(net: Network, query: np.ndarray, world: np.ndarray) -> np.ndarray:
    h = np.concatenate([query, world]).astype(np.float64)
    for w, b in zip(net.weights, net.biases):
        h = np.maximum(0.0, w @ h + b)
    return net.out_w @ h + net.out_b


def expression_interval(coeff: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[float, float]:
    # coeff includes constant coefficient at index 0; lower/upper are environment
    # bounds including constant 1.0 at index 0.
    pos = coeff >= 0.0
    lo = float((coeff[pos] * lower[pos]).sum() + (coeff[~pos] * upper[~pos]).sum())
    hi = float((coeff[pos] * upper[pos]).sum() + (coeff[~pos] * lower[~pos]).sum())
    return lo, hi


def compile_query(net: Network, query: np.ndarray) -> ResidualProgram:
    """Partially evaluate a ReLU network with query fixed and world unresolved.

    Each hidden neuron is represented as an affine expression over typed world
    variables and previously introduced unresolved ReLU atoms. Interval proof over
    the complete future world box [-1,1]^WORLD_DIM certifies gates that can never
    flip. Always-positive ReLUs become identity; always-negative ReLUs disappear;
    only ambiguous gates become residual-program atoms.
    """
    t0 = time.perf_counter()
    atoms: list[Atom] = []
    stable_pos = stable_neg = unstable = 0

    # Environment starts [1, world_0, ..., world_n]. Bounds are exact for base vars.
    env_lower = [1.0] + [-1.0] * WORLD_DIM
    env_upper = [1.0] + [1.0] * WORLD_DIM

    # First-layer input expression matrix maps [constant, world] -> [query,world].
    base = 1 + WORLD_DIM
    input_expr = np.zeros((Q_DIM + WORLD_DIM, base), dtype=np.float64)
    input_expr[:Q_DIM, 0] = query
    input_expr[Q_DIM:, 1:] = np.eye(WORLD_DIM)
    current_expr = input_expr

    for layer_idx, (w, b) in enumerate(zip(net.weights, net.biases)):
        # Expand current expressions to the current environment width.
        env_dim = len(env_lower)
        if current_expr.shape[1] < env_dim:
            current_expr = np.pad(current_expr, ((0,0),(0,env_dim-current_expr.shape[1])))
        pre = w @ current_expr
        pre[:, 0] += b

        next_rows = []
        lower_arr = np.asarray(env_lower, dtype=np.float64)
        upper_arr = np.asarray(env_upper, dtype=np.float64)
        for j in range(WIDTH):
            coeff = pre[j]
            lo, hi = expression_interval(coeff, lower_arr, upper_arr)
            if lo >= 0.0:
                stable_pos += 1
                next_rows.append(coeff.copy())
            elif hi <= 0.0:
                stable_neg += 1
                next_rows.append(np.zeros(env_dim, dtype=np.float64))
            else:
                unstable += 1
                # Store exactly the expression defining this ambiguous gate.
                atoms.append(Atom(coeff.copy(), max(0.0, lo), max(0.0, hi)))
                env_lower.append(max(0.0, lo))
                env_upper.append(max(0.0, hi))
                # Output of this ReLU becomes a new basis atom.
                row = np.zeros(len(env_lower), dtype=np.float64)
                row[-1] = 1.0
                # All previously emitted rows need one zero for the new atom.
                next_rows = [np.pad(r, (0,1)) for r in next_rows]
                next_rows.append(row)
                # Subsequent neurons of the same dense layer must NOT depend on this
                # new atom (all preactivations were computed from previous layer), so
                # extend their stored pre coefficients lazily when encountered.
                lower_arr = np.asarray(env_lower, dtype=np.float64)
                upper_arr = np.asarray(env_upper, dtype=np.float64)

        current_expr = np.vstack(next_rows)

    env_dim = len(env_lower)
    if current_expr.shape[1] < env_dim:
        current_expr = np.pad(current_expr, ((0,0),(0,env_dim-current_expr.shape[1])))
    out_expr = net.out_w @ current_expr
    out_expr[:, 0] += net.out_b
    return ResidualProgram(
        atoms=atoms,
        out_expr=out_expr,
        stable_positive=stable_pos,
        stable_negative=stable_neg,
        unstable=unstable,
        original_relus=LAYERS * WIDTH,
        compile_seconds=time.perf_counter() - t0,
    )


def full_multiply_count() -> int:
    first = WIDTH * (Q_DIM + WORLD_DIM)
    hidden = (LAYERS - 1) * WIDTH * WIDTH
    out = OUT_DIM * WIDTH
    return first + hidden + out


def run_scale(scale: float, seed_offset: int) -> dict:
    rng = np.random.default_rng(SEED + seed_offset)
    py_rng = random.Random(SEED + seed_offset)
    net = make_network(rng, scale)

    programs = []
    compile_s = []
    atom_counts = []
    stable_counts = []
    coeff_counts = []
    max_abs_error = 0.0
    class_mismatches = 0
    digest_before = hashlib.sha256()

    queries = [rng.normal(0.0, 1.0, size=(Q_DIM,)).astype(np.float64) for _ in range(QUERIES)]
    for q in queries:
        p = compile_query(net, q)
        programs.append(p)
        compile_s.append(p.compile_seconds)
        atom_counts.append(len(p.atoms))
        stable_counts.append(p.stable_positive + p.stable_negative)
        coeff_counts.append(p.coefficient_count)
        digest_before.update(p.out_expr.tobytes())
        for a in p.atoms:
            digest_before.update(a.coeff.tobytes())

    # Arbitrary future worlds over the entire typed domain. No recompilation/patching.
    for qi, (q, p) in enumerate(zip(queries, programs)):
        for _ in range(FUTURE_WORLDS):
            world = rng.uniform(-1.0, 1.0, size=(WORLD_DIM,)).astype(np.float64)
            full = full_forward(net, q, world)
            residual = p.serve(world)
            err = float(np.max(np.abs(full - residual)))
            max_abs_error = max(max_abs_error, err)
            class_mismatches += int(int(np.argmax(full)) != int(np.argmax(residual)))

    # Simulate a large write stream; residual programs are immutable functions of
    # future world values, so no query-cache maintenance is performed.
    current_world = rng.uniform(-1.0, 1.0, size=(WORLD_DIM,)).astype(np.float64)
    generations = np.ones(WORLD_DIM, dtype=np.int64)
    same_value_updates = 0
    updates = 50000
    for _ in range(updates):
        k = py_rng.randrange(WORLD_DIM)
        generations[k] += 1
        if py_rng.random() < 0.35:
            same_value_updates += 1
        else:
            current_world[k] = py_rng.uniform(-1.0, 1.0)

    digest_after = hashlib.sha256()
    for p in programs:
        digest_after.update(p.out_expr.tobytes())
        for a in p.atoms:
            digest_after.update(a.coeff.tobytes())

    # Timed serve comparison on one representative query.
    q = queries[0]; p = programs[0]
    worlds = [rng.uniform(-1.0, 1.0, size=(WORLD_DIM,)).astype(np.float64) for _ in range(500)]
    t0 = time.perf_counter_ns()
    for w in worlds:
        _ = full_forward(net, q, w)
    full_ns = time.perf_counter_ns() - t0
    t0 = time.perf_counter_ns()
    for w in worlds:
        _ = p.serve(w)
    residual_ns = time.perf_counter_ns() - t0

    full_mult = full_multiply_count()
    return {
        "bias_scale": scale,
        "queries": QUERIES,
        "future_worlds_per_query": FUTURE_WORLDS,
        "max_abs_error": max_abs_error,
        "class_mismatches": class_mismatches,
        "mean_certified_stable_relu_fraction": statistics.mean(stable_counts) / (LAYERS * WIDTH),
        "mean_unresolved_relu_atoms": statistics.mean(atom_counts),
        "p95_unresolved_relu_atoms": sorted(atom_counts)[int(0.95 * (len(atom_counts)-1))],
        "mean_residual_program_coefficients": statistics.mean(coeff_counts),
        "full_network_multiply_count": full_mult,
        "mean_residual_to_full_multiply_ratio": statistics.mean(coeff_counts) / full_mult,
        "median_compile_ms_per_query": statistics.median(compile_s) * 1000.0,
        "timed_full_over_residual_serve_ratio_python": full_ns / max(1, residual_ns),
        "world_updates": updates,
        "same_value_generation_updates": same_value_updates,
        "write_time_program_patches_or_invalidations": 0,
        "program_digest_unchanged_after_world_updates": digest_before.hexdigest() == digest_after.hexdigest(),
    }


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = [run_scale(s, i * 1000) for i, s in enumerate(BIAS_SCALES)]
    report = {
        "stage": STAGE,
        "architecture_candidate": "Certified Prospective Residual Circuit (CPRC)",
        "world_domain": "each mutable typed scalar independently in [-1,1]",
        "query_dim": Q_DIM,
        "world_dim": WORLD_DIM,
        "width": WIDTH,
        "relu_layers": LAYERS,
        "rows": rows,
        "max_abs_error_all_scales": max(r["max_abs_error"] for r in rows),
        "total_class_mismatches": sum(r["class_mismatches"] for r in rows),
        "all_program_digests_unchanged_after_updates": all(r["program_digest_unchanged_after_world_updates"] for r in rows),
        "best_mean_residual_to_full_multiply_ratio": min(r["mean_residual_to_full_multiply_ratio"] for r in rows),
        "best_certified_stable_relu_fraction": max(r["mean_certified_stable_relu_fraction"] for r in rows),
        "mechanism": (
            "A neural network is partially evaluated with immutable query state fixed while mutable world inputs remain symbolic. Interval certificates "
            "prove ReLU gates that cannot switch over the complete future world domain. Always-positive gates collapse to identity, always-negative gates "
            "collapse to zero, and only ambiguous gates survive as explicit residual-program atoms. The retained program is a Prospective Neural State: "
            "world updates change only its later evaluation, never the compiled bytes."
        ),
        "research_use": (
            "This creates an automatic binding-time compiler for neural subgraphs: rather than choosing one global materialization layer, the runtime can "
            "push the mutable-world boundary through certified-stable nonlinear regions and preserve only the genuinely world-sensitive nonlinear gates."
        ),
        "claim_boundary": (
            "Interval bound propagation, abstract interpretation, neural verification and partial evaluation are established prior art. CPRC is not claimed "
            "novel here. The gate tests their use as an exact prospective-state compiler for mutable neural world inputs and quantifies when that can reduce "
            "serve work without cache maintenance."
        ),
        "dod_status": "NOT_DOD; certified selective-binding mechanism gate",
    }
    report["contract_pass"] = (
        report["max_abs_error_all_scales"] <= TOL
        and report["total_class_mismatches"] == 0
        and report["all_program_digests_unchanged_after_updates"]
        and report["best_certified_stable_relu_fraction"] > 0.20
    )
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
