"""RSI001: revision-sufficient interface test, NOT a new repair mechanism.

The repaired interface deliberately does not retain the earlier numerical
context. All current state, model, canonical source and ID/generation lineage
are fixed within each pair. Adding a receipt/log changes that interface.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

ALPHABET = (-3, -1, 1, 3)
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32, "float64": torch.float64}


def tensor_bytes(x: torch.Tensor) -> bytes:
    return x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


def identical(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and tensor_bytes(a) == tensor_bytes(b)


def delta_write(s: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                beta: float) -> torch.Tensor:
    """DeltaNet recurrence, alpha=1. Operation ordering is explicit."""
    return s + torch.outer(k, beta * (v - s.T @ k))


def fraction_suffix(context: tuple[int, ...], erase: bool) -> tuple:
    x = [Fraction(0 if erase else z) for z in context]
    writes = [tuple(x)]
    for stage in range(3):
        # Invertible lower-triangular mixing followed by an injective rational
        # nonlinearity. Each stage is recorded, not merely its final answer.
        z = [x[i] + (x[i-1] if i else 0) + Fraction(stage+i+1, 8)
             for i in range(len(x))]
        x = [v/(1+abs(v)) for v in z]
        writes.append(tuple(x))
    return tuple(writes)


def encode_receipt(context: tuple[int, ...]) -> int:
    if len(context) != 4 or any(x not in ALPHABET for x in context):
        raise ValueError("Receipt demo requires four four-valued coordinates")
    return sum(ALPHABET.index(x) << (2*i) for i, x in enumerate(context))


def decode_receipt(code: int) -> tuple[int, ...]:
    if not isinstance(code, int) or isinstance(code, bool) or not 0 <= code < 256:
        raise ValueError("Receipt code must be one unsigned byte")
    return tuple(ALPHABET[(code >> (2*i)) & 3] for i in range(4))


def exact_screen() -> dict:
    contexts = list(itertools.product(ALPHABET, repeat=4))
    interfaces = [fraction_suffix(c, True) for c in contexts]
    targets = [fraction_suffix(c, False) for c in contexts]
    fixed_receipt = all(fraction_suffix(decode_receipt(encode_receipt(c)), False) == target
                        for c, target in zip(contexts, targets))
    # This control is invertible in exact arithmetic; beta need not be zero.
    half = [tuple(Fraction(x, 2) for x in c) for c in contexts]
    assert len(set(half)) == len(contexts)
    assert all(tuple(2*x for x in row) == c for row, c in zip(half, contexts))
    assert len(set(interfaces)) == 1 and len(set(targets)) == 256 and fixed_receipt
    assert len(set(target[-1] for target in targets)) == 256
    return dict(histories=256, distinct_current_interfaces=1,
                distinct_rebuild_trajectories=256, distinct_final_rebuild_states=256,
                minimum_auxiliary_bits=8, exact_packed_receipt_bits=8,
                receipt_repairs_all=fixed_receipt,
                rational_half_gate_distinct_states=len(set(half)),
                formula_lower_bound="ceil(log2(distinct targets in a fixed-interface fiber))",
                new_theorem_claim=False)


def setup(seed: int):
    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=.25, size=(3, 8, 8))
    feedback = rng.normal(scale=.2, size=(3, 8, 8))
    inputs = rng.normal(scale=.2, size=(4, 3, 8))
    a = rng.integers(-6, 7, size=8).astype(np.float64)/4
    b = a+1
    return tuple(torch.tensor(x, dtype=torch.float64) for x in (weights, feedback, inputs, a, b))


def neural_build(context: torch.Tensor, weights: torch.Tensor,
                 feedback: torch.Tensor, inputs: torch.Tensor,
                 erase: bool):
    if context.shape != (8,) or context.dtype != torch.float64:
        raise ValueError("Context must be eight float64 coordinates")
    state = torch.zeros((3, 4, 8), dtype=torch.float64)
    state[0, 0] = context
    keys = torch.eye(4, dtype=torch.float64)
    q = torch.full((4,), .25, dtype=torch.float64)
    if erase:
        state[0] = delta_write(state[0], keys[0], torch.zeros(8, dtype=torch.float64), 1.)
    boundary = state.clone()
    history = []
    for step in range(4):
        for layer in range(3):
            below = inputs[step, 0] if layer == 0 else state[layer-1].T @ q
            own = state[layer].T @ q
            value = torch.tanh(weights[layer] @ below + feedback[layer] @ own + inputs[step, layer])
            state[layer] = delta_write(state[layer], keys[step], value, .25)
        history.append(state.clone())
    return boundary, torch.stack(history), state.clone()


def numpy_build(context, weights, feedback, inputs, erase):
    """Separate implementation; tolerant cross-library comparison is diagnostic."""
    context, weights, feedback, inputs = [np.asarray(x) for x in (context, weights, feedback, inputs)]
    s = np.zeros((3, 4, 8), dtype=np.float64)
    s[0, 0] = context
    if erase:
        s[0, 0] = 0  # Equivalent exact coordinate-key overwrite, independently expressed.
    boundary = s.copy()
    history = []
    for step in range(4):
        for layer in range(3):
            below = inputs[step, 0] if layer == 0 else s[layer-1].mean(axis=0)
            own = s[layer].mean(axis=0)
            value = np.tanh(weights[layer].dot(below) + feedback[layer].dot(own) + inputs[step, layer])
            s[layer, step] += .25*(value-s[layer, step])
        history.append(s.copy())
    return boundary, np.stack(history), s.copy()


def interface_bytes(current: torch.Tensor, seed: int) -> bytes:
    # Identity/edge lineage is complete for this constructed program, but is not
    # a hidden encoding of old activation VALUES. Changing this assumption
    # (e.g. adding a recoverable raw-prefix reference) changes the theorem input.
    metadata = dict(model_seed=seed, source_id="canonical-pod", generation=7,
                    old_payload=dict(key=0, beta=1, value=[0]*8), revision="OMIT",
                    prefix_event_id="context-arrival", prefix_generation=1,
                    dependency_edges=[("context-arrival", "initial-0"), ("initial-0", "source-write")]+
                    [("source-write" if t == 0 and l == 0 else
                      f"initial-{l}" if t == 0 else f"state-{t-1}-{l}",
                      f"state-{t}-{l}") for t in range(4) for l in range(3)]+
                    [(f"state-{t}-{l-1}", f"state-{t}-{l}") for t in range(4) for l in (1,2)]+
                    [(f"exogenous-{t}-{l}", f"state-{t}-{l}") for t in range(4) for l in range(3)])
    return json.dumps(metadata, sort_keys=True).encode()+tensor_bytes(current)


def neural_screen(seed: int) -> dict:
    w, f, x, a, b = setup(seed)
    old_a = neural_build(a, w, f, x, True)
    old_b = neural_build(b, w, f, x, True)
    fresh_a = neural_build(a, w, f, x, False)
    fresh_b = neural_build(b, w, f, x, False)
    assert all(identical(u, v) for u, v in zip(old_a, old_b))
    assert interface_bytes(old_a[-1], seed) == interface_bytes(old_b[-1], seed)
    assert not identical(fresh_a[-1], fresh_b[-1])
    repeated = neural_build(a, w, f, x, True)
    assert all(identical(u,v) for u,v in zip(old_a, repeated))
    # A conventional reduced checkpoint contains the erased row. Known zero
    # background makes this sufficient; arbitrary background would cost extra.
    repairs = []
    independent_maxabs = 0.
    for c, target in [(a, fresh_a), (b, fresh_b)]:
        receipt = c.clone()  # Saved BEFORE the destructive write, not inferred now.
        repaired = neural_build(receipt, w, f, x, False)
        assert all(identical(u,v) for u,v in zip(repaired, target))
        independent = numpy_build(c.numpy(), w.numpy(), f.numpy(), x.numpy(), False)
        discrepancy = max(float(np.max(np.abs(u.numpy()-v))) for u,v in zip(target, independent))
        independent_maxabs = max(independent_maxabs, discrepancy)
        assert discrepancy < 1e-12  # Operator cross-check only, NOT the repair acceptance gate.
        repairs.append(True)
    partial = old_a[-1].clone()
    partial[0] = fresh_a[-1][0]
    partial_error = float(torch.max(torch.abs(partial-fresh_a[-1])))
    assert partial_error > 1e-8
    rng = np.random.default_rng(10000+seed)
    lens = torch.tensor(rng.normal(size=(7,96)), dtype=torch.float64)
    audit_a = torch.tanh(lens @ old_a[-1].reshape(-1))
    audit_b = torch.tanh(lens @ old_b[-1].reshape(-1))
    assert identical(audit_a, audit_b)
    return dict(seed=seed, contexts=[a.tolist(), b.tolist()],
        full_interface_byte_identical=True,
        old_all_write_trajectories_identical=True, repeat_forward_exact=True,
        distinct_never_final_states=True,
        never_write_maxabs=float(torch.max(torch.abs(fresh_a[1]-fresh_b[1]))),
        never_final_maxabs=float(torch.max(torch.abs(fresh_a[-1]-fresh_b[-1]))),
        final_layer_maxabs=[float(torch.max(torch.abs(fresh_a[-1][l]-fresh_b[-1][l]))) for l in range(3)],
        exact_receipt_plus_replay=repairs,
        stale_upper_state_maxabs=partial_error,
        numpy_torch_operator_crosscheck_maxabs=independent_maxabs,
        present_state_readouts_identical=True, jlens_implemented=False,
        stored_receipt_array_bytes=64, current_state_array_bytes=768,
        downstream_replayed_writes=12, known_zero_background=True,
        pretrained_backbone=False)


def finite_screen(dtype_name: str) -> dict:
    dtype = DTYPES[dtype_name]
    value = torch.tensor(1., dtype=dtype)
    values = []
    for _ in range(256):
        values.append(value.clone())
        value = torch.nextafter(value, torch.tensor(float("inf"), dtype=dtype))
    inputs = torch.stack(values)
    outputs = inputs + .5*(torch.ones_like(inputs)-inputs)
    groups = defaultdict(list)
    for i, y in enumerate(outputs):
        groups[tensor_bytes(y)].append(i)
    colliding = [indices for indices in groups.values() if len(indices)>1]
    assert len({tensor_bytes(v) for v in inputs}) == 256 and colliding
    i, j = colliding[0][:2]
    exact = [(Fraction.from_float(float(v))+1)/2 for v in inputs]
    assert len(set(exact)) == 256
    restored = 2*outputs-1
    assert not identical(restored, inputs)
    return dict(dtype=dtype_name, inputs=256, distinct_float_outputs=len(groups),
        colliding_output_buckets=len(colliding), maximum_preimages=max(map(len,groups.values())),
        first_collision_input_indices=[i,j],
        first_collision_input_hex=[float(inputs[i]).hex(),float(inputs[j]).hex()],
        identical_output_hex=float(outputs[i]).hex(),
        first_collision_target_gap=float(inputs[j])-float(inputs[i]),
        rational_outputs_distinct=256, gate=.5, all_inputs_normal_positive=True,
        inverse_reconstructs_all_exactly=False,
        low_precision_deployment_failure_frequency="NOT_MEASURED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("rsi001-results.json"))
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = dict(experiment="RSI-001", status="interface_falsification_not_invention",
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exact=exact_screen(), neural=[neural_screen(s) for s in range(5)],
        finite=[finite_screen(d) for d in DTYPES],
        full_system_gates="NOT_EVALUATED", trained_backbones=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(exact=result["exact"],
        neural_pairs=len(result["neural"]), finite=result["finite"]),indent=2))

if __name__ == "__main__":
    main()
